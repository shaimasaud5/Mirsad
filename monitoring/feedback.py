"""
Admin-only feedback analytics. Definitions (also shown in the dashboard):

  eligible answers  completed assistant answers (status 'ok') in the selected date window and execution kind
  rated             eligible answers that CURRENTLY carry a rating (a changed or cleared rating is not double counted)
  Helpful %         helpful / rated        (unrated answers are never counted as helpful)
  coverage          rated / eligible

An answer can involve several model calls (tool loop). Those calls share one turn_id in `requests`; the rating attaches
to the answer, so it is counted once and the turn's calls are summarised, never multiplied.
User ratings are a signal for investigation, not proof of accuracy or of routing quality.
"""
import json
import sqlite3
import time

import pandas as pd

LOW_SAMPLE = 10
REASON_KEYS = ("incorrect", "incomplete", "misunderstood", "other", "not_specified")
LEGACY = "legacy (unknown)"
EXCERPT = 160


def _kind_ok(mode: str, kind: str) -> bool:
    if kind == "all":
        return True
    return mode.startswith("legacy") if kind == "legacy" else mode == kind


def _turns(conn: sqlite3.Connection) -> dict[str, dict]:
    df = pd.read_sql_query(
        "SELECT request_id, ts, turn_id, status, execution_mode, routing_mode, route_source, llm_route, classifier_model, "
        "route, model, latency_ms FROM requests WHERE turn_id IS NOT NULL ORDER BY ts, id", conn)
    out = {}
    for turn, g in df.groupby("turn_id", sort=False):
        first = g.iloc[0]
        val = lambda c: None if pd.isna(first[c]) else first[c]  # noqa: E731
        out[turn] = {
            "execution_mode": val("execution_mode") or LEGACY, "routing_mode": val("routing_mode"),
            "route_source": val("route_source"), "llm_route": val("llm_route"), "classifier_model": val("classifier_model"),
            "route": val("route"), "model": val("model"), "calls": int(len(g)),
            "models": sorted({str(m) for m in g["model"].dropna()}),
            "request_ids": [str(r) for r in g["request_id"]],
        }
    return out


def _stat(rated: list[dict], eligible: int) -> dict:
    helpful = sum(1 for r in rated if r["rating"] == "helpful")
    n = len(rated)
    return {"eligible": eligible, "rated": n, "helpful": helpful, "not_helpful": n - helpful,
            "helpful_pct": (helpful / n * 100) if n else None, "coverage_pct": (n / eligible * 100) if eligible else None,
            "low_sample": n < LOW_SAMPLE}


def _breakdown(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key) or "unknown"), []).append(r)
    out = []
    for k, g in sorted(groups.items()):
        st = _stat([r for r in g if r["rating"]], len(g))
        out.append({"key": k, **st})
    return out


def _excerpt(text: str | None) -> str:
    t = " ".join(str(text or "").split())
    return t[:EXCERPT] + ("…" if len(t) > EXCERPT else "")


def _question(conn: sqlite3.Connection, conversation_id: str, message_id: int) -> str:
    r = conn.execute("SELECT content FROM messages WHERE conversation_id=? AND role='user' AND id<? ORDER BY id DESC LIMIT 1",
                     (conversation_id, message_id)).fetchone()
    return r[0] if r else ""


def report(conn: sqlite3.Connection, *, minutes: float | None = None, kind: str = "live", rating: str | None = None,
           reason: str | None = None, limit: int = 100) -> dict:
    """Summary, breakdowns and the feedback list under ONE set of filters (date window + execution kind)."""
    since = time.time() - minutes * 60 if minutes else 0.0
    turns = _turns(conn)
    rows = []
    for mid, cid, uid, turn, ts, content, rt, rs, cm, upd in conn.execute(
            "SELECT m.id, m.conversation_id, m.user_id, m.turn_id, m.ts, m.content, f.rating, f.reason, f.comment, f.updated_at "
            "FROM messages m LEFT JOIN message_feedback f ON f.message_id=m.id "
            "WHERE m.role='assistant' AND m.status='ok' AND m.ts>=? ORDER BY m.id DESC", (since,)):
        t = turns.get(turn) or {}
        rows.append({"message_id": mid, "conversation_id": cid, "user_id": uid, "turn_id": turn, "answered_at": ts,
                     "answer": content, "rating": rt, "reason": rs, "comment": cm, "rated_at": upd,
                     "execution_mode": t.get("execution_mode", LEGACY), "route": t.get("route"), "model": t.get("model"),
                     "llm_route": t.get("llm_route"), "route_source": t.get("route_source"),
                     "routing_mode": t.get("routing_mode"), "classifier_model": t.get("classifier_model"),
                     "calls": t.get("calls", 0)})

    by_exec = {}
    for r in rows:                                        # before the kind filter: lets the admin see what is being excluded
        m = "legacy" if r["execution_mode"].startswith("legacy") else r["execution_mode"]
        e = by_exec.setdefault(m, {"eligible": 0, "rated": 0})
        e["eligible"] += 1
        e["rated"] += 1 if r["rating"] else 0

    rows = [r for r in rows if _kind_ok(r["execution_mode"], kind)]
    rated = [r for r in rows if r["rating"]]
    reasons = dict.fromkeys(REASON_KEYS, 0)
    for r in rated:
        if r["rating"] == "not_helpful":
            reasons[r["reason"] or "not_specified"] += 1

    listed = [r for r in rated if (not rating or r["rating"] == rating)
              and (not reason or (r["rating"] == "not_helpful" and (r["reason"] or "not_specified") == reason))]
    listed.sort(key=lambda r: r["rated_at"] or 0, reverse=True)
    items = []
    for r in listed[:max(1, min(int(limit), 500))]:
        items.append({**{k: r[k] for k in ("message_id", "conversation_id", "rating", "reason", "comment", "rated_at", "answered_at",
                                           "execution_mode", "route", "model", "llm_route", "route_source", "routing_mode", "calls")},
                      "question": _excerpt(_question(conn, r["conversation_id"], r["message_id"])),
                      "answer": _excerpt(r["answer"])})
    return {"definitions": {"kind": kind, "minutes": minutes, "low_sample_below": LOW_SAMPLE},
            "summary": {**_stat(rated, len(rows)), "unrated": len(rows) - len(rated)},
            "reasons": reasons, "by_execution": by_exec,
            "by_model": _breakdown(rows, "model"), "by_route": _breakdown(rows, "route"),
            "items": items, "items_total": len(listed)}


def detail(conn: sqlite3.Connection, message_id: int) -> dict | None:
    """One rated (or unrated) answer with its question, the turn's model calls and who asked. Admin use only."""
    r = conn.execute(
        "SELECT m.id, m.conversation_id, m.user_id, m.turn_id, m.ts, m.content, m.steps_json, m.files_json, u.email, "
        "f.rating, f.reason, f.comment, f.created_at, f.updated_at FROM messages m "
        "LEFT JOIN app_users u ON u.user_id=m.user_id LEFT JOIN message_feedback f ON f.message_id=m.id "
        "WHERE m.id=? AND m.role='assistant'", (message_id,)).fetchone()
    if r is None:
        return None
    mid, cid, uid, turn, ts, answer, steps, files, email, rating, reason, comment, created, updated = r
    calls = []
    if turn:
        df = pd.read_sql_query(
            "SELECT request_id, ts, status, execution_mode, routing_mode, route_source, llm_route, classifier_model, route, model, "
            "classification_reused, latency_ms, error FROM requests WHERE turn_id=? ORDER BY ts, id", conn, params=(turn,))
        calls = json.loads(df.astype(object).where(df.notna(), None).to_json(orient="records"))
    return {"message_id": mid, "conversation_id": cid, "user": email or uid, "turn_id": turn, "answered_at": ts,
            "question": _question(conn, cid, mid), "answer": answer, "files": json.loads(files or "[]"),
            "steps": json.loads(steps or "[]"), "calls": calls,
            "feedback": {"rating": rating, "reason": reason, "comment": comment, "created_at": created, "updated_at": updated}
            if rating else None}
