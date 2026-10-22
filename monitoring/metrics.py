"""
Metric calculations used by the dashboard (and the tests). Three different things are counted, never mixed:

  user turn        one submitted user message and its whole agent run  (table `turns`; legacy: `messages`)
  model call       one request to the answering model                  (table `requests`, one row per call record)
  provider attempt one HTTP call to the provider inside a model call   (table `provider_attempts`; failover/retry)

Grouping of calls into turns uses the verified (user_id, turn_id) pair only - never prompt text or timestamps.
Records without both ids (written before turn tracking) cannot be grouped: they are excluded from turn metrics and
counted as `ungroupable_calls`.

Token usage kinds (column `usage_kind`):
  provider  both counts reported by the provider's usage block      -> the only kind that is summed into token/cost totals
  mock      simulated numbers from MIRSAD_MOCK                        -> reported separately, never priced
  unknown   a successful call whose usage is missing / not trustworthy -> counted, never turned into 0
  none      failed or blocked call: its consumption is unknown       -> counted, never invented
Local classifier (Qwen/Ollama) time is reported separately; its token usage is not recorded at all.

Costs are ESTIMATED inference cost at list price (config.MODEL_PRICES), not an invoice. See proxy/cost.py.

Percentiles: linear interpolation between closest ranks (numpy default, "type 7"), over the samples shown as `n`.
"""
import sqlite3
import time

import numpy as np
import pandas as pd

import config
from proxy import cost
from proxy.cost import savings_percent

LEGACY = "legacy (unknown)"
NO_CALL = "no model call"             # a turn that never reached the proxy (no call record exists)
TURN_OUTCOMES = ("completed", "failed", "blocked", "step_limit", "unknown")
MESSAGE_OUTCOME = {"ok": "completed", "error": "failed", "blocked": "blocked", "step_limit": "step_limit"}
LOW_SAMPLE = 20                       # a p95 over fewer samples is flagged as indicative only

_NUMERIC = ("input_tokens", "output_tokens", "latency_ms", "model_call_ms", "classifier_ms", "est_cost_usd",
            "baseline_big_usd", "provider_attempts", "classification_reused", "failover_used")
_TEXT = ("user_id", "turn_id", "token_source", "model", "route", "route_source", "pricing_version", "status")


def _connect(path):
    return sqlite3.connect(f"file:{path or config.DB_PATH}?mode=ro", uri=True, timeout=5)


def load(db_path=None, since_minutes: float | None = None, user_id: str | None = None) -> pd.DataFrame:
    """Call records. Time basis: when the call was recorded (`ts`).
    user_id=None -> every row (global/admin view); otherwise ONLY that user's rows.
    Legacy rows (user_id NULL) are therefore visible to the global view only."""
    try:
        conn = _connect(db_path)
        if user_id is None:
            df = pd.read_sql_query("SELECT * FROM requests ORDER BY ts", conn)
        else:
            df = pd.read_sql_query("SELECT * FROM requests WHERE user_id = ? ORDER BY ts", conn, params=(user_id,))
        conn.close()
    except Exception:
        return pd.DataFrame()
    if since_minutes and not df.empty:
        df = df[df["ts"] >= time.time() - since_minutes * 60]
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        # rows written before the routing migration have no execution_mode: keep them, label them
        df["execution_mode"] = df["execution_mode"].fillna(LEGACY) if "execution_mode" in df else LEGACY
    return df


def load_attempts(db_path=None, limit: int | None = 50, request_ids=None) -> pd.DataFrame:
    """Provider attempts (credential alias, outcome, status, duration, failover). Never contains key values.
    request_ids restricts them to the selected call records, so attempts follow the same filters as the calls."""
    try:
        conn = _connect(db_path)
        sql = "SELECT * FROM provider_attempts ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        df = pd.read_sql_query(sql, conn)
        conn.close()
    except Exception:
        return pd.DataFrame()
    if request_ids is not None and not df.empty:
        df = df[df["request_id"].isin(set(request_ids))]
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
    return df


def filter_execution(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind: 'all' | 'live' | 'mock' | 'legacy'."""
    if df.empty or kind == "all" or "execution_mode" not in df:
        return df
    if kind == "legacy":
        return df[df["execution_mode"].str.startswith("legacy")]
    return df[df["execution_mode"] == kind]


# ---------------------------------------------------------------- statistics helpers
def stat(values) -> dict:
    """n / average / p50 / p95 of the finite values; None (not 0) when there are no samples."""
    v = pd.to_numeric(pd.Series(values, dtype="float64"), errors="coerce").dropna()
    v = v[np.isfinite(v)]
    if v.empty:
        return {"n": 0, "avg": None, "p50": None, "p95": None, "low_sample": True}
    return {"n": int(len(v)), "avg": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "low_sample": len(v) < LOW_SAMPLE}


def _floats(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype="float64")


def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in _NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    for c in _TEXT:
        if c not in d:
            d[c] = None
    if "execution_mode" not in d:
        d["execution_mode"] = LEGACY
    d["execution_mode"] = d["execution_mode"].fillna(LEGACY)
    return d


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived per-call columns: is_model_call, usage_kind, in_tok/out_tok, est_cost, base_cost, cost_basis,
    classify_ms, call_ms, call_ms_derived. Stored values are never modified."""
    if df.empty:
        return df
    d = _ensure(df)
    ok = d["status"] == "success"
    mock = d["execution_mode"] == "mock"
    # a call record that reached the answering model: success, or an error after at least one provider attempt
    # (error with provider_attempts == 0 = no key configured; NULL = legacy error rows, which all followed a call)
    d["is_model_call"] = ok | ((d["status"] == "error") & ~(d["provider_attempts"] == 0))

    both = d["input_tokens"].notna() & d["output_tokens"].notna()
    kind = pd.Series("none", index=d.index)
    kind[ok & mock] = "mock"
    kind[ok & ~mock] = "unknown"
    kind[ok & ~mock & (d["token_source"] == "estimate")] = "estimate"
    kind[ok & ~mock & (d["token_source"] == "provider") & both] = "provider"
    d["usage_kind"] = kind
    prov = kind == "provider"
    d["in_tok"] = d["input_tokens"].where(kind.isin(["provider", "mock"]))
    d["out_tok"] = d["output_tokens"].where(kind.isin(["provider", "mock"]))

    # cost: recorded estimate when present, else a CURRENT-RATE estimate of the same provider-reported tokens
    recorded = prov & d["est_cost_usd"].notna()
    est = d["est_cost_usd"].where(recorded)
    need = prov & ~recorded
    if need.any():
        est[need] = _floats(cost.call_cost(m, i, o) for m, i, o in zip(d.loc[need, "model"], d.loc[need, "in_tok"], d.loc[need, "out_tok"]))
    base = d["baseline_big_usd"].where(prov & d["baseline_big_usd"].notna())
    need_b = prov & base.isna()
    if need_b.any():
        base[need_b] = _floats(cost.baseline_cost(i, o) for i, o in zip(d.loc[need_b, "in_tok"], d.loc[need_b, "out_tok"]))
    d["est_cost"], d["base_cost"] = est.astype("float64"), base.astype("float64")
    d["cost_basis"] = np.where(recorded, "recorded", np.where(est.notna(), "current-rate", None))

    # durations (ms); each measures exactly one thing
    fresh = d["classifier_ms"].notna() & (d["classification_reused"].fillna(0) == 0)
    d["classify_ms"] = d["classifier_ms"].where(fresh)
    derived = d["latency_ms"] - d["classifier_ms"].where(fresh, 0.0)       # proxy time minus the classification inside it
    derived = derived.where(derived >= 0)
    d["call_ms_derived"] = ok & d["model_call_ms"].isna() & derived.notna()
    d["call_ms"] = d["model_call_ms"].where(ok).fillna(derived.where(ok))
    return d


# ---------------------------------------------------------------- call-level summary
def _usage_summary(d: pd.DataFrame) -> dict:
    prov, mock = d[d["usage_kind"] == "provider"], d[d["usage_kind"] == "mock"]
    return {"provider_calls": len(prov), "provider_input": int(prov["in_tok"].sum()), "provider_output": int(prov["out_tok"].sum()),
            "mock_calls": len(mock), "mock_input": int(mock["in_tok"].sum()), "mock_output": int(mock["out_tok"].sum()),
            "unknown_calls": int((d["usage_kind"].isin(["unknown", "estimate"])).sum()),
            "failed_calls_unknown": int((d["is_model_call"] & (d["status"] != "success")).sum()),
            "classifier_tokens": None}      # local classifier usage is not recorded (time is: see latency.classification)


def _cost_summary(d: pd.DataFrame) -> dict:
    prov = d[d["usage_kind"] == "provider"]
    priced = prov[prov["est_cost"].notna()]
    matched = prov[prov["est_cost"].notna() & prov["base_cost"].notna()]
    hyp, est_m = float(matched["base_cost"].sum()), float(matched["est_cost"].sum())
    diff = hyp - est_m
    by_model = []
    for m, g in prov.groupby(prov["model"].fillna("unknown")):
        p = cost.price_for(m)
        pr = g[g["est_cost"].notna()]
        by_model.append({"model": m, "calls": len(g), "input_tokens": int(g["in_tok"].sum()), "output_tokens": int(g["out_tok"].sum()),
                         "priced_calls": len(pr), "est_cost": float(pr["est_cost"].sum()) if len(pr) else None,
                         "input_price": p["input"] if p else None, "output_price": p["output"] if p else None})
    return {
        "currency": config.PRICING_CURRENCY, "unit": config.PRICING_UNIT, "pricing_version": config.PRICING_VERSION,
        "retrieved": config.PRICING_RETRIEVED, "source_url": config.PRICING_SOURCE_URL,
        "estimated_cost": float(priced["est_cost"].sum()) if len(priced) else None,
        "priced_calls": len(priced), "unpriced_calls": len(prov) - len(priced),
        "current_rate_calls": int((priced["cost_basis"] == "current-rate").sum()),
        "excluded_no_usage_calls": int((d["is_model_call"] & (d["usage_kind"] != "provider") & (d["usage_kind"] != "mock")).sum()),
        "mock_calls_unpriced": int((d["usage_kind"] == "mock").sum()),
        "baseline": {"model": config.BIG_MODEL, "matched_calls": len(matched), "estimated": est_m if len(matched) else None,
                     "hypothetical": hyp if len(matched) else None, "difference": diff if len(matched) else None,
                     "pct": savings_percent(diff, hyp) if len(matched) else None},
        "by_model": by_model,
    }


def _latency_summary(d: pd.DataFrame, attempts: pd.DataFrame | None) -> dict:
    out = {"classification": stat(d["classify_ms"]), "model_call": stat(d["call_ms"]),
           "model_call_derived": int(d["call_ms_derived"].sum()),
           "proxy_total": stat(d["latency_ms"].where(d["status"] == "success"))}
    if attempts is not None and not attempts.empty:
        a = attempts
        out["attempt"] = stat(a["duration_ms"])
        out["attempt_success"] = stat(a.loc[a["outcome"] == "success", "duration_ms"])
        # time spent in attempts that preceded the final one (failover / retry overhead), per request that needed >1 attempt
        multi = a.groupby("request_id").filter(lambda g: len(g) > 1)
        over = multi[multi["attempt_no"] < multi.groupby("request_id")["attempt_no"].transform("max")]
        out["failover_overhead"] = stat(over.groupby("request_id")["duration_ms"].sum())
    else:
        out["attempt"] = out["attempt_success"] = out["failover_overhead"] = stat([])
    return out


def summarize(df: pd.DataFrame, attempts: pd.DataFrame | None = None) -> dict:
    """Call-level summary of the selected call records. Turn metrics: summarize_turns()."""
    if df.empty:
        return {"total": 0}
    d = annotate(df)
    calls = d[d["is_model_call"]]
    ok = d[d["status"] == "success"]
    failed = calls[calls["status"] == "error"]
    known = calls[calls["provider_attempts"].notna() & (calls["execution_mode"] != "mock")]
    fresh_cls = d[d["classify_ms"].notna()]
    return {
        "total": len(d),
        "live": int((d["execution_mode"] == "live").sum()), "mock": int((d["execution_mode"] == "mock").sum()),
        "legacy": int(d["execution_mode"].astype(str).str.startswith("legacy").sum()),
        "calls": {
            "model_calls": len(calls), "succeeded": len(ok), "failed": len(failed), "blocked": int((d["status"] == "blocked").sum()),
            "call_error_rate": (len(failed) / len(calls)) if len(calls) else None,
            "provider_attempts": int(known["provider_attempts"].sum()), "attempts_known_calls": len(known),
            "attempts_unknown_calls": int((calls["provider_attempts"].isna() & (calls["execution_mode"] != "mock")).sum()),
            "retried_calls": int((known["provider_attempts"] > 1).sum()),
            "credential_failovers": int((d["failover_used"].fillna(0) == 1).sum()) if "failover_used" in d else 0,
        },
        "routing": {"fallbacks": int((d["route_source"] == "rules_fallback").sum()),
                    "classifications": len(fresh_cls), "reused": int((d["classification_reused"].fillna(0) == 1).sum()),
                    "simple_share": float((ok["route"] == "simple").mean()) if len(ok) else None},
        "tokens": _usage_summary(d), "cost": _cost_summary(d), "latency": _latency_summary(d, attempts),
    }


# ---------------------------------------------------------------- turns
def _kind_ok(mode: str, kind: str) -> bool:
    if kind == "all":
        return True
    if mode == NO_CALL:
        return kind == "live"       # a turn that never reached the proxy had no call to label; it is shown with live traffic
    return mode.startswith("legacy") if kind == "legacy" else mode == kind


def load_turns(db_path=None, since_minutes: float | None = None, kind: str = "all", user_id: str | None = None) -> tuple[pd.DataFrame, int]:
    """One row per user turn + the number of call records that cannot be grouped into a turn.
    Time basis: when the turn started (`turns.started_at`; legacy: its first call). Execution kind: of its calls."""
    calls = annotate(load(db_path, user_id=user_id))
    try:
        conn = _connect(db_path)
        try:
            known = pd.read_sql_query("SELECT * FROM turns" + (" WHERE user_id=?" if user_id else ""), conn,
                                      params=(user_id,) if user_id else ())
        except Exception:
            known = pd.DataFrame()
        try:
            msgs = pd.read_sql_query("SELECT turn_id, user_id, status FROM messages WHERE role='assistant' AND turn_id IS NOT NULL", conn)
        except Exception:
            msgs = pd.DataFrame()
        conn.close()
    except Exception:
        known, msgs = pd.DataFrame(), pd.DataFrame()

    grouped = calls[calls["user_id"].notna() & calls["turn_id"].notna()] if not calls.empty else calls
    ungroupable = 0
    if not calls.empty:                                   # same record type + window as everything else on the page
        scope = filter_execution(calls, kind)
        if since_minutes:
            scope = scope[scope["ts"] >= time.time() - since_minutes * 60]
        ungroupable = int((scope["user_id"].isna() | scope["turn_id"].isna()).sum())
    rec = {(r.user_id, r.turn_id): r for r in known.itertuples()} if not known.empty else {}
    msg_status = {(r.user_id, r.turn_id): r.status for r in msgs.itertuples()} if not msgs.empty else {}
    by_key = {k: g for k, g in grouped.groupby(["user_id", "turn_id"], sort=False)} if not grouped.empty else {}

    rows = []
    for key in list(by_key) + [k for k in rec if k not in by_key]:
        g = by_key.get(key)
        r = rec.get(key)
        cc = g[g["is_model_call"]] if g is not None else None
        modes = sorted(set(g["execution_mode"])) if g is not None else [NO_CALL]
        outcome, source = "unknown", None
        if r is not None and r.outcome:
            outcome, source = r.outcome, "turns"
        elif key in msg_status:
            outcome, source = MESSAGE_OUTCOME.get(msg_status[key], "unknown"), "message"
        prov = cc[cc["usage_kind"] == "provider"] if cc is not None else None
        rows.append({
            "user_id": key[0], "turn_id": key[1], "execution_mode": modes[0] if len(modes) == 1 else "mixed",
            "started_at": float(r.started_at) if r is not None and pd.notna(r.started_at) else (float(g["ts"].min()) if g is not None else np.nan),
            "duration_ms": float(r.duration_ms) if r is not None and pd.notna(r.duration_ms) else np.nan,
            "outcome": outcome, "outcome_source": source,
            "model_calls": len(cc) if cc is not None else 0,
            "blocked_calls": int((g["status"] == "blocked").sum()) if g is not None else 0,
            "provider_attempts": int(cc["provider_attempts"].sum()) if cc is not None and len(cc) else 0,
            "attempts_complete": bool(cc["provider_attempts"].notna().all()) if cc is not None and len(cc) else True,
            "input_tokens": int(prov["in_tok"].sum()) if prov is not None else 0,
            "output_tokens": int(prov["out_tok"].sum()) if prov is not None else 0,
            "usage_complete": bool((cc["usage_kind"] == "provider").all()) if cc is not None and len(cc) else True,
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t, ungroupable
    t = t[[_kind_ok(m, kind) for m in t["execution_mode"]]]
    if since_minutes:
        t = t[t["started_at"] >= time.time() - since_minutes * 60]
    return t.sort_values("started_at", na_position="first").reset_index(drop=True), ungroupable


def summarize_turns(turns: pd.DataFrame, ungroupable: int = 0) -> dict:
    n = len(turns)
    out = {"turns": n, "ungroupable_calls": int(ungroupable)}
    for o in TURN_OUTCOMES:
        out[o] = int((turns["outcome"] == o).sum()) if n else 0
    out["completion_rate"] = (out["completed"] / (n - out["unknown"])) if n - out["unknown"] > 0 else None
    out["calls_per_turn"] = stat(turns["model_calls"]) if n else stat([])
    out["duration"] = stat(turns["duration_ms"]) if n else stat([])
    out["no_duration_turns"] = int(turns["duration_ms"].isna().sum()) if n else 0
    out["attempts_per_turn_avg"] = float(turns["provider_attempts"].mean()) if n else None
    return out


# ---------------------------------------------------------------- alerts
def alert_data(df: pd.DataFrame) -> dict:
    """Numbers behind the red banner. Live traffic only; the error rate is failed / (succeeded + failed) model calls
    in the last ALERT_WINDOW_MIN minutes (blocked loops are reported separately, not as errors)."""
    data = {"window_min": config.ALERT_WINDOW_MIN, "error_threshold": config.ALERT_ERROR_RATE, "error_rate": None, "blocked": 0}
    if df.empty:
        return data
    live = df[df["execution_mode"] == "live"]
    recent = live[live["ts"] >= time.time() - config.ALERT_WINDOW_MIN * 60]
    calls = recent[recent["status"].isin(["success", "error"])]
    if len(calls) >= 3:
        data["error_rate"] = float((calls["status"] == "error").mean())
    data["blocked"] = int((recent["status"] == "blocked").sum())
    return data


def alerts(df: pd.DataFrame) -> list[str]:
    a = alert_data(df)
    out = []
    if a["error_rate"] is not None and a["error_rate"] > a["error_threshold"]:
        out.append(f"Error rate {a['error_rate']:.0%} in the last {a['window_min']} min "
                   f"(threshold {a['error_threshold']:.0%}). Check provider status / rate limits.")
    if a["blocked"]:
        out.append(f"{a['blocked']} agent loop(s) blocked in the last {a['window_min']} min. "
                   "An agent is repeating a failing tool call.")
    return out
