"""
Per-user conversation storage (SQLite). EVERY query is scoped by user_id inside the SQL itself, so a conversation that
belongs to someone else behaves exactly like one that does not exist (callers answer 404, never 403, to avoid leaking ids).
"""
import json
import secrets
import sqlite3
import time

from proxy.db import _lock

MAX_TITLE = 60


def _title(text: str) -> str:
    t = " ".join(str(text or "").split())
    return (t[:MAX_TITLE] + "…") if len(t) > MAX_TITLE else (t or "New chat")


def new_session_id() -> str:
    return "sess_" + secrets.token_hex(4)


def create(conn: sqlite3.Connection, user_id: str, first_message: str) -> dict:
    cid, now = "c_" + secrets.token_hex(8), time.time()
    with _lock:
        conn.execute("INSERT INTO conversations (id, user_id, title, session_id, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                     (cid, user_id, _title(first_message), new_session_id(), now, now))
        conn.commit()
    return get(conn, cid, user_id)


def get(conn: sqlite3.Connection, conv_id: str, user_id: str) -> dict | None:
    r = conn.execute("SELECT id, title, session_id, created_at, updated_at FROM conversations WHERE id=? AND user_id=?",
                     (conv_id, user_id)).fetchone()
    return None if r is None else dict(zip(("id", "title", "session_id", "created_at", "updated_at"), r))


def list_for(conn: sqlite3.Connection, user_id: str, limit: int = 100) -> list[dict]:
    rows = conn.execute("SELECT id, title, updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                        (user_id, limit)).fetchall()
    return [{"id": i, "title": t, "updated": u} for i, t, u in rows]


def safe_error_code(content: str | None) -> str:
    """Raw provider/proxy error text can contain internals, so users only ever see one of these codes (the page shows a
    translated, actionable sentence). The raw text stays in the monitoring records for administrators."""
    c = str(content or "").lower()
    if c in ("busy", "unavailable", "failed"):
        return c
    if "rate" in c or "temporar" in c or "429" in c or "503" in c or "quota" in c:
        return "busy"
    if "cannot reach" in c or "connect" in c or "timeout" in c or "timed out" in c or "unreachable" in c:
        return "unavailable"
    return "failed"


def messages(conn: sqlite3.Connection, conv_id: str, user_id: str, limit: int = 500) -> list[dict] | None:
    """None when the conversation is not the user's. Technical details (route, model, cost, steps) are NOT returned."""
    if get(conn, conv_id, user_id) is None:
        return None
    rows = conn.execute(
        "SELECT m.id, m.role, m.content, m.status, m.files_json, m.ts, f.rating, f.reason, f.comment "
        "FROM messages m LEFT JOIN message_feedback f ON f.message_id=m.id AND f.user_id=m.user_id "
        "WHERE m.conversation_id=? AND m.user_id=? ORDER BY m.id LIMIT ?", (conv_id, user_id, limit)).fetchall()
    out = []
    for mid, role, content, status, files, ts, rating, reason, comment in rows:
        m = {"role": role, "content": content, "ts": ts}
        if role == "assistant":
            if status == "error":
                m["content"] = safe_error_code(content)
            m.update(id=mid, status=status, files=json.loads(files or "[]"),
                     feedback={"rating": rating, "reason": reason, "comment": comment} if rating else None)
        out.append(m)
    return out


def add_message(conn: sqlite3.Connection, conv_id: str, user_id: str, role: str, content: str, *, status: str | None = None,
                steps: list | None = None, files: list | None = None, turn_id: str | None = None) -> int | None:
    """Returns the new message id, or None when the conversation is not the user's."""
    if get(conn, conv_id, user_id) is None:
        return None
    now = time.time()
    with _lock:
        cur = conn.execute("INSERT INTO messages (conversation_id, user_id, role, content, status, steps_json, summary_json, "
                           "turn_id, ts, files_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (conv_id, user_id, role, content, status, json.dumps(steps or [], ensure_ascii=False),
                            "{}", turn_id, now, json.dumps(files or [], ensure_ascii=False)))
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?", (now, conv_id, user_id))
        conn.commit()
    return cur.lastrowid


def rotate_session(conn: sqlite3.Connection, conv_id: str, user_id: str) -> str | None:
    """After a loop block the old agent session stays blocked by design; the next message gets a fresh one."""
    sid = new_session_id()
    with _lock:
        cur = conn.execute("UPDATE conversations SET session_id=? WHERE id=? AND user_id=?", (sid, conv_id, user_id))
        conn.commit()
    return sid if cur.rowcount else None


def delete(conn: sqlite3.Connection, conv_id: str, user_id: str) -> bool:
    if get(conn, conv_id, user_id) is None:
        return False
    with _lock:
        conn.execute("DELETE FROM message_feedback WHERE conversation_id=? AND user_id=?", (conv_id, user_id))
        conn.execute("DELETE FROM messages WHERE conversation_id=? AND user_id=?", (conv_id, user_id))
        conn.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id))
        conn.commit()
    return True


def history_for_agent(conn: sqlite3.Connection, conv_id: str, user_id: str, max_messages: int = 20,
                      max_chars: int = 12000) -> list[dict]:
    """Earlier user messages and SUCCESSFUL answers only, newest-bounded; read from the database, never from the browser."""
    rows = conn.execute("SELECT role, content FROM messages WHERE conversation_id=? AND user_id=? "
                        "AND (role='user' OR status='ok') ORDER BY id DESC LIMIT ?", (conv_id, user_id, max_messages)).fetchall()
    out, total = [], 0
    for role, content in rows:                      # newest first, stop when the character budget is used
        total += len(content or "")
        if total > max_chars:
            break
        out.append({"role": role, "content": content or ""})
    return out[::-1]
