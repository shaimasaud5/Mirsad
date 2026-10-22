"""
Answer feedback (Helpful / Not helpful). One CURRENT rating per user and assistant answer, enforced by a UNIQUE
constraint, so changing a rating can never be counted twice. Ownership is checked inside the SQL: the answer must be an
assistant message of the caller's own conversation, otherwise nothing is read, written or deleted (callers answer 404).
Feedback lives in its own table and is never read by history_for_agent, so it cannot become model context.
"""
import re
import sqlite3
import time

from proxy.db import _lock

RATINGS = ("helpful", "not_helpful")
REASONS = ("incorrect", "incomplete", "misunderstood", "other")
MAX_COMMENT = 1000
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_comment(text: str | None) -> str | None:
    text = _CONTROL.sub("", str(text or "")).strip()
    return text[:MAX_COMMENT] or None


def _row(r) -> dict | None:
    return None if r is None else {"rating": r[0], "reason": r[1], "comment": r[2]}


def get(conn: sqlite3.Connection, user_id: str, message_id: int) -> dict | None:
    return _row(conn.execute("SELECT rating, reason, comment FROM message_feedback WHERE user_id=? AND message_id=?",
                             (user_id, message_id)).fetchone())


def owns_answer(conn: sqlite3.Connection, user_id: str, message_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM messages m JOIN conversations c ON c.id=m.conversation_id AND c.user_id=m.user_id "
        "WHERE m.id=? AND m.user_id=? AND m.role='assistant' AND m.status='ok'", (message_id, user_id)).fetchone() is not None


def save(conn: sqlite3.Connection, user_id: str, message_id: int, rating: str, reason: str | None,
         comment: str | None) -> dict | None:
    """Create or replace the caller's rating. None when the answer is not the caller's (or is not a completed answer)."""
    if rating not in RATINGS or (reason is not None and (rating != "not_helpful" or reason not in REASONS)):
        raise ValueError("invalid feedback")
    now = time.time()
    with _lock:
        conn.execute(
            "INSERT INTO message_feedback (user_id, conversation_id, message_id, turn_id, rating, reason, comment, created_at, updated_at) "
            "SELECT m.user_id, m.conversation_id, m.id, m.turn_id, ?, ?, ?, ?, ? "
            "FROM messages m JOIN conversations c ON c.id=m.conversation_id AND c.user_id=m.user_id "
            "WHERE m.id=? AND m.user_id=? AND m.role='assistant' AND m.status='ok' "
            "ON CONFLICT(user_id, message_id) DO UPDATE SET rating=excluded.rating, reason=excluded.reason, "
            "comment=excluded.comment, updated_at=excluded.updated_at",
            (rating, reason, clean_comment(comment), now, now, message_id, user_id))
        conn.commit()
    return get(conn, user_id, message_id)


def clear(conn: sqlite3.Connection, user_id: str, message_id: int) -> bool:
    """Remove the caller's rating (idempotent). False when the answer is not the caller's."""
    if not owns_answer(conn, user_id, message_id):
        return False
    with _lock:
        conn.execute("DELETE FROM message_feedback WHERE user_id=? AND message_id=?", (user_id, message_id))
        conn.commit()
    return True
