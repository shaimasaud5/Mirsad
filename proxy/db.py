"""
SQLite storage: one row per request that reaches the proxy.
The dashboard only READS this table; the proxy only WRITES to it.
"""
import sqlite3
import threading
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT,
    ts                REAL,      -- unix time (seconds)
    session_id        TEXT,
    status            TEXT,      -- success | error | blocked
    route             TEXT,      -- simple | complex
    score             INTEGER,   -- router complexity score
    reason            TEXT,      -- why the router decided
    model             TEXT,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    latency_ms        REAL,
    actual_cost       REAL,
    baseline_cost     REAL,
    est_saved_cost    REAL,      -- only for blocked loops (estimated)
    has_tool_calls    INTEGER,   -- did the model ask for a tool?
    error             TEXT
);
"""

# Columns added after the first release. Added with ALTER TABLE (never drops or rewrites data);
# rows written before the migration simply have NULL here and are shown as "legacy".
NEW_COLUMNS = {
    "turn_id": "TEXT",                 # one id per user message (agent header)
    "routing_mode": "TEXT",            # rules | shadow | local_llm
    "route_source": "TEXT",            # rules | local_llm | rules_fallback
    "fallback_reason": "TEXT",         # why local_llm fell back to rules
    "rules_route": "TEXT",             # what the keyword rules said
    "llm_route": "TEXT",               # what the local classifier said
    "classifier_model": "TEXT",
    "classifier_prompt_id": "TEXT",
    "classifier_ms": "REAL",           # time spent classifying (0 when the turn's classification was reused)
    "classifier_reason": "TEXT",
    "classifier_error": "TEXT",
    "classifier_truncated": "INTEGER", # classifier input was cut (current request or context overflow)
    "context_truncated": "INTEGER",    # earlier-conversation context was bounded
    "classification_reused": "INTEGER",
    "execution_mode": "TEXT",          # live | mock
    "token_source": "TEXT",            # provider | estimate | mock-estimate
    "response_model": "TEXT",          # model name reported by the provider response
    "credential_alias": "TEXT",        # primary | backup_N (never the key); credential that answered / last tried
    "provider_attempts": "INTEGER",    # provider calls made for this request (details: provider_attempts table)
    "failover_used": "INTEGER",        # 1 when a backup credential was tried after a 401
}

ATTEMPTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL,
    request_id      TEXT,
    turn_id         TEXT,
    session_id      TEXT,
    attempt_no      INTEGER,
    credential_alias TEXT,   -- primary | backup_N | env  (never a key value)
    model           TEXT,
    outcome         TEXT,    -- success | credential_invalid | rate_limited | permission_denied | bad_request | provider_outage | error
    http_status     INTEGER,
    duration_ms     REAL,
    failover        INTEGER, -- 1 = a different credential was tried because of this attempt
    failover_reason TEXT,
    retry_after_s   REAL,
    error           TEXT     -- redacted
);
"""

_lock = threading.Lock()


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    for col, typ in NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")
    conn.commit()


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")   # lets the dashboard read while we write
    conn.execute(SCHEMA)
    conn.execute(ATTEMPTS_SCHEMA)
    _migrate(conn)
    return conn


def insert_attempts(conn: sqlite3.Connection, base: dict, model: str, attempts) -> None:
    rows = [(base["ts"], base["request_id"], base.get("turn_id"), base["session_id"], a.n, a.alias, model,
             a.outcome, a.http_status, a.duration_ms, int(a.failover), a.failover_reason, a.retry_after_s, a.error)
            for a in attempts]
    if not rows:
        return
    with _lock:
        conn.executemany(
            "INSERT INTO provider_attempts (ts, request_id, turn_id, session_id, attempt_no, credential_alias, model, "
            "outcome, http_status, duration_ms, failover, failover_reason, retry_after_s, error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()


def insert(conn: sqlite3.Connection, row: dict) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with _lock:
        conn.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", list(row.values()))
        conn.commit()


def clear(conn: sqlite3.Connection) -> None:
    with _lock:
        conn.execute("DELETE FROM requests")
        conn.execute("DELETE FROM provider_attempts")
        conn.commit()
