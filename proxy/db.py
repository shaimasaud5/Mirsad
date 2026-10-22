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
    "user_id": "TEXT",                 # verified owner (Supabase user id); NULL = legacy row, admin-only
    "conversation_id": "TEXT",
}

# Added by the metrics-accuracy release (same rules: additive ALTER TABLE, backup first, NULL on older rows).
METRICS_COLUMNS = {
    "model_call_ms": "REAL",        # answering-model call only: provider dispatch -> final attempt returned (failover and
                                    # Retry-After waits included; classification, loop guard and DB write excluded)
    "est_cost_usd": "REAL",         # estimated inference cost at list price when written; NULL = unavailable (never 0)
    "baseline_big_usd": "REAL",     # the same tokens priced at BIG_MODEL's rates (hypothetical comparison)
    "pricing_version": "TEXT",      # config.PRICING_VERSION used for the two columns above
}

# One row per user turn (one submitted message and its whole agent run), written by the web server.
TURNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id         TEXT NOT NULL,
    user_id         TEXT NOT NULL,                        -- verified owner, from the server-side session
    conversation_id TEXT,
    session_id      TEXT,
    started_at      REAL,                                 -- wall clock when the server received the submission
    duration_ms     REAL,                                 -- monotonic clock: receipt -> agent finished (completed/failed/blocked)
    outcome         TEXT,                                 -- completed | failed | blocked | step_limit
    PRIMARY KEY (user_id, turn_id)
);
CREATE INDEX IF NOT EXISTS idx_turns_started ON turns(started_at);
"""

# Application data owned by users. Role lives HERE (never in Supabase metadata, which users can edit).
AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_users (
    user_id       TEXT PRIMARY KEY,                       -- Supabase auth user id (verified, never from the browser)
    email         TEXT,
    display_name  TEXT,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    disabled      INTEGER NOT NULL DEFAULT 0,
    created_at    REAL,
    last_login_at REAL
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    sid_hash      TEXT PRIMARY KEY,                       -- sha256 of the opaque cookie value
    user_id       TEXT NOT NULL,
    tokens_enc    BLOB NOT NULL,                          -- Fernet(JSON{access_token, refresh_token})
    access_exp    REAL,
    last_verified REAL,
    csrf_token    TEXT NOT NULL,
    recovery      INTEGER NOT NULL DEFAULT 0,             -- 1 = opened from a password-reset link
    created_at    REAL,
    last_seen     REAL
);
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT,
    session_id  TEXT,                                     -- agent/proxy session (rotated after a loop block)
    created_at  REAL,
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    role            TEXT NOT NULL,                        -- user | assistant
    content         TEXT,
    status          TEXT,
    steps_json      TEXT,
    summary_json    TEXT,
    turn_id         TEXT,
    ts              REAL,
    files_json      TEXT                                  -- files PROVEN to have been read for this answer (tool-verified)
);
-- One current rating per user and assistant answer. Kept apart from `messages` so it can never become model history.
CREATE TABLE IF NOT EXISTS message_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,                        -- from the server-side session, never from the browser
    conversation_id TEXT NOT NULL,
    message_id      INTEGER NOT NULL,                     -- messages.id of the assistant answer
    turn_id         TEXT,                                 -- links to requests.turn_id (many model calls -> one rating)
    rating          TEXT NOT NULL CHECK (rating IN ('helpful', 'not_helpful')),
    reason          TEXT CHECK (reason IS NULL OR reason IN ('incorrect', 'incomplete', 'misunderstood', 'other')),
    comment         TEXT,
    created_at      REAL,
    updated_at      REAL,
    UNIQUE (user_id, message_id),
    CHECK (reason IS NULL OR rating = 'not_helpful')
);
CREATE INDEX IF NOT EXISTS idx_feedback_message ON message_feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_requests_user ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
"""

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

_lock = threading.RLock()   # shared by every writer on the process-wide connection


def _backup_before_migration(conn: sqlite3.Connection, path: Path, suffix: str = "pre-auth",
                             tables: tuple[str, ...] = ("requests",)) -> None:
    """One-time safety copy (sqlite online backup) before columns are added to a table that has data."""
    target = path.with_name(f"{path.name}-backup-{suffix}")
    if target.exists():
        return
    if not any(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables):
        return
    dst = sqlite3.connect(target)
    try:
        conn.backup(dst)
    finally:
        dst.close()


def _migrate_messages(conn: sqlite3.Connection, path: Path | None = None) -> None:
    """Feedback release: messages.files_json. Additive only; a pre-feedback copy is taken first when data exists."""
    if "files_json" in {r[1] for r in conn.execute("PRAGMA table_info(messages)")}:
        return
    if path is not None:
        _backup_before_migration(conn, path, "pre-feedback", ("messages", "requests"))
    conn.execute("ALTER TABLE messages ADD COLUMN files_json TEXT")
    conn.commit()


def _migrate(conn: sqlite3.Connection, path: Path | None = None) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    missing = {c: t for c, t in NEW_COLUMNS.items() if c not in existing}
    if missing and path is not None:
        _backup_before_migration(conn, path)
    for col, typ in missing.items():
        conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")
    conn.commit()


def _migrate_metrics(conn: sqlite3.Connection, path: Path | None = None) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    missing = {c: t for c, t in METRICS_COLUMNS.items() if c not in existing}
    if missing and path is not None:
        _backup_before_migration(conn, path, "pre-metrics")
    for col, typ in missing.items():
        conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")
    conn.commit()


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")   # lets the dashboard read while we write
    conn.execute(SCHEMA)
    conn.execute(ATTEMPTS_SCHEMA)
    if path.exists() and not {"model_call_ms"} <= {r[1] for r in conn.execute("PRAGMA table_info(requests)")}:
        _backup_before_migration(conn, path, "pre-metrics")   # BEFORE any column is added, so it holds the original data
    _migrate(conn, path)
    _migrate_metrics(conn, path)
    conn.executescript(TURNS_SCHEMA)
    _premigrate_messages(conn, path)         # old `messages` table: add files_json BEFORE the schema script runs
    conn.executescript(AUTH_SCHEMA)          # after _migrate: idx_requests_user needs requests.user_id
    return conn


def _premigrate_messages(conn: sqlite3.Connection, path: Path) -> None:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'").fetchone():
        _migrate_messages(conn, path)


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


def insert_turn(conn: sqlite3.Connection, *, turn_id: str, user_id: str, conversation_id: str | None, session_id: str | None,
                started_at: float, duration_ms: float, outcome: str) -> None:
    with _lock:
        conn.execute("INSERT OR REPLACE INTO turns (turn_id, user_id, conversation_id, session_id, started_at, duration_ms, outcome) "
                     "VALUES (?,?,?,?,?,?,?)", (turn_id, user_id, conversation_id, session_id, started_at, duration_ms, outcome))
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
        conn.execute("DELETE FROM turns")
        conn.commit()
