"""
Application users and roles, stored in Mirsad's own SQLite (table app_users).

The role is NEVER taken from Supabase metadata, signup data, cookies, browser storage or request bodies:
  * every new account is created with role 'user'
  * the only way to become 'admin' is `python -m scripts.manage_admin grant ...` run on the server machine
    (it needs filesystem access to the database; no service-role key is involved)
"""
import re
import sqlite3
import time

from proxy.db import _lock
_CONTROL = re.compile(r"[\x00-\x1f\x7f<>]")


def clean_name(name, fallback: str = "") -> str:
    name = _CONTROL.sub("", str(name or "")).strip()
    return name[:80] or fallback


def upsert_login(conn: sqlite3.Connection, user_id: str, email: str, name: str | None) -> dict:
    """Creates the user with role 'user' on first sight; later logins refresh email/name/last_login only."""
    now = time.time()
    email = (email or "").strip().lower()[:254]
    with _lock:
        row = conn.execute("SELECT role, display_name FROM app_users WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO app_users (user_id, email, display_name, role, created_at, last_login_at) "
                         "VALUES (?,?,?,'user',?,?)", (user_id, email, clean_name(name, email.split("@")[0]), now, now))
        else:
            conn.execute("UPDATE app_users SET email=?, last_login_at=?, display_name=COALESCE(NULLIF(display_name,''),?) "
                         "WHERE user_id=?", (email, now, clean_name(name, email.split("@")[0]), user_id))
        conn.commit()
    return get(conn, user_id)


def get(conn: sqlite3.Connection, user_id: str) -> dict | None:
    r = conn.execute("SELECT user_id, email, display_name, role, disabled, created_at, last_login_at "
                     "FROM app_users WHERE user_id=?", (user_id,)).fetchone()
    return None if r is None else dict(zip(
        ("user_id", "email", "display_name", "role", "disabled", "created_at", "last_login_at"), r))


def list_all(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT u.user_id, u.email, u.display_name, u.role, u.disabled, u.last_login_at, "
        "(SELECT COUNT(*) FROM requests r WHERE r.user_id = u.user_id) "
        "FROM app_users u ORDER BY u.created_at").fetchall()
    keys = ("user_id", "email", "display_name", "role", "disabled", "last_login_at", "requests")
    return [dict(zip(keys, r)) for r in rows]


def find_by_email(conn: sqlite3.Connection, email: str) -> dict | None:
    r = conn.execute("SELECT user_id FROM app_users WHERE lower(email)=?", ((email or "").strip().lower(),)).fetchone()
    return None if r is None else get(conn, r[0])


def set_role(conn: sqlite3.Connection, user_id: str, role: str) -> None:
    if role not in ("user", "admin"):
        raise ValueError("role must be 'user' or 'admin'")
    with _lock:
        conn.execute("UPDATE app_users SET role=? WHERE user_id=?", (role, user_id))
        conn.commit()
