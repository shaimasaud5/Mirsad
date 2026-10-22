"""
Server-managed sessions.

The browser holds ONE opaque random cookie value (HttpOnly). Everything else lives on the server:
  auth_sessions.sid_hash   sha256(cookie)         - a database leak does not reveal usable cookies
  auth_sessions.tokens_enc Fernet(access+refresh) - Supabase tokens are encrypted at rest and never reach the browser
Identity therefore always comes from this verified server-side record, never from anything the browser sends.

Lifecycle
  idle timeout (MIRSAD_SESSION_IDLE_HOURS) and absolute lifetime (MIRSAD_SESSION_MAX_DAYS) end a session;
  the Supabase access token is refreshed shortly before it expires (single-flight per session, because refresh
  tokens are single-use); a refresh/validation rejected by Supabase ends the session (revoked, banned or deleted user);
  a transient Supabase outage keeps the session until the access token itself expires.
"""
import base64
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import config
from auth import users
from auth.supabase import SupabaseAuth, SupabaseError
from proxy.db import _lock as _shared_db_lock

REFRESH_MARGIN_S = 60
TOUCH_EVERY_S = 30


@dataclass
class AuthSession:
    sid_hash: str
    user_id: str
    access_token: str
    refresh_token: str
    access_exp: float
    csrf_token: str
    recovery: bool
    created_at: float
    last_seen: float


def hash_sid(sid: str) -> str:
    return hashlib.sha256(sid.encode("utf-8")).hexdigest()


def make_fernet(secret: str) -> Fernet:
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"mirsad-session-v1", info=b"tokens").derive(secret.encode())
    return Fernet(base64.urlsafe_b64encode(key))


class SessionManager:
    def __init__(self, conn: sqlite3.Connection, supa: SupabaseAuth, secret: str, clock=time.time):
        if not secret or len(secret) < 16:
            raise ValueError("MIRSAD_SESSION_SECRET must be set (16+ characters)")
        self.conn, self.supa, self.now = conn, supa, clock
        self._fernet = make_fernet(secret)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._db_lock = _shared_db_lock

    # ---------------------------------------------------------------- helpers
    def _lock_for(self, h: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(h, threading.Lock())

    def _encrypt(self, access: str, refresh: str) -> bytes:
        return self._fernet.encrypt(json.dumps({"a": access, "r": refresh}).encode())

    def _decrypt(self, blob: bytes) -> tuple[str, str] | None:
        try:
            d = json.loads(self._fernet.decrypt(bytes(blob)))
            return d["a"], d["r"]
        except (InvalidToken, ValueError, KeyError, TypeError):
            return None

    def _row(self, h: str):
        return self.conn.execute(
            "SELECT sid_hash, user_id, tokens_enc, access_exp, last_verified, csrf_token, recovery, created_at, last_seen "
            "FROM auth_sessions WHERE sid_hash=?", (h,)).fetchone()

    def _delete(self, h: str) -> None:
        with self._db_lock:
            self.conn.execute("DELETE FROM auth_sessions WHERE sid_hash=?", (h,))
            self.conn.commit()

    @staticmethod
    def _expiry(supa_session: dict, now: float) -> float:
        exp = supa_session.get("expires_at")
        if isinstance(exp, (int, float)) and exp > now:
            return float(exp)
        return now + float(supa_session.get("expires_in") or 3600)

    # ---------------------------------------------------------------- create / revoke
    def create(self, supa_session: dict, *, recovery: bool = False) -> tuple[str, dict]:
        """supa_session = a token response from Supabase (already verified by Supabase). Returns (cookie value, user)."""
        user = supa_session.get("user") or {}
        uid = str(user.get("id") or "")
        access, refresh = supa_session.get("access_token"), supa_session.get("refresh_token")
        if not (uid and access and refresh):
            raise SupabaseError(502, "bad_session")
        meta = user.get("user_metadata") or {}
        app_user = users.upsert_login(self.conn, uid, user.get("email") or "", meta.get("name") or meta.get("full_name"))
        if app_user["disabled"]:
            raise PermissionError("account disabled")
        sid = secrets.token_urlsafe(32)
        now = self.now()
        with self._db_lock:
            self.conn.execute(
                "INSERT INTO auth_sessions (sid_hash, user_id, tokens_enc, access_exp, last_verified, csrf_token, recovery, "
                "created_at, last_seen) VALUES (?,?,?,?,?,?,?,?,?)",
                (hash_sid(sid), uid, self._encrypt(access, refresh), self._expiry(supa_session, now), now,
                 secrets.token_urlsafe(24), int(recovery), now, now))
            self.conn.commit()
        return sid, app_user

    def revoke(self, sid: str | None, *, remote: bool = True) -> None:
        if not sid:
            return
        h = hash_sid(sid)
        row = self._row(h)
        if row and remote:
            tokens = self._decrypt(row[2])
            if tokens:
                try:
                    self.supa.logout(tokens[0])         # scope=local: other devices stay signed in
                except Exception:  # noqa: BLE001  - local logout must succeed even if Supabase is unreachable
                    pass
        self._delete(h)

    def set_recovery(self, sid_hash: str, value: bool) -> None:
        with self._db_lock:
            self.conn.execute("UPDATE auth_sessions SET recovery=? WHERE sid_hash=?", (int(value), sid_hash))
            self.conn.commit()

    def purge_expired(self) -> int:
        now = self.now()
        with self._db_lock:
            cur = self.conn.execute("DELETE FROM auth_sessions WHERE last_seen < ? OR created_at < ?",
                                    (now - config.SESSION_IDLE_HOURS * 3600, now - config.SESSION_MAX_DAYS * 86400))
            self.conn.commit()
            return cur.rowcount

    # ---------------------------------------------------------------- load (expiry, refresh, revalidation)
    def load(self, sid: str | None) -> AuthSession | None:
        if not sid or len(sid) > 128:
            return None
        h = hash_sid(sid)
        with self._lock_for(h):
            row = self._row(h)
            if row is None:
                return None
            _, uid, blob, access_exp, last_verified, csrf, recovery, created, seen = row
            now = self.now()
            if now - seen > config.SESSION_IDLE_HOURS * 3600 or now - created > config.SESSION_MAX_DAYS * 86400:
                self._delete(h)
                return None
            tokens = self._decrypt(blob)
            if tokens is None:                       # secret rotated or data tampered with
                self._delete(h)
                return None
            access, refresh = tokens

            if access_exp - now < REFRESH_MARGIN_S:
                try:
                    fresh = self.supa.refresh(refresh)
                    access, refresh = fresh["access_token"], fresh["refresh_token"]
                    access_exp, last_verified = self._expiry(fresh, now), now
                    with self._db_lock:
                        self.conn.execute("UPDATE auth_sessions SET tokens_enc=?, access_exp=?, last_verified=? WHERE sid_hash=?",
                                          (self._encrypt(access, refresh), access_exp, last_verified, h))
                        self.conn.commit()
                except SupabaseError as exc:
                    if exc.status < 500:             # refresh token invalid/expired/revoked, user banned or deleted
                        self._delete(h)
                        return None
                    if access_exp <= now:            # outage and the access token is already expired
                        return None
                except (KeyError, TypeError):
                    self._delete(h)
                    return None
            elif now - (last_verified or 0) > config.AUTH_REVALIDATE_S:
                try:
                    self.supa.get_user(access)
                    last_verified = now
                    with self._db_lock:
                        self.conn.execute("UPDATE auth_sessions SET last_verified=? WHERE sid_hash=?", (now, h))
                        self.conn.commit()
                except SupabaseError as exc:
                    if exc.status < 500:             # session revoked server-side (e.g. signed out elsewhere/globally)
                        self._delete(h)
                        return None

            if now - seen > TOUCH_EVERY_S:
                with self._db_lock:
                    self.conn.execute("UPDATE auth_sessions SET last_seen=? WHERE sid_hash=?", (now, h))
                    self.conn.commit()
                seen = now
            return AuthSession(h, uid, access, refresh, access_exp, csrf, bool(recovery), created, seen)
