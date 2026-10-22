"""Authentication, roles and per-user isolation. Supabase is replaced by an in-memory fake: no network, no emails."""
import itertools
import logging
import os
import re
import secrets
import sqlite3

import pytest
from fastapi.testclient import TestClient

import config
from auth import chatstore, users
from auth.security import RateLimiter
from auth.sessions import SessionManager, hash_sid
from auth.supabase import SupabaseError
from monitoring import metrics
from proxy import db
from proxy import main as proxy_main
from scripts import manage_admin
from ui import web_server as ws

INTERNAL = {"Authorization": "Bearer " + os.environ["MIRSAD_INTERNAL_TOKEN"]}
proxy = TestClient(proxy_main.app)
PW = "correct-horse-battery"


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeSupa:
    """Behaves like the parts of Supabase Auth that Mirsad uses, and records what Mirsad sent."""
    configured = True

    def __init__(self):
        self.users, self.access, self.refresh_tokens = {}, {}, {}
        self.calls, self.confirm_required, self.outage, self.refresh_rejects = [], True, False, False
        self._ids = itertools.count(1)

    # -- test helpers
    def add_user(self, email, name="Test", password=PW, confirmed=True):
        uid = f"00000000-0000-4000-8000-{next(self._ids):012d}"
        self.users[email] = {"id": uid, "email": email, "password": password, "confirmed": confirmed,
                             "user_metadata": {"name": name}}
        return uid

    def _issue(self, u):
        a, r = "acc" + secrets.token_hex(24), "ref" + secrets.token_hex(24)
        self.access[a], self.refresh_tokens[r] = u["email"], u["email"]
        return {"access_token": a, "refresh_token": r, "expires_in": 3600, "token_type": "bearer",
                "user": {"id": u["id"], "email": u["email"], "user_metadata": dict(u["user_metadata"])}}

    def revoke_access(self, token):
        self.access.pop(token, None)

    def _up(self):
        if self.outage:
            raise SupabaseError(503, "upstream_unavailable")

    # -- the SupabaseAuth surface
    def signup(self, email, password, name, redirect_to):
        self._up()
        self.calls.append(("signup", {"email": email, "name": name, "redirect_to": redirect_to}))
        if email not in self.users:
            self.add_user(email, name, password, confirmed=not self.confirm_required)
        u = self.users[email]
        if not self.confirm_required:
            return {"user": {}, "session": self._issue(u)}
        return {"user": {"id": u["id"], "email": email}, "session": None}

    def login(self, email, password):
        self._up()
        u = self.users.get(email)
        if not u or u["password"] != password:
            raise SupabaseError(400, "invalid_credentials")
        if not u["confirmed"]:
            raise SupabaseError(400, "email_not_confirmed")
        return self._issue(u)

    def refresh(self, refresh_token):
        self._up()
        self.calls.append(("refresh", {}))
        email = self.refresh_tokens.pop(refresh_token, None)
        if email is None or self.refresh_rejects:
            raise SupabaseError(400, "refresh_token_not_found")
        return self._issue(self.users[email])

    def get_user(self, access_token):
        self._up()
        self.calls.append(("get_user", {}))
        email = self.access.get(access_token)
        if email is None:
            raise SupabaseError(401, "bad_jwt")
        u = self.users[email]
        return {"id": u["id"], "email": email, "user_metadata": dict(u["user_metadata"])}

    def logout(self, access_token):
        self.calls.append(("logout", {"scope": "local"}))
        self.access.pop(access_token, None)

    def recover(self, email, redirect_to):
        self._up()
        self.calls.append(("recover", {"email": email, "redirect_to": redirect_to}))

    def resend_signup(self, email, redirect_to):
        self.calls.append(("resend", {"email": email}))

    def verify_token_hash(self, token_hash, type_):
        self._up()
        if token_hash != "good-token-hash":
            raise SupabaseError(401, "otp_expired")
        return self._issue(self.users["hash@example.com"])

    def update_password(self, access_token, password):
        self._up()
        email = self.access.get(access_token)
        if email is None:
            raise SupabaseError(401, "bad_jwt")
        if password == "weakweak":
            raise SupabaseError(422, "weak_password")
        self.users[email]["password"] = password
        self.calls.append(("update_password", {}))
        return {}


@pytest.fixture()
def env(monkeypatch):
    fake, clock = FakeSupa(), Clock()
    monkeypatch.setattr(ws, "supa", fake)
    monkeypatch.setattr(ws, "limiter", RateLimiter())
    monkeypatch.setattr(ws, "_manager", SessionManager(ws.conn, fake, config.SESSION_SECRET, clock=clock))
    for table in ("auth_sessions", "app_users", "message_feedback", "conversations", "messages", "requests", "provider_attempts"):
        ws.conn.execute(f"DELETE FROM {table}")
    ws.conn.commit()
    proxy_main.guard.reset()
    proxy_main.router.reset()
    agent_calls = []

    def fake_run_agent(task, session_id=None, history=None, *, user_id=None, conversation_id=None):
        agent_calls.append({"task": task, "history": history, "user_id": user_id, "conversation_id": conversation_id,
                            "session_id": session_id})
        turn = "turn_" + secrets.token_hex(4)
        r = proxy.post("/v1/chat/completions",
                       headers={**INTERNAL, "X-Mirsad-User-Id": user_id, "X-Mirsad-Conversation-Id": conversation_id,
                                "X-Session-ID": session_id, "X-Turn-ID": turn},
                       json={"model": "mirsad-auto", "messages": [{"role": "user", "content": task}]})
        assert r.status_code == 200
        return {"answer": f"answer to: {task}", "steps": [], "status": "ok", "session_id": session_id, "turn_id": turn}

    monkeypatch.setattr(ws, "run_agent", fake_run_agent)
    return type("Env", (), {"supa": fake, "clock": clock, "agent_calls": agent_calls})


def browser():
    return TestClient(ws.app)           # its own cookie jar = its own browser


def sign_in(env, email, name="Test User"):
    """Returns (client, csrf, user_id) for a confirmed user who logged in through the real endpoint."""
    uid = env.supa.add_user(email, name)
    c = browser()
    r = c.post("/api/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    s = c.get("/api/auth/session").json()
    return c, s["csrf_token"], uid


def post(c, csrf, path, **kw):
    return c.post(path, headers={"X-CSRF-Token": csrf}, **kw)


def make_admin(uid):
    users.set_role(ws.conn, uid, "admin")


# ===================================================================== anonymous access
PROTECTED = [("get", "/api/conversations"), ("get", "/api/conversations/c_x"),
             ("get", "/api/metrics"), ("get", "/api/health"), ("get", "/api/admin/feedback"), ("get", "/api/admin/feedback/1"),
             ("delete", "/api/messages/1/feedback"), ("get", "/api/admin/users"), ("post", "/api/chat"), ("post", "/api/admin/reset"),
             ("delete", "/api/conversations/c_x"), ("post", "/api/auth/password")]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_anonymous_requests_are_rejected(env, method, path):
    r = getattr(browser(), method)(path, **({"json": {"task": "hi"}} if method == "post" and path == "/api/chat" else {}))
    assert r.status_code == 401 and r.json()["error"] == "not_authenticated"


def test_public_surface_is_minimal(env):
    c = browser()
    assert c.get("/healthz").json() == {"status": "ok"}
    s = c.get("/api/auth/session").json()
    assert s == {"authenticated": False}
    assert c.get("/").status_code == 200 and c.get("/auth/callback").status_code == 200
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_missing_configuration_is_a_setup_error_never_anonymous_access(env, monkeypatch):
    c = browser()
    monkeypatch.setattr(config, "SUPABASE_URL", "")
    monkeypatch.setattr(config, "INTERNAL_TOKEN", "")
    s = c.get("/api/auth/session").json()
    assert s["authenticated"] is False and set(s["setup_error"]) == {"SUPABASE_URL", "MIRSAD_INTERNAL_TOKEN"}
    for path in ("/api/admin/feedback", "/api/metrics", "/api/conversations"):
        r = c.get(path)
        assert r.status_code == 503 and r.json()["error"] == "setup_error"
    assert c.post("/api/auth/login", json={"email": "a@b.co", "password": "x"}).status_code == 503


# ===================================================================== signup / login / logout
def test_signup_requires_confirmation_and_sends_only_the_name(env):
    c = browser()
    r = c.post("/api/auth/signup", json={"name": "Sara", "email": "Sara@Example.com", "password": PW})
    assert r.json() == {"status": "confirmation_required"} and not c.cookies
    op, sent = env.supa.calls[0]
    assert op == "signup" and sent["name"] == "Sara" and sent["email"] == "sara@example.com"
    assert sent["redirect_to"] == config.PUBLIC_URL + "/auth/callback"
    assert c.post("/api/auth/login", json={"email": "sara@example.com", "password": PW}).json()["error"] == "email_not_confirmed"


def test_signup_without_email_confirmation_signs_in_as_plain_user(env):
    env.supa.confirm_required = False
    c = browser()
    r = c.post("/api/auth/signup", json={"name": "Sara", "email": "sara@example.com", "password": PW})
    assert r.status_code == 200 and r.json()["user"]["role"] == "user"
    assert c.get("/api/auth/session").json()["user"]["role"] == "user"


@pytest.mark.parametrize("extra", [{"role": "admin"}, {"data": {"role": "admin"}}, {"user_id": "x"}, {"app_metadata": {"role": "admin"}}])
def test_forged_signup_fields_are_rejected(env, extra):
    r = browser().post("/api/auth/signup", json={"name": "Eve", "email": "eve@example.com", "password": PW, **extra})
    assert r.status_code == 422 and not env.supa.calls and not env.supa.users


def test_new_accounts_always_get_the_user_role_even_with_an_admin_like_name(env):
    env.supa.confirm_required = False
    c = browser()
    c.post("/api/auth/signup", json={"name": "admin", "email": "admin@example.com", "password": PW})
    assert users.find_by_email(ws.conn, "admin@example.com")["role"] == "user"
    assert c.get("/api/metrics").status_code == 403


def test_a_role_planted_in_supabase_metadata_is_ignored(env):
    uid = env.supa.add_user("mallory@example.com", "Mallory")
    env.supa.users["mallory@example.com"]["user_metadata"].update(role="admin", is_admin=True)
    c = browser()
    assert c.post("/api/auth/login", json={"email": "mallory@example.com", "password": PW}).status_code == 200
    assert users.get(ws.conn, uid)["role"] == "user" and c.get("/api/admin/users").status_code == 403


def test_login_errors_are_generic_and_rate_limited(env):
    env.supa.add_user("a@example.com")
    c = browser()
    assert c.post("/api/auth/login", json={"email": "a@example.com", "password": "nope"}).json()["error"] == "invalid_credentials"
    assert c.post("/api/auth/login", json={"email": "ghost@example.com", "password": "nope"}).json()["error"] == "invalid_credentials"
    codes = [c.post("/api/auth/login", json={"email": "a@example.com", "password": "nope"}).status_code for _ in range(10)]
    assert codes[-1] == 429


def test_session_cookie_is_httponly_and_tokens_never_reach_the_browser(env):
    env.supa.add_user("a@example.com")
    c = browser()
    r = c.post("/api/auth/login", json={"email": "a@example.com", "password": PW})
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "samesite=lax" in cookie.lower() and "Path=/" in cookie
    assert "Secure" not in cookie                                    # plain http://localhost development
    access = next(iter(env.supa.access))
    blob = r.text + str(dict(r.headers)) + c.get("/api/auth/session").text
    assert access not in blob and next(iter(env.supa.refresh_tokens)) not in blob
    sid = c.cookies.get("mirsad_sid")
    row = ws.conn.execute("SELECT sid_hash, tokens_enc FROM auth_sessions").fetchone()
    assert row[0] == hash_sid(sid) and sid not in str(row)           # only the hash of the cookie is stored
    assert access.encode() not in bytes(row[1])                      # tokens are encrypted at rest


def test_secure_cookie_and_host_prefix_on_https(env, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_URL", "https://mirsad.example.com")
    env.supa.add_user("a@example.com")
    r = browser().post("/api/auth/login", json={"email": "a@example.com", "password": PW}, headers={"Origin": "https://mirsad.example.com"})
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("__Host-mirsad_sid=") and "Secure" in cookie and "HttpOnly" in cookie


def test_logout_needs_csrf_revokes_the_session_and_clears_the_cookie(env):
    c, csrf, _ = sign_in(env, "a@example.com")
    assert c.post("/api/auth/logout").status_code == 403            # authenticated mutation without CSRF token
    assert c.get("/api/conversations").status_code == 200
    r = post(c, csrf, "/api/auth/logout")
    assert r.status_code == 200 and ("logout", {"scope": "local"}) in env.supa.calls
    assert 'mirsad_sid=""' in r.headers["set-cookie"] or "Max-Age=0" in r.headers["set-cookie"]
    assert ws.conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0
    assert c.get("/api/conversations").status_code == 401


# ===================================================================== CSRF / origin
def test_cross_site_requests_are_rejected(env):
    c, csrf, _ = sign_in(env, "a@example.com")
    evil = {"Origin": "https://evil.example"}
    assert c.post("/api/auth/login", json={"email": "a@example.com", "password": PW}, headers=evil).status_code == 403
    assert c.post("/api/chat", json={"task": "hi"}, headers={**evil, "X-CSRF-Token": csrf}).status_code == 403
    assert c.post("/api/chat", json={"task": "hi"}, headers={"Sec-Fetch-Site": "cross-site", "X-CSRF-Token": csrf}).status_code == 403
    assert c.post("/api/chat", json={"task": "hi"}, headers={"X-CSRF-Token": csrf}).status_code == 200   # same-site control


def test_authenticated_mutations_require_the_csrf_token(env):
    c, csrf, _ = sign_in(env, "a@example.com")
    assert c.post("/api/chat", json={"task": "hi"}).status_code == 403
    assert c.post("/api/chat", json={"task": "hi"}, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert post(c, csrf, "/api/chat", json={"task": "hi"}).status_code == 200
    assert c.get("/api/conversations").status_code == 200           # reads need no token


# ===================================================================== expiry, refresh, revocation
def test_idle_and_absolute_expiry(env):
    c, _, _ = sign_in(env, "a@example.com")
    env.clock.advance(config.SESSION_IDLE_HOURS * 3600 - 10)
    assert c.get("/api/conversations").status_code == 200           # still inside the idle window (and refreshes it)
    env.clock.advance(config.SESSION_IDLE_HOURS * 3600 + 10)
    assert c.get("/api/conversations").status_code == 401
    assert ws.conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_absolute_lifetime_ends_an_active_session(env):
    c, _, _ = sign_in(env, "a@example.com")
    for _ in range(int(config.SESSION_MAX_DAYS * 24 / 11) + 2):
        env.clock.advance(11 * 3600)                                # never idle for 12h, but old
        if c.get("/api/conversations").status_code == 401:
            break
    assert c.get("/api/conversations").status_code == 401


def test_access_token_is_refreshed_before_it_expires(env):
    c, _, _ = sign_in(env, "a@example.com")
    old_access = next(iter(env.supa.access))
    env.clock.advance(3600 - 30)
    assert c.get("/api/conversations").status_code == 200
    assert ("refresh", {}) in env.supa.calls
    assert len(env.supa.access) == 2 and old_access in env.supa.access        # a second access token was issued


def test_rejected_refresh_ends_the_session(env):
    c, _, _ = sign_in(env, "a@example.com")
    env.supa.refresh_rejects = True
    env.clock.advance(3600 - 30)
    assert c.get("/api/conversations").status_code == 401


def test_session_revoked_at_supabase_is_noticed_within_the_revalidation_window(env):
    c, _, _ = sign_in(env, "a@example.com")
    env.supa.revoke_access(next(iter(env.supa.access)))             # signed out elsewhere / banned / deleted
    assert c.get("/api/conversations").status_code == 200           # not yet re-checked (inside the window)
    env.clock.advance(config.AUTH_REVALIDATE_S + 1)
    assert c.get("/api/conversations").status_code == 401


def test_supabase_outage_does_not_log_users_out_early(env):
    c, _, _ = sign_in(env, "a@example.com")
    env.supa.outage = True
    env.clock.advance(config.AUTH_REVALIDATE_S + 1)
    assert c.get("/api/conversations").status_code == 200


def test_garbage_forged_and_tampered_cookies(env):
    c, _, uid = sign_in(env, "a@example.com")
    for value in ("", "x" * 500, "not-a-real-session", uid, "admin"):
        b = browser()
        b.cookies.set("mirsad_sid", value)
        assert b.get("/api/conversations").status_code == 401
    ws.conn.execute("UPDATE auth_sessions SET tokens_enc=?", (b"garbage",))
    ws.conn.commit()
    assert c.get("/api/conversations").status_code == 401           # undecryptable record = no session


def test_disabled_account_is_locked_out_immediately(env):
    c, _, uid = sign_in(env, "a@example.com")
    ws.conn.execute("UPDATE app_users SET disabled=1 WHERE user_id=?", (uid,))
    ws.conn.commit()
    assert c.get("/api/conversations").status_code == 401


# ===================================================================== recovery and callbacks
def test_recover_never_reveals_whether_an_account_exists(env):
    env.supa.add_user("real@example.com")
    a = browser().post("/api/auth/recover", json={"email": "real@example.com"}).json()
    b = browser().post("/api/auth/recover", json={"email": "ghost@example.com"}).json()
    assert a == b == {"status": "sent"}
    assert env.supa.calls[0][1]["redirect_to"] == config.PUBLIC_URL + "/auth/callback"


def test_password_reset_flow_locks_the_app_until_a_new_password_is_set(env):
    uid = env.supa.add_user("r@example.com")
    issued = env.supa._issue(env.supa.users["r@example.com"])      # what the emailed link makes Supabase issue
    c = browser()
    r = c.post("/api/auth/callback", json={"access_token": issued["access_token"],
                                           "refresh_token": issued["refresh_token"], "type": "recovery"})
    assert r.status_code == 200 and r.json()["recovery"] is True
    s = c.get("/api/auth/session").json()
    assert s["recovery"] is True
    assert c.get("/api/conversations").json()["error"] == "password_reset_required"
    assert post(c, s["csrf_token"], "/api/auth/password", json={"password": "weakweak"}).json()["error"] == "weak_password"
    assert post(c, s["csrf_token"], "/api/auth/password", json={"password": "a-brand-new-password"}).status_code == 200
    assert c.get("/api/conversations").status_code == 200
    assert env.supa.users["r@example.com"]["password"] == "a-brand-new-password"
    assert users.get(ws.conn, uid)["role"] == "user"


def test_callback_rejects_invalid_links_and_verifies_with_supabase(env):
    c = browser()
    bad = c.post("/api/auth/callback", json={"access_token": "a" * 40, "refresh_token": "b" * 40, "type": "signup"})
    assert bad.status_code == 400 and bad.json()["error"] == "invalid_link"
    assert c.post("/api/auth/callback", json={"access_token": "short", "refresh_token": "x"}).status_code == 400
    assert c.post("/api/auth/callback", json={"token_hash": "bad", "type": "signup"}).status_code == 400
    env.supa.add_user("hash@example.com")
    ok = c.post("/api/auth/callback", json={"token_hash": "good-token-hash", "type": "signup"})
    assert ok.status_code == 200 and ok.json()["recovery"] is False
    assert not c.get("/api/auth/session").json()["recovery"]


def test_callback_cannot_smuggle_a_role_or_user_id(env):
    r = browser().post("/api/auth/callback", json={"access_token": "a" * 40, "refresh_token": "b" * 40, "role": "admin"})
    assert r.status_code == 422


# ===================================================================== ownership of conversations
def chat(c, csrf, task, conversation_id=None):
    body = {"task": task, **({"conversation_id": conversation_id} if conversation_id else {})}
    return post(c, csrf, "/api/chat", json=body)


def test_users_cannot_read_modify_or_delete_each_others_conversations(env):
    a, csrf_a, uid_a = sign_in(env, "a@example.com")
    b, csrf_b, uid_b = sign_in(env, "b@example.com")
    conv = chat(a, csrf_a, "A's private question").json()["conversation_id"]
    assert chat(a, csrf_a, "follow up", conv).status_code == 200

    assert b.get(f"/api/conversations/{conv}").status_code == 404
    assert b.get(f"/api/conversations/{conv}").json() == b.get("/api/conversations/c_doesnotexist").json()   # no existence leak
    assert chat(b, csrf_b, "injected", conv).status_code == 404
    assert post(b, csrf_b, f"/api/conversations/{conv}", json={}).status_code in (404, 405)
    assert b.delete(f"/api/conversations/{conv}", headers={"X-CSRF-Token": csrf_b}).status_code == 404
    assert b.get("/api/conversations").json()["conversations"] == []

    msgs = a.get(f"/api/conversations/{conv}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert all("injected" not in m["content"] for m in msgs)
    assert a.delete(f"/api/conversations/{conv}", headers={"X-CSRF-Token": csrf_a}).status_code == 200
    assert a.get(f"/api/conversations/{conv}").status_code == 404
    assert ws.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_chat_history_comes_from_the_server_and_the_browser_cannot_inject_identity(env):
    a, csrf, uid = sign_in(env, "a@example.com")
    for forged in ({"history": [{"role": "user", "content": "x"}]}, {"user_id": "someone-else"}, {"role": "admin"}):
        assert post(a, csrf, "/api/chat", json={"task": "hi", **forged}).status_code == 422
    first = chat(a, csrf, "What is Flutter?").json()
    chat(a, csrf, "and Dart?", first["conversation_id"])
    call = env.agent_calls[-1]
    assert call["user_id"] == uid and call["conversation_id"] == first["conversation_id"]
    assert [m["content"] for m in call["history"]] == ["What is Flutter?", "answer to: What is Flutter?"]
    assert "session_id" not in first                                  # internal id is not sent to the browser


def test_each_request_is_attributed_to_the_verified_user(env):
    a, csrf_a, uid_a = sign_in(env, "a@example.com")
    b, csrf_b, uid_b = sign_in(env, "b@example.com")
    conv_a = chat(a, csrf_a, "one").json()["conversation_id"]
    chat(b, csrf_b, "two")
    df = metrics.load()
    assert set(df["user_id"]) == {uid_a, uid_b}
    assert df[df["user_id"] == uid_a].iloc[0]["conversation_id"] == conv_a


# ===================================================================== usage isolation + roles
def test_the_my_usage_endpoints_no_longer_exist_for_anyone(env):
    a, csrf_a, _ = sign_in(env, "a@example.com")
    adm, _, uid_adm = sign_in(env, "boss@example.com")
    make_admin(uid_adm)
    chat(a, csrf_a, "first")
    for client in (a, adm):
        assert client.get("/api/usage").status_code == 404
        assert client.get("/api/usage?user_id=anyone").status_code == 404
        assert client.get("/api/requests/req_1").status_code == 404


def test_per_user_request_records_stay_separate_and_only_an_admin_sees_them(env):
    a, csrf_a, uid_a = sign_in(env, "a@example.com")
    b, csrf_b, uid_b = sign_in(env, "b@example.com")
    adm, _, uid_adm = sign_in(env, "boss@example.com")
    make_admin(uid_adm)
    chat(a, csrf_a, "first"); chat(a, csrf_a, "second"); chat(b, csrf_b, "only b")
    assert len(metrics.load(user_id=uid_a)) == 2 and len(metrics.load(user_id=uid_b)) == 1
    assert not set(metrics.load(user_id=uid_a)["request_id"]) & set(metrics.load(user_id=uid_b)["request_id"])
    assert len(metrics.load(user_id="someone-else")) == 0
    assert {r["user_id"] for r in adm.get("/api/metrics").json()["recent"]} == {uid_a, uid_b}
    for client in (a, b):                                                     # ordinary users: no global view in any form
        assert client.get("/api/metrics").status_code == 403


def test_one_browser_switching_accounts_never_shows_the_previous_accounts_data(env):
    """Two accounts in ONE cookie jar (the way two tabs behave): the newest login owns the cookie, and the old tab's
    CSRF token stops working instead of acting as the new user."""
    a, csrf_a, uid_a = sign_in(env, "a@example.com")
    chat(a, csrf_a, "A's private question")
    env.supa.add_user("b@example.com", "B")
    assert a.post("/api/auth/login", json={"email": "b@example.com", "password": PW}).status_code == 200
    assert a.get("/api/conversations").json()["conversations"] == []         # B's list, not A's
    assert a.get("/api/auth/session").json()["user"]["email"] == "b@example.com"
    assert chat(a, csrf_a, "stale tab of A").status_code == 403              # A's tab cannot post as B


def test_ordinary_users_cannot_reach_global_monitoring_or_reset(env, monkeypatch):
    monkeypatch.setattr(ws.httpx, "get", lambda url, **kw: proxy.get("/health", headers=kw["headers"]))
    a, csrf, _ = sign_in(env, "a@example.com")
    chat(a, csrf, "hello")
    for path in ("/api/metrics", "/api/admin/users", "/api/admin/feedback", "/api/admin/feedback/1"):
        r = a.get(path)
        assert r.status_code == 403 and r.json()["error"] == "forbidden"
    assert post(a, csrf, "/api/admin/reset").status_code == 403
    assert ws.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1          # nothing was deleted
    h = a.get("/api/health").json()["proxy"]
    assert set(h) <= {"status", "mock_mode", "execution", "routing_mode"}                # no credentials/classifier details


def test_admin_sees_everything_and_can_reset_with_csrf(env, monkeypatch):
    monkeypatch.setattr(ws.httpx, "get", lambda url, **kw: proxy.get("/health", headers=kw["headers"]))   # never a real port 8000
    a, csrf_a, _ = sign_in(env, "a@example.com")
    adm, csrf_adm, uid_adm = sign_in(env, "boss@example.com")
    make_admin(uid_adm)
    chat(a, csrf_a, "from a")
    m = adm.get("/api/metrics").json()
    assert m["summary"]["total"] == 1 and m["recent"][0]["user_id"]
    users_json = adm.get("/api/admin/users").json()["users"]
    assert {u["email"]: u["role"] for u in users_json} == {"a@example.com": "user", "boss@example.com": "admin"}
    assert adm.post("/api/admin/reset").status_code == 403                                # CSRF still required for admins
    assert "credentials" in adm.get("/api/health").json()["proxy"]


def test_admin_reset_clears_monitoring_data(env, monkeypatch):
    adm, csrf, uid = sign_in(env, "boss@example.com")
    make_admin(uid)
    chat(adm, csrf, "x")
    monkeypatch.setattr(ws.httpx, "post", lambda url, **kw: proxy.post("/admin/reset", headers=kw["headers"]))
    assert post(adm, csrf, "/api/admin/reset").json()["status"] == "reset"
    assert ws.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


def test_legacy_records_without_an_owner_are_admin_only(env):
    legacy = {"request_id": "req_legacy1", "ts": 1.0, "session_id": "old", "status": "success", "route": "simple",
              "model": "m", "input_tokens": 5, "output_tokens": 5, "latency_ms": 1.0, "actual_cost": 0.1, "baseline_cost": 0.2}
    db.insert(ws.conn, legacy)
    a, csrf_a, uid_a = sign_in(env, "a@example.com")
    adm, _, uid_adm = sign_in(env, "boss@example.com")
    make_admin(uid_adm)
    assert len(metrics.load(user_id=uid_a)) == 0
    assert a.get("/api/metrics").status_code == 403
    assert adm.get("/api/metrics").json()["summary"]["total"] == 1


# ===================================================================== the proxy on port 8000
def test_proxy_rejects_callers_without_the_internal_token(env):
    body = {"model": "mirsad-auto", "messages": [{"role": "user", "content": "hi"}]}
    uid = {"X-Mirsad-User-Id": "victim"}
    assert proxy.post("/v1/chat/completions", json=body).status_code == 401
    assert proxy.post("/v1/chat/completions", json=body, headers=uid).status_code == 401           # forged owner, no token
    wrong = {"Authorization": "Bearer wrong-token", **uid}
    assert proxy.post("/v1/chat/completions", json=body, headers=wrong).status_code == 401
    assert proxy.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer ", **uid}).status_code == 401
    assert proxy.get("/v1/models").status_code == 401
    assert proxy.post("/admin/reset").status_code == 401
    assert ws.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0


def test_proxy_requires_an_attributed_user_and_validates_the_id(env):
    body = {"model": "mirsad-auto", "messages": [{"role": "user", "content": "hi"}]}
    assert proxy.post("/v1/chat/completions", json=body, headers=INTERNAL).status_code == 400
    bad = {**INTERNAL, "X-Mirsad-User-Id": "a'; DROP TABLE requests;--"}
    assert proxy.post("/v1/chat/completions", json=body, headers=bad).status_code == 400
    ok = {**INTERNAL, "X-Mirsad-User-Id": "user-1"}
    assert proxy.post("/v1/chat/completions", json=body, headers=ok).status_code == 200
    assert metrics.load().iloc[-1]["user_id"] == "user-1"


def test_proxy_without_a_configured_token_refuses_everything(env, monkeypatch):
    monkeypatch.setattr(config, "INTERNAL_TOKEN", "")
    r = proxy.post("/v1/chat/completions", headers={"Authorization": "Bearer ", "X-Mirsad-User-Id": "u"},
                   json={"model": "m", "messages": []})
    assert r.status_code == 503 and r.json()["error"]["type"] == "setup_error"


def test_proxy_health_is_minimal_for_the_public(env):
    assert proxy.get("/health").json() == {"status": "ok"}
    full = proxy.get("/health", headers=INTERNAL).json()
    assert "routing_mode" in full and "credentials" in full


def test_loop_guard_and_routing_state_are_per_user(env):
    def turn(cid):
        return [{"role": "assistant", "content": None, "tool_calls": [{"id": cid, "type": "function",
                 "function": {"name": "fetch_report", "arguments": '{"report_name": "q3"}'}}]},
                {"role": "tool", "tool_call_id": cid, "content": "ERROR: 503"}]

    def call(user, msgs):
        return proxy.post("/v1/chat/completions", headers={**INTERNAL, "X-Mirsad-User-Id": user, "X-Session-ID": "shared-name"},
                          json={"model": "m", "messages": msgs}).status_code
    base = [{"role": "user", "content": "get report"}]
    assert call("alice", base + turn("a1") + turn("a2")) == 200            # alice: 2 failures
    assert call("bob", base + turn("b1")) == 200                           # same session NAME, different user: his own counter (1)
    assert call("alice", base + turn("a1") + turn("a2") + turn("a3")) == 429
    assert call("bob", base + turn("b1") + turn("b2")) == 200              # bob unaffected by alice's block


def test_agent_sends_internal_credentials_and_requires_an_owner(monkeypatch):
    from agent import agent as agent_mod
    captured = {}

    class FakeChat:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(agent_mod, "ChatOpenAI", FakeChat)
    monkeypatch.setattr(agent_mod, "create_agent", lambda *a, **k: object())
    agent_mod.build_agent("sess_1", "turn_1", "user-9", "c_1")
    assert captured["api_key"] == config.INTERNAL_TOKEN
    assert captured["default_headers"]["X-Mirsad-User-Id"] == "user-9" and captured["default_headers"]["X-Turn-ID"] == "turn_1"
    with pytest.raises(RuntimeError):
        agent_mod.build_agent("sess_1", "turn_1", None)
    monkeypatch.setattr(config, "INTERNAL_TOKEN", "")
    with pytest.raises(RuntimeError):
        agent_mod.build_agent("sess_1", "turn_1", "user-9")


# ===================================================================== admin tooling
def test_first_admin_can_only_be_granted_locally_and_the_last_admin_is_protected(env, capsys):
    uid = env.supa.add_user("me@example.com")
    assert manage_admin.main(["grant", "me@example.com"]) == 1                # never signed in: no row yet
    c = browser()
    c.post("/api/auth/login", json={"email": "me@example.com", "password": PW})
    assert c.get("/api/metrics").status_code == 403
    assert manage_admin.main(["grant", "ME@example.com"]) == 0
    assert c.get("/api/metrics").status_code == 200                           # effective on the next request, no re-login
    assert manage_admin.main(["revoke", "me@example.com"]) == 1               # refuses to remove the last admin
    assert manage_admin.main(["list"]) == 0 and "admin  me@example.com" in capsys.readouterr().out
    assert users.get(ws.conn, uid)["role"] == "admin"


# ===================================================================== secrets and logs
def test_secrets_never_reach_responses_logs_or_errors(env, caplog):
    caplog.set_level(logging.DEBUG)
    env.supa.add_user("a@example.com")
    c = browser()
    r1 = c.post("/api/auth/login", json={"email": "a@example.com", "password": PW})
    csrf = c.get("/api/auth/session").json()["csrf_token"]
    r2 = chat(c, csrf, "hello")
    r3 = c.post("/api/auth/login", json={"email": "a@example.com", "password": "wrong-password-123"})
    secrets_ = [next(iter(env.supa.access)), next(iter(env.supa.refresh_tokens)), config.INTERNAL_TOKEN,
                config.SESSION_SECRET, config.SUPABASE_PUBLISHABLE_KEY, PW, "wrong-password-123", c.cookies.get("mirsad_sid")]
    blob = " ".join([r1.text, r2.text, r3.text, str(dict(r2.headers)), str(dict(r3.headers)), caplog.text,
                     proxy.get("/health").text, ws.app.openapi_url or ""])
    for s in secrets_:
        assert s not in blob
    assert config.SUPABASE_PUBLISHABLE_KEY not in c.get("/").text


# ===================================================================== storage safety
def test_rate_limiter_blocks_chat_floods(env, monkeypatch):
    monkeypatch.setattr(config, "CHAT_RATE_PER_MIN", 3)
    a, csrf, _ = sign_in(env, "a@example.com")
    codes = [chat(a, csrf, f"m{i}").status_code for i in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]


def test_migration_backs_up_and_preserves_existing_records(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, ts REAL, session_id TEXT, "
                "status TEXT, route TEXT, score INTEGER, reason TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER, "
                "latency_ms REAL, actual_cost REAL, baseline_cost REAL, est_saved_cost REAL, has_tool_calls INTEGER, error TEXT)")
    old.executemany("INSERT INTO requests (request_id, ts, status) VALUES (?,?,?)", [("r1", 1.0, "success"), ("r2", 2.0, "error")])
    old.commit(); old.close()
    conn = db.connect(path)
    assert conn.execute("SELECT request_id, user_id FROM requests ORDER BY id").fetchall() == [("r1", None), ("r2", None)]
    backup = sqlite3.connect(tmp_path / "old.db-backup-pre-auth")
    assert backup.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 2
    assert "user_id" not in [r[1] for r in backup.execute("PRAGMA table_info(requests)")]   # pre-migration copy
    assert {"app_users", "auth_sessions", "conversations", "messages"} <= {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    db.connect(path)                                                          # idempotent


def test_conversation_store_scopes_every_query_by_owner():
    conn = ws.conn
    c = chatstore.create(conn, "owner-1", "hello")
    assert chatstore.get(conn, c["id"], "owner-2") is None
    assert chatstore.add_message(conn, c["id"], "owner-2", "user", "x") is None
    assert chatstore.messages(conn, c["id"], "owner-2") is None
    assert chatstore.rotate_session(conn, c["id"], "owner-2") is None
    assert chatstore.delete(conn, c["id"], "owner-2") is False
    assert chatstore.get(conn, c["id"], "owner-1") is not None
