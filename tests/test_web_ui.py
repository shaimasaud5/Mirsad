"""Tests for the web UI server (ui/web_server.py), offline. Auth/permission cases live in test_auth.py."""
import json

from fastapi.testclient import TestClient

import config
from test_auth import chat, env, make_admin, sign_in  # noqa: F401  (env is a pytest fixture)
from ui import web_server

client = TestClient(web_server.app)


def test_page_logo_and_bundled_libraries_are_served():
    page = client.get("/")
    assert "<title>Mirsad</title>" in page.text
    for img in ("logo-light.png", "logo-dark.png", "favicon.png", "apple-touch-icon.png"):
        assert client.get(f"/{img}").status_code == 200
    assert client.get("/secret.png").status_code == 404
    for lib in ("marked.min.js", "purify.min.js", "chart.umd.js"):
        assert client.get(f"/vendor/{lib}").status_code == 200
    # no secret ever ships inside the page, and the browser is never handed a Supabase key at all
    for secret in (config.INTERNAL_TOKEN, config.SESSION_SECRET, config.SUPABASE_PUBLISHABLE_KEY, config.SUPABASE_URL):
        assert secret and secret not in page.text


def test_security_headers_are_present():
    h = client.get("/").headers
    assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in h["content-security-policy"] and h["referrer-policy"] == "no-referrer"
    assert client.get("/api/auth/session").headers["cache-control"] == "no-store"


def test_admin_metrics_are_json_safe_after_a_routed_request(env):
    adm, csrf, uid = sign_in(env, "boss@example.com")
    make_admin(uid)
    assert chat(adm, csrf, "What is 2+2?").status_code == 200
    data = adm.get("/api/metrics?kind=all").json()
    json.dumps(data, allow_nan=False)                     # no NaN sneaks into the response
    assert data["summary"]["total"] >= 1 and data["recent"][0]["route"] == "simple"
    assert set(data["alert_data"]) >= {"window_min", "blocked", "error_rate"}


def test_chat_reply_has_no_technical_summary_but_admin_telemetry_is_recorded(env):
    c, csrf, uid = sign_in(env, "a@example.com")
    adm, _, uid_adm = sign_in(env, "boss@example.com")
    make_admin(uid_adm)
    r = chat(c, csrf, "What is 2+2?").json()
    assert r["answer"].startswith("answer to:") and r["status"] == "ok" and r["conversation_id"].startswith("c_")
    assert set(r) == {"answer", "status", "conversation_id", "message_id", "files", "feedback"}
    stored = c.get(f"/api/conversations/{r['conversation_id']}").json()["messages"]
    blob = json.dumps(stored) + json.dumps(r)
    for leaked in ("summary", "route", "model", "cost", "baseline", "classifier", "failover", "tokens", "steps", "gpt"):
        assert leaked not in blob.lower().replace("answer to:", "")
    row = adm.get("/api/metrics?kind=all").json()["recent"][0]                    # the admin still gets all of it
    assert row["route"] == "simple" and row["user_id"] == uid and row["usage_kind"] == "mock" and row["est_cost"] is None   # mock is never priced
    assert ws_rows(uid) == 1


def ws_rows(uid):
    return web_server.conn.execute("SELECT COUNT(*) FROM requests WHERE user_id=? AND turn_id IS NOT NULL", (uid,)).fetchone()[0]


def test_users_get_a_safe_error_not_the_raw_provider_text(env, monkeypatch):
    c, csrf, _ = sign_in(env, "a@example.com")
    raw = "litellm.RateLimitError: GroqException 429 gsk_SECRET at /srv/app/proxy/groq.py"
    monkeypatch.setattr(web_server, "run_agent", lambda *a, **k: {"answer": raw, "steps": [], "status": "error",
                                                                  "session_id": k["conversation_id"], "turn_id": "turn_x"})
    r = chat(c, csrf, "hello").json()
    assert r["status"] == "error" and r["answer"] == "busy" and "gsk_" not in json.dumps(r)
    again = c.get(f"/api/conversations/{r['conversation_id']}").json()["messages"][-1]
    assert again["content"] == "busy" and "gsk_" not in json.dumps(again)
