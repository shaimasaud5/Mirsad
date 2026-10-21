"""Tests for the web UI server (ui/web_server.py), offline."""
import json

from fastapi.testclient import TestClient

from proxy.main import app as proxy_app
from ui import web_server

client = TestClient(web_server.app)
proxy = TestClient(proxy_app)


def test_page_logo_and_bundled_libraries_are_served():
    assert "<title>Mirsad</title>" in client.get("/").text
    assert client.get("/logo.svg").status_code == 200
    for lib in ("marked.min.js", "purify.min.js", "chart.umd.js"):
        assert client.get(f"/vendor/{lib}").status_code == 200


def test_metrics_after_a_routed_request_are_json_safe():
    proxy.post("/admin/reset")
    proxy.post("/v1/chat/completions", headers={"X-Session-ID": "ui-test", "X-Turn-ID": "turn-ui"},
               json={"model": "mirsad-auto", "messages": [{"role": "user", "content": "What is 2+2?"}]})
    data = client.get("/api/metrics").json()
    json.dumps(data, allow_nan=False)                     # no NaN sneaks into the response
    assert data["summary"]["total"] >= 1
    assert data["recent"][0]["route"] == "simple"
    assert set(data["alert_data"]) >= {"window_min", "blocked", "error_rate"}


def test_chat_returns_agent_result_with_turn_summary(monkeypatch):
    proxy.post("/admin/reset")

    def fake_run_agent(task, session_id=None, history=None):
        proxy.post("/v1/chat/completions", headers={"X-Session-ID": "s1", "X-Turn-ID": "t1"},
                   json={"model": "mirsad-auto", "messages": [{"role": "user", "content": task}]})
        return {"answer": "4", "steps": [], "status": "ok", "session_id": "s1", "turn_id": "t1"}

    monkeypatch.setattr(web_server, "run_agent", fake_run_agent)
    r = client.post("/api/chat", json={"task": "What is 2+2?", "history": []}).json()
    assert r["answer"] == "4"
    assert r["summary"]["route"] == "simple" and r["summary"]["calls"] == 1
    assert r["summary"]["baseline_cost"] >= r["summary"]["actual_cost"]
