"""End-to-end tests of the HTTP proxy, offline (mock model)."""
import os
from fastapi.testclient import TestClient

from monitoring import metrics
from proxy.main import app

client = TestClient(app, headers={"Authorization": "Bearer " + os.environ["MIRSAD_INTERNAL_TOKEN"], "X-Mirsad-User-Id": "test-user"})


def _tool_turn(cid, ok):
    return [{"role": "assistant", "content": None, "tool_calls": [{
                "id": cid, "type": "function",
                "function": {"name": "fetch_report", "arguments": "{\"report_name\": \"q3\"}"}}]},
            {"role": "tool", "tool_call_id": cid, "content": "ok" if ok else "ERROR: 503"}]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_simple_request_is_routed_and_logged():
    client.post("/admin/reset")
    r = client.post("/v1/chat/completions", headers={"X-Session-ID": "t1"},
                    json={"model": "x", "messages": [{"role": "user", "content": "Say hi"}]})
    assert r.status_code == 200
    assert r.headers["X-Mirsad-Route"] == "simple"
    assert r.json()["choices"][0]["message"]["content"]
    row = metrics.load().iloc[-1]
    assert row["status"] == "success" and row["route"] == "simple"
    assert row["token_source"] == "mock-estimate" and row["execution_mode"] == "mock"
    assert row["est_cost_usd"] is None or row["est_cost_usd"] != row["est_cost_usd"]    # mock usage is never priced
    assert row["model_call_ms"] > 0 and row["latency_ms"] >= row["model_call_ms"]


def test_loop_is_blocked_with_429_and_logged():
    client.post("/admin/reset")
    msgs = [{"role": "user", "content": "get the report"}]
    for i in range(3):
        msgs += _tool_turn(f"c{i}", ok=False)
    r = client.post("/v1/chat/completions", headers={"X-Session-ID": "t2"},
                    json={"model": "x", "messages": msgs})
    assert r.status_code == 429
    assert "Loop blocked" in r.json()["error"]["message"]
    s = metrics.summarize(metrics.load())
    assert s["calls"]["blocked"] == 1 and s["calls"]["model_calls"] == 0     # blocked: no model call, no invented saving
    assert s["cost"]["estimated_cost"] is None
