"""Routing modes, per-turn classification, classifier failures, loop guard and DB migration.
All offline: the Ollama call is stubbed or served by httpx.MockTransport; answers come from the mock model."""
import os
import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

import config
from monitoring import metrics
from proxy import classifier, db
from proxy import main as pm
from proxy.classifier import ClassifierResult
from proxy.routing import TurnRouter

client = TestClient(pm.app, headers={"Authorization": "Bearer " + os.environ["MIRSAD_INTERNAL_TOKEN"], "X-Mirsad-User-Id": "test-user"})


class Stub:
    """Stands in for the Ollama classifier and counts calls."""
    def __init__(self, label="complex", error=None):
        self.label, self.error, self.calls = label, error, 0

    def __call__(self, messages):
        self.calls += 1
        if self.error:
            return ClassifierResult(error=self.error, duration_ms=3.0)
        return ClassifierResult(label=self.label, reason="stub reason", duration_ms=12.0)


def U(text):
    return {"role": "user", "content": text}


@pytest.fixture(autouse=True)
def _clean():
    classifier.reset_cooldown()
    client.post("/admin/reset")
    yield


def use_router(monkeypatch, mode, stub):
    r = TurnRouter(mode=mode, classify_fn=stub)
    monkeypatch.setattr(pm, "router", r)
    return r


# ------------------------------------------------------------------ modes
def test_rules_mode_never_calls_classifier():
    stub = Stub()
    d = TurnRouter(mode="rules", classify_fn=stub).decide("s", "t1", [U("What is the capital of France?")])
    assert stub.calls == 0 and d.source == "rules" and d.route == "simple" and d.model == config.SMALL_MODEL


def test_local_llm_controls_the_answering_model():
    big = TurnRouter(mode="local_llm", classify_fn=Stub("complex")).decide("s", "t", [U("hi")])
    small = TurnRouter(mode="local_llm", classify_fn=Stub("simple")).decide("s", "t", [U("Design a distributed database")])
    assert (big.model, big.source, big.route) == (config.BIG_MODEL, "local_llm", "complex")
    assert (small.model, small.source, small.route) == (config.SMALL_MODEL, "local_llm", "simple")
    assert big.rules_route == "simple"          # rules disagreed and were overruled


def test_shadow_records_llm_but_rules_decide():
    d = TurnRouter(mode="shadow", classify_fn=Stub("complex")).decide("s", "t", [U("What is Flutter?")])
    assert d.llm_route == "complex" and d.route == "simple" and d.source == "rules"
    assert d.model == config.SMALL_MODEL


# ------------------------------------------------------------------ lifecycle
def test_one_classification_per_turn_and_new_turn_reclassifies():
    stub = Stub()
    r = TurnRouter(mode="local_llm", classify_fn=stub)
    msgs = [U("Plan a migration")]
    a = r.decide("s", "turn-1", msgs)
    tool_step = msgs + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    b = r.decide("s", "turn-1", tool_step)             # tool step of the same turn
    assert stub.calls == 1 and not a.reused and b.reused and b.model == a.model
    r.decide("s", "turn-2", msgs)                      # identical text, new turn -> classified again
    assert stub.calls == 2


def test_no_header_fallback_key_distinguishes_repeated_text_by_position():
    stub = Stub()
    r = TurnRouter(mode="local_llm", classify_fn=stub)
    first = [U("same text")]
    second = first + [{"role": "assistant", "content": "ok"}, U("same text")]
    r.decide("s", None, first); r.decide("s", None, first)
    assert stub.calls == 1
    r.decide("s", None, second)                         # same text, later in the conversation
    assert stub.calls == 2


def test_cache_is_bounded():
    r = TurnRouter(mode="local_llm", classify_fn=Stub(), max_entries=3)
    for i in range(5):
        r.decide("s", f"t{i}", [U("x")])
    assert len(r._cache) == 3


# ------------------------------------------------------------------ failures -> rules fallback
@pytest.mark.parametrize("err", ["timeout", "ollama_unavailable", "invalid_json", "unknown_label",
                                 "context_overflow", "http_error_500"])
def test_classifier_failure_falls_back_to_rules_with_reason(err):
    d = TurnRouter(mode="local_llm", classify_fn=Stub(error=err)).decide(
        "s", "t", [U("Design a fault tolerant system and compare trade-offs")])
    assert d.source == "rules_fallback" and d.fallback_reason == err
    assert d.route == d.rules_route == "complex"        # the rules' decision is used


def _ollama(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(content, tokens=900):
    return lambda req: httpx.Response(200, json={"message": {"content": content}, "prompt_eval_count": tokens})


def test_classify_text_success_and_validation():
    good = json.dumps({"classification": "complex", "reason": "design task"})
    r = classifier.classify_text("Design X", _ollama(_ok(good)))
    assert (r.label, r.reason, r.error) == ("complex", "design task", None)
    assert classifier.classify_text("x", _ollama(_ok("not json"))).error == "invalid_json"
    bad = json.dumps({"classification": "medium", "reason": "?"})
    assert classifier.classify_text("x", _ollama(_ok(bad))).error == "unknown_label"
    assert classifier.classify_text("x", _ollama(_ok(good, tokens=config.CLASSIFIER_NUM_CTX))).error == "context_overflow"
    assert classifier.classify_text("x", _ollama(lambda q: httpx.Response(500))).error == "http_error_500"


def test_timeout_and_unreachable_start_cooldown():
    def boom(req):
        raise httpx.ReadTimeout("slow")
    assert classifier.classify_text("x", _ollama(boom)).error == "timeout"
    assert classifier.classify_text("x", _ollama(_ok("{}"))).error == "classifier_cooldown:timeout"   # skipped, no wait
    classifier.reset_cooldown()

    def refused(req):
        raise httpx.ConnectError("refused")
    assert classifier.classify_text("x", _ollama(refused)).error == "ollama_unavailable"
    classifier.reset_cooldown()


def test_prompt_id_is_stable_v1():
    assert classifier.PROMPT_ID.startswith("v1-") and len(classifier.FEWSHOT) == 9


# ------------------------------------------------------------------ bounded input
def test_input_without_history_equals_plain_request():
    assert classifier.build_input([U("What is Flutter?")]) == ("What is Flutter?", False, False)


def test_context_is_bounded_flagged_and_excludes_tools():
    msgs = []
    for i in range(6):
        msgs += [U(f"q{i} " + "x" * 500), {"role": "assistant", "content": f"a{i}"},
                 {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": "TOOLOUTPUT"}]
    msgs.append(U("and now the follow-up?"))
    text, cur_trunc, ctx_trunc = classifier.build_input(msgs)
    assert ctx_trunc and not cur_trunc
    assert "TOOLOUTPUT" not in text and "...[cut]" in text
    assert text.count("\nUser:") + text.count("\nAssistant:") <= config.CONTEXT_MAX_MESSAGES
    assert text.endswith("and now the follow-up?") and classifier.CURRENT_HEADER in text


def test_long_current_request_is_flagged_not_silent():
    text, cur_trunc, _ = classifier.build_input([U("a" * (config.CLASSIFIER_MAX_CHARS + 10))])
    assert cur_trunc and len(text) == config.CLASSIFIER_MAX_CHARS


# ------------------------------------------------------------------ HTTP: proxy end to end (mock answers)
def post(messages, session="sess", turn="turn-A"):
    return client.post("/v1/chat/completions", headers={"X-Session-ID": session, "X-Turn-ID": turn},
                       json={"model": "mirsad-auto", "messages": messages})


def tool_turn(cid, ok, args='{"report_name": "q3"}'):
    return [{"role": "assistant", "content": None, "tool_calls": [{
        "id": cid, "type": "function", "function": {"name": "fetch_report", "arguments": args}}]},
        {"role": "tool", "tool_call_id": cid, "content": "ok" if ok else "ERROR: 503"}]


def test_http_local_llm_row_is_complete_and_step_reuses_classification(monkeypatch):
    stub = Stub("complex")
    use_router(monkeypatch, "local_llm", stub)
    msgs = [U("Read sales.csv and tell me the total revenue.")]
    r1 = post(msgs)
    assert r1.status_code == 200 and r1.headers["X-Mirsad-Route-Source"] == "local_llm"
    step2 = msgs + [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "k1", "type": "function", "function": {"name": "read_file", "arguments": '{"name":"sales.csv"}'}}]},
        {"role": "tool", "tool_call_id": "k1", "content": "month,region\n..."}]
    post(step2)
    df = metrics.load()
    a, b = df.iloc[-2], df.iloc[-1]
    assert stub.calls == 1
    assert a["model"] == b["model"] == config.BIG_MODEL and a["route"] == "complex"
    assert (a["classification_reused"], b["classification_reused"]) == (0, 1)
    assert a["classifier_ms"] == 12.0 and b["classifier_ms"] == 0.0
    assert a["routing_mode"] == "local_llm" and a["execution_mode"] == "mock" and a["token_source"] == "mock-estimate"
    assert a["llm_route"] == "complex" and a["rules_route"] == "simple" and a["turn_id"] == "turn-A"
    post(msgs, turn="turn-B")                           # same text, new turn
    assert stub.calls == 2


def test_http_classifier_failure_is_recorded_and_not_hidden(monkeypatch):
    use_router(monkeypatch, "local_llm", Stub(error="timeout"))
    r = post([U("What is Flutter?")])
    assert r.status_code == 200
    row = metrics.load().iloc[-1]
    assert row["route_source"] == "rules_fallback" and row["fallback_reason"] == "timeout"
    assert row["model"] == config.SMALL_MODEL


# ------------------------------------------------------------------ loop guard
def test_failures_counted_once_per_execution_and_fourth_call_is_blocked(monkeypatch):
    stub = Stub("simple")
    use_router(monkeypatch, "local_llm", stub)
    base = [U("fetch the report")]
    msgs = list(base)
    for i in range(2):
        msgs += tool_turn(f"c{i}", ok=False)
    assert post(msgs).status_code == 200
    assert post(msgs).status_code == 200                 # the SAME 2 executions re-sent: still 2, not 4
    assert (lambda d: d[next(iter(d))])(pm.guard._failures["test-user:sess"]) == 2   # guard state is namespaced per user
    msgs += tool_turn("c2", ok=False)                    # third distinct failing execution
    r = post(msgs)
    assert r.status_code == 429 and r.json()["error"]["type"] == "loop_blocked"
    row = metrics.load().iloc[-1]
    assert row["status"] == "blocked" and row["classification_reused"] == 1   # no new classification
    assert stub.calls == 1


def test_different_inputs_are_not_the_same_failure(monkeypatch):
    use_router(monkeypatch, "rules", Stub())
    msgs = [U("fetch reports")]
    for i in range(4):
        msgs += tool_turn(f"d{i}", ok=False, args=json.dumps({"report_name": f"r{i}"}))
    assert post(msgs).status_code == 200


# ------------------------------------------------------------------ storage migration
def test_old_database_is_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, ts REAL, "
                "session_id TEXT, status TEXT, route TEXT, score INTEGER, reason TEXT, model TEXT, "
                "input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL, actual_cost REAL, "
                "baseline_cost REAL, est_saved_cost REAL, has_tool_calls INTEGER, error TEXT)")
    old.execute("INSERT INTO requests (request_id, ts, status, route, model) VALUES ('r1', 1.0, 'success', 'simple', 'm')")
    old.commit(); old.close()
    conn = db.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    assert set(db.NEW_COLUMNS) <= cols
    assert conn.execute("SELECT request_id, execution_mode FROM requests").fetchall() == [("r1", None)]
    db.connect(path)                                     # idempotent
    df = metrics.load(path)
    assert df.iloc[0]["execution_mode"].startswith("legacy")
    assert metrics.summarize(df)["legacy"] == 1
