"""Answering-model configuration: the SMALL model must be cheaper than BIG per input AND output token, BIG is unchanged,
routing picks the right model, the pick is reused within a turn, and telemetry prices every call with ITS model.
Offline: stubbed classifier, mock answers / fake provider. The live comparison lives in eval/answering_models.py."""
import os

import pytest
from fastapi.testclient import TestClient

import config
from eval import answering_models as am
from monitoring import metrics
from proxy import cost
from proxy import main as pm
from proxy import provider
from proxy.routing import TurnRouter
from test_failover import FakeProvider, make_pool
from test_metrics import groq_response
from test_routing import Stub, U, use_router

OLD_SMALL = "groq/qwen/qwen3.8-27b"


def test_small_model_is_cheaper_than_big_for_input_and_output_and_big_is_unchanged():
    assert config.BIG_MODEL == "groq/openai/gpt-oss-120b"
    small, big = config.MODEL_PRICES[config.SMALL_MODEL], config.MODEL_PRICES[config.BIG_MODEL]
    assert small["input"] < big["input"] and small["output"] < big["output"]
    assert config.SMALL_MODEL == "groq/openai/gpt-oss-20b" and (small["input"], small["output"]) == (0.075, 0.30)
    assert (big["input"], big["output"]) == (0.15, 0.60)


def test_every_priced_model_carries_a_source_and_the_replaced_model_keeps_its_price():
    for m, p in config.MODEL_PRICES.items():
        assert p["source"].startswith("https://console.groq.com/docs/model/") and p["input"] >= 0 and p["output"] >= 0
    assert OLD_SMALL in config.MODEL_PRICES            # historical rows that used it are still priced with ITS rates


def test_simple_goes_to_the_economical_model_complex_to_big():
    simple = TurnRouter(mode="local_llm", classify_fn=Stub("simple")).decide("s", "t1", [U("What is the capital of France?")])
    complex_ = TurnRouter(mode="local_llm", classify_fn=Stub("complex")).decide("s", "t2", [U("Design a distributed database")])
    assert (simple.route, simple.model) == ("simple", config.SMALL_MODEL) and simple.model == "groq/openai/gpt-oss-20b"
    assert (complex_.route, complex_.model) == ("complex", "groq/openai/gpt-oss-120b")


def test_selection_is_reused_during_tool_steps_of_one_turn_and_each_call_is_priced_with_its_model(monkeypatch):
    client = TestClient(pm.app, headers={"Authorization": "Bearer " + os.environ["MIRSAD_INTERNAL_TOKEN"], "X-Mirsad-User-Id": "sel-user"})
    client.post("/admin/reset")
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(provider, "_sleep", lambda s: None)
    monkeypatch.setattr(pm, "pool", make_pool("gsk_FAKE_selection_key_0001"))
    stub = Stub("simple")
    use_router(monkeypatch, "local_llm", stub)
    seen = []

    def fake(**kw):
        seen.append(kw["model"])
        return groq_response({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "total_tokens": 2_000_000})
    monkeypatch.setattr(provider, "_completion", fake)
    msgs = [U("Read sales.csv and tell me the total revenue.")]
    step2 = msgs + [{"role": "assistant", "content": None, "tool_calls": [{"id": "k1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "k1", "content": "month,region"}]
    for m in (msgs, step2):
        assert client.post("/v1/chat/completions", headers={"X-Session-ID": "ss", "X-Turn-ID": "turn-sel"},
                           json={"model": "mirsad-auto", "messages": m}).status_code == 200
    assert stub.calls == 1 and seen == [config.SMALL_MODEL, config.SMALL_MODEL]
    df = metrics.annotate(metrics.load())
    assert list(df["model"]) == [config.SMALL_MODEL] * 2 and list(df["classification_reused"]) == [0, 1]
    small = config.MODEL_PRICES[config.SMALL_MODEL]
    assert list(df["est_cost"]) == pytest.approx([small["input"] + small["output"]] * 2)
    big = config.MODEL_PRICES[config.BIG_MODEL]
    assert list(df["base_cost"]) == pytest.approx([big["input"] + big["output"]] * 2)
    assert (df["est_cost"] < df["base_cost"]).all()                  # the hypothetical comparison is now positive-savings
    # a complex turn is answered (and priced) by the unchanged BIG model
    use_router(monkeypatch, "local_llm", Stub("complex"))
    client.post("/v1/chat/completions", headers={"X-Session-ID": "ss2", "X-Turn-ID": "turn-big"}, json={"model": "mirsad-auto", "messages": msgs})
    assert seen[-1] == config.BIG_MODEL
    row = metrics.annotate(metrics.load()).iloc[-1]
    assert row["model"] == config.BIG_MODEL and row["est_cost"] == pytest.approx(big["input"] + big["output"])


def test_historical_rows_keep_their_original_model_and_are_priced_by_it(tmp_path):
    from proxy import db
    c = db.connect(tmp_path / "h.db")
    db.insert(c, {"request_id": "r_old", "ts": 1.0, "session_id": "s", "user_id": "u", "turn_id": "T", "status": "success",
                  "route": "simple", "model": OLD_SMALL, "execution_mode": "live", "input_tokens": 1_000_000, "output_tokens": 1_000_000,
                  "token_source": "provider", "provider_attempts": 1})
    df = metrics.annotate(metrics.load(tmp_path / "h.db"))
    assert df.iloc[0]["model"] == OLD_SMALL                          # identity untouched by the configuration change
    assert df.iloc[0]["est_cost"] == pytest.approx(0.80 + 4.00)      # old model's own rates, not the new SMALL_MODEL's


# ------------------------------------------------------------------ evaluation harness (offline checks of the criteria)
def test_eval_cases_cover_the_required_scenarios_within_the_authorized_budget():
    ids = {c["id"] for c in am.CASES}
    assert {"en_fact", "ar_fact", "translate", "format", "file_read", "missing_file"} <= ids
    assert 2 * len(am.CASES) <= am.MAX_RUNS and am.MAX_CALLS_PER_RUN == 3


def _read_steps(name, result):
    return [{"type": "tool_call", "tool": "read_file", "args": {"name": name}}, {"type": "tool_result", "tool": "read_file", "result": result}]


def test_eval_criteria_reject_wrong_invalid_and_fabricated_outputs():
    notes = (config.AGENT_FILES_DIR / "team_notes.txt").read_text(encoding="utf-8")
    assert am.check_en_fact("The capital of France is Paris.", []) == []
    assert am.check_en_fact("Lyon.", [])
    assert am.check_en_fact("Paris", [{"type": "tool_call", "tool": "hack", "args": {}}])          # unknown tool
    assert am.check_en_fact("Paris", [{"type": "tool_call", "tool": "read_file", "args": {}}])     # missing argument
    assert am.check_ar_fact("عاصمتها الرياض", []) == [] and am.check_ar_fact("جدة", [])
    assert am.check_format("- apples\n- oranges\n- bananas", []) == [] and am.check_format("apples, oranges, bananas", [])
    assert am.check_file_read("Plans: Pro Plan marketing in Jeddah and Riyadh hiring.", _read_steps("team_notes.txt", notes)) == []
    assert am.check_file_read("Plans: Pro Plan marketing in Jeddah.", [])                          # no tool use: unverified
    assert am.check_file_read("Pro Plan marketing in Jeddah", _read_steps("team_notes.txt", "made up text"))   # not the real file content
    assert am.check_missing_file("I couldn’t find budget_2031.csv.", _read_steps("budget_2031.csv", "ERROR: file not found")) == []
    assert am.check_missing_file("Summary: revenue was 4,200 SAR.", _read_steps("budget_2031.csv", "ERROR: file not found"))   # fabricated
    assert am.check_missing_file("The file does not exist.", [])                                   # never tried the tool
