import json

import pytest

from proxy import cost
from proxy.loop_guard import LoopGuard
from proxy.router import classify, latest_user_text


# ---------------------------------------------------------------- router
@pytest.mark.parametrize("prompt", [
    "Translate hello to French.",
    "Read sales.csv and tell me the total revenue.",
    "What is 15% of 200?",
])
def test_simple_prompts_go_to_small_model(prompt):
    assert classify(prompt).route == "simple"


@pytest.mark.parametrize("prompt", [
    "Implement a binary search in Python.",
    "Debug this: def f(x): return x/0",
    "Design a schema for a clinic waitlist system.",
])
def test_complex_prompts_go_to_big_model(prompt):
    assert classify(prompt).route == "complex"


def test_router_explains_its_decision():
    d = classify("Debug this: def f(x): return x/0")
    assert "debug" in d.reason_text and "contains code" in d.reason_text


def test_router_uses_latest_user_message():
    msgs = [{"role": "user", "content": "Implement quicksort"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks!"}]
    assert latest_user_text(msgs) == "thanks!"


# ------------------------------------------------------------ loop guard
def _history(n_failures, tool="fetch_report", args=None, ok_last=False):
    msgs = [{"role": "user", "content": "go"}]
    for i in range(n_failures):
        cid = f"c{i}"
        ok = ok_last and i == n_failures - 1
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": cid, "type": "function",
            "function": {"name": tool, "arguments": json.dumps(args or {"x": 1})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid,
                     "content": "ok" if ok else "ERROR: timeout"})
    return msgs


def test_guard_blocks_after_threshold():
    g = LoopGuard(threshold=3)
    assert not g.check("s", _history(2)).blocked
    assert g.check("s", _history(3)).blocked


def test_guard_does_not_double_count_resent_history():
    g = LoopGuard(threshold=3)
    for _ in range(5):                      # same 2-failure history sent 5 times
        assert not g.check("s", _history(2)).blocked


def test_guard_resets_after_success():
    g = LoopGuard(threshold=3)
    assert not g.check("s", _history(3, ok_last=True)).blocked


def test_guard_isolated_per_session():
    g = LoopGuard(threshold=3)
    g.check("a", _history(2))
    assert not g.check("b", _history(2)).blocked


# ------------------------------------------------------------------ cost
def test_cost_formula_uses_separate_input_and_output_rates(monkeypatch):
    monkeypatch.setattr(cost.config, "MODEL_PRICES", {"m": {"input": 2.0, "output": 10.0}})
    assert cost.call_cost("m", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert cost.call_cost("m", 500_000, 0) == pytest.approx(1.0)          # zero output tokens is a real value


def test_unknown_price_or_usage_is_unavailable_not_zero(monkeypatch):
    monkeypatch.setattr(cost.config, "MODEL_PRICES", {"m": {"input": 2.0, "output": 10.0}})
    assert cost.call_cost("other", 10, 10) is None
    assert cost.call_cost("m", None, 10) is None and cost.call_cost("m", 10, float("nan")) is None


def test_savings_percent_is_weighted_and_na_for_a_zero_baseline():
    assert cost.savings_percent(25, 100) == pytest.approx(25)
    assert cost.savings_percent(-50, 100) == pytest.approx(-50)           # negative savings are allowed
    assert cost.savings_percent(0, 0) is None and cost.savings_percent(1, None) is None
