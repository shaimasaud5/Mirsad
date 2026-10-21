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
def test_cost_formula():
    # 1M input + 1M output tokens at GPT-4o prices = 2.50 + 10.00
    actual, baseline = cost.request_costs("complex", 1_000_000, 1_000_000)
    assert actual == pytest.approx(12.5) and baseline == pytest.approx(12.5)


def test_simple_route_is_cheaper_than_baseline():
    actual, baseline = cost.request_costs("simple", 120, 40)
    assert actual == pytest.approx(0.000042)
    assert baseline == pytest.approx(0.0007)


def test_savings_percent_is_weighted_not_averaged():
    # 7 simple requests (94% each) + 3 complex (0% each)
    saved = 7 * (0.0007 - 0.000042)
    baseline = 7 * 0.0007 + 3 * 0.008
    assert cost.savings_percent(saved, baseline) == pytest.approx(15.9, abs=0.1)
