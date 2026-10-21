"""
Cost math. Formula for any tier:
    cost = input_tokens / 1M * input_price  +  output_tokens / 1M * output_price

actual_cost   = priced at the tier the router chose
baseline_cost = priced at GPT-4o (as if there were no router)
saved         = baseline_cost - actual_cost

Overall savings % must be computed as  sum(saved) / sum(baseline),
NOT as the average of per-request percentages (that would be misleading).
"""
import config


def price(tier: str, input_tokens: int, output_tokens: int) -> float:
    p = config.PRICES[tier]
    return input_tokens / 1_000_000 * p["input"] + output_tokens / 1_000_000 * p["output"]


def request_costs(route: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
    """Returns (actual_cost, baseline_cost) in USD."""
    actual = price(route, input_tokens, output_tokens)
    baseline = price(config.BASELINE_TIER, input_tokens, output_tokens)
    return actual, baseline


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate (~4 characters per token) for requests that never
    reached a model (blocked loops). Clearly labelled 'estimated' on the dashboard."""
    chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        chars += len(str(content or ""))
        for tc in m.get("tool_calls") or []:
            chars += len(str(tc.get("function", {})))
    return max(1, chars // 4)


def savings_percent(total_saved: float, total_baseline: float) -> float:
    return 0.0 if total_baseline <= 0 else total_saved / total_baseline * 100
