"""
Cost math: ESTIMATED inference cost at list price (not money charged, not an invoice).

    cost = input_tokens / 1M * input_price  +  output_tokens / 1M * output_price      (per model, config.MODEL_PRICES)

  - A call's cost needs BOTH token counts reported by the provider and a configured price for the model.
    Otherwise it is None ("unavailable"): never 0.
  - baseline = the same recorded tokens priced at config.BIG_MODEL's rates. It is hypothetical: another model could
    produce different token counts or tool calls, so it says nothing about real savings or answer quality.
  - Overall savings % = sum(difference) / sum(baseline) over MATCHED calls, never the average of per-call percentages.
"""
import config


def price_for(model: str | None) -> dict | None:
    return config.MODEL_PRICES.get(model) if model else None


def call_cost(model: str | None, input_tokens, output_tokens) -> float | None:
    """Estimated cost in USD, or None when the price or either token count is unknown."""
    p = price_for(model)
    if p is None or input_tokens is None or output_tokens is None:
        return None
    if input_tokens != input_tokens or output_tokens != output_tokens:      # NaN
        return None
    return float(input_tokens) / 1_000_000 * p["input"] + float(output_tokens) / 1_000_000 * p["output"]


def baseline_cost(input_tokens, output_tokens) -> float | None:
    """The same tokens priced at the configured BIG model's rates (hypothetical)."""
    return call_cost(config.BIG_MODEL, input_tokens, output_tokens)


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate (~4 characters per token). Used only by the offline mock model, whose numbers are
    labelled 'mock-estimate' and never mixed with provider-reported usage."""
    chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        chars += len(str(content or ""))
        for tc in m.get("tool_calls") or []:
            chars += len(str(tc.get("function", {})))
    return max(1, chars // 4)


def savings_percent(difference, baseline) -> float | None:
    """difference / baseline * 100, or None (shown as N/A) when the baseline is zero or unavailable. May be negative."""
    if baseline is None or difference is None or baseline <= 0:
        return None
    return difference / baseline * 100
