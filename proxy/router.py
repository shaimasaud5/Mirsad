"""
Complexity router: decides whether a request goes to the SMALL or BIG model.

Rule-based on purpose: free, instant, and every decision can be explained
("score 4 because: keyword 'debug', contains code").

Scoring rules (each adds points):
  +3  per complex keyword found (max 2 counted). One clear intent word such as
      "implement" or "debug" is enough on its own to need the big model.
  +3  prompt contains code (``` fence, 'def ', 'class ', 'SELECT ', etc.)
  +2  prompt longer than 400 characters
  +1  prompt longer than 1000 characters (on top of the +2)
  +1  multi-step wording ("first ... then", numbered steps)
  +1  two or more questions in one prompt
Score >= ROUTER_THRESHOLD  ->  "complex"  (big model)
otherwise                  ->  "simple"   (small model)
"""
import re
from dataclasses import dataclass, field

import config

COMPLEX_KEYWORDS = [
    "implement", "debug", "refactor", "optimize", "optimise", "analyze", "analyse",
    "design", "architecture", "algorithm", "prove", "explain why", "compare",
    "trade-off", "tradeoff", "root cause", "step by step", "write a function",
    "write code", "write a script", "strategy", "evaluate", "diagnose", "plan",
    "pros and cons", "derive", "architect", "forecast", "unit test", "mitigate",
    "security", "investigate", "recommend", "how would you", "reason about",
]

CODE_PATTERNS = [r"```", r"\bdef \w+\(", r"\bclass \w+", r"\bimport \w+",
                 r"\bSELECT\b.+\bFROM\b", r"=>", r"\w+\(.*\)\s*\{", r"Traceback \(most recent"]

MULTI_STEP_PATTERNS = [r"\bfirst\b.*\bthen\b", r"\bthen\b.*\bfinally\b",
                       r"(^|\n)\s*1[.)]\s.+\n\s*2[.)]\s"]


@dataclass
class RouteDecision:
    route: str            # "simple" or "complex"
    score: int
    reasons: list = field(default_factory=list)

    @property
    def reason_text(self) -> str:
        return ", ".join(self.reasons) if self.reasons else "no complexity signals"


def classify(text: str, threshold: int | None = None) -> RouteDecision:
    threshold = config.ROUTER_THRESHOLD if threshold is None else threshold
    text = text or ""
    lower = text.lower()
    score, reasons = 0, []

    # "s?" lets "trade-off" match "trade-offs", "plan" match "plans", etc.
    found = [k for k in COMPLEX_KEYWORDS if re.search(r"\b" + re.escape(k) + r"s?\b", lower)]
    if found:
        counted = found[:2]
        score += 3 * len(counted)
        reasons.append("keywords: " + ", ".join(counted))

    if any(re.search(p, text, flags=re.IGNORECASE | re.DOTALL) for p in CODE_PATTERNS):
        score += 3
        reasons.append("contains code")

    if len(text) > 400:
        score += 2
        reasons.append(f"long prompt ({len(text)} chars)")
    if len(text) > 1000:
        score += 1

    if any(re.search(p, lower, flags=re.DOTALL) for p in MULTI_STEP_PATTERNS):
        score += 1
        reasons.append("multi-step")

    if text.count("?") >= 2:
        score += 1
        reasons.append("multiple questions")

    route = "complex" if score >= threshold else "simple"
    return RouteDecision(route=route, score=score, reasons=reasons)


def latest_user_text(messages: list[dict]) -> str:
    """The router scores the latest *user* message: that is the task.
    Tool results that follow it (file contents etc.) belong to the same task,
    so every step of one task goes to the same model."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):  # OpenAI "content parts" format
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content or ""
    return ""
