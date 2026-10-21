"""
Mirsad's routing decision: which ANSWERING model handles a user turn.

  rules     : keyword rules decide (proxy/router.py)
  shadow    : rules decide; the local LLM's recommendation is recorded next to it
  local_llm : the local LLM's classification decides; rules are the recorded fallback

One classification per user turn. A turn is identified by the agent's X-Turn-ID header (a fresh id for
every user message, so a repeated text is still a new turn). Clients that send no header fall back to
(session, position of the latest user message, hash of its text). Every model/tool step of the same
turn reuses the cached decision, so the answering model never changes in the middle of a tool loop.
"""
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace

import config
from proxy import classifier
from proxy.router import classify as rules_classify

ROUTE_TO_MODEL = lambda route: config.BIG_MODEL if route == "complex" else config.SMALL_MODEL  # noqa: E731


@dataclass(frozen=True)
class Routing:
    route: str                      # "simple" | "complex" - the route actually used
    model: str                      # answering model chosen by Mirsad
    mode: str                       # configured routing mode
    source: str                     # "rules" | "local_llm" | "rules_fallback"
    fallback_reason: str | None = None
    rules_route: str | None = None
    rules_score: int | None = None
    rules_reason: str | None = None
    llm_route: str | None = None
    llm_reason: str | None = None
    llm_ms: float | None = None
    classifier_error: str | None = None
    classifier_truncated: bool = False
    context_truncated: bool = False
    classifier_model: str | None = None
    classifier_prompt_id: str | None = None
    reused: bool = False
    turn_id: str | None = None

    @property
    def reason_text(self) -> str:
        if self.source == "local_llm":
            return f"local LLM: {self.llm_reason}"
        if self.source == "rules_fallback":
            return f"rules fallback ({self.fallback_reason}): {self.rules_reason}"
        if self.mode == "shadow" and self.llm_route:
            return f"rules: {self.rules_reason} | shadow LLM said {self.llm_route}"
        return f"rules: {self.rules_reason}"


def _turn_key(session_id: str, turn_id: str | None, messages: list[dict]):
    if turn_id:
        return (session_id, "turn", turn_id)
    idx = classifier.last_user_index(messages)
    text = classifier._text(messages[idx]) if idx >= 0 else ""
    return (session_id, "msg", idx, hashlib.sha1(text.encode("utf-8")).hexdigest()[:12])


class TurnRouter:
    def __init__(self, mode: str | None = None, classify_fn=None, max_entries: int | None = None):
        self.mode = mode or config.ROUTING_MODE
        self._classify = classify_fn or classifier.classify_messages
        self._max = max_entries or config.TURN_CACHE_MAX
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    # ---- cache helpers
    def lookup(self, session_id, turn_id, messages) -> Routing | None:
        key = _turn_key(session_id, turn_id, messages)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        return replace(hit, reused=True) if hit is not None else None

    def _store(self, key, routing):
        with self._lock:
            self._cache[key] = routing
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def reset(self):
        with self._lock:
            self._cache.clear()

    # ---- decisions
    def rules_only(self, messages, turn_id=None, fallback_reason=None) -> Routing:
        """Cheap, network-free decision (used when a request is blocked before any classification)."""
        text = classifier._text(messages[classifier.last_user_index(messages)]) if messages and \
            classifier.last_user_index(messages) >= 0 else ""
        d = rules_classify(text)
        return Routing(route=d.route, model=ROUTE_TO_MODEL(d.route), mode=self.mode,
                       source="rules_fallback" if fallback_reason else "rules",
                       fallback_reason=fallback_reason, rules_route=d.route, rules_score=d.score,
                       rules_reason=d.reason_text, turn_id=turn_id)

    def decide(self, session_id: str, turn_id: str | None, messages: list[dict]) -> Routing:
        cached = self.lookup(session_id, turn_id, messages)
        if cached is not None:
            return cached
        key = _turn_key(session_id, turn_id, messages)
        base = self.rules_only(messages, turn_id)
        routing = base
        if self.mode in ("shadow", "local_llm"):
            res = self._classify(messages)
            info = dict(llm_route=res.label, llm_reason=res.reason, llm_ms=res.duration_ms,
                        classifier_error=res.error, classifier_truncated=res.truncated,
                        context_truncated=res.context_truncated,
                        classifier_model=config.CLASSIFIER_MODEL, classifier_prompt_id=classifier.PROMPT_ID)
            if self.mode == "local_llm":
                if res.label:
                    routing = replace(base, route=res.label, model=ROUTE_TO_MODEL(res.label),
                                      source="local_llm", **info)
                else:   # classifier failed -> rules decide, with an explicit recorded reason
                    routing = replace(base, source="rules_fallback", fallback_reason=res.error or "unknown", **info)
            else:       # shadow: rules keep control
                routing = replace(base, **info)
        self._store(key, routing)
        return routing
