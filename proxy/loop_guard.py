"""
Loop guard: stops an agent that keeps repeating the same failing tool call.

How it works
------------
In the OpenAI message format, a tool call looks like:
    assistant: tool_calls=[{id, function: {name, arguments}}]
    tool:      tool_call_id=<id>, content="ERROR: ..."   (or a normal result)

Every request the agent sends contains the WHOLE conversation, so the guard
remembers which tool_call_ids it has already counted (no double counting).

For each new tool result:
  * content starts with "ERROR"  -> fingerprint = hash(tool name + arguments),
                                     failures[session][fingerprint] += 1
  * success                       -> reset that fingerprint to 0
If any fingerprint reaches LOOP_THRESHOLD, the request is blocked before it
reaches a model, so no tokens are spent.
"""
import hashlib
import json
import threading
from collections import defaultdict
from dataclasses import dataclass

import config


@dataclass
class GuardResult:
    blocked: bool
    tool_name: str = ""
    failures: int = 0
    fingerprint: str = ""


def fingerprint(tool_name: str, arguments: str) -> str:
    try:  # normalize JSON so {"a":1} and { "a": 1 } match
        arguments = json.dumps(json.loads(arguments or "{}"), sort_keys=True)
    except (ValueError, TypeError):
        arguments = (arguments or "").strip()
    raw = f"{tool_name}|{arguments}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class LoopGuard:
    def __init__(self, threshold: int | None = None):
        self.threshold = config.LOOP_THRESHOLD if threshold is None else threshold
        self._failures = defaultdict(lambda: defaultdict(int))   # session -> fp -> count
        self._seen = defaultdict(set)                             # session -> tool_call_ids
        self._names = {}                                          # fp -> tool name
        self._lock = threading.Lock()

    def check(self, session_id: str, messages: list[dict]) -> GuardResult:
        # Map tool_call_id -> (name, arguments) from assistant messages
        calls = {}
        for msg in messages:
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                calls[tc.get("id")] = (fn.get("name", "?"), fn.get("arguments", ""))

        with self._lock:
            for msg in messages:
                if msg.get("role") != "tool":
                    continue
                call_id = msg.get("tool_call_id")
                if call_id in self._seen[session_id] or call_id not in calls:
                    continue
                self._seen[session_id].add(call_id)
                name, args = calls[call_id]
                fp = fingerprint(name, args)
                self._names[fp] = name
                content = msg.get("content")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
                if str(content or "").strip().upper().startswith("ERROR"):
                    self._failures[session_id][fp] += 1
                else:
                    self._failures[session_id][fp] = 0

            worst_fp, worst = "", 0
            for fp, count in self._failures[session_id].items():
                if count > worst:
                    worst_fp, worst = fp, count

        if worst >= self.threshold:
            return GuardResult(True, self._names.get(worst_fp, "?"), worst, worst_fp)
        return GuardResult(False, failures=worst)

    def reset(self, session_id: str | None = None):
        with self._lock:
            if session_id is None:
                self._failures.clear(); self._seen.clear()
            else:
                self._failures.pop(session_id, None); self._seen.pop(session_id, None)
