"""
Mock model for MIRSAD_MOCK=1 (offline development / emergency demo fallback).

It imitates a tool-using model well enough to exercise the whole pipeline:
  * user mentions a known file      -> asks for read_file(<file>)
  * user mentions a "report"        -> asks for fetch_report(...)  (fails -> retries -> loop!)
  * last message is a tool result   -> retries the same call if it failed, else answers
Token counts are estimates. Never present mock numbers as real results.
"""
import json
import random
import re
import time
import uuid

from proxy.cost import estimate_tokens

KNOWN_FILES = ["sales.csv", "support_tickets.csv", "team_notes.txt"]


def _response(model, content=None, tool_call=None, in_tok=0):
    message = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_call:
        name, args = tool_call
        message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}]
        finish = "tool_calls"
    out_tok = max(5, len(content or json.dumps(tool_call)) // 4)
    return {"id": "mock-" + uuid.uuid4().hex[:8], "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                      "total_tokens": in_tok + out_tok}}


def complete(model: str, messages: list[dict], tools: list | None) -> dict:
    time.sleep(random.uniform(0.15, 0.6) * (2 if "120b" in model else 1))  # fake latency
    in_tok = estimate_tokens(messages)
    tool_names = {t.get("function", {}).get("name") for t in (tools or [])}
    last = messages[-1] if messages else {}

    if last.get("role") == "tool":
        # find the call this result answers
        call = None
        for m in reversed(messages):
            for tc in m.get("tool_calls") or []:
                if tc.get("id") == last.get("tool_call_id"):
                    call = tc["function"]
        if call and str(last.get("content", "")).upper().startswith("ERROR"):
            return _response(model, tool_call=(call["name"], json.loads(call["arguments"] or "{}")),
                             in_tok=in_tok)                       # naive retry -> loop
        return _response(model, content="[mock] Based on the tool output, here is a short "
                                         "summary of the data you asked about.", in_tok=in_tok)

    user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(user, list):
        user = " ".join(p.get("text", "") for p in user if isinstance(p, dict))
    if "read_file" in tool_names:
        for f in KNOWN_FILES:
            if f in user:
                return _response(model, tool_call=("read_file", {"name": f}), in_tok=in_tok)
    if "fetch_report" in tool_names and re.search(r"\breport\b", user, re.I):
        return _response(model, tool_call=("fetch_report", {"report_name": "q3_revenue"}),
                         in_tok=in_tok)
    return _response(model, content=f"[mock reply from {model}] Simulated answer.", in_tok=in_tok)
