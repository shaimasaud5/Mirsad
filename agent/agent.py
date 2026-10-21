"""
A real LangChain agent whose model traffic goes through Mirsad.

The ONLY Mirsad-specific part is `base_url` (+ a session header): the agent
thinks it is talking to OpenAI, but every request hits our proxy first.

Try it from the terminal (proxy must be running):
    python -m agent.agent "Read sales.csv and tell me the total revenue"
"""
import sys
import uuid

import openai
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

import config
from agent.tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are a business data assistant. Use the tools to answer. "
    "If a tool returns a temporary error, you may retry it. Keep answers short."
)
MAX_GRAPH_STEPS = 16   # ~8 model calls; a safety net on top of the proxy's loop guard


def build_agent(session_id: str, turn_id: str | None = None):
    # X-Turn-ID is new for every user message: the proxy classifies once per turn and reuses the
    # chosen answering model for every tool step of this run (even if the text repeats an old one).
    turn_id = turn_id or "turn_" + uuid.uuid4().hex[:10]
    model = ChatOpenAI(
        base_url=config.PROXY_URL,          # <-- the whole integration is this line
        api_key="mirsad",                 # the real Groq key lives only in the proxy
        model="mirsad-auto",              # Mirsad's routing decision picks the real model
        max_retries=0,                      # don't auto-retry a blocked (429) request
        timeout=60,
        default_headers={"X-Session-ID": session_id, "X-Turn-ID": turn_id},
    )
    return create_agent(model, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


def run_agent(task: str, session_id: str | None = None, history: list | None = None) -> dict:
    """Runs one task. Returns {answer, steps, status, session_id}.
    status: 'ok' | 'blocked' | 'error' | 'step_limit'."""
    session_id = session_id or "sess_" + uuid.uuid4().hex[:6]
    turn_id = "turn_" + uuid.uuid4().hex[:10]
    agent = build_agent(session_id, turn_id)
    messages = list(history or []) + [{"role": "user", "content": task}]
    steps, answer, status = [], "", "ok"

    try:
        # stream() lets us record each step as it happens, even if a later step fails
        for update in agent.stream({"messages": messages},
                                   config={"recursion_limit": MAX_GRAPH_STEPS},
                                   stream_mode="updates"):
            for node_output in update.values():
                for msg in (node_output or {}).get("messages", []):
                    for tc in getattr(msg, "tool_calls", None) or []:
                        steps.append({"type": "tool_call", "tool": tc["name"], "args": tc["args"]})
                    if msg.type == "tool":
                        steps.append({"type": "tool_result", "tool": msg.name,
                                      "result": str(msg.content)[:300]})
                    if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                        answer = msg.content if isinstance(msg.content, str) else str(msg.content)
    except openai.RateLimitError as exc:        # HTTP 429 from Mirsad = loop guard (provider limits arrive as 503)
        is_loop = isinstance(exc.body, dict) and exc.body.get("type") == "loop_blocked"
        status, answer = ("blocked" if is_loop else "error"), _error_message(exc)
    except openai.APIStatusError as exc:        # HTTP 502 etc. = upstream model problem
        status, answer = "error", _error_message(exc)
    except openai.APIConnectionError:
        status, answer = "error", "Cannot reach the Mirsad proxy. Is it running on port 8000?"
    except GraphRecursionError:
        status, answer = "step_limit", "Agent stopped: reached its maximum number of steps."

    return {"answer": answer or "(no answer)", "steps": steps,
            "status": status, "session_id": session_id, "turn_id": turn_id}


def _error_message(exc) -> str:
    try:
        return exc.body["message"] if isinstance(exc.body, dict) and "message" in exc.body \
            else exc.response.json()["error"]["message"]
    except Exception:
        return str(exc)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "List the files you can read."
    result = run_agent(task)
    for s in result["steps"]:
        print("  ", s)
    print(f"[{result['status']}] {result['answer']}")
