"""
Mirsad proxy.

OpenAI-compatible endpoint: POST /v1/chat/completions
The Mirsad agent (agent/agent.py) points its base_url here: http://localhost:8000/v1

Pipeline for every request:
  1. Receive   - start timer, read X-Session-ID and X-Turn-ID headers
  2. Guard     - same failing tool call >= LOOP_THRESHOLD times?  -> 429, no model call
  3. Route     - Mirsad decides small or big answering model (proxy/routing.py):
                   rules | shadow | local_llm (local Qwen via Ollama CLASSIFIES; it never answers)
                 one classification per user turn, reused by every tool/model step of that turn
  4. Call      - LiteLLM -> Groq answering model (or mock reply when MIRSAD_MOCK=1)
  5. Measure   - tokens (provider usage when available), latency, priced cost vs a hypothetical GPT-4o baseline
  6. Save      - one row in SQLite
  7. Respond   - OpenAI-format JSON back to the agent

Run:  uvicorn proxy.main:app --port 8000
"""
import threading
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

import config
from litellm import ModelResponse

from proxy import classifier, cost, db, mock
from proxy.credentials import CredentialPool
from proxy.internal import ProxyAuthError, caller_identity, require_internal, token_valid
from proxy.loop_guard import LoopGuard
from proxy.provider import call_groq
from proxy.routing import TurnRouter

conn = db.connect()
guard = LoopGuard()
router = TurnRouter()
pool = CredentialPool()     # GROQ_API_KEY (primary) + optional GROQ_API_KEY_BACKUP_1..9; only aliases are ever shown

# Only these fields are forwarded to the model provider
FORWARD_KEYS = ("tools", "tool_choice", "temperature", "top_p", "stop", "response_format")


@asynccontextmanager
async def lifespan(_app):
    if config.ROUTING_MODE != "rules":   # load the local classifier in the background (cold start ~1 min)
        threading.Thread(target=classifier.warm_up, daemon=True).start()
    yield


app = FastAPI(title="Mirsad Proxy", lifespan=lifespan)


@app.exception_handler(ProxyAuthError)
async def _auth_error(_request, exc: ProxyAuthError):
    return JSONResponse(status_code=exc.status, content={"error": {"message": exc.message, "type": exc.err_type}},
                        headers={"WWW-Authenticate": "Bearer"} if exc.status == 401 else None)


def _openai_error(status: int, message: str, err_type: str, headers: dict) -> JSONResponse:
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": err_type}})


def _ollama_status() -> str:
    if config.ROUTING_MODE == "rules":
        return "not used"
    try:
        r = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=1.5)
        names = [m.get("name") for m in r.json().get("models", [])]
        return "ready" if config.CLASSIFIER_MODEL in names else "model not downloaded"
    except Exception:
        return "unreachable"


@app.get("/health")
def health(request: Request):
    if not token_valid(request):          # public callers only learn that the service is up
        return {"status": "ok"}
    return {"status": "ok", "mock_mode": config.MOCK_MODE,
            "execution": "mock" if config.MOCK_MODE else "live",
            "routing_mode": config.ROUTING_MODE, "classifier_model": config.CLASSIFIER_MODEL,
            "classifier_status": _ollama_status(),
            "credentials": pool.status(), "provider_max_attempts": config.PROVIDER_MAX_ATTEMPTS,
            "small_model": config.SMALL_MODEL, "big_model": config.BIG_MODEL}


@app.get("/v1/models")
def models(request: Request):
    require_internal(request)
    return {"object": "list", "data": [{"id": "mirsad-auto", "object": "model"}]}


@app.post("/admin/reset")
def reset(request: Request):
    """Clears logged data, loop counters and cached turn classifications (handy before a demo).
    Internal only: the web server exposes it to admins, never to ordinary users."""
    require_internal(request)
    db.clear(conn)
    guard.reset()
    router.reset()
    return {"status": "reset"}


def _routing_fields(d) -> dict:
    return dict(route=d.route, score=d.rules_score, reason=d.reason_text, model=d.model,
                turn_id=d.turn_id, routing_mode=d.mode, route_source=d.source,
                fallback_reason=d.fallback_reason, rules_route=d.rules_route, llm_route=d.llm_route,
                classifier_model=d.classifier_model, classifier_prompt_id=d.classifier_prompt_id,
                classifier_ms=0.0 if d.reused else d.llm_ms, classifier_reason=d.llm_reason,
                classifier_error=d.classifier_error, classifier_truncated=int(d.classifier_truncated),
                context_truncated=int(d.context_truncated), classification_reused=int(d.reused),
                execution_mode="mock" if config.MOCK_MODE else "live")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, x_session_id: str | None = Header(default=None),
                           x_turn_id: str | None = Header(default=None)):
    # ---- 0. Authenticate the caller (internal token) and take the verified user id ----------
    require_internal(request)
    user_id, conversation_id = caller_identity(request)

    # ---- 1. Receive ------------------------------------------------------
    start = time.perf_counter()
    body = await request.json()
    messages = body.get("messages", [])
    session_id = x_session_id or "no-session"
    skey = f"{user_id}:{session_id}"      # loop counters and turn cache are per user: one user can never touch another's
    request_id = "req_" + uuid.uuid4().hex[:8]
    row = {"request_id": request_id, "ts": time.time(), "session_id": session_id,
           "user_id": user_id, "conversation_id": conversation_id,
           "input_tokens": 0, "output_tokens": 0, "actual_cost": 0.0,
           "baseline_cost": 0.0, "est_saved_cost": 0.0, "has_tool_calls": 0}

    # ---- 2. Loop guard (before any classification: a blocked task spends nothing) ----
    verdict = guard.check(skey, messages)
    if verdict.blocked:
        decision = router.lookup(skey, x_turn_id, messages) or \
            router.rules_only(messages, x_turn_id, fallback_reason="blocked_before_classification")
        row.update(_routing_fields(decision))
        headers = {"X-Mirsad-Request-Id": request_id, "X-Mirsad-Route": decision.route}
        est_in = cost.estimate_tokens(messages)
        _, est_baseline = cost.request_costs(
            config.BASELINE_TIER, est_in, config.BLOCKED_OUTPUT_TOKENS_ESTIMATE)
        row.update(status="blocked", est_saved_cost=est_baseline,
                   latency_ms=(time.perf_counter() - start) * 1000,
                   error=f"loop blocked: '{verdict.tool_name}' failed {verdict.failures} times")
        db.insert(conn, row)
        return _openai_error(
            429,
            f"[Mirsad] Loop blocked: tool '{verdict.tool_name}' failed "
            f"{verdict.failures} times with the same input. Stopping to save tokens.",
            "loop_blocked", {**headers, "X-Mirsad-Status": "blocked"})

    # ---- 3. Route: Mirsad picks the answering model (classifier runs once per user turn) ----
    decision = await run_in_threadpool(router.decide, skey, x_turn_id, messages)
    model = decision.model
    row.update(_routing_fields(decision))
    headers = {"X-Mirsad-Request-Id": request_id, "X-Mirsad-Route": decision.route,
               "X-Mirsad-Route-Source": decision.source}

    # ---- 4. Call the answering model ------------------------------------
    kwargs = {k: body[k] for k in FORWARD_KEYS if k in body}
    kwargs["max_tokens"] = min(int(body.get("max_tokens") or config.MAX_TOKENS), config.MAX_TOKENS)
    attempts = []
    if config.MOCK_MODE:
        response = ModelResponse(**mock.complete(model, messages, kwargs.get("tools")))
        row.update(credential_alias="mock", provider_attempts=0, failover_used=0)
    else:
        # Same model and same messages on every attempt; only the credential can change, and only after a 401.
        outcome = await run_in_threadpool(call_groq, model, messages, kwargs, pool)
        attempts, response = outcome.attempts, outcome.response
        row.update(credential_alias=outcome.alias, provider_attempts=len(attempts),
                   failover_used=int(outcome.failover_used))
        if not outcome.ok:   # never replaced by a mock reply
            row.update(status="error", latency_ms=(time.perf_counter() - start) * 1000,
                       error=f"{outcome.error_type}: {outcome.message}"[:400])
            db.insert(conn, row)
            db.insert_attempts(conn, row, model, attempts)
            extra = {"Retry-After": str(int(outcome.retry_after) + 1)} if outcome.retry_after is not None else {}
            return _openai_error(outcome.http_status, outcome.message, outcome.error_type,
                                 {**headers, **extra, "X-Mirsad-Status": "error"})

    # ---- 5. Measure ------------------------------------------------------
    latency_ms = (time.perf_counter() - start) * 1000
    usage = getattr(response, "usage", None)
    prov_in = int(getattr(usage, "prompt_tokens", 0) or 0)
    prov_out_raw = getattr(usage, "completion_tokens", None)
    in_tok = prov_in or cost.estimate_tokens(messages)
    out_tok = int(prov_out_raw or 0)
    if config.MOCK_MODE:
        token_source = "mock-estimate"
    else:
        token_source = "provider" if prov_in and prov_out_raw is not None else "estimate"
    actual, baseline = cost.request_costs(decision.route, in_tok, out_tok)
    choice = response.choices[0]
    has_tools = bool(getattr(choice.message, "tool_calls", None))

    # ---- 6. Save ---------------------------------------------------------
    row.update(status="success", input_tokens=in_tok, output_tokens=out_tok,
               latency_ms=latency_ms, actual_cost=actual, baseline_cost=baseline,
               has_tool_calls=int(has_tools), error=None, token_source=token_source,
               response_model=getattr(response, "model", None))
    db.insert(conn, row)
    db.insert_attempts(conn, row, model, attempts)

    # ---- 7. Respond (OpenAI format) -------------------------------------
    payload = response.model_dump()
    return JSONResponse(content=payload, headers={**headers, "X-Mirsad-Status": "success"})
