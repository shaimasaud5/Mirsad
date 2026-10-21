"""
Calls the answering model (Groq through LiteLLM) with API-key failover.

What fails over and what does not
  401 invalid/revoked key  -> disable that credential for this process, try the next one (same model, same messages)
  429 rate limit           -> NEVER rotate keys: Groq limits are per organization, a sibling key shares the limit.
                              Honor Retry-After with one bounded wait on the SAME key, otherwise return a clear error.
  403 permission/account   -> no automatic failover: the cause is not identified
  400/404/422 request/model, 5xx/timeouts/connection (provider outage) -> no failover: another key does not fix them
Retries are not multiplied: LiteLLM / OpenAI SDK retries are switched off (num_retries=0, max_retries=0) and this
module owns the single, bounded attempt budget (config.PROVIDER_MAX_ATTEMPTS, shared by failover and the one rate-limit retry).
Nothing here ever falls back to the mock model, and no key value is logged, stored or returned.
"""
import logging
import time
from dataclasses import dataclass, field

import litellm

import config
from proxy.credentials import CredentialPool, redact

log = logging.getLogger("mirsad.provider")

_completion = litellm.completion     # replaced by tests with a fake provider
_sleep = time.sleep


@dataclass
class Attempt:
    n: int
    alias: str
    outcome: str                 # success | credential_invalid | rate_limited | permission_denied | bad_request | provider_outage | error
    http_status: int | None
    duration_ms: float
    failover: bool = False       # a different credential was tried (or will be) because of this attempt
    failover_reason: str | None = None
    retry_after_s: float | None = None
    error: str | None = None     # redacted, short


@dataclass
class ProviderOutcome:
    response: object = None
    attempts: list = field(default_factory=list)
    error_type: str | None = None
    http_status: int = 502
    message: str = ""
    retry_after: float | None = None
    alias: str | None = None     # credential that produced the answer (or the last one tried)

    @property
    def ok(self) -> bool:
        return self.response is not None

    @property
    def failover_used(self) -> bool:
        return any(a.failover for a in self.attempts)


# ---------------------------------------------------------------- classification
def _status(exc) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else None


def _retry_after(exc) -> float | None:
    for holder in (getattr(exc, "response", None), ):
        headers = getattr(holder, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    return None


def classify(exc) -> str:
    status = _status(exc)
    if isinstance(exc, litellm.AuthenticationError) or status == 401:
        return "credential_invalid"
    if isinstance(exc, litellm.RateLimitError) or status == 429:
        return "rate_limited"
    if isinstance(exc, litellm.PermissionDeniedError) or status == 403:
        return "permission_denied"
    if isinstance(exc, (litellm.BadRequestError, litellm.NotFoundError)) or status in (400, 404, 422):
        return "bad_request"
    if isinstance(exc, (litellm.Timeout, litellm.APIConnectionError, litellm.ServiceUnavailableError,
                        litellm.InternalServerError)) or (status is not None and status >= 500):
        return "provider_outage"
    return "error"


def _fail(outcome: ProviderOutcome, error_type: str, status: int, message: str, **kw) -> ProviderOutcome:
    outcome.error_type, outcome.http_status, outcome.message = error_type, status, message
    for k, v in kw.items():
        setattr(outcome, k, v)
    return outcome


# ---------------------------------------------------------------- main entry
def call_groq(model: str, messages: list, kwargs: dict, pool: CredentialPool, *,
              max_attempts: int | None = None, retry_after_max: float | None = None) -> ProviderOutcome:
    max_attempts = max_attempts or config.PROVIDER_MAX_ATTEMPTS
    retry_after_max = config.PROVIDER_RETRY_AFTER_MAX_S if retry_after_max is None else retry_after_max
    out = ProviderOutcome()
    secrets = pool.secrets()
    common = dict(num_retries=0, max_retries=0, timeout=config.PROVIDER_TIMEOUT_S)   # no hidden SDK/LiteLLM retries

    if not model.startswith("groq/"):    # not a Groq model: single attempt, credentials come from the environment
        return _single_attempt_env(model, messages, kwargs, common, out, secrets)

    if pool.configured == 0:
        return _fail(out, "provider_not_configured", 503,
                     "[Mirsad] No Groq API key is configured (GROQ_API_KEY). Mock mode was NOT used.")
    rate_retry_used = False
    while len(out.attempts) < max_attempts:
        active = pool.active()
        if not active:
            break
        cred = active[0]
        started = time.perf_counter()
        n = len(out.attempts) + 1
        try:
            response = _completion(model=model, messages=messages, api_key=cred.key, **common, **kwargs)
        except Exception as exc:  # noqa: BLE001
            ms = (time.perf_counter() - started) * 1000
            kind, status = classify(exc), _status(exc)
            detail = redact(f"{type(exc).__name__}: {exc}", secrets)[:200]
            att = Attempt(n, cred.alias, kind, status, ms, error=detail)
            out.attempts.append(att)
            out.alias = cred.alias
            log.info("provider attempt %d alias=%s outcome=%s status=%s %.0fms", n, cred.alias, kind, status, ms)

            if kind == "credential_invalid":
                pool.disable(cred.alias, "401 invalid or revoked")
                nxt = pool.active()
                will_try = bool(nxt) and len(out.attempts) < max_attempts
                att.failover = will_try
                att.failover_reason = "401 invalid or revoked credential" if will_try else None
                log.warning("credential %s rejected (401); disabled for this process; next=%s",
                            cred.alias, nxt[0].alias if nxt else "none")
                continue
            if kind == "rate_limited":
                ra = _retry_after(exc)
                att.retry_after_s = ra
                if ra is not None and ra <= retry_after_max and not rate_retry_used and len(out.attempts) < max_attempts:
                    rate_retry_used = True
                    log.info("rate limited; waiting %.1fs and retrying the SAME credential %s", ra, cred.alias)
                    _sleep(ra)
                    continue
                wait = f" Retry after about {ra:.0f} s." if ra is not None else ""
                return _fail(out, "provider_rate_limited", 503,
                             "[Mirsad] Groq rate limit reached (HTTP 429). Limits are shared across the whole "
                             "organization, so switching API keys would not help." + wait, retry_after=ra)
            if kind == "permission_denied":
                return _fail(out, "provider_permission_denied", 502,
                             "[Mirsad] Groq denied the request (HTTP 403). The cause (account restriction, model "
                             "access, region) is not identified, so no automatic key failover was attempted.")
            if kind == "bad_request":
                return _fail(out, "provider_bad_request", 502,
                             f"[Mirsad] Groq rejected the request or model (HTTP {status}). Another API key "
                             f"would not change this. {detail[:120]}")
            if kind == "provider_outage":
                return _fail(out, "provider_outage", 502,
                             "[Mirsad] Groq is unavailable or timed out. Another API key would not fix a provider outage.")
            return _fail(out, "provider_error", 502, f"[Mirsad] Unexpected provider error. {detail[:120]}")
        else:
            ms = (time.perf_counter() - started) * 1000
            out.attempts.append(Attempt(n, cred.alias, "success", 200, ms))
            out.response, out.alias = response, cred.alias
            log.info("provider attempt %d alias=%s outcome=success %.0fms", n, cred.alias, ms)
            return out

    tried = ", ".join(a.alias for a in out.attempts if a.outcome == "credential_invalid") or "none"
    if not pool.active():
        return _fail(out, "provider_credentials_exhausted", 502,
                     f"[Mirsad] All configured Groq credentials were rejected (HTTP 401): {tried}. "
                     "Replace the keys in .env and restart the proxy. Mock mode was NOT used.")
    return _fail(out, "provider_attempt_limit", 502,
                 f"[Mirsad] Stopped after {len(out.attempts)} provider attempts (limit {max_attempts}).")


def _single_attempt_env(model, messages, kwargs, common, out, secrets):
    started = time.perf_counter()
    try:
        response = _completion(model=model, messages=messages, **common, **kwargs)
    except Exception as exc:  # noqa: BLE001
        kind = classify(exc)
        out.attempts.append(Attempt(1, "env", kind, _status(exc), (time.perf_counter() - started) * 1000,
                                    error=redact(f"{type(exc).__name__}: {exc}", secrets)[:200]))
        return _fail(out, "provider_error", 502, f"[Mirsad] Provider error ({kind}); no key failover for non-Groq models.")
    out.attempts.append(Attempt(1, "env", "success", 200, (time.perf_counter() - started) * 1000))
    out.response, out.alias = response, "env"
    return out
