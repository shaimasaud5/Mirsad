"""
Authentication of callers of the proxy.

The proxy is an INTERNAL service: only Mirsad's own web server (through the LangChain agent) may call it, presenting the
shared secret MIRSAD_INTERNAL_TOKEN as `Authorization: Bearer ...`. The browser never receives this secret, so browser
users cannot reach /v1 or /admin on port 8000 directly, and cannot forge ownership: the user id travels in
X-Mirsad-User-Id and is only honoured when the internal token is valid (the web server sets it from the verified session).

No token configured = setup error (HTTP 503) on every protected route. There is no anonymous mode.
"""
import hmac
import re

from fastapi import Request

import config

USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
CONV_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ProxyAuthError(Exception):
    def __init__(self, status: int, message: str, err_type: str):
        self.status, self.message, self.err_type = status, message, err_type


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def token_valid(request: Request) -> bool:
    expected = config.INTERNAL_TOKEN
    return bool(expected) and hmac.compare_digest(_bearer(request).encode(), expected.encode())


def require_internal(request: Request) -> None:
    if not config.INTERNAL_TOKEN:
        raise ProxyAuthError(503, "[Mirsad] Setup error: MIRSAD_INTERNAL_TOKEN is not configured, so the proxy refuses "
                                  "all requests. Set it in .env (same value for the web server and the proxy) and restart.",
                             "setup_error")
    if not token_valid(request):
        raise ProxyAuthError(401, "[Mirsad] Unauthorized: this proxy only accepts requests from the Mirsad web server.",
                             "unauthorized")


def caller_identity(request: Request) -> tuple[str, str | None]:
    """(user_id, conversation_id) from headers; only call after require_internal()."""
    uid = request.headers.get("x-mirsad-user-id", "")
    if not USER_ID.match(uid):
        raise ProxyAuthError(400, "[Mirsad] Missing or invalid X-Mirsad-User-Id: every request must be attributed to a "
                                  "verified user.", "missing_user")
    conv = request.headers.get("x-mirsad-conversation-id")
    return uid, (conv if conv and CONV_ID.match(conv) else None)
