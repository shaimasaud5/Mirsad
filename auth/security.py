"""Cookie settings, same-origin / CSRF checks, a tiny rate limiter and security headers."""
import hmac
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit

from fastapi import Request

import config

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


# ---------------------------------------------------------------- cookies
def cookie_secure() -> bool:
    return config.PUBLIC_URL.startswith("https://")


def cookie_name() -> str:
    # __Host- requires Secure + Path=/ + no Domain; used automatically once the site runs on https
    return "__Host-mirsad_sid" if cookie_secure() else "mirsad_sid"


def set_session_cookie(response, sid: str) -> None:
    response.set_cookie(cookie_name(), sid, httponly=True, secure=cookie_secure(), samesite="lax", path="/",
                        max_age=int(config.SESSION_MAX_DAYS * 86400))


def clear_session_cookie(response) -> None:
    response.delete_cookie(cookie_name(), path="/", httponly=True, secure=cookie_secure(), samesite="lax")


# ---------------------------------------------------------------- CSRF / origin
def _origin(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}".lower() if p.scheme and p.netloc else ""


def same_origin_ok(request: Request) -> bool:
    """Browsers always send Origin on cross-site POST/PUT/DELETE. It must be this site (PUBLIC_URL or the host the
    request really arrived on). Sec-Fetch-Site is a second signal. Non-browser clients (no Origin) are judged by the
    CSRF token / cookie instead."""
    origin = request.headers.get("origin")
    if origin is not None:
        own = f"{request.url.scheme}://{request.headers.get('host', '')}".lower()
        return origin.lower() in (own, _origin(config.PUBLIC_URL))
    site = request.headers.get("sec-fetch-site")
    return site in (None, "same-origin", "none")


def csrf_ok(request: Request, expected: str) -> bool:
    got = request.headers.get("x-csrf-token", "")
    return bool(expected) and hmac.compare_digest(got.encode(), expected.encode())


# ---------------------------------------------------------------- rate limit (in memory, per process)
class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._clock = clock

    def allow(self, key: str, limit: int, window_s: float) -> bool:
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self._hits) > 5000:               # keep memory bounded
                for k in [k for k, v in self._hits.items() if not v][:1000]:
                    self._hits.pop(k, None)
            return True


# ---------------------------------------------------------------- headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'self'; form-action 'self'"),
}
