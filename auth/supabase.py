"""
Stateless client for Supabase Auth (GoTrue) REST endpoints, using only the PUBLISHABLE key.

Stateless on purpose: no auth client, token or session is stored on this object, so nothing mutable is ever
shared between users. Every call receives the tokens it needs and returns plain data.

Endpoints (https://supabase.com/docs/reference/self-hosting-auth/introduction and the GoTrue OpenAPI spec):
  POST /signup  /token?grant_type=password|refresh_token  /recover  /resend  /verify  /logout?scope=local
  GET|PUT /user
Headers: `apikey: <publishable key>` always; `Authorization: Bearer <user access token>` only for /user and /logout
(a publishable key is not a JWT and is never sent as a bearer token).

Nothing here logs or returns raw upstream response bodies: errors are reduced to (status, code) and a fixed message.
"""
import logging

import httpx

import config

log = logging.getLogger("mirsad.auth")


class SupabaseError(Exception):
    """status = HTTP status from Supabase (503 when unreachable); code = GoTrue error_code when known."""

    def __init__(self, status: int, code: str = "", message: str = ""):
        super().__init__(f"supabase {status} {code}".strip())
        self.status, self.code, self.public = status, code, message


class NotConfigured(RuntimeError):
    pass


def _err(resp: httpx.Response) -> SupabaseError:
    code = ""
    try:
        body = resp.json()
        code = str(body.get("error_code") or body.get("code") or body.get("error") or "")[:60]
    except Exception:  # noqa: BLE001
        pass
    return SupabaseError(resp.status_code, code)


class SupabaseAuth:
    def __init__(self, url: str | None = None, key: str | None = None, http: httpx.Client | None = None):
        self.url = (config.SUPABASE_URL if url is None else url).rstrip("/")
        self.key = config.SUPABASE_PUBLISHABLE_KEY if key is None else key
        self._http = http or httpx.Client(timeout=httpx.Timeout(10.0))

    @property
    def configured(self) -> bool:
        return bool(self.url.startswith("http") and self.key)

    # ---------------------------------------------------------------- plumbing
    def _call(self, method: str, path: str, *, json=None, params=None, access_token: str | None = None):
        if not self.configured:
            raise NotConfigured("Supabase is not configured (SUPABASE_URL / SUPABASE_PUBLISHABLE_KEY)")
        headers = {"apikey": self.key, "Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        try:
            resp = self._http.request(method, f"{self.url}/auth/v1{path}", json=json, params=params, headers=headers)
        except httpx.HTTPError as exc:
            log.warning("supabase unreachable (%s)", type(exc).__name__)       # class only: no URL/headers/body
            raise SupabaseError(503, "upstream_unavailable") from None
        if resp.status_code >= 400:
            raise _err(resp)
        if resp.status_code == 204 or not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            raise SupabaseError(502, "bad_response") from None

    # ---------------------------------------------------------------- operations
    def signup(self, email: str, password: str, name: str, redirect_to: str) -> dict:
        """Only `name` is sent as user metadata, never any client-supplied role/claims.
        Returns {"user": {...}, "session": {...} | None}: no session while email confirmation is required."""
        data = self._call("POST", "/signup", params={"redirect_to": redirect_to},
                          json={"email": email, "password": password, "data": {"name": name}})
        if data.get("access_token"):
            return {"user": data.get("user") or {}, "session": data}
        return {"user": data.get("user") or data, "session": None}

    def login(self, email: str, password: str) -> dict:
        return self._call("POST", "/token", params={"grant_type": "password"},
                          json={"email": email, "password": password})

    def refresh(self, refresh_token: str) -> dict:
        return self._call("POST", "/token", params={"grant_type": "refresh_token"},
                          json={"refresh_token": refresh_token})

    def get_user(self, access_token: str) -> dict:
        return self._call("GET", "/user", access_token=access_token)

    def logout(self, access_token: str) -> None:
        self._call("POST", "/logout", params={"scope": "local"}, access_token=access_token)   # this device only

    def recover(self, email: str, redirect_to: str) -> None:
        self._call("POST", "/recover", params={"redirect_to": redirect_to}, json={"email": email})

    def resend_signup(self, email: str, redirect_to: str) -> None:
        self._call("POST", "/resend", params={"redirect_to": redirect_to}, json={"type": "signup", "email": email})

    def verify_token_hash(self, token_hash: str, type_: str) -> dict:
        return self._call("POST", "/verify", json={"type": type_, "token_hash": token_hash})

    def update_password(self, access_token: str, password: str) -> dict:
        return self._call("PUT", "/user", access_token=access_token, json={"password": password})
