"""
Mirsad web UI: login, chat with answer feedback, and (admins only) the global dashboard and feedback monitoring.

  * serves the single page   ui/web/index.html
  * authenticates users with Supabase Auth and keeps the session ON THE SERVER (HttpOnly cookie = opaque id)
  * runs the LangChain agent for the verified user; model traffic still goes through the Mirsad proxy, which only
    accepts this server (internal token) and records every request under the verified user id
  * stores conversations in SQLite per user; every read/write is scoped by the session's user id

Run (the proxy must be running on port 8000 in another terminal):
    uvicorn ui.web_server:app --port 8501
Then open http://localhost:8501   (setup: README, "Accounts and permissions")
"""
import logging
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import httpx
import jwt as pyjwt
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import config
from agent.agent import run_agent
from auth import chatstore, feedback, security, users
from auth.sessions import AuthSession, SessionManager
from auth.supabase import NotConfigured, SupabaseAuth, SupabaseError
from monitoring import feedback as feedback_stats
from monitoring import metrics
from proxy import db

log = logging.getLogger("mirsad.web")

WEB_DIR = Path(__file__).resolve().parent / "web"
PROXY_BASE = config.PROXY_URL.rsplit("/v1", 1)[0]

app = FastAPI(title="Mirsad UI", docs_url=None, redoc_url=None, openapi_url=None)   # no public API explorer
conn = db.connect()
supa = SupabaseAuth()
limiter = security.RateLimiter()
_manager: SessionManager | None = None


# ---------------------------------------------------------------- setup / errors
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str = "", **extra):
        self.status, self.code, self.message, self.extra = status, code, message, extra


@app.exception_handler(ApiError)
async def _api_error(_request, exc: ApiError):
    return JSONResponse(status_code=exc.status, content={"error": exc.code, "message": exc.message, **exc.extra})


def setup_problems() -> list[str]:
    """Names (never values) of the settings that are missing. Missing configuration = no access, never anonymous access."""
    missing = []
    if not (config.SUPABASE_URL.startswith("http") and config.SUPABASE_PUBLISHABLE_KEY):
        missing += [n for n, v in (("SUPABASE_URL", config.SUPABASE_URL), ("SUPABASE_PUBLISHABLE_KEY", config.SUPABASE_PUBLISHABLE_KEY))
                    if not v or (n == "SUPABASE_URL" and not v.startswith("http"))]
    if len(config.SESSION_SECRET) < 16:
        missing.append("MIRSAD_SESSION_SECRET")
    if not config.INTERNAL_TOKEN:
        missing.append("MIRSAD_INTERNAL_TOKEN")
    return missing


def manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager(conn, supa, config.SESSION_SECRET)
    return _manager


def require_setup() -> None:
    missing = setup_problems()
    if missing:
        raise ApiError(503, "setup_error", "Server setup is incomplete. Set the missing values in .env and restart.",
                       missing=missing)


# ---------------------------------------------------------------- security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    if request.method not in security.SAFE_METHODS and path.startswith("/api/") and not security.same_origin_ok(request):
        resp = JSONResponse(status_code=403, content={"error": "bad_origin", "message": "Cross-site request rejected."})
    else:
        resp = await call_next(request)
    for k, v in security.SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    if path.startswith("/api/") or path.startswith("/auth/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- identity (always from the server-side session)
class Principal:
    def __init__(self, sess: AuthSession, user: dict):
        self.sess, self.user_id = sess, sess.user_id
        self.email, self.name, self.role = user["email"], user["display_name"], user["role"]
        self.csrf, self.recovery = sess.csrf_token, sess.recovery

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _authenticate(request: Request, *, admin: bool = False, allow_recovery: bool = False, csrf: bool | None = None) -> Principal:
    require_setup()
    sid = request.cookies.get(security.cookie_name())
    sess = manager().load(sid)
    if sess is None:
        raise ApiError(401, "not_authenticated", "Please sign in.")
    user = users.get(conn, sess.user_id)
    if user is None or user["disabled"]:
        manager().revoke(sid, remote=False)
        raise ApiError(401, "not_authenticated", "Please sign in.")
    p = Principal(sess, user)
    if p.recovery and not allow_recovery:
        raise ApiError(403, "password_reset_required", "Set a new password to continue.")
    needs_csrf = (request.method not in security.SAFE_METHODS) if csrf is None else csrf
    if needs_csrf and not security.csrf_ok(request, p.csrf):
        raise ApiError(403, "bad_csrf", "Missing or invalid CSRF token.")
    if admin and not p.is_admin:
        raise ApiError(403, "forbidden", "Administrator access required.")      # role comes from app_users only
    return p


def require_user(request: Request) -> Principal:
    return _authenticate(request)


def require_admin(request: Request) -> Principal:
    return _authenticate(request, admin=True)


def require_user_or_recovery(request: Request) -> Principal:
    return _authenticate(request, allow_recovery=True)


# ---------------------------------------------------------------- helpers
def _clean(value):
    """JSON-safe values: NaN/NaT -> None, numpy/pandas scalars -> Python."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    cols = [c for c in cols if c in df.columns]
    return [{c: _clean(v) for c, v in row.items()} for row in df[cols].to_dict("records")]


def _alert_data() -> dict:
    """Same rules as monitoring.metrics.alerts, returned as numbers so the page can show them in Arabic or English."""
    recent = metrics.load(since_minutes=config.ALERT_WINDOW_MIN)
    data = {"window_min": config.ALERT_WINDOW_MIN, "error_threshold": config.ALERT_ERROR_RATE,
            "error_rate": None, "blocked": 0}
    if recent.empty:
        return data
    if len(recent) >= 3:
        data["error_rate"] = float((recent["status"] == "error").mean())
    data["blocked"] = int((recent["status"] == "blocked").sum())
    return data


def _proxy_headers() -> dict:
    return {"Authorization": f"Bearer {config.INTERNAL_TOKEN}"}


def _proxy_health() -> dict | None:
    try:
        r = httpx.get(f"{PROXY_BASE}/health", headers=_proxy_headers(), timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _user_json(p: Principal) -> dict:
    return {"name": p.name, "email": p.email, "role": p.role}


def _dashboard_payload(df: pd.DataFrame) -> dict:
    out = {"summary": {k: _clean(v) for k, v in metrics.summarize(df).items()}}
    if df.empty:
        return out
    ok = df[df["status"] == "success"]
    if not ok.empty:
        cum = ok[["time", "actual_cost", "baseline_cost"]].copy()
        cum[["actual_cost", "baseline_cost"]] = cum[["actual_cost", "baseline_cost"]].cumsum()
        step = max(1, len(cum) // 200)                      # keep the chart light
        cum = pd.concat([cum.iloc[::step], cum.iloc[[-1]]]).drop_duplicates()
        out["cost_series"] = _records(cum, ["time", "actual_cost", "baseline_cost"])
        lat = ok.assign(m=ok["model"].astype(str).str.split("/").str[-1]).groupby("m")["latency_ms"]
        out["latency_by_model"] = [{"model": m, "avg": _clean(g.mean()), "p95": _clean(g.quantile(0.95))} for m, g in lat]
        tok = ok.groupby("route")[["input_tokens", "output_tokens"]].sum().reset_index()
        out["tokens_by_route"] = _records(tok, ["route", "input_tokens", "output_tokens"])
    mix = df.groupby(["route", "status"]).size().reset_index(name="count")
    out["route_status"] = _records(mix, ["route", "status", "count"])
    out["recent"] = _records(df.iloc[::-1].head(50), ADMIN_REQUEST_COLS)
    return out


# ---------------------------------------------------------------- page + static
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/auth/callback")
def auth_callback_page():
    """Email links land here. Tokens arrive in the URL FRAGMENT (#...), which browsers never send to the server, so they
    cannot appear in server logs; the page's script posts them to /api/auth/callback in a JSON body."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}          # public and minimal on purpose


BRAND_FILES = ("logo-light.png", "logo-dark.png", "favicon.png", "apple-touch-icon.png")


@app.get("/{name}.png")
def brand_image(name: str):
    if f"{name}.png" not in BRAND_FILES:
        raise HTTPException(status_code=404)
    return FileResponse(WEB_DIR / f"{name}.png", media_type="image/png")


app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")   # marked, DOMPurify, Chart.js


# ---------------------------------------------------------------- auth API
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,63}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9._\-]{20,4096}$")


class SignupIn(BaseModel):
    model_config = ConfigDict(extra="forbid")      # role/metadata fields from the browser are rejected, not ignored
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(max_length=320)
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=320)
    password: str = Field(min_length=1, max_length=128)


class EmailIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(max_length=320)


class CallbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_token: str | None = Field(default=None, max_length=4096)
    refresh_token: str | None = Field(default=None, max_length=4096)
    token_hash: str | None = Field(default=None, max_length=512)
    type: str | None = Field(default=None, max_length=20)


class PasswordIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=8, max_length=128)


def _email(value: str) -> str:
    value = (value or "").strip().lower()
    if not EMAIL_RE.match(value):
        raise ApiError(422, "invalid_email", "Enter a valid email address.")
    return value


def _ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limit(request: Request, bucket: str, email: str, per_email: int = 6, per_ip: int = 30, window: float = 600) -> None:
    if not (limiter.allow(f"{bucket}:ip:{_ip(request)}", per_ip, window) and limiter.allow(f"{bucket}:em:{email}", per_email, window)):
        raise ApiError(429, "rate_limited", "Too many attempts. Wait a few minutes and try again.")


def _supa_failure(exc: Exception) -> ApiError:
    if isinstance(exc, NotConfigured):
        return ApiError(503, "setup_error", "Supabase is not configured.")
    if isinstance(exc, SupabaseError):
        if exc.status == 429:
            return ApiError(429, "rate_limited", "Too many attempts. Wait a few minutes and try again.")
        if exc.status >= 500:
            return ApiError(503, "auth_unavailable", "The sign-in service is unreachable. Try again shortly.")
    return ApiError(502, "auth_failed", "Authentication failed.")


def _start_session(request: Request, supa_session: dict, *, recovery: bool = False) -> JSONResponse:
    try:
        sid, user = manager().create(supa_session, recovery=recovery)
    except PermissionError:
        raise ApiError(403, "account_disabled", "This account is disabled.") from None
    except SupabaseError:
        raise ApiError(502, "auth_failed", "Authentication failed.") from None
    resp = JSONResponse({"status": "signed_in", "recovery": recovery,
                         "user": {"name": user["display_name"], "email": user["email"], "role": user["role"]}})
    security.set_session_cookie(resp, sid)
    return resp


@app.get("/api/auth/session")
def auth_session(request: Request):
    missing = setup_problems()
    if missing:
        return {"authenticated": False, "setup_error": missing}
    try:
        p = _authenticate(request, allow_recovery=True, csrf=False)
    except ApiError:
        return {"authenticated": False}
    return {"authenticated": True, "recovery": p.recovery, "user": _user_json(p), "csrf_token": p.csrf}


@app.post("/api/auth/signup")
def auth_signup(body: SignupIn, request: Request):
    require_setup()
    email = _email(body.email)
    _limit(request, "signup", email, per_email=4)
    name = users.clean_name(body.name)
    if not name:
        raise ApiError(422, "invalid_name", "Enter your name.")
    try:
        result = supa.signup(email, body.password, name, redirect_to=f"{config.PUBLIC_URL}/auth/callback")
    except SupabaseError as exc:
        if exc.code in ("weak_password",) or exc.status == 422:
            raise ApiError(400, "weak_password", "Choose a stronger password (at least 8 characters).") from None
        raise _supa_failure(exc) from None
    except NotConfigured as exc:
        raise _supa_failure(exc) from None
    if result["session"]:                       # email confirmation is disabled in this Supabase project
        return _start_session(request, result["session"])
    return {"status": "confirmation_required"}  # identical whether or not the address already exists


@app.post("/api/auth/login")
def auth_login(body: LoginIn, request: Request):
    require_setup()
    email = _email(body.email)
    _limit(request, "login", email, per_email=8)
    try:
        session = supa.login(email, body.password)
    except SupabaseError as exc:
        if exc.code == "email_not_confirmed":
            raise ApiError(403, "email_not_confirmed", "Confirm your email first, then sign in.") from None
        if exc.status in (400, 401, 422):
            raise ApiError(401, "invalid_credentials", "Incorrect email or password.") from None
        raise _supa_failure(exc) from None
    except NotConfigured as exc:
        raise _supa_failure(exc) from None
    return _start_session(request, session)


@app.post("/api/auth/recover")
def auth_recover(body: EmailIn, request: Request):
    require_setup()
    email = _email(body.email)
    _limit(request, "recover", email, per_email=3)
    try:
        supa.recover(email, redirect_to=f"{config.PUBLIC_URL}/auth/callback")
    except (SupabaseError, NotConfigured) as exc:
        if isinstance(exc, SupabaseError) and exc.status == 429:
            raise _supa_failure(exc) from None
        log.info("recover request not delivered (%s)", getattr(exc, "status", "n/a"))   # never reveal whether the account exists
    return {"status": "sent"}


@app.post("/api/auth/resend")
def auth_resend(body: EmailIn, request: Request):
    require_setup()
    email = _email(body.email)
    _limit(request, "resend", email, per_email=3)
    try:
        supa.resend_signup(email, redirect_to=f"{config.PUBLIC_URL}/auth/callback")
    except (SupabaseError, NotConfigured) as exc:
        if isinstance(exc, SupabaseError) and exc.status == 429:
            raise _supa_failure(exc) from None
    return {"status": "sent"}


@app.post("/api/auth/callback")
def auth_callback(body: CallbackIn, request: Request):
    """Completes email confirmation / password recovery. The browser reads the URL fragment and posts it here; the
    tokens are NOT trusted: they are verified with Supabase before a session is created."""
    require_setup()
    _limit(request, "callback", _ip(request), per_email=20)
    kind = body.type if body.type in ("signup", "recovery", "magiclink", "email", "invite", "email_change") else "signup"
    try:
        if body.token_hash:
            session = supa.verify_token_hash(body.token_hash, "recovery" if kind == "recovery" else "email" if kind in ("signup", "email") else kind)
        else:
            a, r = body.access_token or "", body.refresh_token or ""
            if not (TOKEN_RE.match(a) and TOKEN_RE.match(r)):
                raise ApiError(400, "invalid_link", "This link is invalid or has expired.")
            user = supa.get_user(a)                          # authoritative check with Supabase
            try:
                exp = pyjwt.decode(a, options={"verify_signature": False}).get("exp")   # scheduling only, not trust
            except pyjwt.PyJWTError:
                exp = None
            session = {"access_token": a, "refresh_token": r, "user": user, "expires_at": exp}
    except SupabaseError as exc:
        if exc.status in (400, 401, 403, 404, 422):
            raise ApiError(400, "invalid_link", "This link is invalid or has expired.") from None
        raise _supa_failure(exc) from None
    except NotConfigured as exc:
        raise _supa_failure(exc) from None
    return _start_session(request, session, recovery=(kind == "recovery"))


@app.post("/api/auth/logout")
def auth_logout(request: Request):
    sid = request.cookies.get(security.cookie_name())
    if sid and not setup_problems():
        try:
            _authenticate(request, allow_recovery=True)       # CSRF-checked
        except ApiError as exc:
            if exc.code == "bad_csrf":
                raise
        manager().revoke(sid)
    resp = JSONResponse({"status": "signed_out"})
    security.clear_session_cookie(resp)
    return resp


@app.post("/api/auth/password")
def auth_password(body: PasswordIn, p: Principal = Depends(require_user_or_recovery)):
    try:
        supa.update_password(p.sess.access_token, body.password)
    except SupabaseError as exc:
        if exc.code == "weak_password" or exc.status in (400, 422):
            raise ApiError(400, "weak_password", "Choose a different, stronger password.") from None
        raise _supa_failure(exc) from None
    if p.recovery:
        manager().set_recovery(p.sess.sid_hash, False)       # the reset link only unlocks the app after a new password
    return {"status": "updated"}


# ---------------------------------------------------------------- conversations (owner only)
def _own_conversation(conv_id: str, p: Principal) -> dict:
    conv = chatstore.get(conn, conv_id, p.user_id)           # scoped by user_id inside the SQL
    if conv is None:
        raise ApiError(404, "not_found", "Conversation not found.")   # same answer for "missing" and "someone else's"
    return conv


@app.get("/api/conversations")
def list_conversations(p: Principal = Depends(require_user)):
    return {"conversations": chatstore.list_for(conn, p.user_id)}


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str, p: Principal = Depends(require_user)):
    conv = _own_conversation(conv_id, p)
    return {"id": conv["id"], "title": conv["title"], "messages": chatstore.messages(conn, conv_id, p.user_id) or []}


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str, p: Principal = Depends(require_user)):
    _own_conversation(conv_id, p)
    chatstore.delete(conn, conv_id, p.user_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------- chat
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")     # no user_id / role / history from the browser
    task: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, max_length=64)


@app.post("/api/chat")
def chat(req: ChatRequest, p: Principal = Depends(require_user)):
    # Plain `def`: FastAPI runs it in a worker thread, so a slow agent never blocks the server.
    task = req.task.strip()
    if not task:
        raise ApiError(422, "empty_task", "Type a message first.")
    if not limiter.allow(f"chat:{p.user_id}", config.CHAT_RATE_PER_MIN, 60):
        raise ApiError(429, "rate_limited", "You are sending messages too quickly. Wait a moment.")
    started = time.time()
    conv = _own_conversation(req.conversation_id, p) if req.conversation_id else chatstore.create(conn, p.user_id, task)
    history = chatstore.history_for_agent(conn, conv["id"], p.user_id)          # from the database, not the browser
    chatstore.add_message(conn, conv["id"], p.user_id, "user", task)
    try:
        result = run_agent(task, session_id=conv["session_id"], history=history,
                           user_id=p.user_id, conversation_id=conv["id"])
    except RuntimeError as exc:                   # setup problem inside the agent (e.g. internal token missing)
        log.error("agent setup error: %s", exc)
        raise ApiError(503, "setup_error", "The agent is not configured correctly. Check the server setup.") from None
    # Technical details (route, model, tokens, cost, failover) stay in the monitoring records for administrators.
    # Errors are stored as a safe code, never as raw provider/proxy text.
    status = result["status"]
    stored = chatstore.safe_error_code(result["answer"]) if status == "error" else result["answer"]
    files = result.get("files") or []
    message_id = chatstore.add_message(conn, conv["id"], p.user_id, "assistant", stored, status=status,
                                       steps=result.get("steps"), files=files, turn_id=result.get("turn_id"))
    if status == "blocked":                       # a blocked agent session stays blocked by design: start a fresh one
        chatstore.rotate_session(conn, conv["id"], p.user_id)
    return {"answer": stored, "status": status, "conversation_id": conv["id"], "message_id": message_id,
            "files": files, "feedback": None}


# ---------------------------------------------------------------- answer feedback (owner only)
class FeedbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")     # no user_id / conversation / role fields from the browser
    rating: Literal["helpful", "not_helpful"]
    reason: Literal["incorrect", "incomplete", "misunderstood", "other"] | None = None
    comment: str | None = Field(default=None, max_length=feedback.MAX_COMMENT)


def _feedback_limit(p: Principal) -> None:
    if not limiter.allow(f"feedback:{p.user_id}", 60, 60):
        raise ApiError(429, "rate_limited", "Too many requests. Wait a moment.")


@app.put("/api/messages/{message_id}/feedback")
def put_feedback(message_id: int, body: FeedbackIn, p: Principal = Depends(require_user)):
    _feedback_limit(p)
    if body.reason and body.rating != "not_helpful":
        raise ApiError(422, "invalid_feedback", "A reason can only accompany a Not helpful rating.")
    saved = feedback.save(conn, p.user_id, message_id, body.rating, body.reason, body.comment)   # owner check is in the SQL
    if saved is None:
        raise ApiError(404, "not_found", "Answer not found.")
    return {"feedback": saved}


@app.delete("/api/messages/{message_id}/feedback")
def delete_feedback(message_id: int, p: Principal = Depends(require_user)):
    _feedback_limit(p)
    if not feedback.clear(conn, p.user_id, message_id):
        raise ApiError(404, "not_found", "Answer not found.")
    return {"feedback": None}


# ---------------------------------------------------------------- dashboard (administrators only)
ADMIN_REQUEST_COLS = ["time", "request_id", "user_id", "session_id", "execution_mode", "status", "routing_mode", "route_source",
                      "fallback_reason", "llm_route", "route", "model", "classifier_ms", "credential_alias",
                      "provider_attempts", "token_source", "input_tokens", "output_tokens", "latency_ms",
                      "actual_cost", "baseline_cost", "reason", "error"]
ATTEMPT_COLS = ["time", "request_id", "attempt_no", "credential_alias", "model", "outcome",
                "http_status", "duration_ms", "failover", "failover_reason", "retry_after_s", "error"]


@app.get("/api/metrics")
def dashboard_metrics(minutes: float | None = None, kind: str = "all", p: Principal = Depends(require_admin)):
    df = metrics.filter_execution(metrics.load(since_minutes=minutes), kind)
    out = {"health": _proxy_health(), "alert_data": _alert_data(), **_dashboard_payload(df)}
    att = metrics.load_attempts(limit=20)
    out["attempts"] = _records(att, ATTEMPT_COLS) if not att.empty else []
    return out


@app.get("/api/health")
def health(p: Principal = Depends(require_user)):
    h = _proxy_health()
    if h is None:
        return {"proxy": None}
    if p.is_admin:
        return {"proxy": h}
    return {"proxy": {k: h.get(k) for k in ("status", "mock_mode", "execution", "routing_mode")}}


# ---------------------------------------------------------------- administration (admins only)
@app.get("/api/admin/users")
def admin_users(p: Principal = Depends(require_admin)):
    return {"users": [{**u, "last_login_at": _clean(u["last_login_at"])} for u in users.list_all(conn)]}


@app.get("/api/admin/feedback")
def admin_feedback(minutes: float | None = None, kind: str = "live", rating: str | None = None, reason: str | None = None,
                   limit: int = 100, p: Principal = Depends(require_admin)):
    """Same date window and execution kind as the dashboard (default: live records only) for summary AND list."""
    if kind not in ("all", "live", "mock", "legacy") or rating not in (None, "helpful", "not_helpful")             or reason not in (None, *feedback_stats.REASON_KEYS):
        raise ApiError(422, "invalid_filter", "Unknown filter value.")
    return feedback_stats.report(conn, minutes=minutes, kind=kind, rating=rating, reason=reason, limit=limit)


@app.get("/api/admin/feedback/{message_id}")
def admin_feedback_detail(message_id: int, p: Principal = Depends(require_admin)):
    d = feedback_stats.detail(conn, message_id)
    if d is None:
        raise ApiError(404, "not_found", "Answer not found.")
    return d


@app.post("/api/admin/reset")
def admin_reset(p: Principal = Depends(require_admin)):
    try:
        httpx.post(f"{PROXY_BASE}/admin/reset", headers=_proxy_headers(), timeout=5).raise_for_status()
        return {"status": "reset"}
    except Exception:
        return {"status": "error", "message": "Could not reach the proxy."}
