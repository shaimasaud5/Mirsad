"""
Mirsad web UI (chat + dashboard) - replaces the Streamlit page.

It is a small FastAPI app that:
  * serves the single page   ui/web/index.html
  * runs the LangChain agent  POST /api/chat     (traffic still goes through the Mirsad proxy)
  * feeds the dashboard       GET  /api/metrics  (reads the same SQLite log as before)

The proxy (proxy/main.py) and the agent (agent/agent.py) are unchanged.

Run (the proxy must be running on port 8000 in another terminal):
    uvicorn ui.web_server:app --port 8501
Then open http://localhost:8501
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from agent.agent import run_agent
from monitoring import metrics

WEB_DIR = Path(__file__).resolve().parent / "web"
PROXY_BASE = config.PROXY_URL.rsplit("/v1", 1)[0]

app = FastAPI(title="Mirsad UI")


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


def _turn_summary(session_id: str, turn_id: str | None, since: float) -> dict:
    """What happened for one user message: route, model(s), cost vs GPT-4o, latency."""
    df = metrics.load()
    if df.empty:
        return {}
    if turn_id and "turn_id" in df and (df["turn_id"] == turn_id).any():
        df = df[df["turn_id"] == turn_id]
    else:
        df = df[(df["session_id"] == session_id) & (df["ts"] >= since - 1)]
    if df.empty:
        return {}
    first = df.iloc[0]
    get = lambda c: _clean(first.get(c)) if c in df else None  # noqa: E731
    models = sorted({str(m).split("/", 1)[-1] for m in df["model"].dropna().unique()})
    actual, baseline = float(df["actual_cost"].sum()), float(df["baseline_cost"].sum())
    return {
        "route": get("route"),
        "routing_mode": get("routing_mode"),
        "route_source": get("route_source"),
        "fallback_reason": get("fallback_reason"),
        "llm_route": get("llm_route"),
        "classifier_model": get("classifier_model"),
        "classifier_ms": get("classifier_ms"),
        "execution_mode": get("execution_mode"),
        "models": models,
        "calls": int(len(df)),
        "actual_cost": actual,
        "baseline_cost": baseline,
        "saved_pct": (baseline - actual) / baseline * 100 if baseline > 0 else 0.0,
        "latency_ms": float(df["latency_ms"].fillna(0).sum()),
        "failover_used": bool("failover_used" in df and (df["failover_used"].fillna(0) == 1).any()),
        "token_source": sorted(set(df["token_source"].dropna())) if "token_source" in df else [],
    }


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


def _proxy_health() -> dict | None:
    try:
        return httpx.get(f"{PROXY_BASE}/health", timeout=3).json()
    except Exception:
        return None


# ---------------------------------------------------------------- page
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


BRAND_FILES = ("logo-light.png", "logo-dark.png", "favicon.png", "apple-touch-icon.png")


@app.get("/{name}.png")
def brand_image(name: str):
    if f"{name}.png" not in BRAND_FILES:
        raise HTTPException(status_code=404)
    return FileResponse(WEB_DIR / f"{name}.png", media_type="image/png")


app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")   # marked, DOMPurify, Chart.js


# ---------------------------------------------------------------- chat
class ChatRequest(BaseModel):
    task: str
    session_id: str | None = None
    history: list[dict] = []


@app.post("/api/chat")
def chat(req: ChatRequest):
    # Plain `def`: FastAPI runs it in a worker thread, so a slow agent never blocks the server.
    started = time.time()
    history = [{"role": m.get("role"), "content": m.get("content", "")}
               for m in req.history if m.get("role") in ("user", "assistant")]
    result = run_agent(req.task.strip(), session_id=req.session_id, history=history)
    result["summary"] = _turn_summary(result["session_id"], result.get("turn_id"), started)
    return result


# ---------------------------------------------------------------- dashboard
REQUEST_COLS = ["time", "session_id", "execution_mode", "status", "routing_mode", "route_source",
                "fallback_reason", "llm_route", "route", "model", "classifier_ms", "credential_alias",
                "provider_attempts", "token_source", "input_tokens", "output_tokens", "latency_ms",
                "actual_cost", "baseline_cost", "reason", "error"]
ATTEMPT_COLS = ["time", "request_id", "attempt_no", "credential_alias", "model", "outcome",
                "http_status", "duration_ms", "failover", "failover_reason", "retry_after_s", "error"]


@app.get("/api/metrics")
def dashboard_metrics(minutes: float | None = None, kind: str = "all"):
    df = metrics.filter_execution(metrics.load(since_minutes=minutes), kind)
    out = {"health": _proxy_health(),
           "alert_data": _alert_data(),
           "summary": {k: _clean(v) for k, v in metrics.summarize(df).items()}}
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
        out["latency_by_model"] = [{"model": m, "avg": _clean(g.mean()), "p95": _clean(g.quantile(0.95))}
                                   for m, g in lat]
        tok = ok.groupby("route")[["input_tokens", "output_tokens"]].sum().reset_index()
        out["tokens_by_route"] = _records(tok, ["route", "input_tokens", "output_tokens"])

    mix = df.groupby(["route", "status"]).size().reset_index(name="count")
    out["route_status"] = _records(mix, ["route", "status", "count"])
    out["recent"] = _records(df.iloc[::-1].head(50), REQUEST_COLS)
    att = metrics.load_attempts(limit=20)
    out["attempts"] = _records(att, ATTEMPT_COLS) if not att.empty else []
    return out


@app.get("/api/health")
def health():
    return {"proxy": _proxy_health()}


@app.post("/api/reset")
def reset():
    try:
        httpx.post(f"{PROXY_BASE}/admin/reset", timeout=5).raise_for_status()
        return {"status": "reset"}
    except Exception as exc:
        return {"status": "error", "message": f"Could not reach the proxy: {exc}"}
