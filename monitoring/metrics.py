"""
Metric calculations used by the dashboard (and the tests).
Each metric maps to one monitoring type from the project brief:

  Application  : total requests, error rate, loops blocked
  Performance  : latency per model (avg, p95)
  AI quality   : route mix (offline routing accuracy lives in eval/)
  Cost         : tokens, actual vs baseline cost, savings %, estimated loop savings
"""
import sqlite3
import time

import pandas as pd

import config
from proxy.cost import savings_percent


def load(db_path=None, since_minutes: float | None = None) -> pd.DataFrame:
    path = db_path or config.DB_PATH
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        df = pd.read_sql_query("SELECT * FROM requests ORDER BY ts", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()
    if since_minutes and not df.empty:
        df = df[df["ts"] >= time.time() - since_minutes * 60]
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        # rows written before the routing migration have no execution_mode: keep them, label them
        if "execution_mode" in df:
            df["execution_mode"] = df["execution_mode"].fillna("legacy (unknown)")
    return df


def load_attempts(db_path=None, limit: int = 50) -> pd.DataFrame:
    """Provider attempts (credential alias, outcome, status, duration, failover). Never contains key values."""
    path = db_path or config.DB_PATH
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        df = pd.read_sql_query(f"SELECT * FROM provider_attempts ORDER BY id DESC LIMIT {int(limit)}", conn)
        conn.close()
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s")
    return df


def filter_execution(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind: 'all' | 'live' | 'mock' | 'legacy'."""
    if df.empty or kind == "all" or "execution_mode" not in df:
        return df
    if kind == "legacy":
        return df[df["execution_mode"].str.startswith("legacy")]
    return df[df["execution_mode"] == kind]


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0}
    ok = df[df["status"] == "success"]
    total = len(df)
    errors = int((df["status"] == "error").sum())
    blocked = int((df["status"] == "blocked").sum())
    actual, baseline = ok["actual_cost"].sum(), ok["baseline_cost"].sum()
    col = lambda name: df[name] if name in df else pd.Series(dtype="object")  # noqa: E731
    classified = df[col("classifier_ms").notna() & (col("classification_reused").fillna(0) == 0)] \
        if "classifier_ms" in df else df.iloc[0:0]
    fb = col("route_source") == "rules_fallback"
    return {
        "live": int((col("execution_mode") == "live").sum()),
        "mock": int((col("execution_mode") == "mock").sum()),
        "legacy": int(col("execution_mode").astype(str).str.startswith("legacy").sum()),
        "fallbacks": int(fb.sum()),
        "credential_failovers": int((col("failover_used").fillna(0) == 1).sum()),
        "provider_errors": int(((df["status"] == "error") & (col("provider_attempts").fillna(0) > 0)).sum()),
        "reused": int((col("classification_reused").fillna(0) == 1).sum()),
        "classifier_turns": len(classified),
        "classifier_avg_ms": float(classified["classifier_ms"].mean()) if len(classified) else 0.0,
        "provider_token_rows": int((ok["token_source"] == "provider").sum()) if "token_source" in ok else 0,
        "estimated_token_rows": int((ok["token_source"] != "provider").sum()) if "token_source" in ok else len(ok),
        "total": total,
        "success": len(ok),
        "errors": errors,
        "blocked": blocked,
        "error_rate": errors / total,
        "avg_latency_ms": float(ok["latency_ms"].mean()) if len(ok) else 0.0,
        "p95_latency_ms": float(ok["latency_ms"].quantile(0.95)) if len(ok) else 0.0,
        "input_tokens": int(ok["input_tokens"].sum()),
        "output_tokens": int(ok["output_tokens"].sum()),
        "actual_cost": float(actual),
        "baseline_cost": float(baseline),
        "saved": float(baseline - actual),
        "savings_pct": savings_percent(baseline - actual, baseline),
        "est_loop_savings": float(df["est_saved_cost"].sum()),
        "simple_share": float((ok["route"] == "simple").mean()) if len(ok) else 0.0,
    }


def alerts(df: pd.DataFrame) -> list[str]:
    """Rules that turn the dashboard banner red (evaluated on the last N minutes)."""
    if df.empty:
        return []
    recent = df[df["ts"] >= time.time() - config.ALERT_WINDOW_MIN * 60]
    out = []
    if len(recent) >= 3:
        rate = (recent["status"] == "error").mean()
        if rate > config.ALERT_ERROR_RATE:
            out.append(f"Error rate {rate:.0%} in the last {config.ALERT_WINDOW_MIN} min "
                       f"(threshold {config.ALERT_ERROR_RATE:.0%}). Check provider status / rate limits.")
    n_blocked = int((recent["status"] == "blocked").sum())
    if n_blocked:
        out.append(f"{n_blocked} agent loop(s) blocked in the last {config.ALERT_WINDOW_MIN} min. "
                   "An agent is repeating a failing tool call.")
    return out
