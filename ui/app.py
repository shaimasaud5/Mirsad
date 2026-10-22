"""
Streamlit app with two pages:
  * Chat       - talk to the real LangChain agent (its traffic goes through Mirsad)
  * Dashboard  - live monitoring read from SQLite, refreshes every 2 seconds

Run:  streamlit run ui/app.py
"""
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import httpx
import pandas as pd
import streamlit as st

import config
from agent.agent import run_agent
from monitoring import metrics

# RETIRED: this page showed every request without signing in and cannot authenticate users.
# The supported interface is the web UI:  uvicorn ui.web_server:app --port 8501
st.set_page_config(page_title="Mirsad", page_icon="🛡️", layout="wide")
st.error("RETIRED / متوقفة: this Streamlit page bypassed authentication. Run the web UI instead: "
         "`uvicorn ui.web_server:app --port 8501` (see README, 'Accounts and permissions').")
st.stop()

DEMO_TASKS = {
    "Simple task": "Read sales.csv and tell me the total revenue.",
    "Complex task": ("Analyze sales.csv step by step: first compare revenue by region, "
                     "then explain why Riyadh performs differently and design a plan "
                     "to grow Jeddah next quarter."),
    "Loop scenario": "Fetch the q3_revenue report and summarize it.",
}


def new_session():
    st.session_state.session_id = "sess_" + uuid.uuid4().hex[:6]
    st.session_state.chat = []


def task_summary(session_id: str, turn_id: str | None, since: float) -> str:
    """Concise status under a response: who classified, who answered, live/mock, token source, cost."""
    df = metrics.load()
    if df.empty:
        return ""
    if turn_id and "turn_id" in df and (df["turn_id"] == turn_id).any():
        df = df[df["turn_id"] == turn_id]
    else:
        df = df[(df["session_id"] == session_id) & (df["ts"] >= since - 1)]
    if df.empty:
        return ""
    first = df.iloc[0]
    answered = ", ".join(sorted(str(m).split("/", 1)[-1] for m in df["model"].dropna().unique()))
    parts = []
    if first.get("routing_mode") in ("local_llm", "shadow"):
        cls = f"{str(first.get('classifier_model') or '?')} → {first.get('llm_route') or 'no result'}"
        if pd.notna(first.get("classifier_ms")) and not first.get("classification_reused"):
            cls += f" ({first['classifier_ms']:.0f} ms)"
        parts.append(f"Classifier: {cls}")
    parts.append(f"Routing: {first.get('routing_mode') or 'legacy'} → {first['route']}"
                 + (f" ⚠ rules fallback ({first['fallback_reason']})" if first.get("route_source") == "rules_fallback" else ""))
    reused = int(df["classification_reused"].fillna(0).sum()) if "classification_reused" in df else 0
    parts.append(f"Answering model: {answered} · {len(df)} call(s), classification reused ×{reused}")
    if "failover_used" in df and (df["failover_used"].fillna(0) == 1).any():
        parts.append("🔑 API-key failover used (credential, not model, changed)")
    mode = str(first.get("execution_mode") or "legacy")
    parts.append("🟡 MOCK answers (not a real model)" if mode == "mock" else "🟢 live" if mode == "live" else mode)
    srcs = ", ".join(sorted(set(df["token_source"].dropna()))) if "token_source" in df else ""
    parts.append(f"tokens: {srcs or 'estimated (legacy)'}")
    parts.append(f"priced ${df['actual_cost'].sum():.6f} vs hypothetical all-GPT-4o ${df['baseline_cost'].sum():.6f}")
    return " · ".join(parts)


# ---------------------------------------------------------------- Chat page
def chat_page():
    if "session_id" not in st.session_state:
        new_session()

    st.title("Agent chat")
    st.caption(f"Session `{st.session_state.session_id}` · every model call goes through "
               f"the Mirsad proxy at `{config.PROXY_URL}`")

    with st.sidebar:
        st.subheader("Demo scenarios")
        for label, task in DEMO_TASKS.items():
            if st.button(label, width="stretch"):
                st.session_state.pending = task
        st.divider()
        if st.button("New session", width="stretch"):
            new_session(); st.rerun()

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("steps"):
                with st.expander(f"Agent steps ({len(turn['steps'])})"):
                    for s in turn["steps"]:
                        st.code(str(s), language="text")
            if turn.get("summary"):
                st.caption(turn["summary"])

    task = st.chat_input("Ask the agent something…") or st.session_state.pop("pending", None)
    if not task:
        return

    st.session_state.chat.append({"role": "user", "content": task})
    history = [{"role": t["role"], "content": t["content"]}
               for t in st.session_state.chat[:-1] if t.get("status") in (None, "ok")]
    started = time.time()
    with st.spinner("Agent is working…"):
        result = run_agent(task, session_id=st.session_state.session_id, history=history)

    content = result["answer"]
    if result["status"] == "blocked":
        content = f"🛑 **{content}**\n\n_A new session was started._"
    elif result["status"] != "ok":
        content = f"⚠️ {content}"
    st.session_state.chat.append({
        "role": "assistant", "content": content, "steps": result["steps"],
        "status": result["status"],
        "summary": task_summary(st.session_state.session_id, result.get("turn_id"), started)})
    if result["status"] == "blocked":   # the old session stays blocked by design
        chat = st.session_state.chat
        new_session()
        st.session_state.chat = chat
    st.rerun()


# ----------------------------------------------------------- Dashboard page
def dashboard_page():
    st.title("Mirsad · Monitoring")

    c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
    window = c1.selectbox("Time window", ["Last 15 min", "Last hour", "All time"], index=2)
    minutes = {"Last 15 min": 15, "Last hour": 60, "All time": None}[window]
    kind = c2.selectbox("Execution", ["all", "live", "mock", "legacy"], index=0,
                        help="live = real provider answers · mock = simulated answers · "
                             "legacy = rows recorded before this field existed (unknown)")
    try:
        health = httpx.get(config.PROXY_URL.replace("/v1", "/health"), timeout=3).json()
        c3.success(f"Proxy up · answers: {'MOCK (simulated)' if health['mock_mode'] else 'live Groq models'} · "
                   f"routing: {health.get('routing_mode', '?')} · classifier: "
                   f"{health.get('classifier_model', '?')} ({health.get('classifier_status', '?')})")
    except Exception:
        c3.error("Proxy is not reachable")
    if c4.button("Reset data"):
        try:
            httpx.post(config.PROXY_URL.replace("/v1", "/admin/reset"), timeout=5)
        except Exception:
            st.warning("Could not reach the proxy to reset.")

    live_panel(minutes, kind)


@st.fragment(run_every="2s")
def live_panel(minutes, kind="all"):
    df = metrics.filter_execution(metrics.load(since_minutes=minutes), kind)
    s = metrics.summarize(df)

    for msg in metrics.alerts(metrics.load(since_minutes=config.ALERT_WINDOW_MIN)):
        st.error("🚨 " + msg)

    if s["total"] == 0:
        st.info("No requests yet. Send a task from the Chat page.")
        return

    if s["mock"] or s["legacy"]:
        st.warning(f"This view includes {s['mock']} mock and {s['legacy']} legacy (unknown-origin) request(s); "
                   f"{s['live']} live. Use the Execution filter to separate them.")

    st.subheader("Cost (priced equivalents, not provider invoices)")
    k = st.columns(4)
    k[0].metric("Hypothetical savings", f"{s['savings_pct']:.1f}%",
                help="sum(saved) / sum(baseline). Baseline is HYPOTHETICAL: the same tokens priced as if every "
                     "request went to GPT-4o. Groq free tier costs nothing, so this is not measured spend.")
    k[1].metric("Priced cost (routed)", f"${s['actual_cost']:.5f}",
                help="Tokens × OpenAI list-price tier of the chosen route (config.PRICES). Not a Groq invoice.")
    k[2].metric("Hypothetical baseline (all GPT-4o)", f"${s['baseline_cost']:.5f}")
    k[3].metric("Loop savings (estimated)", f"${s['est_loop_savings']:.5f}",
                help="Blocked requests never reached a model; tokens are estimated at ~4 chars/token.")

    st.subheader("Routing & token sources")
    k = st.columns(5)
    k[0].metric("Live / mock / legacy", f"{s['live']} / {s['mock']} / {s['legacy']}")
    k[1].metric("Classifier fallbacks to rules", s["fallbacks"])
    k[2].metric("Classifier avg time", f"{s['classifier_avg_ms']:.0f} ms",
                help=f"Over {s['classifier_turns']} classified turn(s); steps that reused a classification excluded.")
    k[3].metric("Classifications reused", s["reused"])
    k[4].metric("Token rows: provider / estimated", f"{s['provider_token_rows']} / {s['estimated_token_rows']}")

    st.subheader("Application health & performance")
    k = st.columns(5)
    k[0].metric("Requests", s["total"])
    k[1].metric("Error rate", f"{s['error_rate']:.0%}")
    k[2].metric("Loops blocked", s["blocked"])
    k[3].metric("Avg latency", f"{s['avg_latency_ms']:.0f} ms")
    k[4].metric("p95 latency", f"{s['p95_latency_ms']:.0f} ms")

    ok = df[df["status"] == "success"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Cumulative cost: actual vs baseline (USD)**")
        if not ok.empty:
            cum = ok.set_index("time")[["actual_cost", "baseline_cost"]].cumsum()
            st.line_chart(cum)
        st.markdown("**Latency by model (ms)**")
        if not ok.empty:
            lat = ok.groupby(ok["model"].str.split("/").str[-1])["latency_ms"] \
                    .agg(avg="mean", p95=lambda x: x.quantile(0.95))
            st.bar_chart(lat)
    with right:
        st.markdown("**Requests by route and status**")
        mix = df.pivot_table(index="route", columns="status", values="id",
                             aggfunc="count", fill_value=0)
        st.bar_chart(mix)
        st.markdown("**Tokens by route**")
        if not ok.empty:
            st.bar_chart(ok.groupby("route")[["input_tokens", "output_tokens"]].sum())

    st.subheader("Provider credential failover")
    st.caption(f"Failovers: {s['credential_failovers']} · requests failed at the provider: {s['provider_errors']}. "
               "Only credential aliases are shown (primary, backup_1…), never key values. "
               "Loop-guard blocks are counted separately above.")
    att = metrics.load_attempts()
    if att.empty:
        st.info("No live provider attempts recorded yet (mock answers make no provider calls).")
    else:
        acols = ["time", "request_id", "attempt_no", "credential_alias", "model", "outcome", "http_status",
                 "duration_ms", "failover", "failover_reason", "retry_after_s", "error"]
        st.dataframe(att[[c for c in acols if c in att.columns]], width="stretch", hide_index=True)

    st.subheader("Recent requests")
    cols = ["time", "session_id", "turn_id", "execution_mode", "status", "routing_mode", "route_source",
            "fallback_reason", "llm_route", "route", "model", "classifier_ms", "classification_reused",
            "credential_alias", "provider_attempts", "failover_used", "token_source", "input_tokens", "output_tokens", "latency_ms", "actual_cost", "baseline_cost",
            "reason", "error"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols].iloc[::-1].head(50), width="stretch", hide_index=True)


pg = st.navigation([st.Page(chat_page, title="Chat", icon="💬", default=True),
                    st.Page(dashboard_page, title="Dashboard", icon="📊")])
pg.run()
