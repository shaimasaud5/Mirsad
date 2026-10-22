"""Monitoring accuracy: turns vs model calls vs provider attempts, token/usage kinds, estimated cost and the hypothetical
baseline, separate latency measurements, filters, and admin-only access. Isolated temp databases and mocked providers only:
no real user activity, no network, no paid inference."""
import json
import os
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from litellm import ModelResponse

import config
from monitoring import feedback as fb_stats
from monitoring import metrics
from proxy import db
from proxy import main as pm
from proxy import provider
from test_auth import chat, env, make_admin, sign_in  # noqa: F401  (env is a pytest fixture)
from test_failover import FakeProvider, make_pool
from ui import web_server as ws

SMALL, BIG, OTHER = "groq/test/small", "groq/test/big", "groq/test/unpriced"
PRICES = {SMALL: {"input": 1.0, "output": 2.0}, BIG: {"input": 3.0, "output": 6.0}}   # USD per 1M tokens
M = 1_000_000
NOW = time.time()


@pytest.fixture
def prices(monkeypatch):
    monkeypatch.setattr(config, "MODEL_PRICES", dict(PRICES))
    monkeypatch.setattr(config, "BIG_MODEL", BIG)
    return PRICES


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "metrics.db")
    yield c
    c.close()


_n = 0


def call(conn, *, turn, user="u1", status="success", model=SMALL, tin=None, tout=None, src="provider", attempts=1,
         mode="live", model_ms=None, latency=None, cls_ms=None, reused=1, ts=None, est=None, route="simple", failover=0):
    """One model-call record (a row of `requests`)."""
    global _n
    _n += 1
    db.insert(conn, {"request_id": f"req_{_n}", "ts": ts if ts is not None else NOW - 600 + _n * 0.001, "session_id": "s", "user_id": user,
                     "turn_id": turn, "status": status, "route": route, "model": model, "execution_mode": mode,
                     "input_tokens": tin, "output_tokens": tout, "token_source": src, "provider_attempts": attempts,
                     "failover_used": failover, "model_call_ms": model_ms, "latency_ms": latency if latency is not None else model_ms,
                     "classifier_ms": cls_ms, "classification_reused": reused, "est_cost_usd": est,
                     "route_source": "rules", "has_tool_calls": 0})
    return f"req_{_n}"


def turn(conn, tid, outcome, ms, *, user="u1", started=None):
    db.insert_turn(conn, turn_id=tid, user_id=user, conversation_id="c1", session_id="s",
                   started_at=started if started is not None else NOW - 600, duration_ms=ms, outcome=outcome)


def attempt(conn, request_id, n, outcome, ms):
    conn.execute("INSERT INTO provider_attempts (ts, request_id, attempt_no, credential_alias, model, outcome, duration_ms, failover) "
                 "VALUES (?,?,?,?,?,?,?,?)", (NOW - 600, request_id, n, "primary", SMALL, outcome, ms, 0))
    conn.commit()


def seed(conn):
    """T1: one turn, THREE model calls (tool loop), 4 provider attempts (the first call failed over once).
    T2: one call on the big model. T3: failed turn (call failed after 3 attempts). T4: loop-blocked turn.
    T5: older turn known only from `messages`. Plus: unpriced model, missing usage, mock call, two ungroupable legacy rows."""
    r1 = call(conn, turn="T1", tin=M, tout=M // 2, attempts=2, failover=1, model_ms=100, latency=170, cls_ms=50, reused=0, est=2.0)   # recorded
    r2 = call(conn, turn="T1", tin=2 * M, tout=M, model_ms=200, latency=200)
    call(conn, turn="T1", tin=M, tout=0, model_ms=300, latency=300)                       # zero output tokens is a real value
    call(conn, turn="T2", model=BIG, route="complex", tin=M, tout=M, model_ms=400, latency=470, cls_ms=70, reused=0)
    call(conn, turn="T3", status="error", src=None, attempts=3, model_ms=900)             # failed: usage unknown
    call(conn, turn="T4", status="blocked", src=None, attempts=None)                      # loop guard: never reached a model
    call(conn, turn="T5", tin=M, tout=M, model_ms=None, latency=500, cls_ms=120, reused=0)   # older row: duration is derived
    call(conn, turn="T6", model=OTHER, tin=M, tout=M, model_ms=50, latency=50)            # provider usage, but no configured price
    call(conn, turn="T7", src="unknown", tin=None, tout=None, model_ms=60, latency=60)    # success without usage
    call(conn, turn="M1", mode="mock", src="mock-estimate", tin=500, tout=50, attempts=0, model_ms=5, latency=5)
    call(conn, turn=None, user=None, mode=None, tin=M, tout=M, model_ms=None, latency=300)   # legacy, cannot be grouped
    call(conn, turn=None, user=None, mode=None, status="error", src=None, attempts=None)
    attempt(conn, r1, 1, "credential_invalid", 30)
    attempt(conn, r1, 2, "success", 70)
    attempt(conn, r2, 1, "success", 190)
    turn(conn, "T1", "completed", 1000); turn(conn, "T2", "completed", 3000)
    turn(conn, "T3", "failed", 500); turn(conn, "T4", "blocked", 200)
    conn.execute("INSERT INTO messages (conversation_id, user_id, role, content, status, turn_id, ts) VALUES ('c1','u1','assistant','a','ok','T5',?)", (NOW - 600,))
    conn.commit()


# ============================================================ turns vs model calls vs provider attempts
def test_one_turn_with_several_model_calls_and_attempts(conn, prices):
    seed(conn)
    df = metrics.filter_execution(metrics.load(conn_path(conn)), "live")
    turns, ungroup = metrics.load_turns(conn_path(conn), kind="live")
    t1 = turns[turns["turn_id"] == "T1"].iloc[0]
    assert (t1["model_calls"], t1["provider_attempts"], t1["outcome"], t1["duration_ms"]) == (3, 4, "completed", 1000)
    assert (t1["input_tokens"], t1["output_tokens"]) == (4 * M, int(1.5 * M)) and t1["usage_complete"]
    ts = metrics.summarize_turns(turns, ungroup)
    # live turns: T1 T2 T5 completed (T5 from its message), T3 failed, T4 blocked, T6/T7 outcome unknown (mock M1 and the two ungroupable legacy rows are not live turns)
    assert ts["turns"] == 7 and ts["completed"] == 3 and ts["failed"] == 1 and ts["blocked"] == 1 and ts["unknown"] == 2
    assert ungroup == 0                                                    # legacy rows are outside the live view
    s = metrics.summarize(df, metrics.load_attempts(conn_path(conn), limit=None, request_ids=df["request_id"]))
    c = s["calls"]
    assert c["model_calls"] == 8 and c["succeeded"] == 7 and c["failed"] == 1 and c["blocked"] == 1   # blocked is not a model call
    assert c["provider_attempts"] == 2 + 1 + 1 + 3 + 1 * 4 and c["credential_failovers"] == 1   # 3 T1 calls: 2+1+1, T2 1, T3 3, T5/T6/T7 1 each
    assert s["total"] == 9 and c["call_error_rate"] == pytest.approx(1 / 8)


def test_tool_loop_calls_are_not_separate_turns_and_ids_not_text_define_turns(conn, prices):
    for i in range(5):                                       # same prompt text elsewhere, five calls, one turn id
        call(conn, turn="TX", tin=10, tout=5, model_ms=10)
    call(conn, turn="TY", tin=10, tout=5, model_ms=10)
    call(conn, turn="TX", user="u2", tin=10, tout=5, model_ms=10)          # same turn id, different verified user = different turn
    turns, _ = metrics.load_turns(conn_path(conn), kind="live")
    assert len(turns) == 3
    assert sorted(turns["model_calls"]) == [1, 1, 5]


def test_legacy_rows_without_ids_are_excluded_from_turns_and_counted(conn, prices):
    seed(conn)
    turns, ungroup = metrics.load_turns(conn_path(conn), kind="legacy")
    assert turns.empty and ungroup == 2
    assert metrics.summarize_turns(turns, ungroup)["turns"] == 0
    all_turns, ungroup_all = metrics.load_turns(conn_path(conn), kind="all")
    assert ungroup_all == 2 and len(all_turns) == 8                       # 7 live + the mock turn; none invented from legacy rows


# ============================================================ tokens and cost
def test_tokens_and_costs_use_each_models_input_and_output_rates(conn, prices):
    seed(conn)
    s = metrics.summarize(metrics.filter_execution(metrics.load(conn_path(conn)), "live"))
    tk, cs = s["tokens"], s["cost"]
    # provider-reported calls: T1 x3, T2, T5, T6 (OTHER has usage but no price)
    assert tk["provider_calls"] == 6
    assert tk["provider_input"] == (1 + 2 + 1 + 1 + 1 + 1) * M and tk["provider_output"] == int((0.5 + 1 + 0 + 1 + 1 + 1) * M)
    # priced: T1c1 2.0 (recorded) + T1c2 4.0 + T1c3 1.0 + T2 9.0 + T5 3.0 = 19.0 ; OTHER excluded, not zero
    assert cs["priced_calls"] == 5 and cs["unpriced_calls"] == 1
    assert cs["estimated_cost"] == pytest.approx(19.0)
    assert cs["current_rate_calls"] == 4                                   # only the first call carried a stored estimate
    by = {m["model"]: m for m in cs["by_model"]}
    assert by[SMALL]["est_cost"] == pytest.approx(10.0) and by[BIG]["est_cost"] == pytest.approx(9.0)
    assert by[OTHER]["est_cost"] is None and by[OTHER]["input_price"] is None


def test_missing_usage_is_unknown_not_zero_and_failed_calls_are_not_invented(conn, prices):
    seed(conn)
    s = metrics.summarize(metrics.filter_execution(metrics.load(conn_path(conn)), "live"))
    assert s["tokens"]["unknown_calls"] == 1                               # T7: success without usage
    assert s["tokens"]["failed_calls_unknown"] == 1                        # T3: failed after 3 attempts
    d = metrics.annotate(metrics.filter_execution(metrics.load(conn_path(conn)), "live"))
    t7 = d[d["turn_id"] == "T7"].iloc[0]
    assert t7["usage_kind"] == "unknown" and np.isnan(t7["in_tok"]) and np.isnan(t7["est_cost"])
    assert s["tokens"]["classifier_tokens"] is None                        # local classifier usage is not recorded


def test_mock_usage_is_separate_and_never_priced(conn, prices):
    seed(conn)
    s = metrics.summarize(metrics.filter_execution(metrics.load(conn_path(conn)), "mock"))
    assert s["tokens"]["mock_calls"] == 1 and s["tokens"]["mock_input"] == 500 and s["tokens"]["provider_calls"] == 0
    assert s["cost"]["estimated_cost"] is None and s["cost"]["baseline"]["pct"] is None


def test_matched_baseline_uses_same_tokens_at_big_model_rates(conn, prices):
    seed(conn)
    b = metrics.summarize(metrics.filter_execution(metrics.load(conn_path(conn)), "live"))["cost"]["baseline"]
    # matched = priced calls with a priced baseline: T1 (6+12+3), T2 (9), T5 (M,M -> small 3.0 vs big 9.0)
    assert b["matched_calls"] == 5 and b["model"] == BIG
    assert b["hypothetical"] == pytest.approx(6 + 12 + 3 + 9 + 9)
    assert b["estimated"] == pytest.approx(19.0)
    assert b["difference"] == pytest.approx(20.0) and b["pct"] == pytest.approx(20 / 39 * 100)


def _one_call_summary(conn, small_in, small_out, big_prices):
    call(conn, turn="N1", tin=small_in, tout=small_out, model_ms=10)
    return metrics.summarize(metrics.filter_execution(metrics.load(conn_path(conn)), "live"))["cost"]["baseline"]


def test_negative_savings_are_reported_as_negative(conn, monkeypatch):
    monkeypatch.setattr(config, "BIG_MODEL", BIG)
    monkeypatch.setattr(config, "MODEL_PRICES", {SMALL: {"input": 4.0, "output": 8.0}, BIG: {"input": 1.0, "output": 1.0}})
    b = _one_call_summary(conn, M, M, None)
    assert b["hypothetical"] == pytest.approx(2.0) and b["estimated"] == pytest.approx(12.0)
    assert b["difference"] == pytest.approx(-10.0) and b["pct"] == pytest.approx(-500.0)


def test_zero_or_unavailable_baseline_is_na_not_a_percentage(conn, monkeypatch):
    monkeypatch.setattr(config, "BIG_MODEL", BIG)
    monkeypatch.setattr(config, "MODEL_PRICES", {SMALL: {"input": 1.0, "output": 2.0}, BIG: {"input": 0.0, "output": 0.0}})
    b = _one_call_summary(conn, M, M, None)
    assert b["hypothetical"] == 0 and b["pct"] is None                     # zero baseline
    monkeypatch.setattr(config, "MODEL_PRICES", {SMALL: {"input": 1.0, "output": 2.0}})
    b = metrics.summarize(metrics.filter_execution(metrics.load(conn_path(conn)), "live"))["cost"]["baseline"]
    assert b["matched_calls"] == 0 and b["pct"] is None and b["difference"] is None   # BIG unpriced


def test_recorded_estimates_are_preserved_when_prices_change_later(conn, monkeypatch):
    monkeypatch.setattr(config, "BIG_MODEL", BIG)
    monkeypatch.setattr(config, "MODEL_PRICES", {SMALL: {"input": 9.0, "output": 9.0}, BIG: {"input": 9.0, "output": 9.0}})
    call(conn, turn="P1", tin=M, tout=M, model_ms=10, est=2.0)             # stored under an older price table
    s = metrics.summarize(metrics.filter_execution(metrics.load(conn_path(conn)), "live"))["cost"]
    assert s["estimated_cost"] == pytest.approx(2.0) and s["current_rate_calls"] == 0


# ============================================================ proxy writes (mocked provider)
@pytest.fixture
def live_proxy(monkeypatch, prices):
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(config, "SMALL_MODEL", SMALL)
    monkeypatch.setattr(config, "BIG_MODEL", BIG)
    monkeypatch.setattr(provider, "_sleep", lambda s: None)
    client = TestClient(pm.app, headers={"Authorization": "Bearer " + os.environ["MIRSAD_INTERNAL_TOKEN"], "X-Mirsad-User-Id": "m-user"})
    client.post("/admin/reset")

    def go(fake, text="hi", turn="tm"):
        monkeypatch.setattr(provider, "_completion", fake)
        monkeypatch.setattr(pm, "pool", make_pool("gsk_FAKE_metrics_key_0001"))
        return client.post("/v1/chat/completions", headers={"X-Session-ID": "sm", "X-Turn-ID": turn},
                           json={"model": "mirsad-auto", "messages": [{"role": "user", "content": text}]})
    go.client = client
    return go


def groq_response(usage):
    d = {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
         "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
    if usage is not None:
        d["usage"] = usage
    return ModelResponse(**d)


def test_proxy_stores_groq_usage_exactly_and_prices_it(live_proxy):
    r = live_proxy(FakeProvider([groq_response({"prompt_tokens": 1234, "completion_tokens": 321, "total_tokens": 1555})]))
    assert r.status_code == 200
    row = metrics.load().iloc[-1]
    assert (row["input_tokens"], row["output_tokens"], row["token_source"]) == (1234, 321, "provider")
    p = PRICES[row["model"]]
    assert row["est_cost_usd"] == pytest.approx(1234 / M * p["input"] + 321 / M * p["output"])
    assert row["baseline_big_usd"] == pytest.approx(1234 / M * 3.0 + 321 / M * 6.0)
    assert row["pricing_version"] == config.PRICING_VERSION and row["model_call_ms"] > 0


def test_proxy_does_not_turn_missing_usage_into_zero_or_an_estimate(live_proxy):
    live_proxy(FakeProvider([groq_response(None)]))
    row = metrics.load().iloc[-1]
    assert row["status"] == "success" and row["token_source"] == "unknown"
    assert pd.isna(row["input_tokens"]) and pd.isna(row["output_tokens"]) and pd.isna(row["est_cost_usd"])
    # a zero prompt count cannot come from a real call: it means "usage not reported", never a free call
    live_proxy(FakeProvider([groq_response({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})]), turn="tm2")
    row = metrics.load().iloc[-1]
    assert row["token_source"] == "unknown" and pd.isna(row["input_tokens"]) and pd.isna(row["est_cost_usd"])


def test_unpriced_model_keeps_tokens_but_has_no_cost(live_proxy, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PRICES", {})
    live_proxy(FakeProvider([groq_response({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})]))
    row = metrics.load().iloc[-1]
    assert row["input_tokens"] == 10 and pd.isna(row["est_cost_usd"]) and pd.isna(row["baseline_big_usd"])


def test_model_call_duration_excludes_local_classification(live_proxy, monkeypatch):
    real = pm.router.decide

    def slow(*a, **k):
        time.sleep(0.12)                                                   # stands in for the local Qwen classification
        return real(*a, **k)
    monkeypatch.setattr(pm.router, "decide", slow)
    live_proxy(FakeProvider([groq_response({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})]))
    row = metrics.load().iloc[-1]
    assert row["latency_ms"] - row["model_call_ms"] >= 100 and row["model_call_ms"] < 100


def test_failed_calls_record_attempts_without_inventing_usage(live_proxy):
    import litellm
    live_proxy(FakeProvider([litellm.Timeout(message="timed out", model="m", llm_provider="groq")]))
    row = metrics.load().iloc[-1]
    assert row["status"] == "error" and row["provider_attempts"] == 1 and pd.isna(row["input_tokens"]) and pd.isna(row["est_cost_usd"])
    assert row["model_call_ms"] >= 0


# ============================================================ latency
def test_percentile_method_is_linear_interpolation_with_sample_counts():
    s = metrics.stat([10, 20, 30, 40])
    assert s["n"] == 4 and s["avg"] == 25 and s["p50"] == 25 and s["p95"] == pytest.approx(38.5)
    assert s["low_sample"] is True and metrics.stat(range(100))["low_sample"] is False
    e = metrics.stat([])
    assert e["n"] == 0 and e["avg"] is None and e["p95"] is None          # no samples: None, not 0 / NaN
    assert metrics.stat([None, float("nan"), 5])["n"] == 1


def test_latency_measurements_are_separate_and_not_double_counted(conn, prices):
    seed(conn)
    df = metrics.filter_execution(metrics.load(conn_path(conn)), "live")
    lat = metrics.summarize(df, metrics.load_attempts(conn_path(conn), limit=None, request_ids=df["request_id"]))["latency"]
    # classification: only calls that really ran the classifier (reused=0): 50, 70, 120
    assert lat["classification"]["n"] == 3 and lat["classification"]["avg"] == pytest.approx(80.0)
    # model-call duration: measured values 100 200 300 400 50 60 + one derived (500 proxy - 120 classification = 380)
    assert lat["model_call"]["n"] == 7 and lat["model_call_derived"] == 1
    assert lat["model_call"]["avg"] == pytest.approx((100 + 200 + 300 + 400 + 50 + 60 + 380) / 7)
    # attempts: 30, 70, 190 ; failover overhead = the earlier failed attempt of the one request that needed 2
    assert lat["attempt"]["n"] == 3 and lat["attempt"]["avg"] == pytest.approx(290 / 3)
    assert lat["failover_overhead"]["n"] == 1 and lat["failover_overhead"]["avg"] == 30
    turns, _ = metrics.load_turns(conn_path(conn), kind="live")
    d = metrics.summarize_turns(turns)["duration"]
    assert d["n"] == 4 and d["avg"] == pytest.approx(1175) and d["p50"] == pytest.approx(750) and d["p95"] == pytest.approx(2700)
    assert metrics.summarize_turns(turns)["no_duration_turns"] == 3        # T5/T6/T7 have no recorded turn duration: not invented


# ============================================================ filters, empty data, completion vs feedback
def test_filters_apply_consistently_to_calls_turns_and_feedback(conn, prices):
    seed(conn)
    path = conn_path(conn)
    for kind, calls_n, turns_n in (("live", 9, 7), ("mock", 1, 1), ("legacy", 2, 0), ("all", 12, 8)):
        assert len(metrics.filter_execution(metrics.load(path), kind)) == calls_n, kind
        assert len(metrics.load_turns(path, kind=kind)[0]) == turns_n, kind
    # time window: everything was recorded ~10 min ago
    assert len(metrics.load(path, since_minutes=0.01)) == 0 and len(metrics.load(path, since_minutes=60)) == 12
    assert metrics.load_turns(path, since_minutes=0.01, kind="all")[0].empty
    assert fb_stats.report(conn, minutes=0.01, kind="all")["summary"]["eligible"] == 0
    # attempts follow the selected calls
    sel = metrics.filter_execution(metrics.load(path), "mock")
    assert metrics.load_attempts(path, limit=None, request_ids=sel["request_id"]).empty


def test_empty_dataset_has_no_nan_and_no_misleading_zeros(conn, prices):
    path = conn_path(conn)
    assert metrics.summarize(metrics.load(path)) == {"total": 0}
    turns, ungroup = metrics.load_turns(path, kind="live")
    ts = metrics.summarize_turns(turns, ungroup)
    assert ts["turns"] == 0 and ts["completion_rate"] is None and ts["duration"]["avg"] is None and ts["calls_per_turn"]["avg"] is None
    json.dumps(ts, allow_nan=False)


def test_technical_completion_is_separate_from_user_feedback(conn, prices):
    seed(conn)
    for mid, (tid, status) in enumerate((("T1", "ok"), ("T2", "ok"), ("T3", "error")), 1):
        conn.execute("INSERT INTO messages (id, conversation_id, user_id, role, content, status, turn_id, ts) "
                     "VALUES (?, 'c1', 'u1', 'assistant', 'a', ?, ?, ?)", (mid + 100, status, tid, NOW - 600))
    conn.execute("INSERT INTO message_feedback (user_id, conversation_id, message_id, turn_id, rating, created_at, updated_at) "
                 "VALUES ('u1','c1',101,'T1','not_helpful',?,?)", (NOW - 600, NOW - 600))
    conn.commit()
    ts = metrics.summarize_turns(metrics.load_turns(conn_path(conn), kind="live")[0])
    assert ts["completed"] == 3                                            # T1 stays technically completed despite Not helpful
    f = fb_stats.report(conn, kind="live")["summary"]
    assert (f["eligible"], f["rated"], f["helpful"], f["not_helpful"]) == (3, 1, 0, 1)   # unrated T2 and T5 are not Helpful
    assert f["helpful_pct"] == 0 and f["coverage_pct"] == pytest.approx(100 / 3)
    conn.execute("DELETE FROM message_feedback"); conn.commit()
    f = fb_stats.report(conn, kind="live")["summary"]
    assert f["rated"] == 0 and f["helpful_pct"] is None                    # no ratings: not 0%, not 100%


def conn_path(c):
    return c.execute("PRAGMA database_list").fetchone()[2]


# ============================================================ admin API
def test_metrics_endpoint_is_admin_only_defaults_to_live_and_records_turns(env):
    user, csrf_u, uid = sign_in(env, "user@example.com")
    adm, _, uid_adm = sign_in(env, "boss@example.com")
    make_admin(uid_adm)
    anon = TestClient(ws.app)
    assert anon.get("/api/metrics").status_code == 401
    assert chat(user, csrf_u, "What is 2+2?").status_code == 200           # runs through the (mock) proxy
    for path in ("/api/metrics", "/api/metrics?kind=all", "/api/admin/feedback"):
        assert user.get(path).status_code == 403
        assert anon.get(path).status_code == 401
    live = adm.get("/api/metrics").json()                                  # no filter given: Live
    assert live["filter"]["kind"] == "live" and live["summary"]["total"] == 0
    mock = adm.get("/api/metrics?kind=mock").json()
    assert mock["summary"]["total"] == 1 and mock["summary"]["calls"]["model_calls"] == 1
    assert mock["turns"]["turns"] == 1 and mock["turns"]["completed"] == 1 and mock["turns"]["duration"]["n"] == 1
    assert mock["summary"]["tokens"]["mock_calls"] == 1 and mock["summary"]["cost"]["estimated_cost"] is None
    assert mock["definitions"]["pricing"]["source_url"].startswith("https://")
    assert adm.get("/api/metrics?kind=bogus").status_code == 422
    json.dumps(live, allow_nan=False); json.dumps(mock, allow_nan=False)
    row = ws.conn.execute("SELECT outcome, duration_ms, user_id FROM turns").fetchone()
    assert row[0] == "completed" and row[1] > 0 and row[2] == uid          # monotonic duration, verified user
