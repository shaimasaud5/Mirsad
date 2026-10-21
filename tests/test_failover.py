"""API-key failover with MOCKED provider responses. No network, no real credentials.
The key values below are obviously fake strings used only to prove they never leak."""
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import config
import litellm
from litellm import ModelResponse

from monitoring import metrics
from proxy import main as pm
from proxy import mock, provider
from proxy.credentials import CredentialPool, redact

K_PRIMARY = "gsk_FAKEPRIMARYKEY_unit_test_0001"
K_BACKUP1 = "gsk_FAKEBACKUPKEY1_unit_test_0002"
K_BACKUP2 = "gsk_FAKEBACKUPKEY2_unit_test_0003"
ALL_KEYS = (K_PRIMARY, K_BACKUP1, K_BACKUP2)

client = TestClient(pm.app)
REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def make_pool(*keys):
    env = {}
    if keys:
        env["GROQ_API_KEY"] = keys[0]
    for i, k in enumerate(keys[1:], 1):
        env[f"GROQ_API_KEY_BACKUP_{i}"] = k
    return CredentialPool(env)


def ok_response(model="groq/qwen/qwen3.8-27b", messages=None):
    return ModelResponse(**mock.complete(model, messages or [{"role": "user", "content": "hi"}], None))


def err401(key):      # providers sometimes echo the credential in their message: it must still never leak
    return litellm.AuthenticationError(message=f"Invalid API Key: {key}", llm_provider="groq", model="m")


def err429(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return litellm.RateLimitError(message="Rate limit reached for model", llm_provider="groq", model="m",
                                  response=httpx.Response(429, headers=headers, request=REQ))


class FakeProvider:
    """Replaces litellm.completion. `script` maps a call number to a response or an exception."""
    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, **kw):
        self.calls.append(kw)
        item = self.script[min(len(self.calls), len(self.script)) - 1]
        item = item(kw) if callable(item) else item
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def live_with_fake_provider(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", False)
    sleeps = []
    monkeypatch.setattr(provider, "_sleep", sleeps.append)
    client.post("/admin/reset")
    yield sleeps


def use(monkeypatch, fake, pool):
    monkeypatch.setattr(provider, "_completion", fake)
    monkeypatch.setattr(pm, "pool", pool)


def post(text="What is Flutter?", session="fo", turn="t1"):
    return client.post("/v1/chat/completions", headers={"X-Session-ID": session, "X-Turn-ID": turn},
                       json={"model": "mirsad-auto", "messages": [{"role": "user", "content": text}]})


MODEL = "groq/qwen/qwen3.8-27b"
MSGS = [{"role": "user", "content": "hello"}]


# ------------------------------------------------------------------ unit: provider.call_groq
def test_primary_succeeds_without_failover(monkeypatch):
    fake = FakeProvider([ok_response()])
    monkeypatch.setattr(provider, "_completion", fake)
    out = provider.call_groq(MODEL, MSGS, {}, make_pool(K_PRIMARY, K_BACKUP1))
    assert out.ok and out.alias == "primary" and not out.failover_used
    assert len(fake.calls) == 1 and fake.calls[0]["api_key"] == K_PRIMARY
    assert fake.calls[0]["num_retries"] == 0 and fake.calls[0]["max_retries"] == 0   # retries not multiplied


def test_401_then_backup_succeeds_same_model_and_context(monkeypatch):
    fake = FakeProvider([err401(K_PRIMARY), ok_response()])
    monkeypatch.setattr(provider, "_completion", fake)
    pool = make_pool(K_PRIMARY, K_BACKUP1)
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    msgs = MSGS + [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "data"}]
    out = provider.call_groq(MODEL, msgs, {"tools": tools, "temperature": 0.2}, pool)
    assert out.ok and out.alias == "backup_1" and out.failover_used
    assert [a.outcome for a in out.attempts] == ["credential_invalid", "success"]
    assert out.attempts[0].failover and out.attempts[0].failover_reason
    first, second = fake.calls
    assert (first["api_key"], second["api_key"]) == (K_PRIMARY, K_BACKUP1)
    for k in ("model", "messages", "tools", "temperature"):       # nothing but the credential changed
        assert first[k] == second[k]
    assert [s["state"] for s in pool.status()] == ["disabled", "active"]


def test_disabled_credential_stays_disabled_for_later_requests(monkeypatch):
    fake = FakeProvider([err401(K_PRIMARY), ok_response(), ok_response()])
    monkeypatch.setattr(provider, "_completion", fake)
    pool = make_pool(K_PRIMARY, K_BACKUP1)
    provider.call_groq(MODEL, MSGS, {}, pool)
    out = provider.call_groq(MODEL, MSGS, {}, pool)
    assert fake.calls[2]["api_key"] == K_BACKUP1 and len(out.attempts) == 1


def test_all_credentials_fail_clear_error_never_mock(monkeypatch):
    fake = FakeProvider([err401(K_PRIMARY), err401(K_BACKUP1), err401(K_BACKUP2)])
    monkeypatch.setattr(provider, "_completion", fake)
    out = provider.call_groq(MODEL, MSGS, {}, make_pool(*ALL_KEYS))
    assert not out.ok and out.error_type == "provider_credentials_exhausted" and out.http_status == 502
    assert "Mock mode was NOT used" in out.message and len(fake.calls) == 3
    again = provider.call_groq(MODEL, MSGS, {}, make_pool(*ALL_KEYS))   # fresh pool: one more full round only
    assert not again.ok


def test_no_backup_configured(monkeypatch):
    fake = FakeProvider([err401(K_PRIMARY)])
    monkeypatch.setattr(provider, "_completion", fake)
    out = provider.call_groq(MODEL, MSGS, {}, make_pool(K_PRIMARY))
    assert not out.ok and out.error_type == "provider_credentials_exhausted" and len(fake.calls) == 1
    assert not out.attempts[0].failover        # nothing to fail over to


def test_nothing_configured_makes_no_provider_call(monkeypatch):
    fake = FakeProvider([ok_response()])
    monkeypatch.setattr(provider, "_completion", fake)
    out = provider.call_groq(MODEL, MSGS, {}, CredentialPool({"GROQ_API_KEY": "REPLACE_WITH_YOUR_PRIMARY_GROQ_KEY"}))
    assert out.error_type == "provider_not_configured" and fake.calls == []


def test_shared_429_does_not_rotate_keys(monkeypatch, live_with_fake_provider):
    fake = FakeProvider([err429(retry_after=120)])
    monkeypatch.setattr(provider, "_completion", fake)
    pool = make_pool(K_PRIMARY, K_BACKUP1, K_BACKUP2)
    out = provider.call_groq(MODEL, MSGS, {}, pool)
    assert out.error_type == "provider_rate_limited" and out.http_status == 503 and out.retry_after == 120
    assert [c["api_key"] for c in fake.calls] == [K_PRIMARY]            # no sibling key was tried
    assert all(s["state"] == "active" for s in pool.status())            # a 429 does not disable credentials
    assert live_with_fake_provider == []                                 # 120 s > bound: no waiting


def test_429_with_short_retry_after_waits_once_on_same_key(monkeypatch, live_with_fake_provider):
    fake = FakeProvider([err429(retry_after=2), ok_response()])
    monkeypatch.setattr(provider, "_completion", fake)
    out = provider.call_groq(MODEL, MSGS, {}, make_pool(K_PRIMARY, K_BACKUP1))
    assert out.ok and live_with_fake_provider == [2.0]
    assert [c["api_key"] for c in fake.calls] == [K_PRIMARY, K_PRIMARY]


def test_429_twice_is_bounded_and_still_no_rotation(monkeypatch, live_with_fake_provider):
    fake = FakeProvider([err429(retry_after=1)])
    monkeypatch.setattr(provider, "_completion", fake)
    out = provider.call_groq(MODEL, MSGS, {}, make_pool(K_PRIMARY, K_BACKUP1))
    assert not out.ok and len(fake.calls) == 2 and {c["api_key"] for c in fake.calls} == {K_PRIMARY}
    assert live_with_fake_provider == [1.0]


@pytest.mark.parametrize("exc,etype", [
    (litellm.PermissionDeniedError(message="blocked", llm_provider="groq", model="m",
                                   response=httpx.Response(403, request=REQ)), "provider_permission_denied"),
    (litellm.BadRequestError(message="bad", model="m", llm_provider="groq"), "provider_bad_request"),
    (litellm.NotFoundError(message="model missing", model="m", llm_provider="groq"), "provider_bad_request"),
    (litellm.ServiceUnavailableError(message="down", llm_provider="groq", model="m"), "provider_outage"),
    (litellm.Timeout(message="slow", model="m", llm_provider="groq"), "provider_outage"),
    (litellm.InternalServerError(message="boom", llm_provider="groq", model="m"), "provider_outage"),
])
def test_other_failures_do_not_trigger_failover(monkeypatch, exc, etype):
    fake = FakeProvider([exc])
    monkeypatch.setattr(provider, "_completion", fake)
    pool = make_pool(K_PRIMARY, K_BACKUP1)
    out = provider.call_groq(MODEL, MSGS, {}, pool)
    assert out.error_type == etype and len(fake.calls) == 1
    assert all(s["state"] == "active" for s in pool.status())


def test_attempt_budget_is_bounded(monkeypatch):
    fake = FakeProvider([err401(K_PRIMARY), err401(K_BACKUP1), err401(K_BACKUP2)])
    monkeypatch.setattr(provider, "_completion", fake)
    out = provider.call_groq(MODEL, MSGS, {}, make_pool(*ALL_KEYS), max_attempts=2)
    assert len(fake.calls) == 2 and out.error_type in ("provider_attempt_limit", "provider_credentials_exhausted")


def test_backward_compatible_single_key():
    pool = CredentialPool({"GROQ_API_KEY": K_PRIMARY})
    assert pool.aliases() == ["primary"] and pool.configured == 1


def test_backup_order_and_ignored_blanks():
    pool = CredentialPool({"GROQ_API_KEY": K_PRIMARY, "GROQ_API_KEY_BACKUP_2": K_BACKUP2,
                           "GROQ_API_KEY_BACKUP_1": "", "GROQ_API_KEY_BACKUP_3": K_PRIMARY})
    assert pool.aliases() == ["primary", "backup_2"]        # blank skipped, duplicate skipped, order kept


# ------------------------------------------------------------------ HTTP: end to end through the proxy
def test_http_failover_is_recorded_with_aliases_only(monkeypatch):
    use(monkeypatch, FakeProvider([err401(K_PRIMARY), ok_response()]), make_pool(K_PRIMARY, K_BACKUP1))
    r = post()
    assert r.status_code == 200
    row = metrics.load().iloc[-1]
    assert row["status"] == "success" and row["credential_alias"] == "backup_1"
    assert row["provider_attempts"] == 2 and row["failover_used"] == 1 and row["execution_mode"] == "live"
    att = metrics.load_attempts().sort_values("attempt_no")
    assert list(att["credential_alias"]) == ["primary", "backup_1"]
    assert list(att["outcome"]) == ["credential_invalid", "success"]
    assert list(att["http_status"]) == [401, 200] and list(att["failover"]) == [1, 0]
    assert (att["duration_ms"] >= 0).all()


def test_http_all_keys_fail_returns_clear_error_not_mock(monkeypatch):
    use(monkeypatch, FakeProvider([err401(K_PRIMARY), err401(K_BACKUP1)]), make_pool(K_PRIMARY, K_BACKUP1))
    r = post()
    assert r.status_code == 502 and r.json()["error"]["type"] == "provider_credentials_exhausted"
    assert "mock" not in r.json()["error"]["message"].lower().replace("mock mode was not used", "")
    row = metrics.load().iloc[-1]
    assert row["status"] == "error" and row["execution_mode"] == "live" and row["provider_attempts"] == 2


def test_http_shared_429_returns_503_with_retry_after_and_is_not_a_loop_block(monkeypatch):
    fake = FakeProvider([err429(retry_after=90)])
    use(monkeypatch, fake, make_pool(K_PRIMARY, K_BACKUP1))
    r = post()
    assert r.status_code == 503 and r.json()["error"]["type"] == "provider_rate_limited"
    assert r.headers["Retry-After"] == "91"
    assert len(fake.calls) == 1
    row = metrics.load().iloc[-1]
    assert row["status"] == "error" and row["status"] != "blocked"      # distinct from loop-guard blocking


def test_secrets_never_appear_in_logs_errors_db_or_health(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    use(monkeypatch, FakeProvider([err401(K_PRIMARY), err401(K_BACKUP1), err401(K_BACKUP2)]), make_pool(*ALL_KEYS))
    r = post()
    health = client.get("/health").text
    att = metrics.load_attempts().to_json()
    rows = metrics.load().to_json()
    blob = " ".join([r.text, str(dict(r.headers)), health, att, rows, caplog.text])
    for k in ALL_KEYS:
        assert k not in blob
    assert "FAKEPRIMARYKEY" not in blob and "authorization" not in blob.lower()
    assert "primary" in att and "backup_1" in att                         # aliases are what we do show


def test_redact_helper():
    s = redact(f"Invalid {K_PRIMARY} Authorization: Bearer abcdef123456 api_key=zzzzzzzzzz", [K_PRIMARY])
    assert K_PRIMARY not in s and "abcdef123456" not in s and "zzzzzzzzzz" not in s


def test_mock_mode_makes_no_provider_calls(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    fake = FakeProvider([ok_response()])
    use(monkeypatch, fake, make_pool(K_PRIMARY))
    assert post().status_code == 200 and fake.calls == []
    assert metrics.load().iloc[-1]["credential_alias"] == "mock"
