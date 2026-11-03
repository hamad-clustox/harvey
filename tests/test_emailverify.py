"""EmailVerify.io: Harvey's default verifier, tried before the others."""

import os
import tempfile

import httpx
import pytest
import pytest_asyncio

from harvey.integrations import email_finder as ef
from harvey.integrations.email_finder import _status_from_verdict, find_email


@pytest_asyncio.fixture
async def state():
    from harvey.state import StateManager
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


class Env:
    def __init__(self, **kw):
        self.emailverify_api_key = kw.get("emailverify", "")
        self.reoon_api_key = kw.get("reoon", "")
        self.zerobounce_api_key = kw.get("zerobounce", "")


def _fake_api(monkeypatch, status_code=200, body=None, seen=None):
    """Route EmailVerify requests to a canned answer."""
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status_code, json=body if body is not None else {})

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ef.httpx, "AsyncClient", client)


# ── response mapping ──

@pytest.mark.asyncio
@pytest.mark.parametrize("status,sub,expected", [
    ("valid", "permitted", "verified"),
    ("invalid", "mailbox_not_found", "invalid"),
    ("catch_all", "none", "risky"),
    ("do_not_mail", "disposable", "invalid"),
    ("do_not_mail", "mailbox_quota_exceeded", "invalid"),
    ("role_based", "permitted", "verified"),
    ("role_based", "none", None),
    ("unknown", "none", None),
    ("skipped", "none", None),
])
async def test_status_mapping(monkeypatch, status, sub, expected):
    _fake_api(monkeypatch, body={"email": "a@acme.test", "status": status, "sub_status": sub})
    verdict = await ef.verify_emailverify("a@acme.test", "k")
    assert _status_from_verdict(verdict) == expected


@pytest.mark.asyncio
async def test_request_shape(monkeypatch):
    seen = []
    _fake_api(monkeypatch, body={"status": "valid", "sub_status": "permitted"}, seen=seen)
    await ef.verify_emailverify("jane@acme.test", "secret-key")
    req = seen[0]
    assert req.method == "GET"
    assert str(req.url).startswith(ef.EMAILVERIFY_URL)
    assert req.url.params["key"] == "secret-key"
    assert req.url.params["email"] == "jane@acme.test"


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401, 402, 429, 500])
async def test_api_errors_are_unavailable(monkeypatch, code):
    _fake_api(monkeypatch, status_code=code, body={"error": "nope"})
    assert await ef.verify_emailverify("a@acme.test", "k") is None


@pytest.mark.asyncio
async def test_network_error_is_unavailable(monkeypatch):
    real_client = httpx.AsyncClient

    def boom(request):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(ef.httpx, "AsyncClient",
                        lambda *a, **k: real_client(*a, **{**k, "transport": httpx.MockTransport(boom)}))
    assert await ef.verify_emailverify("a@acme.test", "k") is None


# ── order: EmailVerify first, others as fallback ──

def _google(monkeypatch):
    async def google_mx(domain):
        return "aspmx.l.google.com"
    monkeypatch.setattr(ef, "get_mx_host", google_mx)
    monkeypatch.setattr(ef, "_hunter_api_key", lambda: "")


@pytest.mark.asyncio
async def test_emailverify_decides_before_reoon(state, monkeypatch):
    _google(monkeypatch)
    calls = []

    async def fake_ev(email, key):
        calls.append("ev")
        return {"status": "valid", "catch_all": False}

    async def fake_reoon(email, key):
        calls.append("reoon")
        return {"status": "invalid", "catch_all": False}

    monkeypatch.setattr(ef, "verify_emailverify", fake_ev)
    monkeypatch.setattr(ef, "verify_reoon", fake_reoon)
    result = await find_email("Jane", "Doe", "acme.com",
                              env=Env(emailverify="k", reoon="k"), state=state)
    assert result.status == "verified"
    assert calls == ["ev"]


@pytest.mark.asyncio
async def test_unknown_falls_through_to_reoon(state, monkeypatch):
    _google(monkeypatch)

    async def fake_ev(email, key):
        return {"status": "unknown", "catch_all": False}

    async def fake_reoon(email, key):
        return {"status": "safe", "catch_all": False}

    monkeypatch.setattr(ef, "verify_emailverify", fake_ev)
    monkeypatch.setattr(ef, "verify_reoon", fake_reoon)
    result = await find_email("Jane", "Doe", "acme.com",
                              env=Env(emailverify="k", reoon="k"), state=state)
    assert result.status == "verified"


@pytest.mark.asyncio
async def test_catch_all_alone_stays_risky(state, monkeypatch):
    _google(monkeypatch)

    async def fake_ev(email, key):
        return {"status": "catch_all", "catch_all": True}

    monkeypatch.setattr(ef, "verify_emailverify", fake_ev)
    result = await find_email("Jane", "Doe", "acme.com", env=Env(emailverify="k"), state=state)
    assert result.status == "risky"
    assert (await state.get_email_pattern("acme.com"))["is_catch_all"] == 1


@pytest.mark.asyncio
async def test_catch_all_settled_by_zerobounce(state, monkeypatch):
    _google(monkeypatch)

    async def fake_ev(email, key):
        return {"status": "catch_all", "catch_all": True}

    async def fake_zb(email, key):
        return {"status": "valid", "catch_all": False}

    monkeypatch.setattr(ef, "verify_emailverify", fake_ev)
    monkeypatch.setattr(ef, "verify_zerobounce", fake_zb)
    result = await find_email("Jane", "Doe", "acme.com",
                              env=Env(emailverify="k", zerobounce="k"), state=state)
    assert result.status == "verified"


@pytest.mark.asyncio
async def test_emailverify_down_uses_fallbacks(state, monkeypatch):
    _google(monkeypatch)

    async def fake_ev(email, key):
        return None

    async def fake_reoon(email, key):
        return {"status": "invalid", "catch_all": False}

    monkeypatch.setattr(ef, "verify_emailverify", fake_ev)
    monkeypatch.setattr(ef, "verify_reoon", fake_reoon)
    result = await find_email("Jane", "Doe", "acme.com",
                              env=Env(emailverify="k", reoon="k"), state=state)
    assert result.status == "invalid"


@pytest.mark.asyncio
async def test_no_emailverify_key_never_calls_it(state, monkeypatch):
    _google(monkeypatch)

    async def fake_ev(email, key):
        raise AssertionError("EmailVerify called without a key")

    async def fake_reoon(email, key):
        return {"status": "safe", "catch_all": False}

    monkeypatch.setattr(ef, "verify_emailverify", fake_ev)
    monkeypatch.setattr(ef, "verify_reoon", fake_reoon)
    result = await find_email("Jane", "Doe", "acme.com", env=Env(reoon="k"), state=state)
    assert result.status == "verified"
