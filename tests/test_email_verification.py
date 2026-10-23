"""Signup email verification must actually verify something.

Open registration was enabled 2026-08-14, so anyone can sign up with any
address. These fences cover the two ways this feature fails silently:

1. **The link is a session token.** A verification URL sits in a mailbox, gets
   forwarded, and is prefetched by mail clients. If it doubled as a login
   credential, seeing the URL would be enough to take the account. Fenced in
   BOTH directions -- a verify token must not authenticate, and a session
   token must not be replayable as a verification link.

2. **"We sent an email" when nothing was sent.** Asserting the endpoint
   returned 201 proves nothing. These tests mock at the httpx TRANSPORT
   boundary, so every line of our own send logic runs, and then assert on the
   outbound POST that actually left: the recipient, and a token that decodes
   as a verification token for that user.

Claims are read back, never taken from the response body that asserted them:
"verified" is checked against the store, not the redirect.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core import email as email_service
from app.core import jwt_auth
from app.core import users as users_store
from app.main import app
from tests._fixture_values import OTHER_PASSWORD, PASSWORD, RESEND_TEST_KEY, WRONG_PASSWORD


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"id": "msg_test"}
        self.text = str(self._payload)

    def json(self):
        return self._payload


def _email() -> str:
    return f"verif-{uuid.uuid4().hex[:10]}@example.com"


def _configured(monkeypatch):
    """Make the app believe email delivery is available."""
    monkeypatch.setenv("RESEND_API_KEY", RESEND_TEST_KEY)
    monkeypatch.setenv("EMAIL_FROM", "The Shovel <noreply@theshovel.ai>")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://theshovel.ai")


def _sent_calls(post: AsyncMock):
    """The outbound Resend requests that actually left the process."""
    return [c for c in post.await_args_list
            if "api.resend.com" in str(c.args[0] if c.args else c.kwargs.get("url", ""))]


# ---------------------------------------------------------------- the fence

def test_verification_token_cannot_authenticate():
    """The security property: an emailed link is not a login."""
    token = jwt_auth.create_purpose_token("u-1", jwt_auth.EMAIL_VERIFY_TYP, 3600)
    with pytest.raises(jwt_auth.InvalidTokenError):
        jwt_auth.decode_token(token)


def test_session_token_cannot_be_replayed_as_a_verification_link():
    """The other direction, so the fix cannot be half-applied."""
    token = jwt_auth.create_token("u-1")
    with pytest.raises(jwt_auth.InvalidTokenError):
        jwt_auth.decode_purpose_token(token, jwt_auth.EMAIL_VERIFY_TYP)


def test_legacy_session_tokens_still_authenticate():
    """Session tokens predate the typ claim; the guard must not evict them."""
    assert jwt_auth.decode_token(jwt_auth.create_token("u-1"))["user_id"] == "u-1"


# ------------------------------------------------- registration actually sends

def test_registration_sends_a_real_verification_email(monkeypatch):
    _configured(monkeypatch)
    addr = _email()
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
               return_value=_Resp()) as post, TestClient(app) as c:
        r = c.post("/v1/users/register",
                   json={"email": addr, "password": PASSWORD})

    assert r.status_code == 201, r.text
    # The reply is the neutral one (it never says whether mail went out);
    # the email itself is asserted from what actually left the process.
    assert r.json()["registration_received"] is True
    assert users_store.get_user_by_email(addr)["email_verified"] is False

    calls = _sent_calls(post)
    assert calls, "no request reached the email provider"
    body = calls[-1].kwargs["json"]
    assert body["to"] == [addr], f"email went to the wrong recipient: {body['to']}"

    # The link must carry a token that is genuinely a verification token for
    # THIS user -- not merely any string in a URL.
    import re
    m = re.search(r"token=([A-Za-z0-9._\-]+)", body["html"])
    assert m, f"no token in the email body: {body['html'][:200]}"
    payload = jwt_auth.decode_purpose_token(m.group(1), jwt_auth.EMAIL_VERIFY_TYP)
    stored = users_store.get_user_by_email(addr)
    assert payload["user_id"] == stored["id"]


def test_unverified_user_cannot_log_in_then_can_after_verifying(monkeypatch):
    """The gate, and its release. Verified state is read back from the store."""
    _configured(monkeypatch)
    addr = _email()
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
               return_value=_Resp()) as post, TestClient(app) as c:
        c.post("/v1/users/register",
               json={"email": addr, "password": PASSWORD})
        blocked = c.post("/v1/users/login",
                         json={"email": addr, "password": PASSWORD})
        assert blocked.status_code == 403, blocked.text

        import re
        body = _sent_calls(post)[-1].kwargs["json"]
        token = re.search(r"token=([A-Za-z0-9._\-]+)", body["html"]).group(1)

        c.get(f"/v1/users/verify-email?token={token}", follow_redirects=False)

        # Read the CLAIM back from the store, not from the redirect.
        assert users_store.get_user_by_email(addr)["email_verified"] is True

        ok = c.post("/v1/users/login",
                    json={"email": addr, "password": PASSWORD})
    assert ok.status_code == 200, ok.text
    assert ok.json()["token"]


def test_tampered_and_expired_links_do_not_verify(monkeypatch):
    _configured(monkeypatch)
    addr = _email()
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
               return_value=_Resp()), TestClient(app) as c:
        c.post("/v1/users/register",
               json={"email": addr, "password": PASSWORD})
        uid = users_store.get_user_by_email(addr)["id"]

        expired = jwt_auth.create_purpose_token(
            uid, jwt_auth.EMAIL_VERIFY_TYP, -10)
        c.get(f"/v1/users/verify-email?token={expired}", follow_redirects=False)
        assert users_store.get_user_by_email(addr)["email_verified"] is False

        c.get("/v1/users/verify-email?token=not-a-token", follow_redirects=False)
        assert users_store.get_user_by_email(addr)["email_verified"] is False

        # A session token for the same user must not verify them either.
        session = jwt_auth.create_token(uid)
        c.get(f"/v1/users/verify-email?token={session}", follow_redirects=False)
        assert users_store.get_user_by_email(addr)["email_verified"] is False


# ------------------------------------------------------------- failure modes

def test_registration_refuses_when_email_is_unconfigured_in_production(monkeypatch):
    """Fail CLOSED: never mint accounts that can never be verified."""
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("EMAIL_FROM", raising=False)
    monkeypatch.setattr("app.routers.users._is_dev_env", lambda: False)
    addr = _email()
    with TestClient(app) as c:
        r = c.post("/v1/users/register",
                   json={"email": addr, "password": PASSWORD})
    assert r.status_code == 503, r.text
    # And it must not leave an orphan behind that blocks a later retry.
    assert users_store.get_user_by_email(addr) is None


def test_send_failure_keeps_the_account_recoverable(monkeypatch):
    """A provider blip must not permanently burn the address.

    Email is unique-constrained, so rolling the user back would make a retry
    409 forever. The account is kept and resend-verification can recover it.
    """
    _configured(monkeypatch)
    addr = _email()
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
               return_value=_Resp(status_code=422, payload={"message": "nope"})):
        with TestClient(app) as c:
            r = c.post("/v1/users/register",
                       json={"email": addr, "password": PASSWORD})
    assert r.status_code == 201, r.text
    assert r.json()["registration_received"] is True  # same reply whether mail went out or not
    assert users_store.get_user_by_email(addr) is not None


def test_resend_verification_is_not_an_account_oracle(monkeypatch):
    """Same 202 whether or not the address exists."""
    _configured(monkeypatch)
    addr = _email()
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
               return_value=_Resp()), TestClient(app) as c:
        c.post("/v1/users/register",
               json={"email": addr, "password": PASSWORD})
        known = c.post("/v1/users/resend-verification", json={"email": addr})
        unknown = c.post("/v1/users/resend-verification",
                         json={"email": "nobody-here@example.invalid"})
    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()


def test_send_email_raises_rather_than_reporting_a_phantom_success(monkeypatch):
    """The transport itself must never return normally on failure."""
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    with pytest.raises(email_service.EmailNotConfigured):
        import asyncio
        asyncio.run(email_service.send_email("a@example.com", "s", "<p>h</p>"))


def test_send_email_raises_when_provider_success_body_is_not_json(monkeypatch):
    """A 200 with a non-JSON body is a send failure, not an empty message id."""
    import asyncio

    class _NotJson:
        status_code = 200
        text = "not-json"

        def json(self):
            raise ValueError("No JSON object could be decoded")

    _configured(monkeypatch)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_NotJson()):
        with pytest.raises(email_service.EmailSendFailed, match="non-JSON"):
            asyncio.run(email_service.send_email("a@example.com", "s", "<p>h</p>"))
