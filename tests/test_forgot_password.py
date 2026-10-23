"""A user who forgot their password can set a new one by email (0735Z item 3f).

The link is a single-purpose token (never a session) plus a fingerprint of
the current password, so it stops working the moment the password changes:
single use without a token table. The request endpoint answers the same
whether or not the address has an account. No test value is a literal.
"""
from __future__ import annotations

import re
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core import email as email_service
from app.core import jwt_auth
from app.core import users as users_store
from app.main import app
from tests._fixture_values import OTHER_PASSWORD, PASSWORD, WEAK_PASSWORD


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.test")
    monkeypatch.delenv("FRONTEND_URL", raising=False)
    with TestClient(app) as c:
        yield c


def _user():
    email = f"reset-{uuid.uuid4().hex[:10]}@example.test"
    return users_store.create_user(email, PASSWORD, email_verified=True), email


def _request(client, email):
    sent = AsyncMock()
    with patch.object(email_service, "send_email", sent):
        r = client.post("/v1/users/forgot-password", json={"email": email})
    return r, sent


def _token_from(sent: AsyncMock) -> str:
    html = sent.await_args.kwargs["html"]
    return re.search(r"\?reset=([A-Za-z0-9_.\-]+)", html).group(1)


def test_an_unknown_and_a_known_address_get_the_same_reply(client):
    _user_row, email = _user()
    known, sent_known = _request(client, email)
    unknown, sent_unknown = _request(client, f"nobody-{uuid.uuid4().hex[:8]}@example.test")
    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()
    assert sent_known.await_count == 1 and sent_unknown.await_count == 0


def test_the_link_sets_a_new_password_once(client):
    user, email = _user()
    _r, sent = _request(client, email)
    token = _token_from(sent)
    ok = client.post("/v1/users/reset-password", json={"token": token, "new_password": OTHER_PASSWORD})
    assert ok.status_code == 200, ok.text
    assert client.post("/v1/users/login", json={"email": email, "password": OTHER_PASSWORD}).status_code == 200
    assert client.post("/v1/users/login", json={"email": email, "password": PASSWORD}).status_code == 401
    again = client.post("/v1/users/reset-password", json={"token": token, "new_password": PASSWORD})
    assert again.status_code == 400, "a used link worked twice"


@pytest.mark.parametrize("mangle", ["fingerprint", "session", "no_fingerprint", "garbage"])
def test_a_bad_link_is_refused_without_changing_the_password(client, mangle):
    user, email = _user()
    _r, sent = _request(client, email)
    token = _token_from(sent)
    if mangle == "fingerprint":
        token = token.rpartition(".")[0] + "." + "0" * 16
    elif mangle == "session":
        token = jwt_auth.create_token(user["id"]) + "." + token.rpartition(".")[2]
    elif mangle == "no_fingerprint":
        token = token.rpartition(".")[0]
    else:
        token = uuid.uuid4().hex
    r = client.post("/v1/users/reset-password", json={"token": token, "new_password": OTHER_PASSWORD})
    assert r.status_code == 400
    assert client.post("/v1/users/login", json={"email": email, "password": PASSWORD}).status_code == 200


def test_a_reset_link_cannot_authenticate():
    token = jwt_auth.create_purpose_token("u1", jwt_auth.PASSWORD_RESET_TYP, 60)
    with pytest.raises(jwt_auth.InvalidTokenError):
        jwt_auth.decode_token(token)


def test_a_weak_new_password_is_refused(client):
    user, email = _user()
    _r, sent = _request(client, email)
    r = client.post("/v1/users/reset-password", json={"token": _token_from(sent), "new_password": WEAK_PASSWORD})
    assert r.status_code == 422
