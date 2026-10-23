"""A user-facing auth error never names an API route.

An error a person reads says what to do in the product ("Resend confirmation
email" on the sign-in page), never which endpoint to call. Checked through
real requests.
"""
import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import OTHER_PASSWORD, PASSWORD, RESEND_TEST_KEY, WRONG_PASSWORD


class _Resp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"id": "msg_test"}


def _configured(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", RESEND_TEST_KEY)
    monkeypatch.setenv("EMAIL_FROM", "The Shovel <noreply@theshovel.ai>")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://theshovel.ai")


def _detail(resp) -> str:
    body = resp.json()
    detail = body.get("detail", body) if isinstance(body, dict) else body
    return detail if isinstance(detail, str) else str(detail)


def test_an_unverified_login_points_at_the_sign_in_page_not_a_route(monkeypatch):
    _configured(monkeypatch)
    addr = f"routes-{uuid.uuid4().hex[:10]}@example.com"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_Resp()), TestClient(app) as c:
        c.post("/v1/users/register", json={"email": addr, "password": PASSWORD})
        blocked = c.post("/v1/users/login", json={"email": addr, "password": PASSWORD})
    assert blocked.status_code == 403, blocked.text
    text = _detail(blocked)
    assert "/v1/" not in text
    assert "Resend confirmation email" in text


def test_registering_an_existing_address_answers_exactly_like_a_new_one(monkeypatch):
    """LANE4 #978: a registration must not reveal that an account exists."""
    _configured(monkeypatch)
    existing = f"routes-{uuid.uuid4().hex[:10]}@example.com"
    fresh = f"routes-{uuid.uuid4().hex[:10]}@example.com"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_Resp()), TestClient(app) as c:
        c.post("/v1/users/register", json={"email": existing, "password": PASSWORD})
        again = c.post("/v1/users/register", json={"email": existing, "password": OTHER_PASSWORD})
        new = c.post("/v1/users/register", json={"email": fresh, "password": PASSWORD})
    assert again.status_code == new.status_code == 201
    assert {**again.json(), "email": "x"} == {**new.json(), "email": "x"}
    assert "already" not in again.text.lower() and existing.split("@")[0] in again.text  # echoes only what was typed


def test_other_auth_failures_name_no_route_either(monkeypatch):
    _configured(monkeypatch)
    addr = f"routes-{uuid.uuid4().hex[:10]}@example.com"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_Resp()), TestClient(app) as c:
        c.post("/v1/users/register", json={"email": addr, "password": PASSWORD})
        responses = [
            c.post("/v1/users/login", json={"email": addr, "password": WRONG_PASSWORD}),
            c.post("/v1/users/login", json={"email": f"nobody-{uuid.uuid4().hex[:6]}@example.com",
                                             "password": PASSWORD}),
        ]
    for resp in responses:
        assert resp.status_code >= 400, resp.text
        assert "/v1/" not in _detail(resp), resp.text


def _unconfigured(monkeypatch):
    for name in ("RESEND_API_KEY", "EMAIL_FROM"):
        monkeypatch.delenv(name, raising=False)


def test_the_reply_does_not_depend_on_mail_configuration(monkeypatch):
    """LANE1 #978: one body for new and existing addresses, mail on or off."""
    bodies = []
    for setup in (_configured, _unconfigured):
        setup(monkeypatch)
        existing = f"routes-{uuid.uuid4().hex[:10]}@example.com"
        fresh = f"routes-{uuid.uuid4().hex[:10]}@example.com"
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_Resp()), TestClient(app) as c:
            c.post("/v1/users/register", json={"email": existing, "password": PASSWORD})
            again = c.post("/v1/users/register", json={"email": existing, "password": OTHER_PASSWORD})
            new = c.post("/v1/users/register", json={"email": fresh, "password": PASSWORD})
        assert again.status_code == new.status_code == 201, (again.text, new.text)
        bodies += [{**again.json(), "email": "x"}, {**new.json(), "email": "x"}]
    assert all(b == bodies[0] for b in bodies), bodies
