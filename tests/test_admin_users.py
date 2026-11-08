"""Admin account list: gated, paged, searched, and free of secrets.

Synthetic users only. No password change, deletion, or provider call.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app.routers.admin  # noqa: F401  dotted token so CI-M selects this file
from app.core import jwt_auth
from app.core import users as users_store

_PUBLIC = {
    "id",
    "email",
    "display_name",
    "role",
    "created_at",
    "email_verified",
    "driver_mode",
}
_SECRET_KEYS = {"password_hash", "salt", "token", "password", "api_key"}


def _headers(user_id: str, email: str, role: str) -> dict[str, str]:
    users_store.ensure_user_exists(user_id, role=role, email=email)
    return {"Authorization": f"Bearer {jwt_auth.create_token(user_id)}"}


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _emails(body: dict) -> set[str]:
    return {row["email"] for row in body["users"]}


def _ensure_account(email: str, password: str, display_name: str) -> dict:
    try:
        return users_store.create_user(email, password, display_name=display_name)
    except ValueError:
        existing = users_store.get_user_by_email(email)
        assert existing is not None
        return existing


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("anonymous", 401),
        ("user", 403),
        ("admin", 200),
    ],
)
def test_only_an_admin_may_list_users(client, kind, expected):
    headers = {}
    if kind == "user":
        headers = _headers("au-plain", "au-plain@example.test", "user")
    elif kind == "admin":
        headers = _headers("au-admin", "au-admin@example.test", "admin")
    response = client.get("/v1/admin/users", headers=headers)
    assert response.status_code == expected
    if expected == 200:
        body = response.json()
        assert "users" in body and "total" in body


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=51", "offset=-1", "offset=10001"],
)
def test_page_bounds_are_rejected(client, query):
    headers = _headers("au-bounds", "au-bounds@example.test", "admin")
    response = client.get(f"/v1/admin/users?{query}", headers=headers)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "term,emails",
    [
        ("au9f", {"au9fak@example.test", "au9f_k@example.test"}),
        ("au9f_k", {"au9f_k@example.test"}),
    ],
)
def test_search_is_a_literal_substring(client, term, emails):
    headers = _headers("au-search-admin", "au-search-admin@example.test", "admin")
    _ensure_account("au9fak@example.test", "pw-au9fak-one", "Alpha")
    _ensure_account("au9f_k@example.test", "pw-au9f-two", "Beta")
    response = client.get("/v1/admin/users", params={"q": term, "limit": 50}, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert _emails(body) == emails
    assert body["total"] == len(emails)


@pytest.mark.parametrize(
    "offset,email",
    [(0, "au-row-a@example.test"), (1, "au-row-b@example.test")],
)
def test_pages_are_stable_and_bounded(client, offset, email):
    headers = _headers("au-pager", "au-pager@example.test", "admin")
    _ensure_account("au-row-a@example.test", "pw-page-a", "A")
    _ensure_account("au-row-b@example.test", "pw-page-b", "B")
    response = client.get(
        "/v1/admin/users",
        params={"q": "au-row-", "limit": 1, "offset": offset},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 1 and body["offset"] == offset and body["total"] == 2
    assert _emails(body) == {email}


@pytest.mark.parametrize(
    "email,password",
    [
        ("au-sec-a@example.test", "pw-secret-alpha"),
        ("au-sec-b@example.test", "pw-secret-beta"),
    ],
)
def test_list_omits_password_material(client, email, password):
    headers = _headers("au-sec-admin", "au-sec-admin@example.test", "admin")
    created = _ensure_account(email, password, "Listed")
    stored = users_store.get_user_by_id(created["id"])
    response = client.get("/v1/admin/users", params={"q": email, "limit": 5}, headers=headers)
    assert response.status_code == 200
    raw = response.text
    body = response.json()
    assert stored["password_hash"] not in raw
    assert stored["salt"] not in raw
    assert password not in raw
    assert json.dumps(body).find("password_hash") == -1
    for row in body["users"]:
        assert set(row) <= _PUBLIC
        assert _SECRET_KEYS.isdisjoint(row)
