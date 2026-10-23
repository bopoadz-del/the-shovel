"""Drive admin routes must not 500 for an API-key principal.

Found by the authenticated route sweep (the "full authenticated probe of
all routes" item PLATFORM_HEALTH_REPORT lists as NOT COVERED). Live prod
2026-08-02, using the operator's own admin key:

    GET /v1/admin/drive/scan -> 500 INTERNAL_ERROR
    KeyError: 'user_id' at admin.py:1381

Root cause: `require_api_key` returns two different principal shapes.
The JWT branch resolves a real user and includes `user_id`. The API-key
branch returns the `auth_manager` key record — `user`/`tier`/`role`, no
`user_id`. `CEREBRUM_MASTER_KEY` is minted with `role: "admin"`, so a
master-key caller sails through `_require_admin` and then dies indexing
`auth["user_id"]`.

The existing Drive tests never caught it because their fixture overrides
`require_api_key` with a dict that HAS `user_id` — a fixture more
generous than production.

Drive tokens are stored per user, so a key principal genuinely has no
Drive connection. The honest answer is the 409 these routes already
document for "not connected", not an unhandled 500.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import require_api_key
from app.routers import admin as admin_mod

# Exactly what auth_manager mints for CEREBRUM_MASTER_KEY — note: no user_id.
API_KEY_PRINCIPAL = {
    "user": "master",
    "tier": "unlimited",
    "role": "admin",
    "valid": True,
}

JWT_PRINCIPAL = {
    "user": "admin@example.com",
    "tier": "admin",
    "valid": True,
    "user_id": "usr_real_admin",
    "role": "admin",
    "email": "admin@example.com",
    "auth_method": "jwt",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _as(principal):
    app.dependency_overrides[require_api_key] = lambda: principal


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_drive_scan_api_key_principal_gets_409_not_500(client):
    """The exact live failure. Old code: KeyError -> 500."""
    _as(API_KEY_PRINCIPAL)

    r = client.get("/v1/admin/drive/scan")

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "API key" in detail
    assert "admin session" in detail


def test_approve_from_drive_api_key_principal_gets_409_not_500(client):
    """Same principal shape reaches the same bug on the POST path."""
    _as(API_KEY_PRINCIPAL)

    r = client.post(
        "/v1/admin/projects/approve-from-drive",
        json={"folder_id": "fldr_123", "name": "Some Project"},
    )

    assert r.status_code == 409, r.text
    assert "API key" in r.json()["detail"]


def test_real_user_principal_still_reaches_drive_and_reports_not_connected(
    client, monkeypatch
):
    """The resolver must not swallow the genuine not-connected path.

    A JWT principal HAS a user_id, so it must get past the resolver and
    produce the ORIGINAL 409 from drive_auth — a different message. If
    both paths returned the same text, this fix would be hiding the real
    Drive state behind a principal check.
    """
    from app.core import drive_auth

    def _not_connected(_user_id):
        raise drive_auth.DriveNotConnected("no token")

    monkeypatch.setattr(drive_auth, "get_access_token", _not_connected)
    _as(JWT_PRINCIPAL)

    r = client.get("/v1/admin/drive/scan")

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "not connected" in detail.lower()
    assert "API key" not in detail  # the principal branch did NOT fire


def test_non_admin_still_403(client):
    """Admin gating is unchanged — the 403 must still win over the 409."""
    _as({"user": "someone", "tier": "standard", "valid": True})

    r = client.get("/v1/admin/drive/scan")

    assert r.status_code == 403


@pytest.mark.parametrize("bad", [{}, {"user_id": None}, {"user_id": ""}])
def test_drive_user_id_rejects_principals_without_a_user(bad):
    with pytest.raises(HTTPException) as exc:
        admin_mod._drive_user_id(bad)
    assert exc.value.status_code == 409


def test_drive_user_id_returns_the_id_when_present():
    assert admin_mod._drive_user_id({"user_id": "usr_1"}) == "usr_1"
