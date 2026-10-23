"""PR C — user-facing /v1/projects/from-drive endpoint.

Contracts:
  1. Endpoint validates inputs (empty folder_id / name → 400).
  2. Returns 409 when the caller's Drive isn't connected.
  3. Creates a row owned by the caller with origin='user_drive_import'
     and a slugged project_id (collision-suffixed if needed).
  4. Available to any authenticated user (no admin gate).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import require_user
from app.core import projects as projects_mod


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _user_auth():
    # Plain non-admin user — proves the endpoint is NOT admin-gated.
    fake = lambda: {"user_id": "system", "role": "user"}
    app.dependency_overrides[require_user] = fake
    yield
    app.dependency_overrides.clear()


def test_from_drive_requires_folder_id(client):
    resp = client.post(
        "/v1/projects/from-drive",
        json={"folder_id": "", "name": "X"},
    )
    assert resp.status_code == 400


def test_from_drive_requires_name(client):
    resp = client.post(
        "/v1/projects/from-drive",
        json={"folder_id": "abc", "name": ""},
    )
    assert resp.status_code == 400


def test_from_drive_returns_409_when_drive_not_connected(client, monkeypatch):
    from app.core import drive_auth

    async def fake_not_connected(_uid):
        raise drive_auth.DriveNotConnected("no token")

    monkeypatch.setattr(drive_auth, "get_access_token", fake_not_connected)
    resp = client.post(
        "/v1/projects/from-drive",
        json={"folder_id": "f1", "name": "My Personal Project"},
    )
    assert resp.status_code == 409
    body = resp.json()
    detail = body.get("detail") or body.get("error", {}).get("message", "")
    assert "not connected" in detail.lower()


def test_from_drive_creates_user_owned_project(client, monkeypatch):
    from app.core import drive_auth
    import app.routers.projects as projects_router

    async def fake_token(_u): return "tok"
    async def fake_worker(**_kw): return None
    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    monkeypatch.setattr(projects_router, "_run_drive_folder_import", fake_worker, raising=False)
    # The endpoint imports the helper lazily inside the function; patch
    # the source module too so the lookup at call time gets the fake.
    import app.routers.admin as admin_mod
    monkeypatch.setattr(admin_mod, "_run_drive_folder_import", fake_worker)

    resp = client.post(
        "/v1/projects/from-drive",
        json={"folder_id": "drive-folder-xyz", "name": "Beach House Reno"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    proj = body["project"]
    assert proj["name"] == "Beach House Reno"
    assert proj["user_id"] == "system"
    assert proj["origin"] == "user_drive_import"
    assert proj["is_approved"] is True
    assert proj["id"] == "beach_house_reno"
    try:
        projects_mod.delete_project(proj["id"])
    except Exception:
        pass


def test_from_drive_slug_collision_appends_suffix(client, monkeypatch):
    from app.core import drive_auth
    import app.routers.admin as admin_mod

    async def fake_token(_u): return "tok"
    async def fake_worker(**_kw): return None
    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    monkeypatch.setattr(admin_mod, "_run_drive_folder_import", fake_worker)

    r1 = client.post("/v1/projects/from-drive",
                     json={"folder_id": "f1", "name": "Twin"})
    r2 = client.post("/v1/projects/from-drive",
                     json={"folder_id": "f2", "name": "Twin"})
    assert r1.status_code == 201 and r2.status_code == 201
    p1, p2 = r1.json()["project"], r2.json()["project"]
    assert p1["id"] == "twin"
    assert p2["id"] == "twin_2"
    try:
        projects_mod.delete_project(p1["id"])
        projects_mod.delete_project(p2["id"])
    except Exception:
        pass
