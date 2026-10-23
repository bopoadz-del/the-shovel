"""PR A — admin-approved projects: is_approved column + Drive endpoints.

Three contracts to lock:
  1. The is_approved column exists on Project, defaults True, surfaces
     in the projects dict + /v1/projects response.
  2. /v1/admin/drive/scan rejects non-admin (403) and rejects when
     Drive isn't connected (409).
  3. /v1/admin/projects/approve-from-drive validates inputs,
     creates a project row with is_approved=True + correct user
     ownership, and slugs the name into a project_id.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import require_api_key, require_user
from app.core import projects as projects_mod
from tests._fixture_values import STUB_TOKEN_13


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _admin_auth():
    """Pin every test to an admin identity. /v1/projects uses
    ``require_user``; admin endpoints use ``require_api_key`` — override
    both so a single fixture covers every route under test."""
    fake = lambda: {"user_id": "system", "role": "admin"}
    app.dependency_overrides[require_api_key] = fake
    app.dependency_overrides[require_user] = fake
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def fresh_project():
    p = projects_mod.create_project(name="Approved Test")
    yield p
    try:
        projects_mod.delete_project(p["id"])
    except Exception:
        pass


def test_project_has_is_approved_field(fresh_project):
    assert "is_approved" in fresh_project
    assert fresh_project["is_approved"] is True


def test_create_project_with_is_approved_false():
    p = projects_mod.create_project(name="Pending", is_approved=False)
    try:
        assert p["is_approved"] is False
    finally:
        projects_mod.delete_project(p["id"])


def test_project_default_origin_is_user_create(fresh_project):
    """PR B — every project surfaces an origin; new ones default to
    user_create so the admin page's origin='admin_drive_approved' filter
    correctly excludes them."""
    assert fresh_project["origin"] == "user_create"


def test_create_project_with_explicit_origin():
    p = projects_mod.create_project(
        name="Drive Approved", origin="admin_drive_approved",
    )
    try:
        assert p["origin"] == "admin_drive_approved"
    finally:
        projects_mod.delete_project(p["id"])


def test_v1_projects_returns_is_approved(client, fresh_project):
    resp = client.get("/v1/projects")
    assert resp.status_code == 200
    rows = resp.json().get("projects", [])
    match = next((r for r in rows if r["id"] == fresh_project["id"]), None)
    assert match is not None
    assert match.get("is_approved") is True


def test_drive_scan_rejects_non_admin(client):
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-user", "role": "user",
    }
    try:
        resp = client.get("/v1/admin/drive/scan")
    finally:
        app.dependency_overrides[require_api_key] = lambda: {
            "user_id": "test-admin", "role": "admin",
        }
    assert resp.status_code == 403


def test_drive_scan_rejects_without_drive_connection(client, monkeypatch):
    from app.core import drive_auth

    async def fake_not_connected(_user_id):
        raise drive_auth.DriveNotConnected("no token")

    monkeypatch.setattr(drive_auth, "get_access_token", fake_not_connected)
    resp = client.get("/v1/admin/drive/scan")
    assert resp.status_code == 409
    body = resp.json()
    detail = body.get("detail") or body.get("error", {}).get("message", "")
    assert "not connected" in detail.lower()


def test_approve_from_drive_validates_inputs(client):
    r1 = client.post("/v1/admin/projects/approve-from-drive",
                     json={"folder_id": "", "name": "Foo"})
    assert r1.status_code == 400
    r2 = client.post("/v1/admin/projects/approve-from-drive",
                     json={"folder_id": "abc", "name": ""})
    assert r2.status_code == 400


def test_approve_from_drive_creates_row_and_slugs_name(client, monkeypatch):
    """The admin-approve endpoint uses the authenticated user's id as
    the row owner. Override auth to 'system' so the FK to users(id)
    resolves (system user is auto-created at startup)."""
    from app.core import drive_auth
    import app.routers.admin as admin_mod

    fake = lambda: {"user_id": "system", "role": "admin"}
    app.dependency_overrides[require_api_key] = fake
    app.dependency_overrides[require_user] = fake

    async def fake_token(_user_id):
        return STUB_TOKEN_13

    async def fake_worker(**_kw):
        return None

    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    monkeypatch.setattr(admin_mod, "_run_drive_folder_import", fake_worker)

    resp = client.post(
        "/v1/admin/projects/approve-from-drive",
        json={"folder_id": "drive-folder-abc", "name": "Example Infra Pack"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    project = body["project"]
    assert project["name"] == "Example Infra Pack"
    assert project["is_approved"] is True
    assert project["user_id"] == "system"
    assert project["id"] == "example_infra_pack"
    assert project["origin"] == "admin_drive_approved"
    try:
        projects_mod.delete_project(project["id"])
    except Exception:
        pass


def test_approve_from_drive_slug_collision_appends_suffix(client, monkeypatch):
    from app.core import drive_auth
    import app.routers.admin as admin_mod

    fake = lambda: {"user_id": "system", "role": "admin"}
    app.dependency_overrides[require_api_key] = fake
    app.dependency_overrides[require_user] = fake

    async def fake_token(_u): return "tok"
    async def fake_worker(**_kw): return None
    monkeypatch.setattr(drive_auth, "get_access_token", fake_token)
    monkeypatch.setattr(admin_mod, "_run_drive_folder_import", fake_worker)

    r1 = client.post("/v1/admin/projects/approve-from-drive",
                     json={"folder_id": "f1", "name": "Collide"})
    r2 = client.post("/v1/admin/projects/approve-from-drive",
                     json={"folder_id": "f2", "name": "Collide"})
    assert r1.status_code == 201 and r2.status_code == 201
    p1 = r1.json()["project"]; p2 = r2.json()["project"]
    assert p1["id"] == "collide"
    assert p2["id"] == "collide_2"
    try:
        projects_mod.delete_project(p1["id"])
        projects_mod.delete_project(p2["id"])
    except Exception:
        pass
