"""List/detail authz consistency (2026-07-24 live finding).

Admins see EVERY project in GET /v1/projects, but the detail page 404'd any
project they didn't own (the operator hit this clicking an eval-fixture
project: "doesn't exist or you don't have access"). Read-only detail access
must match list visibility: admins may open any project; non-admins keep the
owner-or-approved-platform rule; mutation stays owner-only.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import require_user
from app.core import projects as store
from tests._fixture_values import PW_05


@pytest.fixture
def other_users_project():
    # Suite-shared test DB (same pattern as test_admin_approved_projects):
    # a real second user owns the project, mirroring the eval-fixture rows.
    from app.core import users
    try:
        seeder = users.create_user("seed@example.com", PW_05)
    except ValueError:
        seeder = users.get_user_by_email("seed@example.com")
    p = store.create_project(
        name="Fixture Owned Elsewhere", user_id=seeder["id"]
    )
    yield p
    try:
        store.delete_project(p["id"])
    except Exception:
        pass


def _client_as(role: str):
    app.dependency_overrides[require_user] = lambda: {
        "user_id": "operator", "role": role,
    }
    return TestClient(app)


def test_admin_can_open_any_listed_project(other_users_project):
    try:
        with _client_as("admin") as client:
            res = client.get(f"/v1/projects/{other_users_project['id']}")
        assert res.status_code == 200
        assert res.json()["id"] == other_users_project["id"]
    finally:
        app.dependency_overrides.clear()


def test_non_admin_still_cannot_open_others_projects(other_users_project):
    try:
        with _client_as("user") as client:
            res = client.get(f"/v1/projects/{other_users_project['id']}")
        assert res.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_admin_read_access_does_not_extend_to_mutation(other_users_project):
    """Delete keeps its own admin rules — this fix must not widen them."""
    try:
        with _client_as("user") as client:
            res = client.post(
                f"/v1/projects/{other_users_project['id']}/conversations/ws-x/clear"
            )
        assert res.status_code in (403, 404)
    finally:
        app.dependency_overrides.clear()
