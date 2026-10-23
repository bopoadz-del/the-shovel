"""A project's profile: set once by an admin, stored with the project, and
given to the model in driver mode. Synthetic data."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import jwt_auth
from app.core import users as users_store
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


def _admin_headers():
    users_store.ensure_user_exists("profile-admin", role="admin", email="profile-admin@local")
    return {"Authorization": f"Bearer {jwt_auth.create_token('profile-admin')}"}


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_an_admin_sets_the_profile_and_the_driver_sees_it(client):
    from app.agents.driver import context
    from app.core import projects

    created = client.post("/v1/projects", json={"name": "Profile Probe"}, headers=H).json()
    pid, owner = created["id"], created.get("user_id")
    profile = {"discipline": "infrastructure", "contract_form": "design and build",
               "governing_codes": "the project's specification", "empty": ""}
    r = client.put(f"/v1/admin/projects/{pid}/profile", json=profile, headers=_admin_headers())
    assert r.status_code == 200, r.text
    assert r.json()["profile"] == {k: v for k, v in profile.items() if v}
    assert projects.get_project(pid)["profile"]["contract_form"] == "design and build"
    text = context.project_profile(pid, owner)  # read as the project's owner
    assert "Profile Probe" in text and "contract_form: design and build" in text


def test_only_an_admin_sets_a_profile(client):
    pid = client.post("/v1/projects", json={"name": "Profile Guard"}, headers=H).json()["id"]
    r = client.put(f"/v1/admin/projects/{pid}/profile", json={"phase": "tender"}, headers=H)
    assert r.status_code in (401, 403)


def test_an_empty_profile_clears_it(client):
    from app.core import projects

    pid = client.post("/v1/projects", json={"name": "Profile Clear"}, headers=H).json()["id"]
    projects.set_project_profile(pid, {"phase": "construction"})
    assert projects.get_project(pid)["profile"] == {"phase": "construction"}
    projects.set_project_profile(pid, {})
    assert projects.get_project(pid)["profile"] is None
