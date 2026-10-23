"""Admin restore — the missing undo for project Delete (2026-07-26).

The sidebar cleanup archived the REAL corpus projects along with the junk
shells; archive hides a project from retrieval and there was no way back.
Restore flips an archived project to active, optionally keeping it off the
sidebar (the correct end state for corpus/GK rows).
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import projects as store
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(autouse=True)
def _admin_auth():
    from app.dependencies import require_api_key

    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-admin", "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_restore_archived_project(client):
    store.create_project("Restore me", user_id="system", project_id="p_restore_1")
    assert store.archive_project("p_restore_1")
    assert store.get_project("p_restore_1") is None  # archived = invisible

    r = client.post("/v1/admin/projects/p_restore_1/restore", headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "restored"
    proj = store.get_project("p_restore_1")
    assert proj is not None and proj["status"] == "active"


def test_restore_with_sidebar_hide(client):
    store.create_project("Corpus row", user_id="system", project_id="p_restore_2")
    store.archive_project("p_restore_2")

    r = client.post(
        "/v1/admin/projects/p_restore_2/restore?hide_from_sidebar=true", headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["hidden_from_sidebar"] is True
    # active + retrievable, but not in the sidebar listing
    assert store.get_project("p_restore_2")["status"] == "active"
    assert "p_restore_2" not in {p["id"] for p in store.list_projects()}
    assert "p_restore_2" in {p["id"] for p in store.list_projects(include_hidden=True)}


def test_restore_rejects_active_and_missing(client):
    store.create_project("Already active", user_id="system", project_id="p_restore_3")
    assert client.post("/v1/admin/projects/p_restore_3/restore", headers=H).status_code == 409
    assert client.post("/v1/admin/projects/nope_restore/restore", headers=H).status_code == 404
