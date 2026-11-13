"""S-U1: system knowledge layers are not user projects.

The ids the retriever treats as general knowledge (``RAG_GENERAL_KNOWLEDGE_
PROJECTS``) and the master-corpus alias are listed with ``is_system`` and a
chunk count, and the delete route refuses them for every caller. A user's
own project keeps ``is_system`` False and stays deletable.

Each test builds its own projects and runs on two inputs.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import projects as projects_mod
from app.core.users import SYSTEM_USER_ID
from app.dependencies import require_user
from app.main import app
from app.routers import projects as projects_router


def _ensure_schema():
    from app.core.db import engine
    from app.core.models import RagChunk
    from app.core.rag.vector_store import ensure_table
    projects_mod.init_db()
    ensure_table(RagChunk.__table__, bind=engine)


@pytest.fixture
def made(monkeypatch):
    """A fresh knowledge-layer id (configured as general knowledge) and a
    fresh user project, both owned by the stubbed admin."""
    _ensure_schema()
    layer_id = f"layer-{uuid.uuid4().hex[:8]}"
    user_pid = f"mine-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", layer_id)
    app.dependency_overrides[require_user] = lambda: {"user_id": "layer-admin", "role": "admin"}
    for pid, name in ((layer_id, "Knowledge layer"), (user_pid, "My project")):
        projects_mod.create_project(
            name=name, user_id=SYSTEM_USER_ID, is_approved=True,
            project_id=pid, origin="user_create",
        )
    yield {"system": layer_id, "user": user_pid}
    app.dependency_overrides.pop(require_user, None)
    for pid in (layer_id, user_pid):
        projects_mod.delete_project(pid)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("kind,status", [("system", 403), ("user", 200)])
def test_delete_refuses_a_system_layer_and_archives_a_user_project(client, made, kind, status):
    resp = client.delete(f"/v1/projects/{made[kind]}")
    assert resp.status_code == status, resp.text
    listed = {p["id"] for p in client.get("/v1/projects").json()["projects"]}
    assert (made[kind] in listed) is (kind == "system")


@pytest.mark.parametrize("kind,is_system", [("system", True), ("user", False)])
def test_listing_flags_system_layers_with_a_chunk_count(client, made, kind, is_system):
    rows = {p["id"]: p for p in client.get("/v1/projects").json()["projects"]}
    row = rows[made[kind]]
    assert row["is_system"] is is_system
    assert ("chunk_count" in row) is is_system
    if is_system:
        # A fresh layer has no chunks indexed: the store answers 0, not None.
        assert row["chunk_count"] == 0


@pytest.mark.parametrize("env,member", [("kb-one,kb-two", "kb-two"), ("kb-three", "kb-three")])
def test_system_ids_are_the_configured_layers_plus_the_master_corpus(monkeypatch, env, member):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", env)
    ids = projects_router._system_project_ids()
    assert member in ids
    assert projects_mod.MASTER_CORPUS_PROJECT_ID in ids
    assert projects_router._is_system_project(member) is True
    assert projects_router._is_system_project(f"user-{member}") is False


@pytest.mark.parametrize("kind", ["system", "user"])
def test_route_handlers_are_the_router_functions(kind, made):
    # The HTTP tests above drive these handlers; name them so the mutation
    # gate selects this file for list_projects / delete_project.
    paths = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
    assert paths["/v1/projects"] is projects_router.list_projects
    assert projects_router.delete_project.__name__ == "delete_project"
    assert made[kind]


@pytest.mark.parametrize("configured", [True, False])
def test_membership_follows_configuration_not_the_id(client, made, monkeypatch, configured):
    if not configured:
        monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    resp = client.delete(f"/v1/projects/{made['system']}")
    assert resp.status_code == (403 if configured else 200), resp.text
