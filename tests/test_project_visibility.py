"""PR D — project visibility model + RAG project-scope safety.

Visibility model under test:
  * Admin → sees every project.
  * Non-admin → sees own projects + admin-approved platform projects;
                other users' personal projects stay hidden;
                is_approved=False rows stay owner-only.
  * Mutating endpoints (POST/DELETE on a project) remain owner-only.

RAG retriever:
  * retrieve_with_filter must reject empty project_id.
  * store.search must scope strictly to the requested project_id —
    no chunk from another project ever bleeds across.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import require_user, require_api_key
from app.core import projects as projects_mod


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _override(role: str, user_id: str = "system"):
    fake = lambda: {"user_id": user_id, "role": role}
    app.dependency_overrides[require_user] = fake
    app.dependency_overrides[require_api_key] = fake


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    app.dependency_overrides.clear()


# ── visibility model ──────────────────────────────────────────────────────

def test_admin_sees_all_projects(client):
    """Admin gets the full project list regardless of owner/origin."""
    own = projects_mod.create_project(
        name="System Own", user_id="system",
    )
    approved = projects_mod.create_project(
        name="Admin Approved", user_id="system",
        origin="admin_drive_approved",
    )
    # An alien project owned by a different user_id — admin should still
    # see it. (Use a user we know exists or skip the FK by using "system"
    # again; FK is on users.id which we can't easily forge in tests, so
    # use system as the alien-but-real owner.)
    other = projects_mod.create_project(
        name="System Other", user_id="system",
    )
    try:
        _override("admin", "system")
        resp = client.get("/v1/projects")
        assert resp.status_code == 200
        ids = {p["id"] for p in resp.json()["projects"]}
        assert own["id"] in ids
        assert approved["id"] in ids
        assert other["id"] in ids
    finally:
        for p in (own, approved, other):
            try: projects_mod.delete_project(p["id"])
            except Exception: pass


def test_non_admin_sees_own_plus_admin_approved(client):
    """Non-admin sees their own projects + admin_drive_approved rows."""
    mine = projects_mod.create_project(
        name="Mine", user_id="system", origin="user_create",
    )
    approved = projects_mod.create_project(
        name="Platform", user_id="system", origin="admin_drive_approved",
    )
    # Attach two documents so the pilot incomplete-shell filter (≤1 doc)
    # does not suppress this approved project from the non-admin list.
    projects_mod.add_document(approved["id"], "platform-brief.pdf")
    projects_mod.add_document(approved["id"], "platform-spec.pdf")
    try:
        _override("user", "system")
        resp = client.get("/v1/projects")
        assert resp.status_code == 200
        ids = {p["id"] for p in resp.json()["projects"]}
        assert mine["id"] in ids
        assert approved["id"] in ids
    finally:
        for p in (mine, approved):
            try: projects_mod.delete_project(p["id"])
            except Exception: pass


def test_unapproved_row_hidden_from_non_owner():
    """is_approved=False stays owner-only regardless of origin."""
    pending = projects_mod.create_project(
        name="Pending Detection", user_id="system",
        is_approved=False, origin="admin_drive_approved",
    )
    try:
        # list_projects with include_admin_approved should NOT include
        # is_approved=False rows even though origin matches.
        visible = projects_mod.list_projects(
            user_id="other-user", include_admin_approved=True,
        )
        assert all(p["id"] != pending["id"] for p in visible)
    finally:
        try: projects_mod.delete_project(pending["id"])
        except Exception: pass


def test_get_project_allows_admin_approved_for_non_owner():
    """A user must be able to READ an admin-approved project they don't own."""
    approved = projects_mod.create_project(
        name="Shared", user_id="system", origin="admin_drive_approved",
    )
    try:
        # Different user_id, read-only flag on — should return the row.
        got = projects_mod.get_project(
            approved["id"], user_id="alice",
            include_admin_approved=True,
        )
        assert got is not None
        assert got["id"] == approved["id"]
        # Without the flag, the legacy owner-check applies — invisible.
        assert projects_mod.get_project(
            approved["id"], user_id="alice",
        ) is None
    finally:
        try: projects_mod.delete_project(approved["id"])
        except Exception: pass


def test_get_project_blocks_user_create_for_non_owner():
    """Personal user-created rows stay owner-only — no leakage via the
    read-only flag."""
    mine = projects_mod.create_project(
        name="Private", user_id="system", origin="user_create",
    )
    try:
        # Even with include_admin_approved=True, a user_create row owned
        # by someone else must NOT load for a non-owner.
        assert projects_mod.get_project(
            mine["id"], user_id="alice",
            include_admin_approved=True,
        ) is None
    finally:
        try: projects_mod.delete_project(mine["id"])
        except Exception: pass


# ── RAG retriever safety ──────────────────────────────────────────────────

def test_general_knowledge_project_ids_parses_env(monkeypatch):
    """GK id set matches the env knowledge_seed / retriever already use."""
    monkeypatch.setenv(
        "RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb, training_material,",
    )
    assert projects_mod.general_knowledge_project_ids() == frozenset(
        {"curated_kb", "training_material"}
    )
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    assert projects_mod.general_knowledge_project_ids() == frozenset()
    monkeypatch.delenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", raising=False)
    assert "training_material" in projects_mod.general_knowledge_project_ids()


def test_retrieve_with_filter_rejects_empty_project_id():
    """Without a project_id we'd have no project scoping at all —
    that's a programming error, not a graceful-degradation case."""
    from app.core.rag.retriever import retrieve_with_filter

    # Empty / whitespace / None all rejected. Empty string is the most
    # important case — a logic bug somewhere upstream could produce one
    # and we should NOT silently search the whole corpus.
    with pytest.raises(ValueError):
        retrieve_with_filter("some query", "", k=5)
