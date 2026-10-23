"""POST /v1/rag/search must not return another tenant's chunks.

Measured on 8535199: a user who is not a member of a project could post
that project's id and receive its chunk text verbatim --

    POST /v1/rag/search {"query": ..., "project_id": <someone else's>}
    -> 200 {"chunks": [{"text": "..."}], "count": 1}

Every other project read surface answers 404 for both "missing" and "not
yours" (no existence leak): /v1/projects/{id}, /documents, /conversations,
/documents/{id}/preview, /export/*, /connectors, and doc_search all route
through projects.get_project_accessible. This route did not, so the gate
existed and one door skipped it.

404 rather than 403 on purpose: a 403 confirms the project exists, which
turns this route into an id-enumeration oracle.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import jwt_auth
from app.core import users as users_store
from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import PW_01


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _tenant(tag: str):
    user = users_store.create_user(
        f"rag-{tag}-{uuid.uuid4().hex[:8]}@example.test",
        PW_01,
        email_verified=True,
    )
    return user, {"Authorization": f"Bearer {jwt_auth.create_token(user['id'])}"}


@pytest.fixture
def two_tenants(client, dev_key_is_admin):
    _owner, h_owner = _tenant("owner")
    _other, h_other = _tenant("other")
    pid = client.post(
        "/v1/projects", headers=h_owner, json={"name": "Tenancy Probe"}
    ).json()["id"]
    # Only the admin path adds to a project's knowledge base, so the owner's
    # project is seeded by an admin; what is under test is who may SEARCH it.
    r = client.post(
        f"/v1/projects/{pid}/documents",
        headers={"Authorization": "Bearer " + DEV_KEY},
        files={
            "file": (
                "notes.txt",
                b"the private rate is 0.1 percent per day " * 40,
                "text/plain",
            )
        },
    )
    assert r.status_code == 201, r.text
    return pid, h_owner, h_other


def test_owner_can_search_own_project(client, two_tenants):
    pid, h_owner, _ = two_tenants
    r = client.post(
        "/v1/rag/search",
        headers=h_owner,
        json={"query": "private rate percent", "project_id": pid, "k": 5},
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] >= 1


def test_non_member_gets_404_and_no_chunk_text(client, two_tenants):
    pid, _, h_other = two_tenants
    r = client.post(
        "/v1/rag/search",
        headers=h_other,
        json={"query": "private rate percent", "project_id": pid, "k": 5},
    )
    assert r.status_code == 404, r.text
    assert "private rate" not in r.text


def test_unknown_project_is_indistinguishable_from_foreign(client, two_tenants):
    _, h_owner, _ = two_tenants
    r = client.post(
        "/v1/rag/search",
        headers=h_owner,
        json={"query": "x", "project_id": "does-not-exist", "k": 5},
    )
    assert r.status_code == 404


def test_gk_status_uses_the_same_gate(client, two_tenants):
    pid, _, h_other = two_tenants
    r = client.get("/v1/rag/gk-status", headers=h_other, params={"project_id": pid})
    assert r.status_code == 404


def test_unauthenticated_search_is_401_not_404(client, two_tenants):
    """Auth failure must not look like a membership/retrieval 404 (S13)."""
    pid, _, _ = two_tenants
    r = client.post(
        "/v1/rag/search",
        json={"query": "private rate percent", "project_id": pid, "k": 5},
    )
    assert r.status_code == 401, r.text
    assert "chunks" not in r.json()


@pytest.fixture
def master_corpus_world(client, monkeypatch):
    """Backing corpus + a login JWT that is a master-corpus member, not owner.

    Live WATCH-1: origin=user_create source (projects_folder) 404'd for the
    same JWT that can open master_corpus. Unique ids keep this isolated on
    the shared-Postgres CI job.
    """
    from app.core import projects as store
    from app.core.rag.retriever import index_chunks
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    # Shared-Postgres CI seeds training_material; GK merge can fill k=5
    # with FIDIC "percent per day" notes and hide this fixture's secret.
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")

    tag = uuid.uuid4().hex[:10]
    alias = f"mc_alias_{tag}"
    source = f"mc_src_{tag}"
    monkeypatch.setattr(store, "MASTER_CORPUS_PROJECT_ID", alias)
    monkeypatch.setattr(store, "MASTER_CORPUS_SOURCE_PROJECT_ID", source)

    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.create_project(
        name="Source corpus",
        user_id=SYSTEM_USER_ID,
        is_approved=True,
        project_id=source,
        origin="user_create",
    )
    secret = f"corpus-rate-{tag} percent per day"
    index_chunks(source, f"doc-{tag}", [f"{secret} " * 20])

    member = users_store.create_user(
        f"mc-member-{tag}@example.test",
        PW_01,
        email_verified=True,
    )
    headers = {"Authorization": f"Bearer {jwt_auth.create_token(member['id'])}"}
    try:
        yield {
            "alias": alias,
            "source": source,
            "headers": headers,
            "secret": secret,
            "member_id": member["id"],
        }
    finally:
        store.delete_project(source)
        store.delete_project(alias)


def test_master_corpus_jwt_can_search_alias(client, master_corpus_world):
    """Smoke path: login JWT + project_id=master_corpus is membership, not 404."""
    w = master_corpus_world
    r = client.post(
        "/v1/rag/search",
        headers=w["headers"],
        json={"query": w["secret"], "project_id": w["alias"], "k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    assert any(w["secret"] in (c.get("text") or "") for c in body["chunks"])


def test_master_corpus_jwt_can_search_backing_source_id(client, master_corpus_world):
    """S13 / WATCH-1: the physical source id is the same membership as the alias."""
    w = master_corpus_world
    r = client.post(
        "/v1/rag/search",
        headers=w["headers"],
        json={"query": w["secret"], "project_id": w["source"], "k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    assert any(w["secret"] in (c.get("text") or "") for c in body["chunks"])


def test_master_corpus_member_still_404s_on_foreign_and_unknown(
    client, two_tenants, master_corpus_world,
):
    """Membership of the alias does not open a private project; 404s match."""
    foreign_pid, _, _ = two_tenants
    h = master_corpus_world["headers"]
    foreign = client.post(
        "/v1/rag/search",
        headers=h,
        json={"query": "private rate percent", "project_id": foreign_pid, "k": 5},
    )
    missing = client.post(
        "/v1/rag/search",
        headers=h,
        json={"query": "x", "project_id": "does-not-exist", "k": 5},
    )
    assert foreign.status_code == 404, foreign.text
    assert missing.status_code == 404, missing.text
    assert foreign.json().get("detail") == missing.json().get("detail") or (
        "not found" in (foreign.json().get("detail") or "").lower()
        and "not found" in (missing.json().get("detail") or "").lower()
    )
    assert "private rate" not in foreign.text


def test_picker_still_hides_physical_source_id(client, master_corpus_world):
    """UI-PHYS H1: GET /v1/projects/{source} stays 404; only RAG/chat open it."""
    w = master_corpus_world
    alias = client.get(f"/v1/projects/{w['alias']}", headers=w["headers"])
    source = client.get(f"/v1/projects/{w['source']}", headers=w["headers"])
    assert alias.status_code == 200, alias.text
    assert source.status_code == 404, source.text
