"""Project-detail chunk_count reflects vector-store reality.

The legacy path counted chunks from the per-project ``doc_index`` JSON blob,
which bulk-inserted chunks (drive_archive migration) never populate. These
tests assert that ``GET /v1/projects/{id}`` and the vector store itself report
chunk counts from the ``chunks`` table, so a healthy migrated project never
shows 0 chunks.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.db import SessionLocal, engine
from app.core.models import Document, Project, User
from app.core.rag.vector_store import get_store
from app.dependencies import require_api_key, require_user


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _admin_override():
    # Admin endpoints (bulk-insert) use require_api_key; project endpoints use
    # require_user. Override both so tests don't depend on the test key secret.
    fake_admin = lambda: {"user_id": "test-admin", "role": "admin"}  # noqa: E731
    app.dependency_overrides[require_api_key] = fake_admin
    app.dependency_overrides[require_user] = fake_admin
    yield
    app.dependency_overrides.clear()


def _ensure_schema(monkeypatch):
    # Bulk-insert writes through the ACTIVE namespaced store (the 2026-08-05
    # contract fix — the old code wrote the retired legacy ``chunks`` table,
    # which retrieval no longer reads, and these tests used to force the
    # legacy namespace to paper over exactly that mismatch). A fake embedder
    # keeps the store dimension small and model-free.
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core.rag import embeddings as _emb, vector_store as _vs
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.projects import init_db as init_projects_db
    init_projects_db()
    get_store()  # creates the namespaced chunk table
    _ensure_test_admin()


def _ensure_test_admin():
    """Create the test-admin user so project FK constraints stay happy."""
    with SessionLocal() as session:
        if session.get(User, "test-admin") is None:
            session.add(User(
                id="test-admin",
                email="test-admin@example.com",
                password_hash="",
                salt="",
                display_name="Test Admin",
                role="admin",
                created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()


def _wipe(pid: str):
    store = get_store()
    for doc_id in list(store.count_by_doc(pid)):
        store.delete_doc(pid, doc_id)
    with SessionLocal() as session:
        session.query(Document).filter(Document.project_id == pid).delete()
        session.query(Project).filter(Project.id == pid).delete()
        session.commit()


def _bulk_payload(pid: str, docs: list[tuple[str, int]]):
    """Build a bulk-insert payload with ``docs`` as (doc_id, chunk_count)."""
    now = datetime.now(timezone.utc).isoformat()
    vec = [0.0] * get_store().dim  # match the active namespace's store
    projects = [{"id": pid, "name": f"Project {pid}", "user_id": "test-admin",
                 "status": "active", "created_at": now}]
    documents = []
    chunks = []
    for doc_id, n_chunks in docs:
        documents.append({
            "id": doc_id, "project_id": pid,
            "original_name": f"{doc_id}.pdf", "size": 1024,
            "uploaded_at": now,
        })
        for i in range(n_chunks):
            chunks.append({
                "chunk_id": f"{pid}:{doc_id}:{i}",
                "project_id": pid, "doc_id": doc_id,
                "chunk_index": i, "text": f"chunk {i}",
                "embedding": vec, "created_at": now,
            })
    return {"projects": projects, "documents": documents, "chunks": chunks}


def test_vector_store_count_by_doc_returns_bulk_inserted_counts(monkeypatch):
    _ensure_schema(monkeypatch)
    _wipe("chunk_count_bulk")
    from app.core.rag.vector_store import get_store
    from app.core.rag.embeddings import get_embedder

    payload = _bulk_payload("chunk_count_bulk", [
        ("doc_a", 3), ("doc_b", 5),
    ])
    with TestClient(app) as client:
        resp = client.post("/v1/admin/corpus/bulk-insert", json=payload)
    assert resp.status_code == 200

    embedder = get_embedder()
    store = get_store(dim=embedder.dim)
    counts = store.count_by_doc("chunk_count_bulk")
    assert counts == {"doc_a": 3, "doc_b": 5}
    assert store.count("chunk_count_bulk") == 8
    _wipe("chunk_count_bulk")


def test_project_detail_chunk_count_reflects_bulk_inserted_chunks(client, monkeypatch):
    _ensure_schema(monkeypatch)
    _wipe("chunk_count_bulk")
    payload = _bulk_payload("chunk_count_bulk", [
        ("doc_a", 3), ("doc_b", 5),
    ])
    resp = client.post("/v1/admin/corpus/bulk-insert", json=payload)
    assert resp.status_code == 200

    detail = client.get("/v1/projects/chunk_count_bulk").json()
    assert detail["chunk_count"] == 8
    assert detail["indexed_chunks"] == 8
    assert detail["has_indexed_chunks"] is True

    doc_counts = {d["id"]: d["chunk_count"] for d in detail["documents"]}
    assert doc_counts == {"doc_a": 3, "doc_b": 5}
    _wipe("chunk_count_bulk")


def test_project_detail_chunk_count_works_for_large_projects(client, monkeypatch):
    """There is no document-count threshold that skips chunk enrichment.

    The legacy path skipped projects with >500 documents because it
    deserialised a huge JSON blob. The vector-store path is a cheap
    GROUP BY, so it must run for the master corpus too.
    """
    _ensure_schema(monkeypatch)
    _wipe("chunk_count_large")
    # 600 documents is above the old 500-doc threshold.
    docs = [(f"doc_{i:03d}", 1) for i in range(600)]
    payload = _bulk_payload("chunk_count_large", docs)
    resp = client.post("/v1/admin/corpus/bulk-insert", json=payload)
    assert resp.status_code == 200

    detail = client.get("/v1/projects/chunk_count_large").json()
    assert detail["chunk_count"] == 600
    assert detail["indexed_chunks"] == 600
    assert detail["has_indexed_chunks"] is True
    # First page docs get per-document counts too.
    assert all(d.get("chunk_count", 0) == 1 for d in detail["documents"])
    _wipe("chunk_count_large")


def test_project_detail_chunk_count_zero_for_unindexed_docs(client, monkeypatch):
    """Documents with no chunks report 0, while the project total stays correct."""
    _ensure_schema(monkeypatch)
    _wipe("chunk_count_mixed")
    payload = _bulk_payload("chunk_count_mixed", [
        ("indexed_doc", 4), ("bare_doc", 0),
    ])
    resp = client.post("/v1/admin/corpus/bulk-insert", json=payload)
    assert resp.status_code == 200

    detail = client.get("/v1/projects/chunk_count_mixed").json()
    assert detail["chunk_count"] == 4
    doc_counts = {d["id"]: d["chunk_count"] for d in detail["documents"]}
    assert doc_counts["indexed_doc"] == 4
    assert doc_counts["bare_doc"] == 0
    _wipe("chunk_count_mixed")
