"""Admin bulk-insert endpoint contract guards.

Original contract (PR #93) wrote chunks through the static ``RagChunk``
model — the RETIRED legacy ``chunks`` table. Production retrieval reads the
namespaced table (``chunks_v2``), so every chunk loaded via this endpoint
after the namespace split was invisible to retrieval: a silent no-op for
the endpoint's stated purpose. Found 2026-08-05 while preparing the
ab-2023-101 Vol 3 backfill.

Corrected contract, verified here:
  * admin-only (403 for non-admin) — unchanged
  * projects/documents keep additive existing-row-skip semantics — unchanged
  * chunks land in the ACTIVE namespace via VectorStore.upsert_chunks and
    are visible to store.count_by_doc — THE defect shape
  * chunk writes are a per-document idempotent REPLACE: re-sending a doc
    re-writes it, and the store's total for that doc does not grow
  * an embedding whose dimension doesn't match the namespace's store is
    rejected 400 with NOTHING written (fail closed, whole request)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.db import SessionLocal, engine
from app.core.models import Document, Project
from app.core.rag.vector_store import get_store
from app.dependencies import require_api_key

PROJ = "bulk_test_proj"


def _ensure_schema():
    from app.core.projects import init_db as init_projects_db
    init_projects_db()
    # The namespaced chunk table is created by the store itself.
    get_store()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _admin_override():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-admin", "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


def _payload():
    now = datetime.now(timezone.utc).isoformat()
    vec = [0.0] * get_store().dim  # match the ACTIVE namespace, whatever the test embedder is
    return {
        "projects": [
            {"id": PROJ, "name": "Bulk Test", "user_id": "system",
             "status": "active", "created_at": now},
        ],
        "documents": [
            {"id": "bulk_doc_1", "project_id": PROJ,
             "original_name": "test.pdf", "size": 1024, "uploaded_at": now},
            {"id": "bulk_doc_2", "project_id": PROJ,
             "original_name": "test2.pdf", "size": 2048, "uploaded_at": now},
        ],
        "chunks": [
            {"chunk_id": "bulk_c_1", "project_id": PROJ,
             "doc_id": "bulk_doc_1", "chunk_index": 0, "text": "hello",
             "embedding": vec, "created_at": now},
            {"chunk_id": "bulk_c_2", "project_id": PROJ,
             "doc_id": "bulk_doc_1", "chunk_index": 1, "text": "world",
             "embedding": vec, "created_at": now},
            {"chunk_id": "bulk_c_3", "project_id": PROJ,
             "doc_id": "bulk_doc_2", "chunk_index": 0, "text": "other",
             "embedding": vec, "created_at": now},
        ],
    }


def _wipe_test_rows():
    store = get_store()
    for doc_id in ("bulk_doc_1", "bulk_doc_2"):
        store.delete_doc(PROJ, doc_id)
    with SessionLocal() as session:
        session.query(Document).filter(Document.project_id == PROJ).delete()
        session.query(Project).filter(Project.id == PROJ).delete()
        session.commit()


def test_bulk_insert_inserts_in_fk_order(client):
    _ensure_schema()
    _wipe_test_rows()
    resp = client.post("/v1/admin/corpus/bulk-insert", json=_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    counts = body["counts"]
    assert counts["projects"] == 1
    assert counts["documents"] == 2
    assert counts["chunks"] == 3
    assert (counts["projects_seen"], counts["documents_seen"], counts["chunks_seen"]) == (1, 2, 3)
    with SessionLocal() as session:
        assert session.query(Project).filter(Project.id == PROJ).count() == 1
        assert session.query(Document).filter(Document.project_id == PROJ).count() == 2
    _wipe_test_rows()


def test_bulk_insert_chunks_land_in_the_active_namespace(client):
    """THE defect shape: chunks written here must be visible to the store
    production retrieval actually reads. Under the old code they landed in
    the retired legacy table and this assertion fails with {} == {...}."""
    _ensure_schema()
    _wipe_test_rows()
    resp = client.post("/v1/admin/corpus/bulk-insert", json=_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    counts = get_store().count_by_doc(PROJ)
    assert counts == {"bulk_doc_1": 2, "bulk_doc_2": 1}
    assert body["verified_chunks_by_doc"] == {"bulk_doc_1": 2, "bulk_doc_2": 1}
    assert body["namespace_table"] != "chunks", (
        "active-namespace default must never resolve to the retired legacy table"
    )
    _wipe_test_rows()


def test_bulk_insert_is_idempotent_per_document(client):
    _ensure_schema()
    _wipe_test_rows()
    p = _payload()
    first = client.post("/v1/admin/corpus/bulk-insert", json=p).json()["counts"]
    assert (first["projects"], first["documents"], first["chunks"]) == (1, 2, 3)
    second = client.post("/v1/admin/corpus/bulk-insert", json=p).json()
    # projects/documents: additive skip -> zero new rows
    assert (second["counts"]["projects"], second["counts"]["documents"]) == (0, 0)
    # chunks: per-doc REPLACE — re-written, but the store total must not grow
    assert second["counts"]["chunks"] == 3
    assert get_store().count_by_doc(PROJ) == {"bulk_doc_1": 2, "bulk_doc_2": 1}
    _wipe_test_rows()


def test_bulk_insert_rejects_dim_mismatch_writing_nothing(client):
    _ensure_schema()
    _wipe_test_rows()
    p = _payload()
    bad_dim = get_store().dim + 7
    p["chunks"][2]["embedding"] = [0.0] * bad_dim  # ONE bad doc in the batch
    resp = client.post("/v1/admin/corpus/bulk-insert", json=p)
    assert resp.status_code == 400
    assert "dim" in resp.json()["detail"]
    # Fail closed for the WHOLE chunk payload: even the good docs unwritten.
    assert get_store().count_by_doc(PROJ) == {}
    _wipe_test_rows()


def test_bulk_insert_non_admin_blocked(client):
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-user", "role": "user",
    }
    try:
        resp = client.post("/v1/admin/corpus/bulk-insert", json={"projects": [], "documents": [], "chunks": []})
    finally:
        app.dependency_overrides[require_api_key] = lambda: {
            "user_id": "test-admin", "role": "admin",
        }
    assert resp.status_code == 403


def test_bulk_insert_handles_empty_payload(client):
    _ensure_schema()
    resp = client.post("/v1/admin/corpus/bulk-insert", json={"projects": [], "documents": [], "chunks": []})
    assert resp.status_code == 200
    counts = resp.json()["counts"]
    assert counts == {"projects": 0, "documents": 0, "chunks": 0,
                      "projects_seen": 0, "documents_seen": 0, "chunks_seen": 0}


def test_bulk_insert_refuses_the_retired_legacy_namespace(client):
    """A request must never route chunks into the retired ``chunks`` table."""
    p = _payload()
    p["namespace"] = ""
    resp = client.post("/v1/admin/corpus/bulk-insert", json=p)
    assert resp.status_code == 400, resp.text
    assert "retired" in resp.json()["detail"]
    with SessionLocal() as s:
        assert s.get(Project, PROJ) is None, "a refused request wrote rows"
