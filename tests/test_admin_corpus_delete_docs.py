"""admin corpus delete-docs endpoint — export + scoped delete.

Guards POST /v1/admin/corpus/delete-docs:
  * admin-gated (403 for non-admin)
  * export mode returns each doc's metadata + concatenated chunk text and
    deletes NOTHING (the restore bundle)
  * delete mode removes the document row AND its chunks (document-level, no
    project cascade)
  * a doc whose project_id != the request scope is skipped (wrong_project),
    never deleted — the safety rail that stops a bad id-list crossing corpora

Mirrors test_admin_corpus_collections' fixture: seeds the legacy `chunks`
table (RagChunk) and pins RAG_VECTOR_NAMESPACE="" so the store reads the same
table it seeds.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.db import SessionLocal, engine
from app.core.models import Document, Project, RagChunk


def _ensure_schema():
    from app.core.projects import init_db as init_projects_db
    init_projects_db()
    RagChunk.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture(autouse=True)
def _ns(monkeypatch):
    # Seed + read the same table (legacy `chunks`), as the collections test does.
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _require_api_key():
    from app.dependencies import require_api_key
    return require_api_key


def _seed():
    _ensure_schema()
    zero = np.zeros(256, dtype=np.float32)
    with SessionLocal() as s:
        for pid in ("dpr_test", "keep_test"):
            s.query(RagChunk).filter(RagChunk.project_id == pid).delete()
            s.query(Document).filter(Document.project_id == pid).delete()
            s.query(Project).filter(Project.id == pid).delete()
        s.commit()
        for pid in ("dpr_test", "keep_test"):
            s.add(Project(id=pid, name=pid, user_id="system",
                          created_at="2026-07-15T00:00:00Z", status="active"))
        s.flush()
        ids = {}
        for pid, n_docs in (("dpr_test", 2), ("keep_test", 1)):
            ids[pid] = []
            for i in range(n_docs):
                did = str(uuid.uuid4())[:8]
                ids[pid].append(did)
                s.add(Document(id=did, project_id=pid,
                               original_name=f"{pid}/file_{i}.docx",
                               file_path=f"X:\\\\Example Drive\\\\{pid}\\\\file_{i}.docx",
                               doc_type="document", doc_role="other", size=100,
                               uploaded_at="2026-07-15T00:00:00Z"))
            s.flush()
            for i, did in enumerate(ids[pid]):
                for c in range(3):
                    s.add(RagChunk(
                        chunk_id=f"{pid}:{did}:{c}", project_id=pid, doc_id=did,
                        chunk_index=c, text=f"{pid} {did} chunk {c}",
                        embedding=zero, created_at="2026-07-15T00:00:00Z"))
        s.commit()
        return ids


def _post(client, body):
    app.dependency_overrides[_require_api_key()] = lambda: {"user_id": "a", "role": "admin"}
    try:
        return client.post("/v1/admin/corpus/delete-docs", json=body)
    finally:
        app.dependency_overrides.clear()


def test_export_returns_text_and_deletes_nothing(client):
    ids = _seed()
    r = _post(client, {"project_id": "dpr_test", "doc_ids": ids["dpr_test"], "mode": "export"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["mode"] == "export"
    assert b["in_scope"] == 2
    got = {d["id"]: d for d in b["docs"]}
    for did in ids["dpr_test"]:
        assert got[did]["chunk_count"] == 3
        assert "chunk 0" in got[did]["text"]
    # Nothing deleted.
    with SessionLocal() as s:
        assert s.query(Document).filter(Document.project_id == "dpr_test").count() == 2
        assert s.query(RagChunk).filter(RagChunk.project_id == "dpr_test").count() == 6


def test_delete_removes_rows_and_chunks(client):
    ids = _seed()
    r = _post(client, {"project_id": "dpr_test", "doc_ids": ids["dpr_test"], "mode": "delete"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["mode"] == "delete" and b["deleted"] == 2 and not b["failed"]
    with SessionLocal() as s:
        assert s.query(Document).filter(Document.project_id == "dpr_test").count() == 0
        assert s.query(RagChunk).filter(RagChunk.project_id == "dpr_test").count() == 0


def test_wrong_project_is_skipped_not_deleted(client):
    ids = _seed()
    # Ask to delete a keep_test doc but scope to dpr_test — must be refused.
    r = _post(client, {"project_id": "dpr_test", "doc_ids": ids["keep_test"], "mode": "delete"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["deleted"] == 0
    assert ids["keep_test"][0] in b["wrong_project"]
    with SessionLocal() as s:
        assert s.query(Document).filter(Document.project_id == "keep_test").count() == 1
        assert s.query(RagChunk).filter(RagChunk.project_id == "keep_test").count() == 3


def test_non_admin_403(client):
    app.dependency_overrides[_require_api_key()] = lambda: {"user_id": "u", "role": "user"}
    try:
        r = client.post("/v1/admin/corpus/delete-docs",
                        json={"project_id": "dpr_test", "doc_ids": ["x"], "mode": "export"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 403
