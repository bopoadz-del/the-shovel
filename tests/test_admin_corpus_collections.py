"""PR #91 — admin corpus-collections endpoint contract guards.

Verifies the GET /v1/admin/corpus/collections endpoint:
  * gates on admin role (403 for non-admin)
  * returns per-project_id document + chunk counts
  * sorts collections by chunks desc, then documents desc, then project_id
  * emits by_top_folder ONLY for project_ids above the folder_breakdown_min
    threshold (default 50 documents), so ad-hoc projects don't drown the
    response in single-folder tiles
  * tolerates the case where a project_id exists in chunks but not
    documents (legacy import pattern from the Drive corpus seed)

Uses the app's existing SQLite test DB seeded via SQLAlchemy ORM —
avoids reaching into raw sqlite3 to keep the test resilient to schema
drift handled by Alembic / model edits.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.db import SessionLocal, engine
from app.core.models import Document, Project, RagChunk


def _ensure_schema():
    """Make sure both `documents` (via projects init) and `chunks`
    (RagChunk) tables exist. Mirrors the bootstrap path the live app
    uses on first request — calls projects.init_db then ensures the
    RagChunk table via the same checkfirst=True pattern
    vector_store.py uses."""
    from app.core.projects import init_db as init_projects_db
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    init_projects_db()
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    RagChunk.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture(autouse=True)
def _stub_admin(monkeypatch):
    """Force require_api_key to return an admin identity so we can test
    the endpoint's business logic without exercising JWT/users flow."""
    # The endpoint counts the RAG store the retriever actually reads —
    # rag_chunk_table_name(RAG_VECTOR_NAMESPACE), which defaults to the
    # namespaced `chunks_v2`. This fixture seeds the legacy `chunks` table
    # (RagChunk), so pin the namespace to "" (→ `chunks`) to keep the seed
    # and the count on the same table. Production runs namespace="v2" and the
    # same code path counts `chunks_v2`; the drive_archive count post-deploy
    # is the real-world proof.
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "")
    from app.routers import admin as admin_mod
    from app.dependencies import require_api_key as real_require_api_key

    def fake_admin():
        return {"user_id": "test-admin", "role": "admin"}

    def fake_user():
        return {"user_id": "test-user", "role": "user"}

    yield {"admin": fake_admin, "user": fake_user, "real": real_require_api_key}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _seed_corpus():
    """Insert a known-shape mini-corpus:
       - "big_corpus": 60 documents (above folder_breakdown_min), 200 chunks
                       across two top-folder prefixes (40 in 'A/', 20 in 'B/')
       - "small_project": 3 documents (below threshold), 10 chunks
       - "chunks_only_legacy": 0 documents, 5 chunks (legacy Drive imports)

    FK-compliant on PostgreSQL, where chunks.doc_id and chunks.project_id
    are enforced (the_fork_schema.sql) — sqlite's model-created chunks
    table declares no FKs, so the old fabricated doc-ids only blew up on
    the test-postgres job. Chunks therefore reference REAL document ids,
    and the chunks-only legacy corpus points its chunks at a document
    owned by ANOTHER project: documents are counted by
    documents.project_id, so its own document count stays 0 — the same
    dangling shape the legacy Drive imports produced.
    """
    import uuid

    _ensure_schema()

    with SessionLocal() as session:
        # Wipe pre-existing test rows for these ids — keeps the test idempotent
        # against repeated runs in the same SQLite DB.
        for pid in ("big_corpus", "small_project", "chunks_only_legacy"):
            session.query(RagChunk).filter(RagChunk.project_id == pid).delete()
            session.query(Document).filter(Document.project_id == pid).delete()
            session.query(Project).filter(Project.id == pid).delete()
        session.commit()

        # Projects (FK target for documents AND, on Postgres, for chunks —
        # chunks_only_legacy needs a project row even with zero documents).
        for pid, name in (
            ("big_corpus", "Big corpus"),
            ("small_project", "Small project"),
            ("chunks_only_legacy", "Chunks-only legacy"),
        ):
            session.add(Project(
                id=pid, name=name, user_id="system",
                created_at="2026-06-21T00:00:00Z",
                status="active",
            ))
        session.flush()

        # big_corpus: 40 'A/...' + 20 'B/...' = 60 docs
        big_doc_ids = []
        for i in range(40):
            doc_id = str(uuid.uuid4())[:8]
            big_doc_ids.append(doc_id)
            session.add(Document(
                id=doc_id, project_id="big_corpus",
                original_name=f"A/folder/file_{i:03d}.pdf",
                doc_type="document", doc_role="other", size=1024,
                uploaded_at="2026-06-21T00:00:00Z",
            ))
        for i in range(20):
            doc_id = str(uuid.uuid4())[:8]
            big_doc_ids.append(doc_id)
            session.add(Document(
                id=doc_id, project_id="big_corpus",
                original_name=f"B/folder/file_{i:03d}.pdf",
                doc_type="document", doc_role="other", size=1024,
                uploaded_at="2026-06-21T00:00:00Z",
            ))
        # small_project: 3 docs in one folder
        small_doc_ids = []
        for i in range(3):
            doc_id = str(uuid.uuid4())[:8]
            small_doc_ids.append(doc_id)
            session.add(Document(
                id=doc_id, project_id="small_project",
                original_name=f"misc/file_{i}.pdf",
                doc_type="document", doc_role="other", size=512,
                uploaded_at="2026-06-21T00:00:00Z",
            ))
        # Documents must be flushed before chunks reference them — no ORM
        # relationship is declared, so the unit of work won't order the
        # inserts for us and Postgres checks the FK per statement.
        session.flush()

        # chunks rows for each project. 5 chunks per document (chunk_index
        # 0..4) keeps uq_chunks_project_doc_index happy.
        import numpy as np
        zero_vec = np.zeros(256, dtype=np.float32)
        for pid, n_chunks, doc_ids in (
            ("big_corpus", 200, big_doc_ids),
            ("small_project", 10, small_doc_ids),
            # Legacy: chunks whose parent document landed under a different
            # project — its own documents count is 0.
            ("chunks_only_legacy", 5, big_doc_ids),
        ):
            for i in range(n_chunks):
                session.add(RagChunk(
                    chunk_id=f"{pid}-{i}-{uuid.uuid4().hex[:6]}",
                    project_id=pid,
                    doc_id=doc_ids[(i // 5) % len(doc_ids)],
                    chunk_index=i % 5,
                    text=f"chunk {i}",
                    embedding=zero_vec,
                    created_at="2026-06-21T00:00:00Z",
                ))
        session.commit()


def test_admin_corpus_collections_returns_per_project_counts(client, _stub_admin):
    _seed_corpus()
    app.dependency_overrides[__import__("app.dependencies", fromlist=["require_api_key"]).require_api_key] = _stub_admin["admin"]
    try:
        resp = client.get("/v1/admin/corpus/collections")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    cols = {c["project_id"]: c for c in body["collections"]}
    assert "big_corpus" in cols
    assert cols["big_corpus"]["documents"] == 60
    assert cols["big_corpus"]["chunks"] == 200
    assert cols["small_project"]["documents"] == 3
    assert cols["small_project"]["chunks"] == 10
    # chunks-only legacy: 0 docs, 5 chunks — must still appear so operator
    # sees that the index has dangling data without parent documents.
    assert cols["chunks_only_legacy"]["documents"] == 0
    assert cols["chunks_only_legacy"]["chunks"] == 5


def test_folder_breakdown_only_above_threshold(client, _stub_admin):
    _seed_corpus()
    app.dependency_overrides[__import__("app.dependencies", fromlist=["require_api_key"]).require_api_key] = _stub_admin["admin"]
    try:
        resp = client.get("/v1/admin/corpus/collections")
    finally:
        app.dependency_overrides.clear()
    cols = {c["project_id"]: c for c in resp.json()["collections"]}

    # big_corpus has 60 docs >= 50 default threshold -> breakdown present
    assert "by_top_folder" in cols["big_corpus"]
    folders = {f["folder"]: f["docs"] for f in cols["big_corpus"]["by_top_folder"]}
    assert folders == {"A": 40, "B": 20}

    # small_project has 3 docs -> no breakdown
    assert "by_top_folder" not in cols["small_project"]
    # chunks_only_legacy has 0 docs -> no breakdown
    assert "by_top_folder" not in cols["chunks_only_legacy"]


def test_sort_order_chunks_desc_then_documents(client, _stub_admin):
    _seed_corpus()
    app.dependency_overrides[__import__("app.dependencies", fromlist=["require_api_key"]).require_api_key] = _stub_admin["admin"]
    try:
        resp = client.get("/v1/admin/corpus/collections")
    finally:
        app.dependency_overrides.clear()
    all_ids = [c["project_id"] for c in resp.json()["collections"]]
    # The app boot-seeds a general-knowledge collection (training_material,
    # from docs/knowledge/*.md) whose chunk count can land anywhere in the
    # global ordering. The contract this test guards is chunks-desc AMONG the
    # seeded corpora, so filter to them before asserting order.
    seeded = [pid for pid in all_ids
              if pid in {"big_corpus", "small_project", "chunks_only_legacy"}]
    # big_corpus (200 chunks) > small_project (10) > chunks_only_legacy (5)
    assert seeded == ["big_corpus", "small_project", "chunks_only_legacy"], all_ids


def test_non_admin_gets_403(client, _stub_admin):
    app.dependency_overrides[__import__("app.dependencies", fromlist=["require_api_key"]).require_api_key] = _stub_admin["user"]
    try:
        resp = client.get("/v1/admin/corpus/collections")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 403


def test_folder_breakdown_disable(client, _stub_admin):
    _seed_corpus()
    app.dependency_overrides[__import__("app.dependencies", fromlist=["require_api_key"]).require_api_key] = _stub_admin["admin"]
    try:
        resp = client.get("/v1/admin/corpus/collections?folder_breakdown=false")
    finally:
        app.dependency_overrides.clear()
    cols = {c["project_id"]: c for c in resp.json()["collections"]}
    # With breakdown disabled, even the big_corpus must NOT have by_top_folder.
    assert "by_top_folder" not in cols["big_corpus"]


def test_master_corpus_alias_inherits_source_counts(client, _stub_admin):
    """PR #??? — the admin inventory must show the pilot master-corpus alias
    (master_corpus) with the same counts as its backing Drive folder
    corpus (projects_folder), never 0 chunks.
    """
    _ensure_schema()

    import uuid
    import numpy as np

    with SessionLocal() as session:
        # Clean slate for the ids this test owns.
        for pid in ("projects_folder", "master_corpus"):
            session.query(RagChunk).filter(RagChunk.project_id == pid).delete()
            session.query(Document).filter(Document.project_id == pid).delete()
            session.query(Project).filter(Project.id == pid).delete()
        session.commit()

        # Backing corpus: 7 docs, 21 chunks.
        session.add(Project(
            id="projects_folder", name="Master Corpus",
            user_id="system", created_at="2026-06-21T00:00:00Z", status="active",
        ))
        session.add(Project(
            id="master_corpus", name="Master Corpus",
            user_id="system", created_at="2026-06-21T00:00:00Z", status="active",
        ))
        session.flush()
        doc_ids = []
        for i in range(7):
            doc_id = str(uuid.uuid4())[:8]
            doc_ids.append(doc_id)
            session.add(Document(
                id=doc_id, project_id="projects_folder",
                original_name=f"folder/file_{i}.pdf",
                doc_type="document", doc_role="other", size=1024,
                uploaded_at="2026-06-21T00:00:00Z",
            ))
        session.flush()
        zero_vec = np.zeros(256, dtype=np.float32)
        for i in range(21):
            session.add(RagChunk(
                chunk_id=f"projects_folder-{i}-{uuid.uuid4().hex[:6]}",
                project_id="projects_folder",
                doc_id=doc_ids[i // 3],
                chunk_index=i % 3,
                text=f"chunk {i}",
                embedding=zero_vec,
                created_at="2026-06-21T00:00:00Z",
            ))
        session.commit()

    app.dependency_overrides[__import__("app.dependencies", fromlist=["require_api_key"]).require_api_key] = _stub_admin["admin"]
    try:
        resp = client.get("/v1/admin/corpus/collections?folder_breakdown=false")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200, resp.text
    cols = {c["project_id"]: c for c in resp.json()["collections"]}

    assert "projects_folder" in cols
    assert cols["projects_folder"]["documents"] == 7
    assert cols["projects_folder"]["chunks"] == 21

    # Alias must mirror the source counts, not report zero chunks.
    assert "master_corpus" in cols
    assert cols["master_corpus"]["documents"] == 7
    assert cols["master_corpus"]["chunks"] == 21
    assert cols["master_corpus"].get("source_project_id") == "projects_folder"
    assert cols["master_corpus"].get("is_master_corpus_alias") is True
