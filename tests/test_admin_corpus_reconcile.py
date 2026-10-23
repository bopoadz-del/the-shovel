"""Tests for the corpus reconciliation endpoint.

Detects chunks whose ``project_id`` disagrees with their parent document's
``project_id`` — the root cause of the training_material scoping anomaly after
the drive_archive migration.
"""
from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.main import app
from app.core.db import SessionLocal, engine
from app.core.models import (
    EMBEDDING_DIM,
    Document,
    Project,
    User,
    make_rag_chunk_class,
    rag_chunk_class_for,
    rag_chunk_table_name,
)
from app.core.rag.vector_store import ensure_table
from app.dependencies import require_api_key


def _active_namespace() -> str:
    import os
    return (os.getenv("RAG_VECTOR_NAMESPACE", "v2") or "").strip()


def _chunk_cls():
    """Model for the chunk table the RETRIEVER actually reads.

    These tests used to seed the static ``RagChunk`` model, which maps the
    RETIRED legacy ``chunks`` table. Reconcile queried that same table, so the
    pair agreed with each other while disagreeing with production: post-v2
    ``chunks`` holds 0 rows, and the endpoint reported a clean corpus no matter
    how badly the live store was mismatched. Seeding the ACTIVE namespaced
    table is what makes these tests fail against that bug.
    """
    ns = _active_namespace()
    # Reuse whatever width this namespace was already opened at. A namespace
    # owns one table and therefore one vector width, so guessing EMBEDDING_DIM
    # here would trip the dim-conflict guard whenever the live embedder (384)
    # registered the namespace first.
    cls = rag_chunk_class_for(ns) or make_rag_chunk_class(ns, EMBEDDING_DIM, "test")
    # Race-safe: this helper and the app's own _ensure_schema both create the
    # namespaced table, on different engines, and checkfirst=True is
    # check-then-create rather than atomic. The loser used to fail the test
    # with "table ... already exists" (or "database is locked", same race).
    ensure_table(cls.__table__, bind=engine)
    return cls


def _chunk_table() -> str:
    return rag_chunk_table_name(_active_namespace())


def _chunk_dim() -> int:
    """Vector width of the ACTIVE chunk table.

    Must not be hardcoded. _chunk_cls() returns whatever width the namespace
    was opened at -- 384 when the live embedder registered it first -- and
    SQLite does not enforce vector width while PostgreSQL does. Seeding
    np.zeros(256) into a vector(384) column passes locally and fails only in
    the test-postgres CI job.
    """
    ident = getattr(_chunk_cls(), "embedding_identity", None)
    return int(ident["dim"]) if ident else EMBEDDING_DIM


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


def _ensure_schema():
    from app.core.projects import init_db as init_projects_db
    init_projects_db()
    _chunk_cls()
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


def _is_sqlite_locked(exc: BaseException) -> bool:
    return "database is locked" in str(exc).lower()


def _commit_retry(session):
    """SQLite in CI occasionally raises 'database is locked' on the reconcile
    wipe/seed commits while TestClient still holds a connection. Wait and retry
    rather than flake the production-like job."""
    delay = 0.1
    for attempt in range(10):
        try:
            session.commit()
            return
        except OperationalError as exc:
            session.rollback()
            if not _is_sqlite_locked(exc) or attempt == 9:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.5)


def _wipe_retry(work) -> None:
    """Retry the whole DML unit — DELETE can lock before commit() runs.

    production-like CI failed at ``_wipe('reconcile_clean')`` with
    ``DELETE FROM chunks ... database is locked``. ``_commit_retry`` never
    ran because the flush happened on ``.delete()``.
    """
    delay = 0.1
    for attempt in range(10):
        try:
            with SessionLocal() as session:
                work(session)
                session.commit()
            return
        except OperationalError as exc:
            if not _is_sqlite_locked(exc) or attempt == 9:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.5)


def _wipe(*pids: str):
    def _do(session):
        for pid in pids:
            cls = _chunk_cls()
            session.query(cls).filter(cls.project_id == pid).delete()
            session.query(Document).filter(Document.project_id == pid).delete()
            session.query(Project).filter(Project.id == pid).delete()

    _wipe_retry(_do)


def _wipe_all():
    """Delete every RagChunk/Document/Project row for test isolation.

    The reconcile endpoint reports mismatches across the whole corpus, so
    leftovers from earlier tests make exact-count assertions flaky. Wipe
    everything before seeding the reconcile fixture.

    Keep protected project rows (``training_material``, master-corpus
    alias/source). Later boot seed / tests insert documents into
    ``training_material``; dropping that parent trips
    ``documents_project_id_fkey`` on Postgres.
    """
    from app.core.projects import _PURGE_PROTECTED_IDS
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    _ensure_schema()
    protected = {pid for pid in _PURGE_PROTECTED_IDS if pid}

    def _do(session):
        session.query(_chunk_cls()).delete()
        session.query(Document).delete()
        q = session.query(Project)
        if protected:
            q = q.filter(~Project.id.in_(protected))
        q.delete(synchronize_session=False)

    _wipe_retry(_do)
    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    from app.core import projects as store
    for pid in protected:
        # Isolation TRUNCATE + a live TestClient knowledge-seed can both
        # see a missing GK/master row and INSERT the same PK. Use the
        # race-safe helper (IntegrityError → re-fetch) instead of
        # check-then-create.
        store.get_or_create_project(
            pid, user_id=SYSTEM_USER_ID, project_id=pid, origin="system_seed",
        )


def _seed_misplaced_chunks():
    """Create two projects where one chunk is under the wrong project_id."""
    _wipe_all()
    now = datetime.now(timezone.utc).isoformat()
    vec = np.zeros(_chunk_dim(), dtype=np.float32)
    with SessionLocal() as session:
        for pid in ("reconcile_a", "reconcile_b"):
            session.add(Project(
                id=pid, name=pid, user_id="test-admin",
                status="active", created_at=now,
            ))
        session.flush()
        session.add(Document(
            id="doc_a", project_id="reconcile_a",
            original_name="a.pdf", doc_type="document",
            doc_role="other", size=0, uploaded_at=now,
        ))
        session.add(Document(
            id="doc_b", project_id="reconcile_b",
            original_name="b.pdf", doc_type="document",
            doc_role="other", size=0, uploaded_at=now,
        ))
        # doc_a's chunk is wrongly stored under reconcile_b.
        session.add(_chunk_cls()(
            chunk_id="misplaced_1", project_id="reconcile_b",
            doc_id="doc_a", chunk_index=0, text="content",
            embedding=vec, created_at=now,
        ))
        # doc_b's chunk is correctly stored under reconcile_b.
        session.add(_chunk_cls()(
            chunk_id="correct_1", project_id="reconcile_b",
            doc_id="doc_b", chunk_index=0, text="content",
            embedding=vec, created_at=now,
        ))
        # A dangling chunk with no parent document. On PostgreSQL the
        # chunks.doc_id FK is enforced, so we bypass it the same way the
        # drive-archive bulk import did: session_replication_role=replica
        # disables FK checks for this insert only.
        if session.bind.dialect.name == "postgresql":
            session.execute(text("SET session_replication_role = 'replica'"))
            # embedding_model / embedding_dim / embedding_normalized are
            # nullable=False on the namespaced class with PYTHON-side defaults.
            # Raw SQL bypasses those defaults, so omitting them inserts NULL and
            # PostgreSQL rejects the row (NotNullViolation). The legacy `chunks`
            # table this used to target did not carry the identity columns, so
            # the omission only became a defect when the insert was repointed
            # at the active namespaced table.
            ident = getattr(_chunk_cls(), "embedding_identity", {}) or {}
            session.execute(
                text(f"""
                INSERT INTO {_chunk_table()} (chunk_id, project_id, doc_id,
                                    chunk_index, text, embedding, created_at,
                                    embedding_model, embedding_dim,
                                    embedding_normalized)
                VALUES (:chunk_id, :project_id, :doc_id, :chunk_index,
                        :text, :embedding, :created_at,
                        :embedding_model, :embedding_dim,
                        :embedding_normalized)
                """),
                {
                    "chunk_id": "dangling_1",
                    "project_id": "reconcile_a",
                    "doc_id": "doc_missing",
                    "chunk_index": 0,
                    "text": "content",
                    "embedding": vec.tolist(),
                    "created_at": now,
                    "embedding_model": ident.get("model", "test"),
                    "embedding_dim": ident.get("dim", _chunk_dim()),
                    "embedding_normalized": bool(ident.get("normalized", True)),
                },
            )
            session.execute(text("SET session_replication_role = 'origin'"))
        else:
            session.add(_chunk_cls()(
                chunk_id="dangling_1", project_id="reconcile_a",
                doc_id="doc_missing", chunk_index=0, text="content",
                embedding=vec, created_at=now,
            ))
        _commit_retry(session)


def test_reconcile_detects_misplaced_chunks(client):
    _seed_misplaced_chunks()
    resp = client.post("/v1/admin/corpus/reconcile")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is True
    assert body["execute"] is False
    assert body["mismatches_total"] == 1
    assert body["dangling_total"] == 1
    sample = body["mismatches_sample"]
    assert any(m["chunk_id"] == "misplaced_1" and
               m["current_project_id"] == "reconcile_b" and
               m["correct_project_id"] == "reconcile_a"
               for m in sample)
    _wipe("reconcile_a", "reconcile_b")


def test_reconcile_repairs_misplaced_chunks(client):
    _seed_misplaced_chunks()
    resp = client.post("/v1/admin/corpus/reconcile?execute=true")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is False
    assert body["execute"] is True
    assert body["repaired"] == 1
    assert body["mismatches_total"] == 1

    # After repair, the chunk is under the correct project.
    with SessionLocal() as session:
        chunk = session.get(_chunk_cls(), "misplaced_1")
        assert chunk is not None
        assert chunk.project_id == "reconcile_a"
        assert chunk.doc_id == "doc_a"
    _wipe("reconcile_a", "reconcile_b")


def test_reconcile_non_admin_blocked(client):
    from app.dependencies import require_api_key as real_key
    app.dependency_overrides[real_key] = lambda: {
        "user_id": "test-user", "role": "user",
    }
    try:
        resp = client.post("/v1/admin/corpus/reconcile")
    finally:
        app.dependency_overrides[real_key] = lambda: {
            "user_id": "test-admin", "role": "admin",
        }
    assert resp.status_code == 403


def test_wipe_all_keeps_training_material_project(client):
    """Shared-Postgres: wiping the corpus must not drop the GK parent row."""
    from app.core import projects as store
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    ensure_user_exists(SYSTEM_USER_ID, role="admin")
    store.get_or_create_project(
        "General Knowledge",
        user_id=SYSTEM_USER_ID,
        project_id="training_material",
        origin="system_seed",
    )
    _wipe_all()
    assert store.get_project("training_material") is not None


def test_coverage_endpoint_is_queryable(client):
    _wipe_all()
    resp = client.get("/v1/admin/corpus/coverage")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "by_status" in body
    assert body["orphans_action"] == "report_only"
    assert "tombstoned" in body
    assert "ocr_degraded" in body
    assert "office" in body
    assert ".pdf" in body["office"]["by_kind"]
    assert ".xlsx" in body["office"]["by_kind"]
    assert ".pptx" in body["office"]["by_kind"]


def test_reconcile_reports_no_mismatches_when_clean(client):
    _wipe_all()
    now = datetime.now(timezone.utc).isoformat()
    vec = np.zeros(_chunk_dim(), dtype=np.float32)
    with SessionLocal() as session:
        session.add(Project(
            id="reconcile_clean", name="reconcile_clean",
            user_id="test-admin", status="active", created_at=now,
        ))
        session.flush()
        session.add(Document(
            id="doc_clean", project_id="reconcile_clean",
            original_name="clean.pdf", doc_type="document",
            doc_role="other", size=0, uploaded_at=now,
        ))
        session.add(_chunk_cls()(
            chunk_id="clean_chunk", project_id="reconcile_clean",
            doc_id="doc_clean", chunk_index=0, text="content",
            embedding=vec, created_at=now,
        ))
        _commit_retry(session)

    resp = client.post("/v1/admin/corpus/reconcile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["mismatches_total"] == 0
    assert body["dangling_total"] == 0
    assert body["repaired"] == 0
    _wipe("reconcile_clean")
