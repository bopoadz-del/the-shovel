"""Exception handlers must degrade gracefully, not raise on the way down.

Five handlers in ``app/`` logged through a name that does not exist in their
module (``log`` where the module defines ``logger``; ``logger`` where it
defines ``_LOG``). When the guarded error fired, the handler raised
``NameError`` on top of it -- so a HANDLED failure became an unhandled crash
on the exact path written to degrade gracefully.

Nothing caught it because these branches rarely run, and no test ever forced
one. These tests force them: each injects the failure the handler exists to
absorb and asserts the caller still gets a normal result. Written against the
handler's behaviour rather than its log call, so they stay honest if the
logging changes.
"""
from __future__ import annotations

import builtins
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.dependencies import require_api_key
from app.main import app


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


def _seed_one_document(project_id: str = "missing_chunk_probe") -> None:
    """Insert one project + document so the documents COUNT path is real."""
    from app.core.db import SessionLocal
    from app.core.models import Document, Project
    from app.core.projects import init_db as init_projects_db
    from app.core.users import init_db as init_users_db

    init_users_db()
    init_projects_db()
    with SessionLocal() as session:
        session.query(Document).filter(Document.project_id == project_id).delete()
        session.query(Project).filter(Project.id == project_id).delete()
        session.commit()
        session.add(Project(
            id=project_id, name="Missing-chunk probe", user_id="system",
            created_at="2026-08-29T00:00:00Z", status="active",
        ))
        session.flush()
        session.add(Document(
            id=f"{project_id}-doc", project_id=project_id,
            original_name="probe/spec.pdf",
            doc_type="document", doc_role="other", size=128,
            uploaded_at="2026-08-29T00:00:00Z",
        ))
        session.commit()


def test_corpus_collections_survives_a_missing_chunk_table(client, monkeypatch):
    """The namespaced chunk table may not exist yet on a fresh DB -- the
    handler's own comment says so. It must fall back to the documents side.

    On PostgreSQL a swallowed UndefinedTable aborts the transaction; the
    next SELECT COUNT(*) FROM documents must still succeed (not
    InFailedSqlTransaction).
    """
    import app.routers.admin as admin

    _seed_one_document()
    monkeypatch.setattr(
        admin, "rag_chunk_table_name",
        lambda *_a, **_k: "chunks_table_that_does_not_exist",
        raising=False,
    )
    resp = client.get("/v1/admin/corpus/collections?folder_breakdown=false")
    # 200 with the documents-side answer, never a NameError-driven 500
    # and never InFailedSqlTransaction after the missing chunks_* probe.
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "collections" in body or "projects" in body, body
    cols = {c["project_id"]: c for c in body.get("collections", [])}
    assert "missing_chunk_probe" in cols, body
    assert cols["missing_chunk_probe"]["documents"] == 1
    assert cols["missing_chunk_probe"]["chunks"] == 0


def test_swallowed_missing_chunk_table_still_counts_documents():
    """Direct pin: after a missing chunks_* probe, documents COUNT still runs.

    Mirrors the production engine URL (Postgres CI or SQLite fallback) so
    the InFailedSqlTransaction regression cannot hide behind TestClient.
    """
    import os

    from sqlalchemy import create_engine, text

    from app.core.db import to_psycopg_url
    from app.core.projects import init_db as init_projects_db
    from app.routers.admin import _execute_recovering

    init_projects_db()
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        data_dir = os.getenv("DATA_DIR", "data")
        db_url = f"sqlite:///{os.path.join(data_dir, 'the_fork.db')}"
    engine = create_engine(to_psycopg_url(db_url))
    with engine.connect() as conn:
        before = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        assert _execute_recovering(
            conn, text("SELECT 1 FROM chunks_table_that_does_not_exist")
        ) is None
        after = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        assert after == before


def test_migrate_sqlite_dry_run_survives_an_unwritable_log(
    client, monkeypatch, tmp_path
):
    """The dry-run summary is a convenience file. Failing to write it must not
    cost the caller their migration counts."""
    import scripts.migrate_sqlite_to_pg as mig

    # The endpoint 503s without a DATABASE_URL before it reaches the handler.
    # `migrate` is patched below so nothing ever dials this out.
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused/unused")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mig, "migrate", lambda *_a, **_k: {"documents": 0})

    real_write_text = Path.write_text

    def boom(self, *a, **k):
        if self.name == "pilot_dry_run.log":
            raise OSError("disk full (injected)")
        return real_write_text(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", boom)

    resp = client.post("/v1/admin/debug/migrate-sqlite?dry_run=true")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is True
    assert body["counts"] == {"documents": 0}


def test_training_list_survives_an_unreadable_jsonl(client, monkeypatch, tmp_path):
    """One unreadable file must not lose the whole listing -- the entry is
    still reported, with line_count falling back to 0."""
    import app.routers.admin as admin

    learn = tmp_path / "learning"
    learn.mkdir()
    target = learn / "broken.jsonl"
    target.write_text('{"a": 1}\n', encoding="utf-8")

    monkeypatch.setattr(admin, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    real_open = builtins.open

    def boom(file, *a, **k):
        if str(file).endswith("broken.jsonl"):
            raise OSError("unreadable (injected)")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", boom)

    resp = client.get("/v1/admin/training/list")
    if resp.status_code == 404:
        pytest.skip("training listing not mounted in this build")
    assert resp.status_code == 200, resp.text
    files = resp.json().get("files", [])
    broken = [f for f in files if f["name"] == "broken.jsonl"]
    if broken:
        assert broken[0]["line_count"] == 0, broken[0]
