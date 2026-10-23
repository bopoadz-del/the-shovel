"""R4: index stamps ledger columns; /health counts chunks from the table."""
from __future__ import annotations

import importlib
import os

from app.core.ingest_status import EXTRACTOR_VERSION, INDEXED


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects
    from app.core.rag.vector_store import reset_store_cache
    from app.core.rag.embeddings import reset_embedder_cache

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    reset_store_cache()
    reset_embedder_cache()
    import app.core.doc_index as doc_index_mod
    importlib.reload(doc_index_mod)
    return importlib.reload(projects), users_mod


def test_index_document_stamps_indexed_on_reindex(monkeypatch, tmp_path):
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    users.ensure_user_exists("u1")
    projects.create_project(name="P", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]
    path = tmp_path / "note.txt"
    path.write_text("alpha beta gamma " * 40, encoding="utf-8")
    doc = projects.add_document(
        project_id=proj["id"],
        original_name="note.txt",
        stored_as="note.txt",
        file_path=str(path),
        size=path.stat().st_size,
        content_sha256="aa" * 32,
    )
    from app.core import doc_index

    result = doc_index.index_document(
        proj["id"], doc["id"], stamp_as_indexed=True,
    )
    assert result.get("status") == "ok"
    stamped = projects.get_document(doc["id"])
    assert stamped["ingest_status"] == INDEXED
    assert stamped["chunk_count"] > 0
    assert stamped["extractor_version"] == EXTRACTOR_VERSION


def test_backfill_rewrites_zero_column_when_chunks_exist(monkeypatch, tmp_path):
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    users.ensure_user_exists("u1")
    projects.create_project(name="P", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]
    path = tmp_path / "note.txt"
    path.write_text("one two three four five " * 30, encoding="utf-8")
    doc = projects.add_document(
        project_id=proj["id"],
        original_name="note.txt",
        stored_as="note.txt",
        file_path=str(path),
        size=path.stat().st_size,
        content_sha256="bb" * 32,
    )
    from app.core import doc_index

    doc_index.index_document(proj["id"], doc["id"], stamp_as_indexed=True)
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        row = session.get(Document, doc["id"])
        row.chunk_count = 0
        session.commit()
    assert projects.get_document(doc["id"])["chunk_count"] == 0

    report = projects.backfill_chunk_counts_from_table(
        project_id=proj["id"], apply=True,
    )
    assert report["stale"] >= 1
    assert report["updated"] >= 1
    assert projects.get_document(doc["id"])["chunk_count"] > 0


def test_health_corpus_uses_chunk_table_not_column(monkeypatch, tmp_path):
    projects, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    users.ensure_user_exists("u1")
    projects.create_project(name="P", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]
    path = tmp_path / "note.txt"
    path.write_text("health corpus chunk query " * 20, encoding="utf-8")
    doc = projects.add_document(
        project_id=proj["id"],
        original_name="note.txt",
        stored_as="note.txt",
        file_path=str(path),
        size=path.stat().st_size,
        content_sha256="cc" * 32,
    )
    from app.core import doc_index
    from app.core.rag.vector_store import get_store

    doc_index.index_document(proj["id"], doc["id"], stamp_as_indexed=True)
    store = get_store()
    live = store.count(proj["id"])

    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        row = session.get(Document, doc["id"])
        row.chunk_count = 999
        session.commit()
    assert projects.get_document(doc["id"])["chunk_count"] == 999

    from app.core.health_probes import probe_corpus_chunks, reset_corpus_chunk_cache

    reset_corpus_chunk_cache()
    probe = probe_corpus_chunks()
    assert probe["source"] == "exact"
    assert probe["error"] is None
    assert probe["chunks"] == live
    assert probe["chunks"] != 999


def test_health_payload_exposes_corpus_query(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    from fastapi.testclient import TestClient
    from app.main import app

    from app.core.health_probes import reset_corpus_chunk_cache

    with TestClient(app) as client:
        reset_corpus_chunk_cache()
        data = client.get("/health").json()
    assert "corpus" in data
    assert data["corpus"]["source"] in {"exact", "estimate"}
