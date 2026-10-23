"""Regression: the chunks.project_id btree index must exist even when the
`chunks` table predates the index declaration.

Bug: `RagChunk.__table__.create(checkfirst=True)` skips ALL creation (indexes
included) when the table already exists. A prod `chunks` table created before
`idx_chunks_project` was declared therefore never got the btree, so
COUNT/filter-by-project seq-scanned the whole table (~11s on the master corpus,
while pgvector search stayed fast). _ensure_schema must create the indexes
explicitly and idempotently.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import sqlalchemy as sa


def _index_names(url: str, table: str = "chunks") -> set:
    eng = sa.create_engine(url)
    try:
        insp = sa.inspect(eng)
        if table not in insp.get_table_names():
            return set()
        return {ix["name"] for ix in insp.get_indexes(table)}
    finally:
        eng.dispose()


def _dispose(url: str) -> None:
    """Release every engine that may hold the sqlite file open (Windows unlink)."""
    from app.core.rag import vector_store as vs
    from app.core import db as _db
    from app.core.models import RagChunk
    try:
        _db._engine_for_url(url).dispose()
    except Exception:
        pass
    vs._INITIALIZED_NAMESPACES.discard((url, RagChunk.__tablename__))


def test_ensure_schema_creates_project_index_on_fresh_db():
    from app.core.rag import vector_store as vs
    from app.core.models import RagChunk
    d = tempfile.mkdtemp()
    url = f"sqlite:///{os.path.join(d, 'fresh.db')}"
    try:
        vs._INITIALIZED_NAMESPACES.discard((url, RagChunk.__tablename__))
        vs._ensure_schema(url, RagChunk)
        names = _index_names(url)
        assert "idx_chunks_project" in names
        assert "idx_chunks_doc" in names
    finally:
        _dispose(url)
        shutil.rmtree(d, ignore_errors=True)


def test_index_created_on_preexisting_table_without_it():
    """The real prod case: the table already exists WITHOUT the index."""
    from app.core.rag import vector_store as vs
    from app.core.models import RagChunk
    d = tempfile.mkdtemp()
    url = f"sqlite:///{os.path.join(d, 'legacy.db')}"
    try:
        eng = sa.create_engine(url)
        with eng.begin() as conn:
            conn.execute(sa.text(
                "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, project_id TEXT, "
                "doc_id TEXT, chunk_index INTEGER, text TEXT, embedding BLOB, created_at TEXT)"
            ))
        eng.dispose()
        assert "idx_chunks_project" not in _index_names(url)

        vs._INITIALIZED_NAMESPACES.discard((url, RagChunk.__tablename__))
        vs._ensure_schema(url, RagChunk)

        assert "idx_chunks_project" in _index_names(url), \
            "index must be created on a pre-existing table"

        # Idempotent: a second pass must not raise.
        vs._INITIALIZED_NAMESPACES.discard((url, RagChunk.__tablename__))
        vs._ensure_schema(url, RagChunk)
    finally:
        _dispose(url)
        shutil.rmtree(d, ignore_errors=True)


# ── pgvector HNSW ANN index (semantic search must not seq-scan) ──────────────


class _RecConn:
    def __init__(self, sink):
        self._sink = sink

    def execute(self, stmt):
        self._sink.append(str(stmt))


class _RecCtx:
    def __init__(self, sink):
        self._sink = sink

    def __enter__(self):
        return _RecConn(self._sink)

    def __exit__(self, *a):
        return False


class _RecEngine:
    def __init__(self):
        self.executed = []

    def begin(self):
        return _RecCtx(self.executed)


def test_ensure_hnsw_index_issues_create_using_hnsw():
    """The retriever ranks by ``embedding <=> q``; without an HNSW index that is
    a full sequential scan. _ensure_hnsw_index must issue the CREATE INDEX."""
    from app.core.rag import vector_store as vs

    eng = _RecEngine()
    vs._ensure_hnsw_index(eng, "chunks_v2")
    assert any(
        "chunks_v2_embedding_hnsw" in s
        and "using hnsw" in s.lower()
        and "vector_cosine_ops" in s
        and "if not exists" in s.lower()
        for s in eng.executed
    ), eng.executed


def test_ensure_hnsw_index_never_raises_on_db_error():
    """A missing index degrades to full-scan; it must NOT crash startup."""
    from app.core.rag import vector_store as vs

    class _BadEngine:
        def begin(self):
            raise RuntimeError("connection dropped mid-DDL")

    # Must swallow the error (logged, not raised).
    vs._ensure_hnsw_index(_BadEngine(), "chunks_v2")
