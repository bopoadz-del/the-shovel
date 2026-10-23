"""Chunk embeddings live in the heap row, not in TOAST.

pgvector's ``vector`` defaults to EXTERNAL storage; a chunk row passes the
TOAST threshold and the embedding is moved out of line, so every distance the
executor computes outside the HNSW index costs a TOAST-index walk per row
(measured: one exact search over 400 chunks touched 20 MB). The store keeps
the column ``STORAGE MAIN``.
"""
from __future__ import annotations

import os

import pytest

from app.core.rag import vector_store

_ON_POSTGRES = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")


class _Boom:
    def begin(self):
        raise RuntimeError("database unavailable")


def test_the_storage_hint_never_breaks_startup(monkeypatch):
    warned = []
    monkeypatch.setattr(vector_store.logger, "warning", lambda *a, **k: warned.append(a[0]))
    vector_store._ensure_embedding_inline(_Boom(), "chunks_x")  # must not raise
    assert warned and "inline storage" in warned[0]


class _Conn:
    def __init__(self, storage):
        self.storage, self.sql = storage, []

    def execute(self, stmt, params=None):
        self.sql.append(str(stmt))

        class _R:
            def __init__(self, v):
                self.v = v

            def scalar(self):
                return self.v

        return _R(self.storage)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Eng:
    def __init__(self, storage):
        self.conn = _Conn(storage)

    def begin(self):
        return self.conn


def test_an_external_embedding_is_set_to_main_and_main_is_left_alone():
    eng = _Eng("e")
    vector_store._ensure_embedding_inline(eng, "chunks_x")
    assert any("SET STORAGE MAIN" in s for s in eng.conn.sql)
    assert any("lock_timeout" in s for s in eng.conn.sql)

    eng = _Eng("m")
    vector_store._ensure_embedding_inline(eng, "chunks_x")
    assert not any("ALTER TABLE" in s for s in eng.conn.sql)


@pytest.mark.skipif(not _ON_POSTGRES, reason="column storage is a PostgreSQL property (test-postgres job)")
def test_the_store_keeps_the_embedding_in_the_heap_row():
    from sqlalchemy import text

    from app.core.db import get_engine
    from app.core.rag.embeddings import get_embedder

    table = vector_store.get_store(dim=get_embedder().dim)._table_name
    with get_engine().connect() as conn:
        storage = conn.execute(text(
            "SELECT a.attstorage FROM pg_attribute a "
            "WHERE a.attrelid = CAST(:t AS regclass) AND a.attname = 'embedding'"
        ), {"t": table}).scalar()
    assert storage == "m"


@pytest.mark.skipif(not _ON_POSTGRES, reason="column storage is a PostgreSQL property (test-postgres job)")
def test_the_store_keeps_the_tsvector_in_the_heap_row():
    from sqlalchemy import text

    from app.core.db import get_engine
    from app.core.rag.embeddings import get_embedder

    table = vector_store.get_store(dim=get_embedder().dim)._table_name
    with get_engine().connect() as conn:
        opts, storage = conn.execute(text(
            "SELECT c.reloptions, a.attstorage FROM pg_class c JOIN pg_attribute a "
            "ON a.attrelid = c.oid AND a.attname = 'text_search' WHERE c.oid = CAST(:t AS regclass)"
        ), {"t": table}).first()
    assert storage == "m"
    assert f"toast_tuple_target={vector_store._CHUNK_TOAST_TUPLE_TARGET}" in (opts or [])


def test_the_tsvector_hint_never_breaks_startup(monkeypatch):
    warned = []
    monkeypatch.setattr(vector_store.logger, "warning", lambda *a, **k: warned.append(a[0]))
    vector_store._ensure_tsvector_inline(_Boom(), "chunks_x")  # must not raise
    assert warned and "text_search" in warned[0]
