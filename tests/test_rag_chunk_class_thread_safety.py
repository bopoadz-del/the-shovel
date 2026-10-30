"""make_rag_chunk_class / ensure_table under concurrent callers.

At boot the background ``_warm_models`` thread and the first request both
build the same namespace's chunk class and table. Without
``RAG_CHUNK_SCHEMA_LOCK`` the second builder raised "Table ... is already
defined", walked a half-registered mapper (UnmappedClassError), or lost the
CREATE race. These tests hammer the real functions from N threads.
"""
from __future__ import annotations

import sys
import threading
import uuid

import pytest

from app.core import models
from app.core.db import engine
from app.core.models import EMBEDDING_DIM, make_rag_chunk_class
from app.core.rag.vector_store import ensure_table

N_THREADS = 16


@pytest.fixture(autouse=True)
def _tight_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(old)


def _hammer(fn, n=N_THREADS):
    barrier = threading.Barrier(n)
    results, errors = [], []

    def run():
        barrier.wait()
        try:
            results.append(fn())
        except BaseException as exc:  # noqa: BLE001 — collected and asserted
            errors.append(exc)

    threads = [threading.Thread(target=run, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "builder threads deadlocked"
    return results, errors


@pytest.mark.parametrize("round_", range(5))
def test_concurrent_class_builds_return_one_class(round_):
    ns = "tlock" + uuid.uuid4().hex[:10]
    results, errors = _hammer(lambda: make_rag_chunk_class(ns, EMBEDDING_DIM, "test"))
    assert errors == []
    assert len(results) == N_THREADS
    assert len({id(c) for c in results}) == 1
    assert models.rag_chunk_class_for(ns) is results[0]


@pytest.mark.parametrize("round_", range(3))
def test_concurrent_build_and_ensure_creates_table_once(round_):
    from sqlalchemy import inspect

    ns = "tlock" + uuid.uuid4().hex[:10]

    def build_and_ensure():
        cls = make_rag_chunk_class(ns, EMBEDDING_DIM, "test")
        ensure_table(cls.__table__, bind=engine)
        return cls

    results, errors = _hammer(build_and_ensure)
    try:
        assert errors == []
        assert len({id(c) for c in results}) == 1
        with engine.connect() as conn:
            assert results[0].__tablename__ in inspect(conn).get_table_names()
    finally:
        if results:
            results[0].__table__.drop(bind=engine, checkfirst=True)
