"""The lexical leg's log lines never carry the query's words.

Rule: names and query text never in logs. The four lexical-leg lines
(empty leg twice, the PostgreSQL and the SQLite failure) name the query by
its length and a short hash (``_query_ref``) and the database error by its
classes and SQLSTATE (``_error_ref``): SQLAlchemy renders the bound
parameters into an error's message, and FTS5 echoes the MATCH text.

Every input here is built by the test; two inputs per decision.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import uuid

import numpy as np
import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.core.rag import vector_store as vs
from app.core.rag.vector_store import VectorStore, _error_ref, _query_ref

_LOGGER = "app.core.rag.vector_store"
_DIM = 8


def _words(query: str):
    return [w for w in query.split() if len(w) > 3]


def _assert_no_words(messages, query):
    joined = "\n".join(messages)
    for w in _words(query):
        assert w not in joined, f"query word {w!r} in a log line"


# ── _query_ref ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query", [
    "zebrawood lintel tolerance", "quarrystone parapet coping",
])
def test_query_ref_carries_length_and_hash_never_words(query):
    ref = _query_ref(query)
    assert f"chars={len(query)}" in ref
    assert "sha256=" in ref
    _assert_no_words([ref], query)


def test_query_ref_joins_equal_queries_and_splits_different_ones():
    a, b = "zebrawood lintel tolerance", "quarrystone parapet coping"
    assert _query_ref(a) == _query_ref(a)
    assert _query_ref(a) != _query_ref(b)
    assert _query_ref(None) == _query_ref("")


# ── _error_ref ────────────────────────────────────────────────────────────

class _PgDriverError(Exception):
    sqlstate = "42703"


@pytest.mark.parametrize("query,exc,expect", [
    (
        "zebrawood lintel tolerance",
        OperationalError(
            "SELECT 1 WHERE x MATCH :q", {"q": "zebrawood lintel tolerance"},
            sqlite3.OperationalError('fts5: syntax error near "zebrawood"'),
        ),
        ["OperationalError"],
    ),
    (
        "quarrystone parapet coping",
        ProgrammingError(
            "SELECT 1 WHERE t @@ to_tsquery(:q)",
            {"q": "quarrystone parapet coping"},
            _PgDriverError('column "text_search" does not exist quarrystone'),
        ),
        ["ProgrammingError", "_PgDriverError", "sqlstate=42703"],
    ),
])
def test_error_ref_names_classes_never_the_message(query, exc, expect):
    # The message itself carries the query (bound parameters / echoed MATCH).
    assert any(w in str(exc) for w in _words(query))
    ref = _error_ref(exc)
    for part in expect:
        assert part in ref
    _assert_no_words([ref], query)


# ── the four log lines, on a store built here ─────────────────────────────

@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp(prefix="lexlog_")
    db_path = os.path.join(tmpdir, f"v_{uuid.uuid4().hex}.db")
    vs.reset_store_cache()
    s = VectorStore(db_path=db_path, dim=_DIM)
    yield s
    s.close()
    try:
        os.remove(db_path)
        os.rmdir(tmpdir)
    except OSError:
        pass


def _vec(seed: int):
    v = np.random.default_rng(seed).random(_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("query", [
    "zebrawood lintel tolerance", "quarrystone parapet coping",
])
def test_empty_project_warning_has_no_query_words(store, monkeypatch, caplog, query):
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    token = vs.start_turn_memo()
    try:
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            store.search(f"empty_{uuid.uuid4().hex[:6]}", _vec(1), k=3,
                         query_text=query)
    finally:
        vs.end_turn_memo(token)
    lines = [m for m in caplog.messages if "lexical leg empty" in m]
    assert lines, "the empty-leg line was not written"
    _assert_no_words(lines, query)
    assert any(_query_ref(query) in m for m in lines)


@pytest.mark.parametrize("query", [
    "zebrawood lintel tolerance", "quarrystone parapet coping",
])
def test_no_lexeme_match_info_has_no_query_words(store, monkeypatch, caplog, query):
    pid = f"proj_{uuid.uuid4().hex[:6]}"
    store.upsert_chunks(pid, "doc_a", ["anchor bolt embedment depth"],
                        np.stack([_vec(2)]))
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    token = vs.start_turn_memo()
    try:
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            store.search(pid, _vec(3), k=3, query_text=query)
    finally:
        vs.end_turn_memo(token)
    lines = [m for m in caplog.messages if "lexical leg empty" in m]
    assert lines, "the no-lexeme line was not written"
    _assert_no_words(lines, query)
    assert any(_query_ref(query) in m for m in lines)


class _RaisingSession:
    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **kw):
        raise self._exc


def _bound_error(cls, query, driver_exc):
    return cls("SELECT 1 WHERE x MATCH :q", {"q": query}, driver_exc)


@pytest.mark.parametrize("query", [
    "zebrawood lintel tolerance", "quarrystone parapet coping",
])
def test_sqlite_match_failure_has_no_query_words(store, monkeypatch, caplog, query):
    exc = _bound_error(
        OperationalError, query,
        sqlite3.OperationalError(f'fts5: syntax error near "{query}"'),
    )
    monkeypatch.setattr(store, "_session_factory",
                        lambda: (lambda: _RaisingSession(exc)))
    monkeypatch.setattr(store, "_docs_visibility_ready", lambda s: False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert store._bm25_sqlite("p", query, 3) == []
    lines = [r.getMessage() for r in caplog.records if "FTS5 MATCH failed" in r.getMessage()]
    assert lines, "the FTS5 failure line was not written"
    _assert_no_words(lines, query)
    assert all(r.exc_info is None for r in caplog.records)


@pytest.mark.parametrize("query", [
    "zebrawood lintel tolerance", "quarrystone parapet coping",
])
def test_postgres_leg_failure_has_no_query_words(store, monkeypatch, caplog, query):
    exc = _bound_error(ProgrammingError, query, _PgDriverError(query))

    def _raise(*a, **kw):
        raise exc

    monkeypatch.setattr(store, "_session_factory",
                        lambda: (lambda: _RaisingSession(exc)))
    monkeypatch.setattr(store, "_docs_visibility_ready", lambda s: False)
    monkeypatch.setattr(store, "_bm25_postgres_rows", _raise)
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        assert store._bm25_postgres("p", query, 3) == []
    recs = [r for r in caplog.records if "SEMANTIC-ONLY" in r.getMessage()]
    assert recs, "the PostgreSQL failure line was not written"
    assert recs[0].levelno == logging.ERROR
    _assert_no_words([r.getMessage() for r in recs], query)
    # A traceback would print the error's message, which carries the query.
    assert all(r.exc_info is None for r in recs)
    assert "sqlstate=42703" in recs[0].getMessage()
