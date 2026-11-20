"""The three lookup failure lines never carry terms taken from the question.

Rule: names and query text never in logs. ``chunks_containing_all`` (the
needles), ``identifier_search`` (the identifiers) and ``chunks_following``
log a failure. Each line names its terms by count and a short hash
(``_terms_ref``) and the database error by its classes and SQLSTATE
(``_error_ref``). SQLAlchemy renders the bound LIKE parameters, which hold
the terms, into an error's message.

Every input here is built by the test; two inputs per decision.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import uuid

import pytest
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.core.rag import vector_store as vs
from app.core.rag.vector_store import VectorStore, _terms_ref

_LOGGER = "app.core.rag.vector_store"


class _PgDriverError(Exception):
    sqlstate = "42P01"


class _RaisingSession:
    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **kw):
        raise self._exc

    def rollback(self):
        return None


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp(prefix="termlog_")
    db_path = os.path.join(tmpdir, f"v_{uuid.uuid4().hex}.db")
    vs.reset_store_cache()
    s = VectorStore(db_path=db_path, dim=8)
    yield s
    s.close()
    try:
        os.remove(db_path)
        os.rmdir(tmpdir)
    except OSError:
        pass


def _raise_on_execute(store, monkeypatch, exc):
    monkeypatch.setattr(store, "_session_factory",
                        lambda: (lambda: _RaisingSession(exc)))
    monkeypatch.setattr(store, "_docs_visibility_ready", lambda s: False)
    monkeypatch.setattr(store, "_hidden_doc_ids_among",
                        lambda *a, **kw: set())


def _bound_error(kind, terms):
    """An error whose message carries the bound LIKE parameters, as
    SQLAlchemy renders them."""
    params = {f"n{i}": f"%{t}%" for i, t in enumerate(terms)}
    if kind == "sqlite":
        return OperationalError("SELECT 1 WHERE text LIKE :n0", params,
                                sqlite3.OperationalError("database is locked"))
    return ProgrammingError("SELECT 1 WHERE text LIKE :n0", params,
                            _PgDriverError("relation does not exist"))


def _assert_clean(lines, terms):
    assert lines, "the failure line was not written"
    joined = "\n".join(lines)
    for t in terms:
        assert t.lower() not in joined.lower(), f"term {t!r} in a log line"


# ── _terms_ref ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("terms", [
    ["zebrawood", "lintel"], ["QS-7731", "rev c", "annex 9"],
])
def test_terms_ref_carries_count_and_hash_never_terms(terms):
    ref = _terms_ref(terms)
    assert f"count={len(terms)}" in ref
    assert "sha256=" in ref
    _assert_clean([ref], terms)


@pytest.mark.parametrize("a,b", [
    (["zebrawood", "lintel"], ["lintel", "zebrawood"]),
    # Same count, one letter apart: the hash, not the count, splits them.
    (["qs-7731"], ["qs-7732"]),
])
def test_terms_ref_joins_equal_lists_and_splits_different_ones(a, b):
    assert _terms_ref(a) == _terms_ref(list(a))
    assert _terms_ref(a) != _terms_ref(b)
    assert _terms_ref(None) == _terms_ref([])


# ── the three failure lines, on a store built here ────────────────────────

@pytest.mark.parametrize("kind,needles", [
    ("sqlite", ["zebrawood lintel", "tolerance band"]),
    ("postgres", ["quarrystone coping", "parapet"]),
])
def test_chunks_containing_all_failure_has_no_needles(
        store, monkeypatch, caplog, kind, needles):
    exc = _bound_error(kind, needles)
    assert any(n in str(exc) for n in needles)  # the raw error carries them
    _raise_on_execute(store, monkeypatch, exc)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert store.chunks_containing_all("p", needles) == []
    lines = [m for m in caplog.messages if "chunks_containing_all failed" in m]
    _assert_clean(lines, needles)
    assert _terms_ref(needles) in lines[0]
    assert type(exc).__name__ in lines[0]


@pytest.mark.parametrize("kind,identifiers", [
    ("sqlite", ["QS-7731", "Annex 9"]),
    ("postgres", ["VO 4471", "D549.2"]),
])
def test_identifier_search_failure_has_no_identifiers(
        store, monkeypatch, caplog, kind, identifiers):
    exc = _bound_error(kind, [i.lower() for i in identifiers])
    if kind == "postgres":
        # identifier_search catches OperationalError; wrap the driver error
        # the same way PostgreSQL's connection-level failures arrive.
        exc = OperationalError("SELECT 1 WHERE text LIKE :p0",
                               exc.params, _PgDriverError("server closed"))
    _raise_on_execute(store, monkeypatch, exc)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert store.identifier_search("p", identifiers) == []
    lines = [m for m in caplog.messages if "identifier_search failed" in m]
    _assert_clean(lines, identifiers)
    assert _terms_ref(identifiers) in lines[0]
    if kind == "postgres":
        assert "sqlstate=42P01" in lines[0]


@pytest.mark.parametrize("kind,secret", [
    ("sqlite", "zebrawood lintel"),
    ("postgres", "quarrystone coping"),
])
def test_chunks_following_failure_prints_error_ref_not_message(
        store, monkeypatch, caplog, kind, secret):
    exc = _bound_error(kind, [secret])
    _raise_on_execute(store, monkeypatch, exc)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert store.chunks_following("p", [("doc_x", 3)]) == []
    lines = [m for m in caplog.messages if "chunks_following failed" in m]
    _assert_clean(lines, [secret])
    # Anchors are document ids and chunk indexes, not question text.
    assert "doc_x" in lines[0]
    assert type(exc).__name__ in lines[0]


@pytest.mark.parametrize("project_id,opens", [
    ("", 0),
    (None, 0),
    ("proj_b", 1),
])
def test_chunks_following_reads_the_store_only_with_a_project(
        store, monkeypatch, project_id, opens):
    """No project id, no read: the lookup must not open a session (an
    unscoped read could reach another project's rows). With a project id
    the same anchors do reach the store."""
    opened = []

    def _factory():
        opened.append(1)
        return _RaisingSession(_PgDriverError("unreachable"))

    monkeypatch.setattr(store, "_session_factory", lambda: _factory)
    monkeypatch.setattr(store, "_hidden_doc_ids_among",
                        lambda *a, **kw: set())
    assert store.chunks_following(project_id, [("doc_q", 2)]) == []
    assert len(opened) == opens
