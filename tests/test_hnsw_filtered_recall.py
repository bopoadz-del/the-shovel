"""Filtered HNSW search must not silently return fewer rows than it should.

pgvector applies ``WHERE project_id = ...`` to whatever the HNSW index hands
back, rather than searching within the project. On a multi-tenant table the
index picks its nearest neighbours across EVERY project and the filter then
discards them, so a project can get nothing at all.

Measured on live 2026-08-23 — project 184921da, 13,922 chunks of a
172,809-chunk table, query "contract agreement parties"::

    Index Scan using chunks_v2_embedding_hnsw  (actual rows=0)
      Filter: ((project_id)::text = '184921da'::text)
      Rows Removed by Filter: 39

Zero rows; the same query with the index disabled returned 10. And it is not
all-or-nothing — "effective date of the agreement" returned 3 of 10, which is
the dangerous case: partial recall looks like a working search.

    setting            'contract agreement parties'   'effective date ...'
    default                      0 rows                      3 rows
    ef_search=200                5 rows                      8 rows
    ef_search=1000              10 rows (1.6-2.8s)          10 rows
    iterative_scan              10 rows (191-459ms)         10 rows

These tests pin the mechanism, not the database: they assert the session is
told to scan iteratively, and that a backend which cannot do so degrades
instead of breaking retrieval.
"""
from __future__ import annotations

import pytest

from app.core.rag import vector_store as vs


class _FakeSession:
    def __init__(self, fail: bool = False):
        self.statements: list[str] = []
        self._fail = fail

    def execute(self, stmt):
        sql = str(stmt)
        self.statements.append(sql)
        if self._fail:
            raise RuntimeError('unrecognized configuration parameter "hnsw.iterative_scan"')
        return None


@pytest.fixture(autouse=True)
def _reset_latch():
    """The support flag is a module-level latch; leaking it across tests would
    make results depend on execution order."""
    vs._ITERATIVE_SCAN_SUPPORTED = None
    yield
    vs._ITERATIVE_SCAN_SUPPORTED = None


def test_search_session_is_told_to_scan_iteratively():
    """THE FIX. Without this statement the project filter can leave 0 rows."""
    session = _FakeSession()
    vs._enable_iterative_scan(session)

    assert len(session.statements) == 1
    sql = session.statements[0].lower()
    assert "hnsw.iterative_scan" in sql
    assert "set local" in sql, "must be transaction-scoped, not session-wide"
    assert vs._ITERATIVE_SCAN_SUPPORTED is True


def test_old_pgvector_degrades_instead_of_breaking_retrieval():
    """The GUC does not exist before pgvector 0.8. Retrieval must still work —
    with the old recall — rather than raising into the caller."""
    session = _FakeSession(fail=True)
    vs._enable_iterative_scan(session)          # must not raise
    assert vs._ITERATIVE_SCAN_SUPPORTED is False


def test_an_unsupported_backend_is_not_retried_every_search():
    """Latched: retrying a statement that always fails costs latency on every
    single query."""
    first = _FakeSession(fail=True)
    vs._enable_iterative_scan(first)
    assert len(first.statements) == 1

    second = _FakeSession(fail=True)
    vs._enable_iterative_scan(second)
    assert second.statements == [], "should not re-attempt after the latch"


def test_a_typo_in_the_env_knob_cannot_reach_the_database(monkeypatch):
    """Same rule the RAG ranking knobs follow: a bad value disables the knob
    rather than being sent to Postgres to fail per-query."""
    monkeypatch.setattr(vs, "_ITERATIVE_SCAN_MODE", "relaxed-order")  # hyphen typo
    session = _FakeSession()
    vs._enable_iterative_scan(session)
    assert session.statements == []
    assert vs._ITERATIVE_SCAN_SUPPORTED is False


@pytest.mark.parametrize("mode", ["relaxed_order", "strict_order", "off"])
def test_documented_modes_are_accepted(monkeypatch, mode):
    monkeypatch.setattr(vs, "_ITERATIVE_SCAN_MODE", mode)
    session = _FakeSession()
    vs._enable_iterative_scan(session)
    assert mode in session.statements[0]


def test_empty_mode_disables_the_statement(monkeypatch):
    """An operator can turn it off entirely without editing code."""
    monkeypatch.setattr(vs, "_ITERATIVE_SCAN_MODE", "")
    session = _FakeSession()
    vs._enable_iterative_scan(session)
    assert session.statements == []
