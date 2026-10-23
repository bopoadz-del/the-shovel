"""Deciding whether a project is small reads no table page: it asks the
planner for its row estimate, and the estimate sorts projects into the
right size class. PostgreSQL only; synthetic rows."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import event, text

pytestmark = pytest.mark.skipif(
    os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() not in ("1", "true", "yes"),
    reason="planner estimates are a PostgreSQL property (test-postgres job)",
)


@pytest.fixture
def probe_table():
    from app.core.db import get_engine
    from app.core.rag import vector_store

    table = "qp_size_probe"
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text(f"CREATE TABLE {table} (chunk_id text PRIMARY KEY, project_id text, "
                          "doc_id text, body text)"))
        conn.execute(text(
            f"INSERT INTO {table} SELECT 'b' || g, 'big', 'd' || (g % 50), repeat('x', 1500) "
            "FROM generate_series(1, :n) g"), {"n": vector_store._EXACT_SCAN_MAX_ROWS * 3})
        conn.execute(text(
            f"INSERT INTO {table} SELECT 's' || g, 'small', 'd1', repeat('x', 1500) "
            "FROM generate_series(1, 40) g"))
        conn.execute(text(f"CREATE INDEX ON {table} (project_id, doc_id)"))
        conn.execute(text(f"ANALYZE {table}"))  # statistics, but no VACUUM
    yield engine, table
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


def test_the_estimate_sorts_projects_by_size_and_reads_no_table(probe_table):
    from sqlalchemy.orm import Session

    from app.core.rag import vector_store

    engine, table = probe_table
    store = SimpleNamespace(_table_name=table)
    sent = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        sent.append(statement)

    event.listen(engine, "before_cursor_execute", _record)
    try:
        memo = vector_store.start_turn_memo()
        try:
            with Session(engine) as session:
                small = vector_store.VectorStore._project_is_small(store, session, "small")
                big = vector_store.VectorStore._project_is_small(store, session, "big")
                again = vector_store.VectorStore._project_is_small(store, session, "big")
        finally:
            vector_store.end_turn_memo(memo)
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    assert small is True and big is False and again is False
    on_table = [s for s in sent if table in s]
    assert len(on_table) == 2  # one per project; the repeat came from the turn memo
    assert all(s.lstrip().upper().startswith("EXPLAIN") for s in on_table), on_table
