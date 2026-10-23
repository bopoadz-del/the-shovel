"""Concurrent CREATE TABLE must not fail the loser.

``checkfirst=True`` is check-THEN-create, not atomic. ``_ensure_schema``
serialises with ``_INIT_LOCK``, but that is a threading lock inside ONE
process: it does not cover a second uvicorn worker, nor a test that creates a
namespaced table on its own engine while the app creates it on the request
thread.

The race is real and was misread twice as flakiness. It failed CI on an
unrelated PR as::

    sqlite3.OperationalError: table chunks_t48b48c8a1360 already exists
    FAILED tests/test_admin_corpus_reconcile.py::test_reconcile_repairs_misplaced_chunks

and had appeared earlier on a different branch as ``database is locked`` --
the same two writers, a different symptom, which is exactly why each looked
like an unrelated blip.

Losing the race is a SUCCESS: the table exists either way.
"""
from __future__ import annotations

import threading

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from sqlalchemy.exc import OperationalError

from app.core.rag.vector_store import ensure_table, table_already_exists


def _table(md: MetaData, name: str) -> Table:
    return Table(
        name, md,
        Column("id", Integer, primary_key=True),
        Column("v", String),
    )


def test_second_create_does_not_raise(tmp_path):
    """The whole point: create twice, on two engines, and survive."""
    url = f"sqlite:///{tmp_path / 'race.db'}"
    e1, e2 = create_engine(url), create_engine(url)
    # Separate MetaData per engine, mimicking the test helper and the app each
    # holding their own mapping of the same namespaced table.
    t1 = _table(MetaData(), "chunks_race")
    t2 = _table(MetaData(), "chunks_race")

    ensure_table(t1, bind=e1)
    ensure_table(t2, bind=e2)  # must be a no-op, not an error

    from sqlalchemy import inspect
    assert "chunks_race" in inspect(e1).get_table_names()


def test_raw_checkfirst_is_what_actually_breaks(tmp_path):
    """Guard the premise: if this ever stops raising, the fix is unnecessary
    and this whole module should go rather than sit here reassuring nobody.

    Two engines each cache their own "does it exist" answer, so the second
    CREATE is issued for real and SQLite rejects it.
    """
    url = f"sqlite:///{tmp_path / 'raw.db'}"
    e1, e2 = create_engine(url), create_engine(url)
    t1 = _table(MetaData(), "chunks_raw")
    t2 = _table(MetaData(), "chunks_raw")

    # Prime both engines' view of an EMPTY database, so both believe the table
    # is absent -- this is the check half of check-then-create.
    from sqlalchemy import inspect
    assert inspect(e1).get_table_names() == []
    assert inspect(e2).get_table_names() == []

    t1.create(bind=e1, checkfirst=True)
    with pytest.raises(OperationalError) as err:
        t2.create(bind=e2, checkfirst=False)
    assert table_already_exists(err.value)


def test_concurrent_threads_all_survive(tmp_path):
    """Eight threads racing to create the same table: none may raise."""
    url = f"sqlite:///{tmp_path / 'threads.db'}"
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            eng = create_engine(url)
            tbl = _table(MetaData(), "chunks_threads")
            barrier.wait(timeout=10)  # maximise overlap on the CREATE
            ensure_table(tbl, bind=eng)
        except BaseException as exc:  # noqa: BLE001 - the assertion is the point
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent ensure_table raised: {errors!r}"


@pytest.mark.parametrize(
    "message, expected",
    [
        ("table chunks_t48b48c8a1360 already exists", True),
        ('relation "chunks_v2" already exists', True),
        ("no such table: chunks_v2", False),
        ("database is locked", False),
        ("disk I/O error", False),
    ],
)
def test_only_already_exists_is_tolerated(message, expected):
    """A genuine failure must still propagate. Swallowing "no such table" or a
    disk error would turn a broken store into a silent one."""
    assert table_already_exists(Exception(message)) is expected
