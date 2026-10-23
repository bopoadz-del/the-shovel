"""Schema init must follow the DATABASE, not a one-shot global flag.

THE BUG: each store module carried a module-level `_initialized = False`, and
`_ensure_db()` skipped `init_db()` once it was True. But the database is
per-DATA_DIR / per-DATABASE_URL. So after init ran against one database,
every OTHER database silently got NO schema:

    sqlite3.OperationalError: no such table: projects
    [SQL: INSERT INTO projects (id, name, ...) VALUES (?, ?, ...)]

Surfaced as an order-dependent test failure — `test_backfill_layers` passed
alone and failed after any test that pointed DATA_DIR at a tmp dir — which
made it look like flakiness rather than a real defect.

It is a real defect. `init_db()` is idempotent and creates tables with
`checkfirst=True`; nothing was wrong with it. The guard in front of it was
wrong, and it would misbehave anywhere the database can change after first
use, not only in tests.

Sibling stores (doc_index, hydration_store, rag.budget) already tracked
`_initialized_for_url`. These four did not, and now do.
"""
from __future__ import annotations

import importlib

import pytest

STORES = [
    ("app.core.projects", "projects"),
    ("app.core.users", "users"),
    ("app.core.agent_memory", "conversations"),
    ("app.core.workflows", "workflows"),
]


@pytest.mark.parametrize("module_name,_table", STORES)
def test_store_declares_the_url_it_initialised(module_name, _table):
    """The companion flag must exist — a bare boolean cannot be correct."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "_initialized_for_url"), (
        f"{module_name} tracks only a global _initialized flag; a per-URL "
        "database needs _initialized_for_url or a DB switch silently gets no schema"
    )


@pytest.mark.parametrize("module_name,_table", STORES)
def test_ensure_reinitialises_when_the_database_changes(module_name, _table):
    """Simulate the switch: mark init done for a DIFFERENT url, then ensure.

    With the old guard this was a no-op and the new database kept no schema.
    """
    mod = importlib.import_module(module_name)
    ensure = getattr(mod, "_ensure_db", None) or getattr(mod, "_ensure")

    called = {"n": 0}
    real_init = mod.init_db

    def _spy():
        called["n"] += 1
        return real_init()

    mod.init_db = _spy
    prev_flag = mod._initialized
    prev_url = mod._initialized_for_url
    try:
        # pretend the schema was created in some other database
        mod._initialized = True
        mod._initialized_for_url = "postgresql://somewhere/else"
        ensure()
        assert called["n"] == 1, (
            "init_db was skipped after the database changed — this is exactly "
            "the 'no such table' failure"
        )
    finally:
        mod.init_db = real_init
        mod._initialized = prev_flag
        mod._initialized_for_url = prev_url


@pytest.mark.parametrize("module_name,_table", STORES)
def test_ensure_is_still_a_noop_for_the_same_database(module_name, _table):
    """The guard must keep doing its job: no re-init on every single call."""
    from app.core.db import get_database_url

    mod = importlib.import_module(module_name)
    ensure = getattr(mod, "_ensure_db", None) or getattr(mod, "_ensure")

    called = {"n": 0}
    real_init = mod.init_db
    mod.init_db = lambda: called.__setitem__("n", called["n"] + 1)

    prev_flag = mod._initialized
    prev_url = mod._initialized_for_url
    try:
        mod._initialized = True
        mod._initialized_for_url = get_database_url()
        ensure()
        assert called["n"] == 0, "re-initialised despite the database being unchanged"
    finally:
        mod.init_db = real_init
        mod._initialized = prev_flag
        mod._initialized_for_url = prev_url
