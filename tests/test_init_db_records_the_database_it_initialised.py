"""init_db records the database it actually initialised.

Found by the parallel test run (#820): projects.init_db() created its tables
on the database DATA_DIR pointed at when it started, then re-read the URL to
record which database was ready. When DATA_DIR changed in between (a
background thread initialising while a test switched databases) it marked the
NEW database ready although its tables were never created, and the next
insert failed with "no such table: projects".
"""
from __future__ import annotations

import os

import pytest

# DATA_DIR chooses the database only when DATABASE_URL is unset (SQLite); on
# the PostgreSQL job the URL is explicit and this switch cannot happen.
pytestmark = pytest.mark.skipif(bool(os.getenv("DATABASE_URL")),
                                reason="DATA_DIR picks the database only on SQLite")


def _switch_after(monkeypatch, mod, step_name, target):
    """Run ``mod.<step_name>`` normally, then point DATA_DIR at ``target`` --
    the moment between creating the tables and recording which database is
    ready."""
    real = getattr(mod, step_name)

    def step(*a, **k):
        out = real(*a, **k)
        monkeypatch.setenv("DATA_DIR", str(target))
        return out
    monkeypatch.setattr(mod, step_name, step)


@pytest.mark.parametrize("module_name,last_step", [
    ("app.core.projects", "_patch_legacy_columns"),
    ("app.core.users", "ensure_system_user"),
])
def test_a_data_dir_switch_before_recording_does_not_mark_the_new_database_ready(
        module_name, last_step, tmp_path, monkeypatch):
    import importlib

    from app.core import db

    mod = importlib.import_module(module_name)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("DATA_DIR", str(first))
    monkeypatch.setattr(mod, "_initialized", False)
    monkeypatch.setattr(mod, "_initialized_for_url", None)
    url_at_start = db.get_database_url()
    _switch_after(monkeypatch, mod, last_step, second)

    mod.init_db()

    assert db.get_database_url() != url_at_start, "the switch never happened"
    assert mod._initialized_for_url == url_at_start


def test_after_a_switch_the_new_database_gets_its_tables(tmp_path, monkeypatch):
    from app.core import db, projects

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("DATA_DIR", str(first))
    monkeypatch.setattr(projects, "_initialized", False)
    _switch_after(monkeypatch, projects, "_patch_legacy_columns", second)
    projects.init_db()  # tables in "a"; DATA_DIR now "b"
    monkeypatch.setattr(projects, "_patch_legacy_columns", lambda: None)
    p = projects.create_project("Synthetic Fenwick Reach Works")
    assert projects.get_project(p["id"])["name"] == "Synthetic Fenwick Reach Works"
    assert str(second) in db.get_database_url()
