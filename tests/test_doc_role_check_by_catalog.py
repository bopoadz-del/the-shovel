"""Alembic 0036: one doc_role CHECK on documents, found by definition (RQ-3).

A table built from the original schema carries an auto-named narrow CHECK that
0035's DROP-by-name missed; 0036 must drop every doc_role CHECK whatever its
name and leave the single widened one.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "alembic" / "versions" / "0036_documents_doc_role_check_by_catalog.py"
_ON_POSTGRES = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")


def _load():
    spec = importlib.util.spec_from_file_location("doc_role_check_migration_0036", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("definition, dropped", [
    ("CHECK ((doc_role = ANY (ARRAY['other'::text])))", True),
    ("CHECK ((size >= 0))", False),
])
def test_only_checks_on_the_doc_role_column_are_picked(definition, dropped):
    picked = _load().doc_role_checks([("some_auto_name", definition)])
    assert picked == (["some_auto_name"] if dropped else [])


@pytest.mark.parametrize("definition, dropped", [
    ("CHECK ((doc_roles IS NOT NULL))", False),
    ("CHECK ((doc_role IS NOT NULL))", True),
])
def test_a_column_that_only_starts_with_doc_role_is_not_picked(definition, dropped):
    assert bool(_load().doc_role_checks([("c", definition)])) is dropped


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Bind:
    def __init__(self, dialect, rows):
        self.dialect = type("D", (), {"name": dialect})()
        self._rows = rows

    def execute(self, _stmt):
        return _Result(self._rows)


class _Op:
    def __init__(self, bind):
        self._bind = bind
        self.sql: list[str] = []

    def get_bind(self):
        return self._bind

    def execute(self, stmt):
        self.sql.append(str(stmt))


_ROWS = [
    ("documents_doc_role_check", "CHECK ((doc_role = ANY (ARRAY['other'::text])))"),
    ("ck_documents_doc_role", "CHECK ((doc_role = ANY (ARRAY['other'::text, 'tender'::text])))"),
    ("documents_size_check", "CHECK ((size >= 0))"),
]


@pytest.mark.parametrize("fn", ["upgrade", "downgrade"])
@pytest.mark.parametrize("dialect, drops, adds", [("postgresql", 2, 1), ("sqlite", 0, 0)])
def test_every_doc_role_check_is_replaced_by_one(monkeypatch, fn, dialect, drops, adds):
    mod = _load()
    fake = _Op(_Bind(dialect, _ROWS))
    monkeypatch.setattr(mod, "op", fake)
    getattr(mod, fn)()
    dropped = [s for s in fake.sql if "DROP CONSTRAINT" in s]
    added = [s for s in fake.sql if "ADD CONSTRAINT" in s]
    assert len(dropped) == drops and len(added) == adds
    assert not any("documents_size_check" in s for s in dropped)
    if adds:
        assert f"ADD CONSTRAINT {mod.NAME} " in added[0]


@pytest.mark.parametrize("name, quoted", [("plain", '"plain"'), ('we"ird', '"we""ird"')])
def test_constraint_names_are_quoted(name, quoted):
    assert _load()._quoted(name) == quoted


def _model_check_text() -> str:
    from app.core.models import Document
    check = next(a for a in Document.__table_args__
                 if getattr(a, "name", "") == "ck_documents_doc_role")
    return str(check.sqltext)


def _schema_check_text() -> str:
    schema = (ROOT / "the_fork_schema.sql").read_text(encoding="utf-8")
    return re.search(r"CHECK \(doc_role IN \((.*?)\)\)", schema, re.DOTALL).group(1)


@pytest.mark.parametrize("source", [_model_check_text, _schema_check_text])
def test_vocabulary_matches_the_model_and_the_schema_file(source):
    # A fresh database (schema file) and the ORM (model) carry the same roles.
    assert set(_load().ROLES) == set(re.findall(r"'(\w+)'", source()))


@pytest.mark.skipif(not _ON_POSTGRES, reason="pg_constraint is PostgreSQL (test-postgres job)")
@pytest.mark.parametrize("folder_role, legacy", [("tender", True), ("tender", False)])
def test_on_postgres_a_legacy_narrow_check_no_longer_refuses_folder_roles(folder_role, legacy):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import text

    from app.core.db import get_engine

    mod = _load()
    with get_engine().connect() as conn:
        tx = conn.begin()
        try:
            if legacy:
                # The shape a table built from the original schema carries.
                conn.execute(text(
                    "ALTER TABLE documents ADD CONSTRAINT legacy_doc_role_narrow "
                    "CHECK (doc_role IN ('baseline_schedule', 'daily_report', "
                    "'weekly_report', 'other'))"
                ))
            with Operations.context(MigrationContext.configure(conn)):
                mod.upgrade()
            names = [r[0] for r in conn.execute(text(
                "SELECT conname FROM pg_constraint WHERE conrelid = 'documents'::regclass "
                "AND contype = 'c' AND pg_get_constraintdef(oid) ~ '\\mdoc_role\\M'"
            ))]
            assert names == [mod.NAME]
            conn.execute(text(
                "INSERT INTO projects (id, name, created_at) VALUES ('p0036', 'p0036', 'x')"
            ))
            conn.execute(text(
                "INSERT INTO documents (id, project_id, original_name, uploaded_at, doc_role) "
                "VALUES ('d0036', 'p0036', 'n', 'x', :r)"
            ), {"r": folder_role})
        finally:
            tx.rollback()
