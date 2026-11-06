"""G3: the inventory script refuses an empty id and never issues DML."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "project_inventory.py"

_SRC = _SCRIPT.read_text(encoding="utf-8")


def _load():
    spec = importlib.util.spec_from_file_location("project_inventory", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inventory_refuses_an_empty_project_id(monkeypatch):
    mod = _load()
    monkeypatch.delenv("INVENTORY_PROJECT_ID", raising=False)
    assert mod.main() == 2


def test_counts_queries_the_namespace_chunk_table(monkeypatch):
    mod = _load()

    class _FakeResult:
        def one(self):
            return (7, 7)

        def all(self):
            return []

    class _FakeSession:
        def __init__(self):
            self.calls = []

        def execute(self, stmt, params):
            self.calls.append((str(stmt), params))
            return _FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fake = _FakeSession()
    monkeypatch.setattr(mod, "SessionLocal", lambda: fake)
    chunks, fts, rows = mod._counts(fake, "p1")
    assert chunks in (7,)
    assert fts in (7,)
    assert any("text_search" in c[0] for c in fake.calls)
    assert any("FROM documents" in c[0] for c in fake.calls)


def test_inventory_sql_strings_are_read_only():
    mod = _load()
    dml = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate)\b", re.IGNORECASE)
    for name in ("_CHUNKS_SQL", "_DOCUMENTS_SQL"):
        sql = getattr(mod, name)
        assert sql.lstrip().upper().startswith("SELECT"), (name, sql)
        assert not dml.search(sql), (name, sql)
    # The only interpolation is the namespace-resolved TABLE name; the project
    # id and every other value are bind parameters.
    assert mod._CHUNKS_SQL.count(":p") == 1
    assert mod._DOCUMENTS_SQL.count(":p") == 1
