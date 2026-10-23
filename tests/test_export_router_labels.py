"""Read-only router-label export.

Proves the export takes every project in the connected database, copies the
stored ingestion label unchanged, and leaves a blank label blank even when
the filename looks like a type.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy import create_engine, text

from scripts.export_router_labels import (
    TOKEN_LIMIT,
    active_chunk_table,
    declared_type,
    export_router_labels,
)


def _engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'export.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT)"))
        conn.execute(
            text(
                "CREATE TABLE documents ("
                "id TEXT PRIMARY KEY, project_id TEXT, original_name TEXT, "
                "doc_type TEXT, uploaded_at TEXT, content_sha256 TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE chunks ("
                "doc_id TEXT, chunk_index INTEGER, text TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO projects (id, name) VALUES "
                "('alpha', 'Alpha Works'), ('beta', 'Beta Works')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO documents "
                "(id, project_id, original_name, doc_type, uploaded_at, content_sha256) "
                "VALUES "
                "('d-blank', 'alpha', 'method_statement.pdf', '   ', "
                "'2026-01-01T00:00:00Z', NULL), "
                "('d-stored', 'beta', 'contract.pdf', 'schedule', "
                "'2026-01-02T00:00:00Z', 'stored-content-hash'), "
                "('d-empty', 'alpha', 'scan.pdf', NULL, "
                "'2026-01-03T00:00:00Z', NULL)"
            )
        )
        words = " ".join(f"w{i}" for i in range(TOKEN_LIMIT + 40))
        conn.execute(
            text(
                "INSERT INTO chunks (doc_id, chunk_index, text) VALUES "
                "('d-blank', 0, :first), ('d-blank', 1, :second), "
                "('d-stored', 0, 'ignored because the cap is what matters')"
            ),
            {"first": words, "second": "tail-token"},
        )
        conn.execute(
            text("CREATE TABLE canary (id INTEGER PRIMARY KEY, note TEXT)")
        )
        conn.execute(text("INSERT INTO canary (note) VALUES ('untouched')"))
    return engine


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_blank_label_stays_blank_and_every_project_is_exported(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "")
    engine = _engine(tmp_path)
    summary = export_router_labels(engine, tmp_path / "out")

    assert summary["projects"] == ["alpha", "beta"]
    assert summary["row_count"] == 3
    rows = _rows(tmp_path / "out" / "labels.jsonl")
    by_name = {row["filename"]: row for row in rows}
    blank = by_name["method_statement.pdf"]
    assert blank["declared_type"] == ""
    assert blank["project_id"] == "alpha"
    assert blank["first_512_tokens"][0] == "w0"
    assert blank["first_512_tokens"][-1] == f"w{TOKEN_LIMIT - 1}"
    assert len(blank["first_512_tokens"]) == TOKEN_LIMIT
    assert "tail-token" not in blank["first_512_tokens"]
    assert "method_statement" not in blank["declared_type"]

    stored = by_name["contract.pdf"]
    assert stored["declared_type"] == "schedule"
    assert stored["doc_id"] == "stored-content-hash"
    assert stored["project_id"] == "beta"
    assert stored["ingested_at"] == "2026-01-02T00:00:00Z"

    empty = by_name["scan.pdf"]
    assert empty["declared_type"] == ""
    assert empty["first_512_tokens"] == []
    assert empty["doc_id"] == hashlib.sha256(b"").hexdigest()

    joined = " ".join(f"w{i}" for i in range(TOKEN_LIMIT + 40)) + "\ntail-token"
    assert blank["doc_id"] == hashlib.sha256(joined.encode("utf-8")).hexdigest()

    fields = {
        "doc_id",
        "project_id",
        "declared_type",
        "filename",
        "first_512_tokens",
        "ingested_at",
    }
    assert all(set(row) == fields for row in rows)
    sums = (tmp_path / "out" / "SHA256SUMS").read_text(encoding="utf-8")
    digest = hashlib.sha256((tmp_path / "out" / "labels.jsonl").read_bytes()).hexdigest()
    assert sums == f"{digest}  labels.jsonl\n"
    assert summary["labels_sha256"] == digest
    assert summary["by_declared_type"][""] == 2
    assert summary["by_declared_type"]["schedule"] == 1

    with engine.connect() as conn:
        note = conn.execute(text("SELECT note FROM canary")).scalar()
    assert note == "untouched"


def test_declared_type_does_not_read_filename():
    assert declared_type(None) == ""
    assert declared_type("  ") == ""
    assert declared_type("boq") == "boq"
    assert declared_type(" schedule ") == " schedule "


def test_chunk_table_follows_rag_namespace(monkeypatch):
    from app.core.models import rag_chunk_table_name
    from app.core.rag.vector_store import _rag_vector_namespace

    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "v2")
    assert active_chunk_table() == "chunks_v2"
    assert active_chunk_table() == rag_chunk_table_name(_rag_vector_namespace())
    monkeypatch.setenv("RAG_VECTOR_NAMESPACE", "")
    assert active_chunk_table() == "chunks"
    assert active_chunk_table() == rag_chunk_table_name(_rag_vector_namespace())


def test_cli_refuses_without_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from scripts.export_router_labels import main

    assert main(["--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_filename_is_not_an_argument_of_declared_type():
    """A stored label is copied; a missing label stays blank."""
    assert declared_type(None) == ""
    assert declared_type("") == ""
    assert declared_type("schedule") == "schedule"
