"""Tests for scripts/migrate_sqlite_to_pg.py (Phase 1.4)."""

from __future__ import annotations

import os
import sqlite3
import struct
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from scripts.migrate_sqlite_to_pg import _unpack_embedding, main as migrate_main


def _pg_url() -> str | None:
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgresql"):
        return url
    return None


def _create_legacy_sqlite_stores(data_dir: Path) -> None:
    """Populate minimal legacy SQLite files mirroring pre-unified layout."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "rag").mkdir(exist_ok=True)

    with sqlite3.connect(data_dir / "users.db") as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT,
                salt TEXT,
                display_name TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO users VALUES (
                'u1', 'alice@example.com', 'hash', 'salt', 'Alice', 'user', '2026-01-01T00:00:00+00:00'
            )
            """
        )

    with sqlite3.connect(data_dir / "projects.db") as conn:
        conn.executescript(
            """
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                client TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                aconex_connected INTEGER NOT NULL DEFAULT 0,
                user_id TEXT NOT NULL DEFAULT 'system',
                created_at TEXT NOT NULL
            );
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                original_name TEXT NOT NULL,
                stored_as TEXT,
                file_path TEXT,
                doc_type TEXT NOT NULL DEFAULT 'document',
                doc_role TEXT NOT NULL DEFAULT 'other',
                size INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                content_sha256 TEXT
            );
            CREATE TABLE project_facts (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_document TEXT,
                confidence REAL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                project_id TEXT,
                owner_id TEXT,
                steps TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO projects VALUES (
                'p1', 'Demo', NULL, 'active', 0, 'u1', '2026-01-01T00:00:00+00:00'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO projects VALUES (
                'p-orphan', 'Orphan owner', NULL, 'active', 0, 'missing-user', '2026-01-02T00:00:00+00:00'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO documents VALUES (
                'd1', 'p1', 'spec.pdf', NULL, '/tmp/spec.pdf', 'document', 'other',
                100, '2026-01-01T00:00:00+00:00', NULL
            )
            """
        )

    embedding = struct.pack("<384f", *([0.1] * 384))
    with sqlite3.connect(data_dir / "rag" / "vectors.db") as conn:
        conn.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO chunks VALUES (
                'c1', 'p1', 'd1', 0, 'hello chunk', ?, '2026-01-01T00:00:00+00:00'
            )
            """,
            (embedding,),
        )

    with sqlite3.connect(data_dir / "rag" / "budget.db") as conn:
        conn.execute(
            "CREATE TABLE rag_budget (day TEXT PRIMARY KEY, consumed INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute("INSERT INTO rag_budget VALUES ('2026-01-01', 42)")


@pytest.fixture
def pg_engine(monkeypatch):
    url = _pg_url()
    if url is None:
        pytest.skip("DATABASE_URL PostgreSQL not configured")
    monkeypatch.setenv("DATABASE_URL", url)
    import importlib

    import app.core.db as db_mod

    importlib.reload(db_mod)

    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    yield engine
    engine.dispose()


def test_dry_run_counts_legacy_rows(tmp_path, pg_engine, monkeypatch, capsys):
    legacy = tmp_path / "legacy"
    _create_legacy_sqlite_stores(legacy)
    monkeypatch.setenv("DATABASE_URL", _pg_url())

    rc = migrate_main(
        ["--dry-run", "--data-dir", str(legacy), "--database-url", _pg_url()]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "users: 1" in out
    assert "projects: 2" in out
    assert "documents: 1" in out
    assert "chunks: 1" in out
    assert "rag_budget: 1" in out


def test_execute_migrates_and_is_idempotent(tmp_path, pg_engine, monkeypatch):
    legacy = tmp_path / "legacy"
    _create_legacy_sqlite_stores(legacy)
    url = _pg_url()
    monkeypatch.setenv("DATABASE_URL", url)

    assert migrate_main(["--execute", "--data-dir", str(legacy), "--database-url", url]) == 0
    assert migrate_main(["--execute", "--data-dir", str(legacy), "--database-url", url]) == 0

    with pg_engine.connect() as conn:
        # Alembic seeds the system user; migration adds u1.
        assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar() == 2
        assert (
            conn.execute(text("SELECT COUNT(*) FROM users WHERE id = 'u1'")).scalar()
            == 1
        )
        assert conn.execute(text("SELECT COUNT(*) FROM projects")).scalar() == 2
        orphan_uid = conn.execute(
            text("SELECT user_id FROM projects WHERE id = 'p-orphan'")
        ).scalar()
        assert orphan_uid == "system"
        assert conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar() == 1
        dim = conn.execute(
            text("SELECT vector_dims(embedding) FROM chunks WHERE chunk_id = 'c1'")
        ).scalar()
        assert dim == 256


def test_unpack_embedding_truncates_legacy_384_blob():
    blob = struct.pack("<384f", *([0.1] * 384))
    values = _unpack_embedding(blob)
    assert len(values) == 256
    assert all(pytest.approx(v) == 0.1 for v in values)


def test_unpack_embedding_pads_short_blob():
    blob = struct.pack("<128f", *([0.2] * 128))
    values = _unpack_embedding(blob)
    assert len(values) == 256
    assert all(pytest.approx(v) == 0.2 for v in values[:128])
    assert values[128:] == [0.0] * 128


def test_requires_dry_run_or_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x:y@localhost/db")
    with pytest.raises(SystemExit) as exc:
        migrate_main(["--data-dir", str(tmp_path)])
    assert exc.value.code == 2
