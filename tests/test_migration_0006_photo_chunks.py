"""Round-trip migration 0006 on a fresh in-memory SQLite database."""
import importlib

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


@pytest.fixture
def sqlite_url(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    url = f"sqlite:///{db}"
    monkeypatch.setenv("DATABASE_URL", url)
    import app.core.db as db_mod
    importlib.reload(db_mod)
    yield url
    importlib.reload(db_mod)


@pytest.fixture
def stamped_sqlite_engine(sqlite_url):
    """Create a SQLite DB with alembic_version stamped at 0005 (skipping the
    Postgres-only early migrations). Lets us test forward-from-0005 without
    needing pgvector."""
    engine = create_engine(sqlite_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"))
        conn.execute(text("INSERT INTO alembic_version VALUES ('0005')"))
    return engine


@pytest.fixture
def alembic_cfg():
    return Config("alembic.ini")


def test_upgrade_creates_photo_chunks_and_photos_tables(stamped_sqlite_engine, alembic_cfg):
    command.upgrade(alembic_cfg, "0006")
    insp = inspect(stamped_sqlite_engine)
    tables = set(insp.get_table_names())
    assert "photo_chunks" in tables
    assert "photos" in tables
    pc_cols = {c["name"] for c in insp.get_columns("photo_chunks")}
    assert pc_cols >= {"chunk_id", "project_id", "sha256", "caption", "photo_metadata", "created_at"}
    photos_cols = {c["name"] for c in insp.get_columns("photos")}
    assert photos_cols >= {"sha256", "content_type", "size_bytes", "bytes", "uploaded_at"}


def test_upgrade_does_not_modify_chunks_or_doc_index(stamped_sqlite_engine, alembic_cfg):
    # The point of this migration: avoid touching existing tables.
    # Insert sentinel rows into both doc_index and chunks; verify they survive untouched.
    with stamped_sqlite_engine.begin() as conn:
        conn.execute(text("CREATE TABLE doc_index (project_id TEXT PRIMARY KEY, index_json TEXT, updated_at TEXT)"))
        conn.execute(text("INSERT INTO doc_index VALUES ('p1', '{}', '2026-06-24')"))
        conn.execute(text(
            "CREATE TABLE chunks ("
            "chunk_id TEXT PRIMARY KEY, "
            "project_id TEXT NOT NULL, "
            "doc_id TEXT NOT NULL, "
            "chunk_index INTEGER NOT NULL, "
            "text TEXT NOT NULL, "
            "embedding BLOB, "
            "created_at TEXT NOT NULL"
            ")"
        ))
        conn.execute(text(
            "INSERT INTO chunks VALUES ('c1', 'p1', 'd1', 0, 'hello', NULL, '2026-06-24')"
        ))
    command.upgrade(alembic_cfg, "0006")
    insp = inspect(stamped_sqlite_engine)
    # doc_index must be untouched
    doc_index_cols = {c["name"] for c in insp.get_columns("doc_index")}
    assert "kind" not in doc_index_cols  # explicitly NOT added
    assert "photo_metadata" not in doc_index_cols
    with stamped_sqlite_engine.begin() as conn:
        row = conn.execute(text("SELECT project_id FROM doc_index WHERE project_id = 'p1'")).fetchone()
    assert row is not None  # untouched
    # chunks must also be untouched
    chunks_cols = {c["name"] for c in insp.get_columns("chunks")}
    assert "kind" not in chunks_cols
    assert "photo_metadata" not in chunks_cols
    with stamped_sqlite_engine.begin() as conn:
        crow = conn.execute(text("SELECT chunk_id FROM chunks WHERE chunk_id = 'c1'")).fetchone()
    assert crow is not None  # sentinel row survives


def test_downgrade_drops_both_new_tables(stamped_sqlite_engine, alembic_cfg):
    command.upgrade(alembic_cfg, "0006")
    command.downgrade(alembic_cfg, "0005")
    insp = inspect(stamped_sqlite_engine)
    tables = set(insp.get_table_names())
    assert "photo_chunks" not in tables
    assert "photos" not in tables
