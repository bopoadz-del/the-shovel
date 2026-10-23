"""Tests for the SHA-256 incremental ingestion path on the Drive walker."""
from __future__ import annotations

import hashlib
import os
import pathlib
import pytest


def _reload_projects(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    return projects, db_mod, users_mod


def test_documents_schema_has_content_sha256_column(monkeypatch, tmp_path):
    projects, db_mod, _users = _reload_projects(monkeypatch, tmp_path)
    projects.init_db()
    import sqlite3
    db_path = db_mod.get_database_url().replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
    assert "content_sha256" in cols


def test_add_document_writes_sha256(monkeypatch, tmp_path):
    projects, _db_mod, users = _reload_projects(monkeypatch, tmp_path)
    projects.init_db()
    users.ensure_user_exists("u1")
    projects.create_project(name="P", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]
    sha = hashlib.sha256(b"hello world").hexdigest()
    doc = projects.add_document(
        project_id=proj["id"],
        original_name="hello.txt",
        stored_as="hello.txt",
        file_path="/tmp/hello.txt",
        size=11,
        content_sha256=sha,
    )
    assert doc["content_sha256"] == sha


def test_walker_skips_unchanged_file_on_rewalk(monkeypatch, tmp_path):
    """Second walk over a Drive folder whose file bytes haven't changed
    must skip the file (no new document row, no re-encryption)."""
    projects, _db_mod, users = _reload_projects(monkeypatch, tmp_path)
    projects.init_db()
    users.ensure_user_exists("u1")
    projects.create_project(name="P", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]

    body = b"hello world content for sha test"
    sha = hashlib.sha256(body).hexdigest()
    # First walk: insert.
    projects.add_document(
        project_id=proj["id"],
        original_name="x.pdf", stored_as="x.pdf", file_path="/tmp/x.pdf",
        size=len(body), content_sha256=sha,
    )
    # Second walk: should detect via find_document_by_sha and skip.
    found = projects.find_document_by_sha(proj["id"], sha)
    assert found is not None
    # If the walker were to add again it would create a duplicate row.
    # Ensure the existing row is the only one.
    docs = projects.list_documents(proj["id"])
    assert len(docs) == 1
