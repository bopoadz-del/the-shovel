"""Schema tests for the SQLAlchemy projects store (Phase 1.3b)."""
import importlib
import sqlite3

from app.core import projects as projects_mod


def _reload_stores(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    pm = importlib.reload(projects_mod)
    pm._initialized = False
    return pm, db_mod


def test_init_db_on_fresh_db_has_user_id(monkeypatch, tmp_path):
    pm, db_mod = _reload_stores(monkeypatch, tmp_path)
    pm.init_db()
    db_path = db_mod.get_database_url().replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(projects)").fetchall()]
    conn.close()
    assert "user_id" in cols


def test_init_db_on_fresh_db_has_retrieval_visible_on_documents(monkeypatch, tmp_path):
    pm, db_mod = _reload_stores(monkeypatch, tmp_path)
    pm.init_db()
    db_path = db_mod.get_database_url().replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
    conn.close()
    assert "superseded_by" in cols
    assert "retrieval_visible" in cols
    assert "extractor_version" in cols
    assert "drive_md5" in cols


def test_init_db_on_fresh_db_has_content_sha256_on_documents(monkeypatch, tmp_path):
    pm, db_mod = _reload_stores(monkeypatch, tmp_path)
    pm.init_db()
    db_path = db_mod.get_database_url().replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(documents)").fetchall()]
    conn.close()
    assert "content_sha256" in cols


def test_fresh_project_defaults_user_id_to_system(monkeypatch, tmp_path):
    pm, _ = _reload_stores(monkeypatch, tmp_path)
    pm.init_db()
    p = pm.create_project("Fresh")
    assert p["user_id"] == "system"
