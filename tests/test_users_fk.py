"""FK constraint tests for the SQLAlchemy users store (Phase 1.3a)."""
import importlib

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

import app.core.projects as projects_mod
from tests._fixture_values import PW_08


@pytest.fixture
def users_db(monkeypatch, tmp_path):
    """Fresh unified DB in a tmp dir with users schema initialized."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    users_mod.init_db()
    yield users_mod, db_mod
    # Same rationale as the users fixture in test_users.py: the initialized
    # flag must not outlive the tmp DATA_DIR it was earned against.
    users_mod._initialized = False
    projects_mod._initialized = False


def test_delete_user_with_projects_blocked(users_db):
    """projects.user_id FK must block deleting a user who owns projects."""
    users_mod, db_mod = users_db
    owner = users_mod.create_user("owner@example.com", PW_08)
    uid = owner["id"]

    with db_mod.SessionLocal() as session:
        session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id               TEXT PRIMARY KEY,
                    name             TEXT NOT NULL,
                    client           TEXT,
                    status           TEXT NOT NULL DEFAULT 'active',
                    aconex_connected INTEGER NOT NULL DEFAULT 0,
                    user_id          TEXT NOT NULL
                                       REFERENCES users (id) ON DELETE RESTRICT,
                    created_at       TEXT NOT NULL
                )
                """
            )
        )
        session.execute(
            text(
                "INSERT INTO projects (id, name, user_id, created_at) "
                "VALUES (:id, :name, :uid, :ts)"
            ),
            {
                "id": "proj-fk-test",
                "name": "FK Test Project",
                "uid": uid,
                "ts": "2026-01-01T00:00:00+00:00",
            },
        )
        session.commit()

    with db_mod.SessionLocal() as session:
        with pytest.raises(IntegrityError):
            session.execute(
                text("DELETE FROM users WHERE id = :uid"), {"uid": uid}
            )
            session.commit()


def test_delete_user_with_projects_blocked_unified_schema(users_db, monkeypatch, tmp_path):
    """projects.user_id ON DELETE RESTRICT via projects.init_db() schema."""
    users_mod, db_mod = users_db
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(db_mod)
    pm = importlib.reload(projects_mod)
    pm._initialized = False
    users_mod._initialized = False
    pm.init_db()

    owner = users_mod.create_user("owner2@example.com", PW_08)
    pm.create_project("Owned project", user_id=owner["id"])

    with db_mod.SessionLocal() as session:
        with pytest.raises(IntegrityError):
            session.execute(
                text("DELETE FROM users WHERE id = :uid"), {"uid": owner["id"]}
            )
            session.commit()
