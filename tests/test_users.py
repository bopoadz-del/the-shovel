"""Tests for the users store — Stream A (User Accounts & Multi-Tenancy)."""
import importlib
import pytest
from app.core import users as users_mod
from tests._fixture_values import PW_07, PW_11, PW_12, PW_13


@pytest.fixture
def users(monkeypatch, tmp_path):
    """Relocate the unified DB into a tmp dir and reload store modules."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod

    importlib.reload(db_mod)
    reloaded = importlib.reload(users_mod)
    reloaded._initialized = False
    yield reloaded
    # _initialized is a process global while the engine URL tracks DATA_DIR
    # at call time. Leaving it True after monkeypatch restores DATA_DIR
    # would point later tests at a DB that never ran init_db (the in-suite
    # 'no such table: users' failure in test_validation_pipeline).
    reloaded._initialized = False


def test_init_creates_system_user(users):
    users.init_db()
    sys_user = users.get_user_by_id("system")
    assert sys_user is not None
    assert sys_user["id"] == "system"
    assert sys_user["role"] == "admin"
    assert sys_user["email"] == "system@local"


def test_init_db_is_idempotent(users):
    users.init_db()
    users.init_db()
    assert users.get_user_by_id("system")["id"] == "system"


def test_ensure_system_user_is_idempotent_when_row_already_exists(users):
    users.init_db()
    users.ensure_system_user()
    users.ensure_system_user()
    assert users.get_user_by_id("system")["id"] == "system"


def test_ensure_system_user_survives_duplicate_pk_race(users, monkeypatch):
    """Re-seed must not UniqueViolation when another caller already inserted.

    Isolation TRUNCATE + a live TestClient knowledge-seed can both observe
    a missing ``system`` row. The second INSERT used to raise IntegrityError
    and fail test-postgres setup (users_pkey / id='system').
    """
    users.init_db()
    assert users.get_user_by_id("system") is not None

    from sqlalchemy.orm import Session

    from app.core.models import User

    real_get = Session.get

    def hide_existing_system(self, entity, ident, *args, **kwargs):
        key = ident[0] if isinstance(ident, tuple) else ident
        if entity is User and key == users.SYSTEM_USER_ID:
            return None
        return real_get(self, entity, ident, *args, **kwargs)

    monkeypatch.setattr(Session, "get", hide_existing_system)
    users.ensure_system_user()
    monkeypatch.undo()
    assert users.get_user_by_id("system")["id"] == "system"


def test_ensure_system_user_concurrent_calls_do_not_raise(users):
    """Several threads re-seeding the same PK must all return cleanly."""
    import threading

    from app.core.db import SessionLocal
    from app.core.models import User

    users.init_db()
    with SessionLocal() as session:
        row = session.get(User, users.SYSTEM_USER_ID)
        session.delete(row)
        session.commit()

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            users.ensure_system_user()
        except BaseException as exc:  # noqa: BLE001 — collect any leak
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert users.get_user_by_id("system")["id"] == "system"


def test_create_user_and_password_round_trip(users):
    users.init_db()
    u = users.create_user("alice@example.com", PW_07, display_name="Alice")
    assert u["email"] == "alice@example.com"
    assert u["role"] == "user"
    assert u["id"] != "system"
    assert "password_hash" not in u
    stored = users.get_user_by_email("alice@example.com")
    assert users.verify_password(PW_07, stored["password_hash"], stored["salt"]) is True
    assert users.verify_password("wrong-pw", stored["password_hash"], stored["salt"]) is False


def test_create_user_rejects_duplicate_email(users):
    users.init_db()
    users.create_user("dup@example.com", PW_11)
    with pytest.raises(ValueError):
        users.create_user("DUP@example.com", PW_12)  # case-insensitive


def test_create_user_normalizes_email_lowercase(users):
    users.init_db()
    u = users.create_user("MixedCase@Example.com", PW_13)
    assert u["email"] == "mixedcase@example.com"
