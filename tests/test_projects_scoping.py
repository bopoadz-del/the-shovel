"""user_id scoping at the projects store layer — Stream A."""
import importlib
import pytest
from app.core import projects as projects_mod


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    pm = importlib.reload(projects_mod)
    pm._initialized = False
    pm.init_db()
    for uid in ("alice", "bob", "carol"):
        users_mod.ensure_user_exists(uid)
    return pm


def test_create_defaults_to_system_owner(store):
    p = store.create_project("Default Owner")
    assert store.get_project(p["id"])["user_id"] == "system"


def test_create_with_explicit_owner(store):
    p = store.create_project("Alice Project", user_id="alice")
    assert store.get_project(p["id"])["user_id"] == "alice"


def test_list_projects_filters_by_owner(store):
    store.create_project("A1", user_id="alice")
    store.create_project("B1", user_id="bob")
    alice_names = {p["name"] for p in store.list_projects(user_id="alice")}
    assert alice_names == {"A1"}
    assert len(store.list_projects()) == 2


def test_get_project_returns_none_on_owner_mismatch(store):
    p = store.create_project("Bob Secret", user_id="bob")
    assert store.get_project(p["id"], user_id="alice") is None
    assert store.get_project(p["id"], user_id="bob") is not None
    assert store.get_project(p["id"]) is not None


def test_project_owner_helper(store):
    p = store.create_project("Owned", user_id="carol")
    assert store.project_owner(p["id"]) == "carol"
    assert store.project_owner("missing9") is None
