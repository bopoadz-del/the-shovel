"""Tests for the require_user dependency — Stream A."""
import sys
import importlib
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from app.dependencies import require_user
from app.core import jwt_auth, users as users_store
from tests._dev_key import DEV_KEY
from tests._fixture_values import GARBAGE_TOKEN, PW_06


@pytest.fixture
def isolated_users(monkeypatch, tmp_path):
    """Isolate users DB and jwt secret into a tmp dir for each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Reset module state so they re-init against the tmp path
    import app.core.users as u_mod
    import app.core.jwt_auth as jwt_mod
    u_mod._initialized = False
    jwt_mod._cached_secret = None
    u_mod.init_db()
    yield u_mod, jwt_mod
    # Cleanup
    u_mod._initialized = False
    jwt_mod._cached_secret = None


@pytest.fixture
def app_client(isolated_users):
    app = FastAPI()

    @app.get("/whoami")
    async def whoami(auth: dict = Depends(require_user)):
        return auth

    return TestClient(app)


def test_legacy_api_key_maps_to_system_user(app_client):
    r = app_client.get("/whoami", headers={"Authorization": "Bearer " + DEV_KEY})
    assert r.status_code == 200
    assert r.json()["user_id"] == "system"
    assert r.json()["auth_method"] == "api_key"
    # Identity only — never copy role from the system user row (which IS admin).
    assert r.json()["role"] != "admin"


def test_valid_jwt_resolves_to_user(app_client, isolated_users):
    u_mod, jwt_mod = isolated_users
    u = u_mod.create_user("dep1@example.com", PW_06)
    token = jwt_mod.create_token(u["id"])
    r = app_client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["user_id"] == u["id"]
    assert r.json()["auth_method"] == "jwt"


def test_missing_credentials_401(app_client):
    assert app_client.get("/whoami").status_code in (401, 403)


def test_garbage_bearer_401(app_client):
    r = app_client.get("/whoami", headers={"Authorization": "Bearer " + GARBAGE_TOKEN})
    assert r.status_code == 401


def test_jwt_for_deleted_user_401(app_client, isolated_users):
    _, jwt_mod = isolated_users
    token = jwt_mod.create_token("nonexistent-id")
    r = app_client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
