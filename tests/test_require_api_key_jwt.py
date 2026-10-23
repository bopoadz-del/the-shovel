"""Tests for the JWT-aware require_api_key dependency — Stream B fix.

require_api_key must now accept BOTH:
  - Legacy API keys (the dev key, env-based keys) — unchanged
  - Valid JWTs minted via jwt_auth.create_token — NEW

All returned-dict keys must be present so every caller pattern works:
  user, user_id, role, email, tier, valid, auth_method
"""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from app.dependencies import require_api_key
from app.core import jwt_auth, users as users_store
from tests._dev_key import DEV_KEY
from tests._fixture_values import GARBAGE_TOKEN, PW_04


@pytest.fixture
def isolated_users(monkeypatch, tmp_path):
    """Isolate users DB and JWT secret into a tmp dir for each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
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

    @app.get("/protected")
    async def protected(auth: dict = Depends(require_api_key)):
        return auth

    return TestClient(app)


def test_valid_jwt_is_accepted_and_returns_superset_dict(app_client, isolated_users):
    """A valid JWT must be accepted and return all required keys."""
    u_mod, jwt_mod = isolated_users
    user = u_mod.create_user("apikey-jwt@example.com", PW_04)
    token = jwt_mod.create_token(user["id"])

    r = app_client.get("/protected", headers={"Authorization": f"Bearer {token}"})

    assert r.status_code == 200
    data = r.json()

    # All keys must be present (superset of both legacy and JWT caller expectations)
    assert "user" in data, f"'user' key missing from {data}"
    assert "user_id" in data, f"'user_id' key missing from {data}"
    assert "role" in data, f"'role' key missing from {data}"
    assert "email" in data, f"'email' key missing from {data}"
    assert "tier" in data, f"'tier' key missing from {data}"
    assert "valid" in data, f"'valid' key missing from {data}"
    assert "auth_method" in data, f"'auth_method' key missing from {data}"

    # Values must be correct
    assert data["user_id"] == user["id"]
    assert data["email"] == user["email"]
    assert data["auth_method"] == "jwt"
    assert data["valid"] is True
    assert data["role"] in ("user", "admin")
    assert data["tier"]  # non-empty string
    assert data["user"]  # non-empty (email or id)


def test_the_dev_key_still_works(app_client):
    """the dev key (legacy path) must keep working after the JWT fix."""
    r = app_client.get("/protected", headers={"Authorization": "Bearer " + DEV_KEY})
    assert r.status_code == 200
    data = r.json()
    assert data.get("valid") is True
    assert data.get("user") == "dev"


def test_garbage_bearer_token_returns_401(app_client):
    """A garbage bearer token that's neither a valid JWT nor a known API key → 401."""
    r = app_client.get("/protected", headers={"Authorization": "Bearer " + GARBAGE_TOKEN})
    assert r.status_code == 401


def test_no_credentials_returns_401(app_client):
    """No credentials at all → 401."""
    r = app_client.get("/protected")
    assert r.status_code == 401


def test_jwt_for_nonexistent_user_returns_401(app_client, isolated_users):
    """A JWT whose user_id no longer exists → 401."""
    _, jwt_mod = isolated_users
    token = jwt_mod.create_token("nonexistent-user-id")
    r = app_client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
