"""Tests for JWT session tokens — Stream A."""
import importlib
import time
import pytest
from app.core import jwt_auth as jwt_mod
from tests._fixture_values import ENV_SECRET_07


@pytest.fixture
def jwtmod(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SECRET_KEY", raising=False)
    return importlib.reload(jwt_mod)


def test_round_trip_encode_decode(jwtmod):
    token = jwtmod.create_token("u123")
    payload = jwtmod.decode_token(token)
    assert payload["user_id"] == "u123"


def test_secret_key_persisted_to_disk(jwtmod, tmp_path):
    jwtmod.create_token("u1")           # forces secret resolution
    assert (tmp_path / ".secret_key").exists()
    reloaded = importlib.reload(jwtmod)
    assert reloaded.decode_token(jwtmod.create_token("u2"))["user_id"] == "u2"


def test_explicit_secret_key_env_used(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_07)
    mod = importlib.reload(jwt_mod)
    assert mod.decode_token(mod.create_token("uX"))["user_id"] == "uX"
    assert not (tmp_path / ".secret_key").exists()


def test_expired_token_rejected(jwtmod):
    token = jwtmod.create_token("u1", expires_in=-10)  # already expired
    with pytest.raises(jwtmod.InvalidTokenError):
        jwtmod.decode_token(token)


def test_garbage_token_rejected(jwtmod):
    with pytest.raises(jwtmod.InvalidTokenError):
        jwtmod.decode_token("not-a-real-token")
