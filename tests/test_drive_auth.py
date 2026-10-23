import json, time, os
import pytest
from pathlib import Path
from app.core import drive_auth
from tests._fixture_values import ENV_SECRET_05


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", ENV_SECRET_05)
    return tmp_path


def test_save_load_roundtrip(tmp_data):
    tok = {"access_token": "a", "refresh_token": "r", "expiry": time.time() + 9999, "email": "x@y.z"}
    drive_auth.save_token("u1", tok)
    assert drive_auth.load_token("u1") == tok


def test_load_returns_none_when_absent(tmp_data):
    assert drive_auth.load_token("u1") is None


def test_clear_token(tmp_data):
    drive_auth.save_token("u1", {"access_token": "a"})
    assert drive_auth.clear_token("u1") is True
    assert drive_auth.load_token("u1") is None
    assert drive_auth.clear_token("u1") is False


@pytest.mark.asyncio
async def test_get_access_token_returns_unexpired(tmp_data):
    drive_auth.save_token("u1", {"access_token": "live", "refresh_token": "r",
                           "expiry": time.time() + 9999})
    assert await drive_auth.get_access_token("u1") == "live"


@pytest.mark.asyncio
async def test_get_access_token_refreshes_when_expired(tmp_data, monkeypatch):
    drive_auth.save_token("u1", {"access_token": "old", "refresh_token": "r",
                           "expiry": time.time() - 10})

    async def fake_refresh(refresh_token):
        assert refresh_token == "r"
        return {"access_token": "fresh", "expires_in": 3600}

    monkeypatch.setattr(drive_auth, "_refresh_request", fake_refresh)
    assert await drive_auth.get_access_token("u1") == "fresh"
    assert drive_auth.load_token("u1")["access_token"] == "fresh"


@pytest.mark.asyncio
async def test_get_access_token_raises_when_not_connected(tmp_data):
    with pytest.raises(drive_auth.DriveNotConnected):
        await drive_auth.get_access_token("u1")


@pytest.mark.asyncio
async def test_get_access_token_raises_on_failed_refresh(tmp_data, monkeypatch):
    drive_auth.save_token("u1", {"access_token": "old", "refresh_token": "r",
                           "expiry": time.time() - 10})

    async def boom(refresh_token):
        raise drive_auth.DriveAuthError("refresh failed")

    monkeypatch.setattr(drive_auth, "_refresh_request", boom)
    with pytest.raises(drive_auth.DriveAuthError):
        await drive_auth.get_access_token("u1")
