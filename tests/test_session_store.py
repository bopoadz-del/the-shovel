"""Tests for the session state store — Reasoning Engine Plan 2."""

import os as _os
import time as _time

import pytest

from app.core.session_store import InMemorySessionStore, get_session_store
from app.schemas.project_session import Artifact, Message, ProjectSession


def test_new_session_is_empty():
    s = ProjectSession.new("sess1")
    assert s.id == "sess1"
    assert s.data == {} and s.history == [] and s.artifacts == []
    assert s.created_at and s.updated_at


def test_message_and_artifact_models():
    m = Message(role="user", content="hi", ts="2026-05-20T00:00:00Z")
    assert m.role == "user"
    a = Artifact(name="schedule.xlsx", path="/data/x.xlsx", type="excel")
    assert a.type == "excel"


@pytest.mark.asyncio
async def test_get_or_create_is_idempotent():
    store = InMemorySessionStore()
    a = await store.get_or_create("s1")
    b = await store.get_or_create("s1")
    assert a.id == b.id == "s1"
    assert await store.get("s1") is not None


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    assert await InMemorySessionStore().get("nope") is None


@pytest.mark.asyncio
async def test_save_persists_mutations():
    store = InMemorySessionStore()
    s = await store.get_or_create("s1")
    s.data["activities"] = [{"id": "A"}]
    s.add_message("user", "create a schedule")
    await store.save(s)
    reloaded = await store.get("s1")
    assert reloaded.data["activities"] == [{"id": "A"}]
    assert reloaded.history[0].content == "create a schedule"


@pytest.mark.asyncio
async def test_delete_removes_session():
    store = InMemorySessionStore()
    await store.get_or_create("s1")
    assert await store.delete("s1") is True
    assert await store.get("s1") is None
    assert await store.delete("s1") is False


@pytest.mark.asyncio
async def test_session_expires_after_ttl():
    store = InMemorySessionStore(ttl_seconds=1)
    await store.get_or_create("s1")
    assert await store.get("s1") is not None
    _time.sleep(1.1)
    assert await store.get("s1") is None          # expired and evicted


@pytest.mark.asyncio
async def test_save_refreshes_ttl():
    store = InMemorySessionStore(ttl_seconds=2)
    s = await store.get_or_create("s1")
    _time.sleep(1.2)
    await store.save(s)                            # resets the 2s window
    _time.sleep(1.2)
    assert await store.get("s1") is not None       # still alive — refreshed


def test_factory_returns_in_memory_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    store = get_session_store()
    assert isinstance(store, InMemorySessionStore)


@pytest.mark.skipif(not _os.getenv("REDIS_URL"), reason="no REDIS_URL configured")
@pytest.mark.asyncio
async def test_redis_backend_roundtrip():
    from app.core.session_store import RedisSessionStore
    store = RedisSessionStore(ttl_seconds=60)
    await store.delete("redis_test_sess")
    s = await store.get_or_create("redis_test_sess")
    s.data["x"] = 1
    await store.save(s)
    loaded = await store.get("redis_test_sess")
    assert loaded is not None
    assert loaded.data["x"] == 1
    await store.delete("redis_test_sess")
