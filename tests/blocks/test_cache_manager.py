"""Tests for the cache manager block — Redis and local-fallback paths."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app.blocks.cache_manager import CacheManagerBlock


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_redis_client(monkeypatch):
    """Isolate the shared Redis singleton and ensure REDIS_URL is unset."""
    from app.core import redis_client

    redis_client.reset_for_tests()
    monkeypatch.delenv("REDIS_URL", raising=False)
    yield
    redis_client.reset_for_tests()


# ── Fake async Redis backends ───────────────────────────────────────────────


class _FakeAsyncRedis:
    """Minimal async Redis implementation backed by an in-memory dict."""

    def __init__(self):
        self._store: Dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def flushdb(self) -> None:
        self._store.clear()

    async def info(self) -> Dict[str, Any]:
        return {"used_memory_human": "1M"}

    async def dbsize(self) -> int:
        return len(self._store)


class _FailingFakeRedis:
    """Always raises, forcing the block down to the local-dict fallback."""

    async def get(self, key: str) -> Any:
        raise Exception("Redis unavailable")

    async def setex(self, key: str, ttl: int, value: str) -> Any:
        raise Exception("Redis unavailable")

    async def delete(self, key: str) -> Any:
        raise Exception("Redis unavailable")

    async def exists(self, key: str) -> Any:
        raise Exception("Redis unavailable")

    async def flushdb(self) -> Any:
        raise Exception("Redis unavailable")

    async def info(self) -> Any:
        raise Exception("Redis unavailable")

    async def dbsize(self) -> Any:
        raise Exception("Redis unavailable")


# ── Local-fallback tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_local_set_get_delete_exists_flush_stats():
    block = CacheManagerBlock()

    set_r = await block.set({}, {"key": "k1", "value": "v1", "ttl": 3600})
    assert set_r == {"status": "success", "action": "set", "key": "k1", "ttl": 3600}

    get_r = await block.get({}, {"key": "k1"})
    assert get_r["status"] == "success"
    assert get_r["found"] is True
    assert get_r["value"] == "v1"

    assert await block.exists({}, {"key": "k1"}) == {
        "status": "success",
        "exists": True,
        "key": "k1",
    }

    assert await block.delete({}, {"key": "k1"}) == {
        "status": "success",
        "deleted": True,
        "key": "k1",
    }

    assert await block.exists({}, {"key": "k1"}) == {
        "status": "success",
        "exists": False,
        "key": "k1",
    }

    flush_r = await block.flush()
    assert flush_r["status"] == "success"
    assert flush_r["action"] == "flush"
    assert flush_r["local_entries_cleared"] == 0

    stats = await block.stats()
    assert stats["status"] == "success"
    assert stats["backend"] == "local"
    assert stats["entries"] == 0


@pytest.mark.asyncio
async def test_missing_key_returns_error():
    block = CacheManagerBlock()

    for action in ("get", "set", "delete", "exists"):
        result = await block.process({}, {"action": action})
        assert result["status"] == "error"
        assert "No key provided" in result["error"]


@pytest.mark.asyncio
async def test_process_routes_actions_and_unknown_action():
    block = CacheManagerBlock()

    set_r = await block.process({"key": "p1", "value": "x"}, {"action": "set"})
    assert set_r["status"] == "success"

    get_r = await block.process({"key": "p1"}, {"action": "get"})
    assert get_r["status"] == "success"
    assert get_r["value"] == "x"

    health = await block.process({}, {"action": "health_check"})
    assert health["status"] == "success"
    assert health["redis_connected"] is False

    unknown = await block.process({}, {"action": "nope"})
    assert unknown["status"] == "error"
    assert "Unknown action" in unknown["error"]


# ── Shared Redis client tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_redis_backend_roundtrip(monkeypatch):
    """When the shared async Redis client is available, the block uses it."""
    from app.core import redis_client

    fake = _FakeAsyncRedis()
    monkeypatch.setattr(redis_client, "_async_client", fake)

    block = CacheManagerBlock()

    set_r = await block.set({}, {"key": "r1", "value": {"a": 1}, "ttl": 60})
    assert set_r["status"] == "success"

    get_r = await block.get({}, {"key": "r1"})
    assert get_r["found"] is True
    assert get_r["value"] == {"a": 1}

    exists_r = await block.exists({}, {"key": "r1"})
    assert exists_r["exists"] is True

    delete_r = await block.delete({}, {"key": "r1"})
    assert delete_r["deleted"] is True

    assert (await block.exists({}, {"key": "r1"}))["exists"] is False

    stats = await block.stats()
    assert stats["backend"] == "redis"
    assert stats["keys"] == 0
    assert stats["used_memory_human"] == "1M"

    health = await block.health_check()
    assert health["redis_connected"] is True


@pytest.mark.asyncio
async def test_redis_error_falls_back_to_local(monkeypatch):
    """Redis call failures must not crash the block; local dict takes over."""

    async def _failing_factory():
        return _FailingFakeRedis()

    monkeypatch.setattr("app.blocks.cache_manager.get_redis_client", _failing_factory)

    block = CacheManagerBlock()

    set_r = await block.set({}, {"key": "f1", "value": "fv", "ttl": 60})
    assert set_r["status"] == "success"

    get_r = await block.get({}, {"key": "f1"})
    assert get_r["found"] is True
    assert get_r["value"] == "fv"

    assert (await block.exists({}, {"key": "f1"}))["exists"] is True
    assert (await block.delete({}, {"key": "f1"}))["deleted"] is True
    assert (await block.exists({}, {"key": "f1"}))["exists"] is False

    flush_r = await block.flush()
    assert flush_r["action"] == "flush"

    stats = await block.stats()
    assert stats["backend"] == "local"


@pytest.mark.asyncio
async def test_flushdb_called_when_redis_available(monkeypatch):
    from app.core import redis_client

    fake = _FakeAsyncRedis()
    fake._store["x"] = "1"
    monkeypatch.setattr(redis_client, "_async_client", fake)

    block = CacheManagerBlock()
    flush_r = await block.flush()
    assert flush_r["status"] == "success"
    assert flush_r["action"] == "flush"
    assert fake._store == {}
