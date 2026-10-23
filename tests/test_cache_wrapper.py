from unittest.mock import AsyncMock

import pytest

from app.core.cache_wrapper import cache_get, cache_set, cache_delete


class FakeRedis:
    def __init__(self):
        self._store = {}
        self._ttl = {}

    async def get(self, key):
        return self._store.get(key)

    async def setex(self, key, ttl, value):
        self._store[key] = value
        self._ttl[key] = ttl

    async def delete(self, key):
        self._ttl.pop(key, None)
        return self._store.pop(key, None) is not None

    def ttl_of(self, key):
        return self._ttl.get(key)


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.mark.asyncio
async def test_cache_round_trip_with_redis(monkeypatch, fake_redis):
    monkeypatch.setattr("app.core.redis_client._async_client", fake_redis)
    assert await cache_set("k", {"v": 1}, ttl=42) is True
    assert fake_redis.ttl_of("k") == 42
    assert await cache_get("k") == {"v": 1}
    assert await cache_delete("k") is True
    assert await cache_get("k") is None


@pytest.mark.asyncio
async def test_cache_get_returns_none_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr("app.core.redis_client._async_client", None)
    assert await cache_get("k") is None


@pytest.mark.asyncio
async def test_cache_set_returns_false_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr("app.core.redis_client._async_client", None)
    assert await cache_set("k", {"v": 1}) is False


@pytest.mark.asyncio
async def test_cache_delete_returns_false_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr("app.core.redis_client._async_client", None)
    assert await cache_delete("k") is False


@pytest.mark.asyncio
async def test_cache_get_returns_none_on_redis_exception(monkeypatch, fake_redis):
    fake_redis.get = AsyncMock(side_effect=Exception("boom"))
    monkeypatch.setattr("app.core.redis_client._async_client", fake_redis)
    assert await cache_get("k") is None


@pytest.mark.asyncio
async def test_cache_set_returns_false_on_redis_exception(monkeypatch, fake_redis):
    fake_redis.setex = AsyncMock(side_effect=Exception("boom"))
    monkeypatch.setattr("app.core.redis_client._async_client", fake_redis)
    assert await cache_set("k", {"v": 1}) is False


@pytest.mark.asyncio
async def test_cache_delete_returns_false_on_redis_exception(monkeypatch, fake_redis):
    fake_redis.delete = AsyncMock(side_effect=Exception("boom"))
    monkeypatch.setattr("app.core.redis_client._async_client", fake_redis)
    assert await cache_delete("k") is False
