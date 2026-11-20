"""Redis loss mode for the project session store (P-I1 part 1).

What the service does when Redis is unreachable, pinned in tests:

- At boot: REDIS_URL is set but the ping fails, so ``get_session_store``
  selects the in-memory store. That store lives only in this process.
- At runtime: the Redis store was selected and later calls raise.
  - ``get`` reports the session as absent.
  - ``save`` drops the write.
  - ``delete`` reports False.
  None of them raises into the request.

Each decision is tested with a working input and a failing one. Every test
builds its own fakes; no Redis server is needed.
"""

from __future__ import annotations

import logging

import pytest

from app.core import session_store as ss
from app.schemas.project_session import ProjectSession


class _PingClient:
    def __init__(self, reachable: bool):
        self.reachable = reachable

    def ping(self):
        if not self.reachable:
            raise ConnectionError("connection refused")
        return True


@pytest.mark.parametrize(
    "reachable, expected",
    [(True, ss.RedisSessionStore), (False, ss.InMemorySessionStore)],
)
def test_boot_selection_follows_the_ping(monkeypatch, reachable, expected):
    import redis as _redis

    monkeypatch.setenv("REDIS_URL", "redis://cache.invalid:6379/0")
    monkeypatch.setattr(_redis, "from_url", lambda url, **kw: _PingClient(reachable))
    if reachable:
        store = ss.get_session_store()
    else:
        with pytest.warns(RuntimeWarning, match="falling back to in-memory"):
            store = ss.get_session_store()
    assert type(store) is expected


class _FakeAsyncRedis:
    """A dict-backed async client. With ``broken`` set, every call raises."""

    def __init__(self, broken: bool):
        self.broken = broken
        self.data: dict[str, str] = {}

    def _check(self):
        if self.broken:
            raise ConnectionError("redis went away")

    async def get(self, key):
        self._check()
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self._check()
        self.data[key] = value
        return True

    async def delete(self, key):
        self._check()
        return 1 if self.data.pop(key, None) is not None else 0


def _store_with(monkeypatch, client):
    async def _get_client():
        return client

    monkeypatch.setattr(ss, "get_redis_client", _get_client)
    return ss.RedisSessionStore(ttl_seconds=60)


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [False, True])
async def test_runtime_save_then_get(monkeypatch, caplog, broken):
    client = _FakeAsyncRedis(broken=broken)
    store = _store_with(monkeypatch, client)
    session = ProjectSession.new("sess-a")
    session.data["turn"] = 3

    with caplog.at_level(logging.WARNING, logger=ss.logger.name):
        await store.save(session)  # must not raise either way
        loaded = await store.get("sess-a")

    if broken:
        assert loaded is None
        assert client.data == {}
        assert "LOST" in caplog.text
        assert "treating it as absent" in caplog.text
    else:
        assert loaded is not None and loaded.data["turn"] == 3
        assert "session:sess-a" in client.data
        assert caplog.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [False, True])
async def test_runtime_delete(monkeypatch, caplog, broken):
    client = _FakeAsyncRedis(broken=False)
    store = _store_with(monkeypatch, client)
    await store.save(ProjectSession.new("sess-b"))
    client.broken = broken

    with caplog.at_level(logging.WARNING, logger=ss.logger.name):
        deleted = await store.delete("sess-b")

    assert deleted is (not broken)
    assert ("session:sess-b" in client.data) is broken
    assert ("persist until its TTL" in caplog.text) is broken


@pytest.mark.asyncio
@pytest.mark.parametrize("have_client", [True, False])
async def test_no_client_reads_absent_and_drops_writes(monkeypatch, have_client):
    """``get_redis_client`` returns None while Redis cannot be reached."""
    client = _FakeAsyncRedis(broken=False)

    async def _get_client():
        return client if have_client else None

    monkeypatch.setattr(ss, "get_redis_client", _get_client)
    store = ss.RedisSessionStore(ttl_seconds=60)
    await store.save(ProjectSession.new("sess-c"))
    loaded = await store.get("sess-c")
    assert (loaded is not None) is have_client
    assert (await store.delete("sess-c")) is have_client
