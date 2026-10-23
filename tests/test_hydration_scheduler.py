"""Tests for the cross-process leader lock around the hydration scheduler.

The race the lock fixes: once UVICORN_WORKERS goes above 1 (PILOT.md targets
2 after the RAM upgrade), every worker spawns its own
``hydration_scheduler._loop`` background task at the same UTC hour. Without
coordination they all run ``_run_one_pass`` concurrently, producing torn JSON
writes to ``/tmp/cerebrum_learning_engine.json``, duplicate ``hydration_runs``
rows, and racy ``set_fact`` / ``set_agent_fact`` writes.

These tests verify the scenarios that matter:

1. Lock free: the inner work runs and the lock is released.
2. Lock held by another worker: ``_run_one_pass`` returns immediately and
   does NOT touch the inner work. Critically, the non-owner must also NOT
   call ``delete`` on the owner's key.
3. ``REDIS_URL`` unset (dev mode): the pass runs unconditionally, no shared
   Redis client is requested.
4. Redis configured but unreachable: ``_acquire_leader_lock`` returns the
   ``REDIS_UNAVAILABLE`` sentinel, and ``_run_one_pass`` still executes the
   inner work as a graceful-degradation fallback.

We stub ``app.core.hydration_scheduler.get_redis_client`` so the real shared
client codepath is exercised (``await client.set(...)`` / ``await client.delete(...)``)
without needing a running Redis.
"""

from __future__ import annotations

import pytest


# ── Fake async redis client ────────────────────────────────────────────────


class _FakeAsyncRedis:
    """Minimal async SET/DELETE backend that honours ``nx=True`` semantics.

    Backed by a shared store dict so two ``_FakeAsyncRedis`` instances built
    from the same store simulate two workers talking to the same Redis. That
    is what makes "worker B sees worker A's lock" a real test, not a tautology.
    """

    def __init__(self, store: dict):
        self._store = store
        self.delete_calls: list[str] = []
        self.set_calls: list[tuple[str, str, bool, int]] = []

    async def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex or 0))
        if nx and key in self._store:
            return None  # redis-py returns None when NX fails; falsy
        self._store[key] = value
        return True

    async def delete(self, key):
        self.delete_calls.append(key)
        return 1 if self._store.pop(key, None) is not None else 0


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def reset_redis_client():
    """Drop the shared redis client between tests so state doesn't bleed."""
    from app.core import redis_client
    redis_client.reset_for_tests()
    yield
    redis_client.reset_for_tests()


@pytest.fixture
def fake_redis_store():
    """Shared key/value backing store. Each fresh fixture yields a clean dict
    so test isolation matches a fresh Redis instance."""
    return {}


def _install_fake_get_redis_client(monkeypatch, store: dict) -> dict:
    """Replace ``get_redis_client`` in the scheduler module with a factory
    that hands out ``_FakeAsyncRedis`` instances sharing ``store``.

    Returns a registry of every client built so tests can introspect set/delete
    calls per worker.
    """
    from app.core import hydration_scheduler

    built: dict[str, _FakeAsyncRedis] = {}

    async def _factory():
        client = _FakeAsyncRedis(store)
        built[f"client-{len(built)}"] = client
        return client

    monkeypatch.setattr(hydration_scheduler, "get_redis_client", _factory)
    return built


def _install_get_redis_client_trap(monkeypatch) -> dict:
    """Replace ``get_redis_client`` with a trap that fails if called."""
    from app.core import hydration_scheduler

    calls = {"count": 0}

    async def _trap():
        calls["count"] += 1
        raise AssertionError("get_redis_client must not be invoked when REDIS_URL is unset")

    monkeypatch.setattr(hydration_scheduler, "get_redis_client", _trap)
    return calls


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lock_free_runs_inner_work(monkeypatch, fake_redis_store, reset_redis_client):
    """Happy path: REDIS_URL set, no other worker holds the key. The inner
    work runs and the lock is released in the finally."""
    from app.core import hydration_scheduler

    monkeypatch.setenv("REDIS_URL", "redis://fake")
    clients = _install_fake_get_redis_client(monkeypatch, fake_redis_store)

    ran = {"called": False}

    async def fake_do_pass():
        ran["called"] = True

    monkeypatch.setattr(hydration_scheduler, "_do_hydration_pass", fake_do_pass)

    await hydration_scheduler._run_one_pass()

    assert ran["called"] is True, "lock-free worker must execute inner work"
    # Lock acquired then released → store is empty again
    assert fake_redis_store == {}, "leader must release the lock in finally"
    # Exactly one client was built and it both SET and DELETE'd
    assert len(clients) == 1
    only = next(iter(clients.values()))
    assert any(call[2] is True for call in only.set_calls), "must call SET with nx=True"
    assert len(only.delete_calls) == 1, "owner must release the lock once"


@pytest.mark.asyncio
async def test_lock_held_skips_inner_work(monkeypatch, fake_redis_store, reset_redis_client):
    """The critical concurrency invariant: worker B must NOT run the inner
    work while worker A holds the lock, and worker B must NOT delete A's key
    on its way out."""
    from datetime import datetime, timezone
    from app.core import hydration_scheduler

    monkeypatch.setenv("REDIS_URL", "redis://fake")
    clients = _install_fake_get_redis_client(monkeypatch, fake_redis_store)

    # Pre-seed: pretend worker A acquired the lease for today already.
    today_iso = datetime.now(timezone.utc).date().isoformat()
    fake_redis_store[hydration_scheduler._leader_key(today_iso)] = "worker-a:host-a"

    ran = {"called": False}

    async def fake_do_pass():
        ran["called"] = True

    monkeypatch.setattr(hydration_scheduler, "_do_hydration_pass", fake_do_pass)

    await hydration_scheduler._run_one_pass()

    assert ran["called"] is False, (
        "Second concurrent worker must NOT enter _do_hydration_pass while the "
        "leader lock is held — this is the race the lock exists to prevent."
    )
    # Worker B's client must not have deleted worker A's key.
    assert len(clients) == 1
    only = next(iter(clients.values()))
    assert only.delete_calls == [], "non-owner worker must not release the leader's lock"
    # Worker A's pre-seeded lease is still there.
    assert fake_redis_store[hydration_scheduler._leader_key(today_iso)] == "worker-a:host-a"


@pytest.mark.asyncio
async def test_no_redis_url_runs_unconditionally(monkeypatch, reset_redis_client):
    """Dev mode: REDIS_URL unset → no coordination, the pass always runs.

    Also asserts that ``get_redis_client`` is never even called — we should not
    need a Redis client to do dev work.
    """
    from app.core import hydration_scheduler

    monkeypatch.delenv("REDIS_URL", raising=False)

    redis_client_calls = _install_get_redis_client_trap(monkeypatch)

    ran = {"called": False}

    async def fake_do_pass():
        ran["called"] = True

    monkeypatch.setattr(hydration_scheduler, "_do_hydration_pass", fake_do_pass)

    await hydration_scheduler._run_one_pass()

    assert ran["called"] is True, "dev mode (no REDIS_URL) must run the pass unconditionally"
    assert redis_client_calls["count"] == 0


@pytest.mark.asyncio
async def test_redis_unreachable_returns_sentinel(monkeypatch, reset_redis_client, caplog):
    """Redis URL is set but the shared client is unavailable: the lock helper
    returns the ``REDIS_UNAVAILABLE`` sentinel so the caller can distinguish it
    from "lock already held" and run the pass as a fallback."""
    import logging
    from datetime import datetime, timezone
    from app.core import hydration_scheduler

    monkeypatch.setenv("REDIS_URL", "redis://fake")

    async def _unavailable():
        return None

    monkeypatch.setattr(hydration_scheduler, "get_redis_client", _unavailable)

    today_iso = datetime.now(timezone.utc).date().isoformat()
    with caplog.at_level(logging.WARNING):
        client = await hydration_scheduler._acquire_leader_lock(today_iso)

    assert client is hydration_scheduler.REDIS_UNAVAILABLE
    assert any("Redis configured but unreachable" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_redis_unreachable_runs_fallback(monkeypatch, reset_redis_client, caplog):
    """Redis outage must not skip hydration: ``_run_one_pass`` executes the
    inner work when the lock helper reports ``REDIS_UNAVAILABLE``."""
    import logging
    from datetime import datetime, timezone
    from app.core import hydration_scheduler

    monkeypatch.setenv("REDIS_URL", "redis://fake")

    async def _unavailable():
        return None

    monkeypatch.setattr(hydration_scheduler, "get_redis_client", _unavailable)

    ran = {"called": False}

    async def fake_do_pass():
        ran["called"] = True

    monkeypatch.setattr(hydration_scheduler, "_do_hydration_pass", fake_do_pass)

    with caplog.at_level(logging.WARNING):
        await hydration_scheduler._run_one_pass()

    assert ran["called"] is True, "Redis-unreachable worker must run the pass as a fallback"
    assert any("Redis unavailable; running pass without lock" in rec.message for rec in caplog.records)
