"""The turn's document injection never runs on the loop's default executor.

RED MAIN (CI 06e541d / e312ea2 / PR #394): every loop-lag capture showed
default-executor threads (``asyncio_*``) inside psycopg while the loop was
late. ``_rag_inject_off_process`` -- the one choke point every turn's
injection goes through -- sent ``rag_inject`` to ``asyncio.to_thread``, and
the retrieval-worker wait to ``run_in_executor(None, ...)``: unbounded, unnamed,
beside the bounded retrieval pool. Both now go through ``offload.to_thread_in``
on the named, bounded retrieval pool.

Every test builds its own input; no document text.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core import offload
from app.core.loop_watchdog import LagMonitor

LIMIT_MS = 250


@pytest.fixture
def fresh_retrieval_pool(monkeypatch):
    old = offload._NAMED_POOLS.pop("retrieval", None)
    yield
    pool = offload._NAMED_POOLS.pop("retrieval", None)
    if pool is not None:
        pool.shutdown(wait=True)
    if old is not None:
        offload._NAMED_POOLS["retrieval"] = old


class _NoDefaultExecutor(ThreadPoolExecutor):
    """Installed as the loop's default executor: any use is recorded."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.used: list = []

    def submit(self, fn, *args, **kwargs):
        self.used.append(getattr(fn, "func", fn))
        raise AssertionError("the default executor was used on the turn path")



def _wire(monkeypatch, worker_process: bool, hold_s: float, seen: list):
    import app.agents.runtime as runtime_mod
    import app.core.rag.retrieval_worker as rw

    def slow_rag_inject(**kwargs):
        # The search and its database read held slow: a blocking wait that
        # releases the GIL, as a socket read does.
        seen.append(threading.current_thread().name)
        time.sleep(hold_s)
        return (None, {"held": hold_s})

    def slow_call_sync(job, *args):
        seen.append(threading.current_thread().name)
        time.sleep(hold_s)
        return (None, {"held": hold_s})

    monkeypatch.setattr(runtime_mod, "rag_inject", slow_rag_inject)
    monkeypatch.setattr(rw, "enabled", lambda: worker_process)
    monkeypatch.setattr(rw, "call_sync", slow_call_sync)
    return runtime_mod


@pytest.mark.parametrize("worker_process", [False, True])
def test_injection_runs_on_the_named_retrieval_pool_not_the_default_executor(
        monkeypatch, fresh_retrieval_pool, worker_process):
    monkeypatch.setenv("RETRIEVAL_THREADS", "2")
    seen: list = []
    runtime_mod = _wire(monkeypatch, worker_process, 0.01, seen)
    guard = _NoDefaultExecutor()

    async def run():
        asyncio.get_running_loop().set_default_executor(guard)
        return await runtime_mod._rag_inject_off_process(
            user_message="q", project_id="p", conversation_id="c", user_id="u",
            agent_name="project-assistant")

    result = asyncio.run(run())
    assert result == (None, {"held": 0.01})
    assert guard.used == []
    assert seen and all(n.startswith("retrieval-work") for n in seen), seen


@pytest.mark.parametrize("users,hold_s", [(15, 0.4), (6, 0.2)])
def test_slow_search_and_database_keep_the_loop_heartbeat_under_the_limit(
        monkeypatch, fresh_retrieval_pool, users, hold_s):
    """Many concurrent turns whose injection is held slow: the turns queue on
    the bounded pool, the loop keeps serving heartbeats."""
    monkeypatch.setenv("RETRIEVAL_THREADS", "2")
    seen: list = []
    runtime_mod = _wire(monkeypatch, False, hold_s, seen)

    async def run():
        loop = asyncio.get_running_loop()
        mon = LagMonitor(loop, threshold_ms=LIMIT_MS, tick_ms=20, log=False).start()
        try:
            await asyncio.gather(*(runtime_mod._rag_inject_off_process(
                user_message=f"q{n}", project_id="p", conversation_id=f"c{n}", user_id=f"u{n}",
                agent_name="project-assistant") for n in range(users)))
        finally:
            mon.stop()
        return mon

    started = time.monotonic()
    mon = asyncio.run(run())
    elapsed = time.monotonic() - started
    assert len(seen) == users
    assert set(seen) <= {"retrieval-work_0", "retrieval-work_1"}, set(seen)
    # Bounded: two at a time, so the batch took about users/2 holds.
    assert elapsed >= (users // 2) * hold_s * 0.9
    assert mon.max_lag_ms < LIMIT_MS, (mon.max_lag_ms, mon.stalls[:1])
