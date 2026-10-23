"""off_loop runs sync work in the bounded turn pool as if inline; the loop
watchdog logs who froze the loop."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time

import pytest

from app.core import loop_watchdog, offload

VAR = contextvars.ContextVar("offload_test_var", default="unset")


def test_off_loop_sees_the_callers_context_and_hands_its_writes_back():
    def work(x):
        seen = VAR.get()
        VAR.set("written-in-thread")
        return seen, x * 2, threading.current_thread().name

    async def run():
        VAR.set("from-caller")
        seen, doubled, thread = await offload.off_loop(work, 21)
        return seen, doubled, thread, VAR.get()

    seen, doubled, thread, after = asyncio.run(run())
    assert (seen, doubled, after) == ("from-caller", 42, "written-in-thread")
    assert thread.startswith("turn-io")


def test_off_loop_passes_keyword_arguments_and_raises_errors():
    def work(a, *, b):
        if b == "boom":
            raise ValueError("boom")
        return a + b

    assert asyncio.run(offload.off_loop(work, "x", b="y")) == "xy"
    with pytest.raises(ValueError):
        asyncio.run(offload.off_loop(work, "x", b="boom"))


@pytest.mark.parametrize("env,expected", [
    ({"TURN_IO_WORKERS": "3"}, 3),
    ({"TURN_IO_WORKERS": "", "DB_POOL_SIZE": "7"}, 7),
    ({"TURN_IO_WORKERS": "nope", "DB_POOL_SIZE": "4"}, 4),
    ({"TURN_IO_WORKERS": "", "DB_POOL_SIZE": ""}, 10),
    ({"TURN_IO_WORKERS": "0"}, 1),
])
def test_the_pool_is_sized_by_the_database_pool(monkeypatch, env, expected):
    for k in ("TURN_IO_WORKERS", "DB_POOL_SIZE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert offload._workers() == expected


def test_the_watchdog_logs_a_stall_with_the_blocking_code(monkeypatch, caplog):
    monkeypatch.setattr(loop_watchdog, "_started", None)
    monkeypatch.setenv("LOOP_WATCHDOG_MS", "100")
    monkeypatch.setenv("LOOP_WATCHDOG_TICK_MS", "10")
    logged = []
    monkeypatch.setattr(loop_watchdog._LOG, "warning", lambda fmt, *a: logged.append(fmt % a))

    async def run():
        mon = loop_watchdog.start()
        assert loop_watchdog.start() is mon  # once per process
        await asyncio.sleep(0.05)
        time.sleep(0.35)  # a synchronous call on the loop
        await asyncio.sleep(0.1)
        mon.stop()

    asyncio.run(run())
    assert logged and logged[0].startswith("LOOP_STALL lag=")
    assert "test_the_watchdog_logs_a_stall_with_the_blocking_code" in logged[0]


def test_the_watchdog_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(loop_watchdog, "_started", None)
    monkeypatch.setenv("LOOP_WATCHDOG_MS", "0")
    assert asyncio.run(_start()) is None
    monkeypatch.setenv("LOOP_WATCHDOG_MS", "not-a-number")
    mon = asyncio.run(_start())
    assert mon is not None and abs(mon.threshold - 0.25) < 1e-9
    mon.stop()


async def _start():
    return loop_watchdog.start()


def test_the_sample_names_other_threads_when_the_loop_is_starved():
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, name="busy-worker", daemon=True)
    t.start()
    try:
        out = loop_watchdog.sample_stacks(threading.get_ident())
    finally:
        stop.set()
    assert out.startswith("[loop] ") and "[busy-worker]" in out
