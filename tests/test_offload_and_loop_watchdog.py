"""off_loop runs sync work in the bounded turn pool as if inline; the loop
watchdog logs who froze the loop."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
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
        await asyncio.sleep(0)  # let the loop serve the pending heartbeat
        # Wait on the event the sampler sets, not a fixed sleep: detection is
        # then observed deterministically instead of racing the sampler thread
        # under CI load. The 2s is only a safety bound; it returns on detection.
        assert await asyncio.to_thread(mon.stall_recorded.wait, 2.0)
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


def _watchdog_block_alpha(seconds: float) -> None:
    time.sleep(seconds)  # a synchronous call that freezes the loop


def _watchdog_block_beta(seconds: float) -> None:
    time.sleep(seconds)  # a differently-named synchronous freeze


@pytest.mark.parametrize("blocker,block_s", [
    (_watchdog_block_alpha, 0.30),
    (_watchdog_block_beta, 0.45),
])
def test_worst_stall_always_captures_a_nonempty_loop_stack(blocker, block_s):
    # Regression: the watchdog once recorded a lag with an EMPTY trace (loop id
    # unresolved -> sample_stacks returned ""), so the worst stall told us
    # nothing. The stack must always be captured and must name the blocking
    # frame, whichever sync call froze the loop.
    async def run():
        loop = asyncio.get_running_loop()
        mon = loop_watchdog.LagMonitor(loop, threshold_ms=100, tick_ms=10, log=False).start()
        await asyncio.sleep(0.05)
        blocker(block_s)                 # freeze the loop inside a named function
        await asyncio.sleep(0)           # let the pending heartbeat be served
        assert await asyncio.to_thread(mon.stall_recorded.wait, 2.0)
        mon.stop()
        return mon.stalls

    stalls = asyncio.run(run())
    assert stalls, "a stall must be recorded"
    worst_lag, worst_stack = max(stalls, key=lambda s: s[0])
    assert worst_lag >= 100.0
    assert worst_stack, "the worst stall must carry a non-empty stack"
    assert "[loop]" in worst_stack            # the loop thread was sampled
    assert blocker.__name__ in worst_stack    # and it names the actual blocker


@pytest.mark.parametrize("posted,served_at,now,expected", [
    (10.0, 10.1, 10.6, 100.0),   # served on time; the watchdog woke 500 ms later
    (10.0, 10.4, 10.9, 400.0),   # served late; the loop's own lateness counts
    (10.0, None, 10.6, 600.0),   # not served yet: late until now
])
def test_heartbeat_lag_is_the_loops_lateness_not_the_watchdogs(posted, served_at, now, expected):
    assert loop_watchdog.heartbeat_lag_ms(posted, served_at, now) == pytest.approx(expected)


class _LateWakingEvent(threading.Event):
    """A heartbeat Event whose waiter wakes WAKE_LATE_S after it is set, as a
    watchdog thread starved of the GIL does under load."""
    WAKE_LATE_S = 0.4

    def wait(self, timeout=None):
        got = super().wait(timeout)
        if got:
            time.sleep(self.WAKE_LATE_S)
        return got


class _ThreadingWithLateWaker:
    Event = _LateWakingEvent

    def __getattr__(self, name):
        return getattr(threading, name)


@pytest.mark.parametrize("loop_blocked_s,stall_expected", [
    (0.0, False),    # loop idle and on time: a late-waking watchdog is not a stall
    (0.9, True),     # loop really blocked (longer than the waker gap): still a stall
])
def test_a_late_waking_watchdog_does_not_charge_its_delay_to_the_loop(
        monkeypatch, loop_blocked_s, stall_expected):
    monkeypatch.setattr(loop_watchdog, "threading", _ThreadingWithLateWaker())

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=200,
                                       tick_ms=10, log=False).start()
        await asyncio.sleep(0.05)
        if loop_blocked_s:
            time.sleep(loop_blocked_s)
        await asyncio.sleep(1.0)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    if stall_expected:
        assert mon.stalls and mon.max_lag_ms >= 200
        assert all(stack for _, stack in mon.stalls)
    else:
        assert not mon.stalls
        assert mon.max_lag_ms < 200, mon.max_lag_ms


@pytest.mark.parametrize("block_s", [0.6, 0.9])
def test_a_stall_is_timed_to_when_the_loop_finally_serves_the_heartbeat(block_s):
    """``LagMonitor._run`` keeps waiting on a late heartbeat until the loop
    serves it, and ``_serve`` stamps that moment on the loop, so the recorded
    stall is the whole block, not the threshold at which it was first seen."""
    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=200,
                                       tick_ms=10, log=False).start()
        await asyncio.sleep(0.05)
        time.sleep(block_s)
        await asyncio.sleep(0.3)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    assert mon.stalls, "a blocked loop must record a stall"
    worst = max(lag for lag, _ in mon.stalls)
    assert worst >= block_s * 1000 * 0.8, (worst, block_s)
    assert worst == pytest.approx(mon.max_lag_ms)


# --- Which thread burned the CPU across a stall -------------------------------
# CI 3261ed1 (631 ms): the loop was idle in select() and the 1500-char message
# held only the loop's own frames, so nothing named the thread that kept the
# loop from the GIL. Per-thread CPU over the stall names it.

def _write_task(root, tid: int, comm: str, utime: int, stime: int) -> None:
    d = root / str(tid)
    d.mkdir(parents=True)
    rest = ["S", "1", "1", "1", "0", "-1", "0", "0", "0", "0", "0", str(utime), str(stime), "0", "0"]
    (d / "stat").write_bytes(f"{tid} ({comm}) {' '.join(rest)}\n".encode())


@pytest.mark.parametrize("tasks,expected_ticks", [
    ([(101, "python", 30, 20), (102, "turn-io) x", 7, 0)], {101: 50, 102: 7}),
    ([], {}),
])
def test_thread_cpu_is_read_per_thread_from_the_task_dir(tmp_path, tasks, expected_ticks):
    for tid, comm, ut, st in tasks:
        _write_task(tmp_path, tid, comm, ut, st)
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    got = loop_watchdog.thread_cpu_ms(str(tmp_path))
    assert got == {tid: t * 1000.0 / tick for tid, t in expected_ticks.items()}


@pytest.mark.parametrize("make", ["absent", "a_file"])
def test_thread_cpu_is_empty_where_the_os_has_no_task_dir(tmp_path, make):
    target = tmp_path / make
    if make == "a_file":
        target.write_text("not a directory")
    assert loop_watchdog.thread_cpu_ms(str(target)) == {}


@pytest.mark.parametrize("after,expected", [
    ({1: 100.0, 2: 690.0, 3: 40.0}, "cpu turn-io_2=600ms loop=50ms"),
    ({1: 50.0, 2: 90.0, 3: 40.0}, ""),
])
def test_the_stall_names_the_threads_that_burned_cpu_across_it(after, expected):
    before = {1: 50.0, 2: 90.0, 3: 40.0}
    names = {1: "loop", 2: "turn-io_2", 3: "idle"}
    assert loop_watchdog.busiest_threads(before, after, names) == expected
