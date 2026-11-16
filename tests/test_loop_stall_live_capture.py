"""A loop stall is captured WHILE it happens, and its CPU figures share one clock.

RED MAIN (CI e312ea2, 06e541d, PR #394): the watchdog's stacks were taken when
the stall ENDED -- the sampler is a Python thread and needs the GIL to wake, so
a thread holding the GIL across one long call (turn-io_3 burned 630 ms) was
already in its next call by the time anything was sampled. And the same stall
logged threads=820ms against proc=190ms: two clocks read at two instants.

Every test builds its own input; no document text.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time

import pytest

from app.core import loop_watchdog


def _gil_holding_call(n: int) -> None:
    """One C call that never releases the GIL: a backtracking regex."""
    re.fullmatch(r"(x+x+)+y", "x" * n)


def _size_for(seconds: float) -> int:
    n = 16
    while n < 32:
        t = time.perf_counter()
        _gil_holding_call(n)
        if time.perf_counter() - t >= seconds:
            return n
        n += 1
    return n


@pytest.mark.parametrize("hold", [True, False])
def test_the_live_dump_names_the_call_that_holds_the_gil_during_the_stall(hold):
    """A worker thread holds the GIL inside one C call while the loop waits.
    The live dump (C timer, no GIL) shows the worker IN that call; with no
    such call there is no live dump at all."""
    n = _size_for(0.6) if hold else 0

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(0.1)
        # Like a pool thread, the worker outlives its call, so it still has
        # its name when the stall ends and the dump is formatted.
        done, release = threading.Event(), threading.Event()

        def work():
            _gil_holding_call(n)
            done.set()
            release.wait(5)

        if hold:
            worker = threading.Thread(target=work, name="gil-holder")
            worker.start()
            while not done.is_set():
                await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.6)
        await asyncio.sleep(0.2)
        mon.stop()
        release.set()
        return mon

    mon = asyncio.run(run())
    if hold:
        assert mon.live_dumps, f"no live dump; max_lag={mon.max_lag_ms:.0f} stalls={len(mon.stalls)}"
        _, during = max(mon.live_dumps)
        assert "[gil-holder]" in during and "_gil_holding_call" in during, during
        # The hold is SEEN whichever side of a heartbeat it began on: as a
        # late heartbeat (a stall carrying the live dump) or, when it began
        # while the monitor slept, as the monitor's own late wake.
        if mon.stalls:
            worst = max(mon.stalls)[1]
            assert "DURING" in worst and "_gil_holding_call" in worst
        else:
            assert mon.max_wake_late_ms >= 250, mon.max_wake_late_ms
            assert during.startswith("WAKE_LATE ")
    else:
        assert mon.live_dumps == [] and mon.stalls == []
        assert mon.max_wake_late_ms < 150, mon.max_wake_late_ms


@pytest.mark.parametrize("hold", [True, False])
def test_a_gil_hold_that_begins_while_the_monitor_sleeps_is_not_invisible(hold):
    """The blind spot this closes: the hold starts inside the monitor's tick
    sleep, so no heartbeat is pending and ``max_lag_ms`` stays ~0. The late
    wake and the live dump still name it."""
    n = _size_for(0.6) if hold else 0
    loop = asyncio.new_event_loop()
    mon = loop_watchdog.LagMonitor(loop, threshold_ms=250, tick_ms=400, log=False, live_ms=150)
    # No loop running: drive the sleep path directly on this thread while a
    # worker holds the GIL from inside that sleep.
    release = threading.Event()

    def work():
        time.sleep(0.05)
        if hold:
            _gil_holding_call(n)
        release.wait(5)

    worker = threading.Thread(target=work, name="gil-holder")
    worker.start()
    try:
        mon._sleep_watched(mon._live, {})
    finally:
        release.set()
        worker.join()
        mon._live.close()
        loop.close()
    if hold:
        assert mon.max_wake_late_ms >= 150, mon.max_wake_late_ms
        assert mon.live_dumps and "[gil-holder]" in mon.live_dumps[0][1], mon.live_dumps
        assert "_gil_holding_call" in mon.live_dumps[0][1]
    else:
        assert mon.max_wake_late_ms < 150 and mon.live_dumps == []


@pytest.mark.parametrize("first_closed,second_active", [(False, False), (True, True)])
def test_only_one_monitor_owns_the_process_wide_dump_timer(first_closed, second_active):
    """faulthandler has one timer per process: a second owner is refused
    while the first holds it, and granted once the first lets it go."""
    first = loop_watchdog.LiveDump(150)
    assert first.active
    if first_closed:
        first.close()
    second = loop_watchdog.LiveDump(150)
    try:
        assert second.active is second_active
        if not second_active:
            assert second.take() == ""
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("text,names,expected", [
    ('Thread 0x00000000000000ff (most recent call first):\n'
     '  File "/w/app/x.py", line 7 in slow\n'
     '  File "/w/app/y.py", line 3 in run\n\n'
     'Current thread 0x0000000000000010 (most recent call first):\n'
     '  File "/usr/lib/selectors.py", line 468 in select\n',
     {255: "turn-io_3", 16: "MainThread"},
     "[turn-io_3] /w/app/x.py:7 slow <- /w/app/y.py:3 run || [*MainThread] /usr/lib/selectors.py:468 select"),
    ('Thread 0xabc (most recent call first):\n  File "a.py", line 1 in f\n',
     {}, "[0xabc] a.py:1 f"),
])
def test_the_live_dump_is_one_line_of_named_threads(text, names, expected):
    assert loop_watchdog.format_live_dump(text, names) == expected


def _stat_line(pid: int, utime: int, stime: int) -> bytes:
    rest = ["S", "1", "1", "1", "0", "-1", "0", "0", "0", "0", "0", str(utime), str(stime), "0", "0"]
    return f"{pid} (py) thing) {' '.join(rest)}\n".encode()


@pytest.mark.parametrize("proc_ticks,present", [(70, True), (0, False)])
def test_process_and_thread_cpu_are_read_on_one_clock(tmp_path, proc_ticks, present):
    tasks = tmp_path / "task"
    (tasks / "11").mkdir(parents=True)
    (tasks / "11" / "stat").write_bytes(_stat_line(11, 30, 10))
    stat = tmp_path / "stat"
    if present:
        stat.write_bytes(_stat_line(11, proc_ticks, 0))
    tick = loop_watchdog.os.sysconf("SC_CLK_TCK") if hasattr(loop_watchdog.os, "sysconf") else 100
    proc, threads = loop_watchdog.cpu_snapshot(str(tasks), str(stat))
    assert threads == {11: 40 * 1000.0 / tick}
    if present:
        assert proc == proc_ticks * 1000.0 / tick
    else:  # no /proc/self/stat: the os.times() fallback, never None
        assert isinstance(proc, float) and proc >= 0


@pytest.mark.parametrize("threads_after,expected", [
    # a thread absent from the earlier snapshot is counted apart, not summed
    ({1: 60.0, 2: 630.0}, "sum window=400ms proc=50ms threads=50ms/1 new=1"),
    ({1: 60.0}, "sum window=400ms proc=50ms threads=50ms/1"),
])
def test_a_thread_born_inside_the_window_is_not_summed_as_a_delta(threads_after, expected):
    got = loop_watchdog.stall_cpu_summary({1: 10.0}, threads_after, 50.0, {}, {}, 400.0)
    assert got == expected


@pytest.mark.parametrize("loop_blocked", [False, True])
def test_the_monitors_own_snapshot_is_not_counted_as_loop_lag(monkeypatch, loop_blocked):
    """The per-thread CPU snapshot is slow under GIL contention. It is the
    monitor's own work, done before the heartbeat is stamped: a slow snapshot
    with a free loop is no stall; a loop really blocked as long still is."""
    def slow_snapshot(*_a, **_k):
        time.sleep(0.35)
        return 0.0, {}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", slow_snapshot)
    monkeypatch.setattr(loop_watchdog, "_CPU_SNAPSHOT_S", 0.0)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False).start()
        await asyncio.sleep(0.9)
        if loop_blocked:
            time.sleep(0.8)  # the loop itself, blocked longer than one snapshot
            await asyncio.sleep(0.5)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    if loop_blocked:
        assert mon.max_lag_ms >= 250, mon.max_lag_ms
    else:
        assert mon.max_lag_ms < 250 and mon.stalls == [], (mon.max_lag_ms, mon.stalls[:1])
