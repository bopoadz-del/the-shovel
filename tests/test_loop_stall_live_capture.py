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
    """faulthandler has one timer per process: ``LiveDump.__init__`` refuses a
    second owner while the first holds it, and grants it once the first lets
    it go."""
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


@pytest.mark.parametrize("snapshot_s,stalled", [(0.35, True), (0.0, False)])
def test_the_heartbeat_is_stamped_before_the_monitors_snapshot(monkeypatch, snapshot_s, stalled):
    """Ruling (advisor 04:18Z): the stamp stays BEFORE the CPU snapshot, so a
    slow snapshot still counts as lag, exactly as the timing test measured it
    since #380. The live timer is armed with the stamp, so such a stall
    carries a dump naming what ran during it -- here the snapshot itself."""
    def snapshot(*_a, **_k):
        time.sleep(snapshot_s)
        return 0.0, {}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", snapshot)
    monkeypatch.setattr(loop_watchdog, "_CPU_SNAPSHOT_S", 0.0)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(0.9)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    if stalled:
        assert mon.max_lag_ms >= 250 and mon.stalls, mon.max_lag_ms
        assert any("snapshot" in text for _, text in mon.live_dumps), mon.live_dumps[:1]
    else:
        assert mon.stalls == [] and mon.live_dumps == [], (mon.max_lag_ms, mon.stalls[:1])


@pytest.mark.parametrize("lags,kept", [
    ([5.0, 1.0, 9.0, 3.0], [9.0, 5.0, 3.0]),
    ([2.0, 4.0], [4.0, 2.0]),
])
def test_live_dumps_keep_only_the_worst(lags, kept):
    """A monitor lives as long as the process; its dump list is bounded and
    keeps the latest-served (worst) entries."""
    got: list = []
    for lag in lags:
        loop_watchdog.keep_worst(got, (lag, f"d{lag}"), cap=3)
    assert sorted((lag for lag, _ in got), reverse=True) == kept


@pytest.mark.parametrize("live_ms,armed", [(0.0, False), (150.0, True)])
def test_live_capture_is_off_unless_asked(live_ms, armed):
    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), log=False, live_ms=live_ms)
        try:
            return mon._live is not None and mon._live.active
        finally:
            if mon._live is not None:
                mon._live.close()

    assert asyncio.run(run()) is armed


@pytest.mark.parametrize("cancelled,fired", [(True, False), (False, True)])
def test_a_cancelled_timer_writes_nothing(cancelled, fired):
    """``cancel`` is what makes a dump mean "still unserved": a timer
    cancelled before its delay writes nothing; one left armed writes."""
    live = loop_watchdog.LiveDump(40)
    try:
        live.arm()
        if cancelled:
            live.cancel()
        time.sleep(0.25)
        assert bool(live.take()) is fired
    finally:
        live.close()


@pytest.mark.parametrize("who_cancels,fired", [("refused", True), ("owner", False)])
def test_a_refused_owner_cannot_cancel_the_owners_timer(who_cancels, fired):
    """The timer is process-wide: a monitor refused ownership must not touch it."""
    owner = loop_watchdog.LiveDump(40)
    refused = loop_watchdog.LiveDump(40)
    try:
        assert owner.active and not refused.active
        owner.arm()
        (refused if who_cancels == "refused" else owner).cancel()
        time.sleep(0.25)
        assert bool(owner.take()) is fired
    finally:
        refused.close()
        owner.close()


@pytest.mark.parametrize("monitor_slow,loop_late_ms,near", [(True, 0, False), (False, 190, True)])
def test_a_served_heartbeat_is_never_a_near_miss(monkeypatch, monitor_slow, loop_late_ms, near):
    """Serving a heartbeat cancels its timer (``_serve``). The monitor itself
    may be slow after that -- held 300 ms after every heartbeat here -- and a
    heartbeat already served still never counts as late. A loop really late
    past live_ms (150) but under the limit (250) is a near miss."""
    if monitor_slow:
        real = loop_watchdog.heartbeat_lag_ms

        def slow_after_serve(posted, served_at, now):
            time.sleep(0.3)
            return real(posted, served_at, now)

        monkeypatch.setattr(loop_watchdog, "heartbeat_lag_ms", slow_after_serve)

    async def run():
        mon = loop_watchdog.LagMonitor(asyncio.get_running_loop(), threshold_ms=250,
                                       tick_ms=10, log=False, live_ms=150).start()
        await asyncio.sleep(1.0)
        for _ in range(5 if loop_late_ms else 0):
            time.sleep(loop_late_ms / 1000)  # the loop itself, late under the limit
            await asyncio.sleep(0.05)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    assert mon.stalls == [], mon.stalls[:1]
    # A near miss is counted; its dump is dropped, never kept.
    assert mon.live_dumps == [], mon.live_dumps[:1]
    assert (mon.live_near_misses > 0) is near, mon.live_near_misses
