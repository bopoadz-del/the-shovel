"""STAMP TO POST: the watchdog keeps how long each heartbeat waited between
its stamp and its post to the loop (advisor 07:28Z 18 Nov).

#435 measured it: stamped before the CPU snapshot (/proc reads), every
heartbeat's lag carried the snapshot's cost. #439 e2a33cf moved the snapshot
after the post but before the wait, which left the loop with no heartbeat in
flight for as long as the snapshot took. Advisor 10:10Z 18 Nov: the snapshot
leaves the heartbeat thread entirely. These tests drive a real monitor on a
real loop: (a) a slow snapshot with an idle loop is no stall; (b) a loop truly
blocked is one stall of about the block; (c) a block that starts while a slow
snapshot runs is one stall of the block; (d) a block that begins the instant a
heartbeat is served is seen by the next heartbeat; (e) the loop is never left
without a heartbeat in flight for two ticks. The stamp-to-post, snapshot and
unobserved figures stay diagnostic: nothing gated reads them.

Every test builds its own input; no document text.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from app.core import loop_watchdog
from app.core.loop_watchdog import LagMonitor, stamp_to_post_summary


def _monitor() -> LagMonitor:
    return LagMonitor(asyncio.new_event_loop(), threshold_ms=250, tick_ms=20, log=False)


@pytest.mark.parametrize("stalled", [True, False])
def test_a_heartbeat_keeps_its_stamp_to_post_beside_its_lag(stalled):
    mon = _monitor()
    try:
        mon.note_stamp_to_post(300.0 if stalled else 40.0, 12.0, stalled)
        assert mon.max_stamp_to_post_ms == 12.0
        assert mon.stall_stamp_to_post == ([(300.0, 12.0)] if stalled else [])
        assert mon.worst_lag_stamp_to_post == ((300.0, 12.0) if stalled else (40.0, 12.0))
    finally:
        mon.loop.close()


@pytest.mark.parametrize("second_lag", [500.0, 100.0])
def test_the_worst_lag_pair_follows_the_largest_lag_not_the_largest_post(second_lag):
    mon = _monitor()
    try:
        mon.note_stamp_to_post(300.0, 280.0, True)
        mon.note_stamp_to_post(second_lag, 5.0, second_lag >= 250)
        # The largest stamp-to-post is kept whatever the lag order.
        assert mon.max_stamp_to_post_ms == 280.0
        if second_lag > 300.0:
            assert mon.worst_lag_stamp_to_post == (500.0, 5.0)
            assert mon.stall_stamp_to_post == [(300.0, 280.0), (500.0, 5.0)]
        else:
            assert mon.worst_lag_stamp_to_post == (300.0, 280.0)
            assert mon.stall_stamp_to_post == [(300.0, 280.0)]
    finally:
        mon.loop.close()


@pytest.mark.parametrize("pairs,expected", [
    ([(310.0, 20.0), (650.0, 610.0), (280.0, 270.0)], "worst_stall=lag=650ms/s2p=610ms"),
    ([(400.0, 3.0)], "worst_stall=lag=400ms/s2p=3ms"),
    ([], "worst_stall=none"),
])
def test_the_summary_names_the_worst_stall_and_its_stamp_to_post(pairs, expected):
    line = stamp_to_post_summary(612.4, pairs, (650.0, 610.0))
    assert line.startswith("stamp_to_post_max=612ms ")
    assert expected in line
    assert line.endswith("worst_lag=650ms/s2p=610ms")


@pytest.mark.parametrize("snapshot_max,expected", [
    (351.6, " snapshot_max=352ms"),
    (0.0, " snapshot_max=0ms"),
    (None, ""),
])
def test_the_summary_prints_the_snapshot_as_its_own_figure(snapshot_max, expected):
    line = stamp_to_post_summary(1.0, [], (40.0, 1.0), snapshot_max)
    assert line == "stamp_to_post_max=1ms worst_stall=none worst_lag=40ms/s2p=1ms" + expected


@pytest.mark.parametrize("unobserved_max,expected", [(41.7, " unobserved_max=42ms"), (None, "")])
def test_the_summary_prints_the_unobserved_window(unobserved_max, expected):
    line = stamp_to_post_summary(0.0, [], (12.0, 0.0), 350.0, unobserved_max)
    assert line == ("stamp_to_post_max=0ms worst_stall=none worst_lag=12ms/s2p=0ms"
                    " snapshot_max=350ms" + expected)


def _slow_snapshot(snapshot_s: float, monkeypatch, started=None, fail: bool = False) -> list:
    """``cpu_snapshot`` takes ``snapshot_s`` (it sleeps, as a /proc read
    waiting on the GIL would); ``started`` is set as each one begins. With
    ``fail`` every reading raises."""
    calls: list = []

    def snapshot():
        calls.append(1)
        if started is not None:
            started.set()
        if fail:
            raise OSError("no /proc here")
        if snapshot_s:
            time.sleep(snapshot_s)
        return 0.0, {}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", snapshot)
    monkeypatch.setattr(loop_watchdog, "host_cpu", lambda: {})
    return calls


def _run(mon: LagMonitor, body) -> LagMonitor:
    """Run ``body(mon)`` on the monitor's loop with the monitor started, then
    stop it."""
    async def run():
        await asyncio.sleep(0.15)  # a few heartbeats and readings first
        await body(mon)
        await asyncio.sleep(0.15)

    mon.start()
    try:
        mon.loop.run_until_complete(run())
    finally:
        mon.stop()
        mon.loop.close()
    return mon


async def _block_then_wait(mon: LagMonitor, block_s: float) -> None:
    time.sleep(block_s)  # the loop thread itself is blocked
    deadline = time.monotonic() + 5
    while not mon.stall_recorded.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)


@pytest.mark.parametrize("snapshot_s", [0.35, 0.6])
def test_a_slow_snapshot_with_an_idle_loop_is_not_a_stall(snapshot_s, monkeypatch):
    """(a) The monitor's own snapshot, however slow, is never charged to the
    loop: an idle loop shows no stall and a lag near zero, and the
    snapshot's duration is kept as its own figure."""
    calls = _slow_snapshot(snapshot_s, monkeypatch)

    async def idle(mon):
        await asyncio.sleep(snapshot_s * 3)

    mon = _run(_monitor(), idle)
    assert len(calls) >= 2, "the slow snapshot never ran"
    assert not mon.stalls, mon.stalls
    assert mon.max_lag_ms < 50, mon.max_lag_ms
    assert mon.max_stamp_to_post_ms < 20, mon.max_stamp_to_post_ms
    assert mon.max_snapshot_ms >= snapshot_s * 1000 * 0.9, mon.max_snapshot_ms
    assert mon.snapshots == len(calls)


@pytest.mark.parametrize("block_s", [0.35, 0.5])
def test_a_blocked_loop_with_an_instant_snapshot_is_one_stall_of_the_block(block_s, monkeypatch):
    """(b) The gate keeps its teeth: a loop truly blocked past the limit is
    one stall, and its lag is about the block."""
    _slow_snapshot(0.0, monkeypatch)
    mon = _run(_monitor(), lambda m: _block_then_wait(m, block_s))
    assert len(mon.stalls) == 1, [lag for lag, _ in mon.stalls]
    lag, s2p = mon.stall_stamp_to_post[0]
    assert lag >= block_s * 1000 - 50, (lag, block_s)
    assert lag <= block_s * 1000 + 100, (lag, block_s)
    assert s2p < 20, (lag, s2p)
    assert mon.max_lag_ms == pytest.approx(lag)


@pytest.mark.parametrize("snapshot_s", [0.35, 0.6])
def test_a_blocked_loop_while_a_slow_snapshot_runs_is_one_stall_of_the_block(snapshot_s, monkeypatch):
    """(c) The loop is blocked 350 ms, starting while a slow snapshot is
    running in its thread. The stall is the block: the snapshot neither
    hides it nor deducts from it."""
    block_s = 0.35
    started = threading.Event()
    _slow_snapshot(snapshot_s, monkeypatch, started)

    async def block_inside_a_snapshot(mon):
        started.clear()
        while not started.is_set():
            await asyncio.sleep(0.002)
        await _block_then_wait(mon, block_s)

    mon = _run(_monitor(), block_inside_a_snapshot)
    assert len(mon.stalls) == 1, [lag for lag, _ in mon.stalls]
    lag, s2p = mon.stall_stamp_to_post[0]
    assert lag >= 300, (lag, snapshot_s)
    assert lag <= block_s * 1000 + 100, (lag, snapshot_s)
    assert s2p < 20, (lag, s2p)


@pytest.mark.parametrize("snapshot_s", [0.0, 0.35, 0.6])
def test_a_block_that_begins_right_after_a_heartbeat_is_served_is_seen(snapshot_s, monkeypatch):
    """(d) The blind window of e2a33cf: a 350 ms block that begins the instant
    a heartbeat is served. The next heartbeat is posted one tick later and
    waits out the rest of the block, whatever the snapshot is doing."""
    block_s = 0.35
    _slow_snapshot(snapshot_s, monkeypatch)
    mon = _monitor()
    trigger = threading.Event()
    post = mon.loop.call_soon_threadsafe

    def post_then_block(callback, *args):
        if trigger.is_set() and getattr(callback, "__name__", "") == "_serve":
            trigger.clear()

            def serve_then_block():
                callback(*args)
                time.sleep(block_s)  # begins right after the serve

            return post(serve_then_block)
        return post(callback, *args)

    mon.loop.call_soon_threadsafe = post_then_block

    async def body(mon):
        await asyncio.sleep(snapshot_s * 1.5)  # a slow snapshot is under way
        trigger.set()
        deadline = time.monotonic() + 5
        while not mon.stall_recorded.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

    mon = _run(mon, body)
    assert not trigger.is_set(), "the block was never posted"
    assert len(mon.stalls) == 1, (mon.max_lag_ms, [lag for lag, _ in mon.stalls])
    assert mon.stalls[0][0] >= 250, mon.stalls[0][0]


@pytest.mark.parametrize("tick_ms", [50, 80])
def test_the_loop_is_never_unobserved_for_more_than_two_ticks(tick_ms, monkeypatch):
    """(e) Between one heartbeat's service and the next one's post there is
    the tick sleep and nothing else, even while a 600 ms snapshot runs."""
    _slow_snapshot(0.6, monkeypatch)
    mon = LagMonitor(asyncio.new_event_loop(), threshold_ms=250, tick_ms=tick_ms, log=False)

    async def idle(mon):
        await asyncio.sleep(1.5)

    mon = _run(mon, idle)
    assert mon.max_snapshot_ms >= 600 * 0.9, mon.max_snapshot_ms
    assert tick_ms * 0.9 <= mon.max_unobserved_ms < 2 * tick_ms, mon.max_unobserved_ms
    assert not mon.stalls, mon.stalls


@pytest.mark.parametrize("fail", [True, False])
def test_a_dead_snapshot_never_reaches_the_heartbeat(fail, monkeypatch):
    """Every reading raising leaves the heartbeat as it is: the stall is
    still found, and its line says ``cpu unknown`` instead of figures."""
    _slow_snapshot(0.0, monkeypatch, fail=fail)
    mon = _run(_monitor(), lambda m: _block_then_wait(m, 0.35))
    assert len(mon.stalls) == 1, [lag for lag, _ in mon.stalls]
    assert mon.stalls[0][0] >= 300, mon.stalls[0][0]
    text = mon.stalls[0][1]
    if fail:
        assert text.startswith("cpu unknown :: "), text[:80]
        assert mon.snapshots == 0
    else:
        assert text.startswith("sum window="), text[:80]
        assert mon.snapshots >= 2


@pytest.mark.parametrize("readings,expected", [([350.0, 12.0], (2, 350.0)), ([5.0], (1, 5.0))])
def test_each_snapshot_is_counted_and_the_slowest_kept(readings, expected):
    mon = _monitor()
    try:
        for ms in readings:
            mon.note_snapshot(ms)
        assert (mon.snapshots, mon.max_snapshot_ms) == expected
    finally:
        mon.loop.close()


@pytest.mark.parametrize("fail", [False, True])
def test_the_snapshot_thread_publishes_its_reading_and_never_raises(fail, monkeypatch):
    """``_snap_run`` publishes ``(process, threads, host, t0)``; a reading that
    raises is skipped and leaves nothing published, without an exception."""
    mon = _monitor()

    def snapshot():
        mon._stop.set()  # one pass of the loop
        if fail:
            raise OSError("no /proc here")
        return 7.0, {1: 3.0}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", snapshot)
    monkeypatch.setattr(loop_watchdog, "host_cpu", lambda: {"total": 9})
    try:
        mon._snap_run()
    finally:
        mon.loop.close()
    if fail:
        assert (mon._snap, mon.snapshots) == (None, 0)
    else:
        assert mon._snap[:3] == (7.0, {1: 3.0}, {"total": 9})
        assert mon.snapshots == 1
