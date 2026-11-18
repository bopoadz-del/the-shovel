"""STAMP TO POST: the watchdog keeps how long each heartbeat waited between
its stamp and its post to the loop (advisor 07:28Z 18 Nov).

#435 measured it: stamped before the CPU snapshot (/proc reads), every
heartbeat's lag carried the snapshot's cost. Outcome A (advisor 09:17Z 18
Nov): the heartbeat is stamped at the post and the snapshot runs after the
post. These tests drive a real monitor on a real loop: a slow snapshot with
an idle loop is no stall; a loop truly blocked is still a stall of about the
block; a blocked loop with a slow snapshot is a stall of the block, not the
sum. The stamp-to-post and snapshot figures stay diagnostic: nothing gated
reads them.

Every test builds its own input; no document text.
"""
from __future__ import annotations

import asyncio
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


def _run_monitor(snapshot_s: float, block_s: float, monkeypatch) -> tuple[LagMonitor, list]:
    """A real LagMonitor on a real loop. ``cpu_snapshot`` takes ``snapshot_s``
    (it sleeps, as a /proc read waiting on the GIL would); the loop itself is
    blocked for ``block_s`` (0 = idle the whole time)."""
    calls: list = []

    def snapshot():
        calls.append(1)
        if snapshot_s:
            time.sleep(snapshot_s)
        return 0.0, {}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", snapshot)
    monkeypatch.setattr(loop_watchdog, "host_cpu", lambda: {})
    loop = asyncio.new_event_loop()
    mon = LagMonitor(loop, threshold_ms=250, tick_ms=20, log=False)

    async def run():
        # Several snapshots run before the block (and the whole time if idle).
        await asyncio.sleep(max(0.15, snapshot_s * 3))
        if block_s:
            time.sleep(block_s)  # the loop thread itself is blocked
            deadline = time.monotonic() + 5
            while not mon.stall_recorded.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)

    mon.start()
    try:
        loop.run_until_complete(run())
    finally:
        mon.stop()
        loop.close()
    return mon, calls


@pytest.mark.parametrize("snapshot_s", [0.35, 0.6])
def test_a_slow_snapshot_with_an_idle_loop_is_not_a_stall(snapshot_s, monkeypatch):
    """The monitor's own snapshot, however slow, is never charged to the loop:
    an idle loop shows no stall and a lag near zero, and the snapshot's
    duration is kept as its own figure."""
    mon, calls = _run_monitor(snapshot_s, 0.0, monkeypatch)
    assert len(calls) >= 2, "the slow snapshot never ran"
    assert not mon.stalls, mon.stalls
    assert mon.max_lag_ms < 50, mon.max_lag_ms
    assert mon.max_stamp_to_post_ms < 20, mon.max_stamp_to_post_ms
    assert mon.max_snapshot_ms >= snapshot_s * 1000 * 0.9, mon.max_snapshot_ms
    assert mon.snapshots == len(calls)


@pytest.mark.parametrize("block_s", [0.35, 0.5])
def test_a_blocked_loop_with_an_instant_snapshot_is_still_a_stall_of_the_block(block_s, monkeypatch):
    """The gate keeps its teeth: a loop truly blocked past the limit is a
    stall, and its lag is about the block."""
    mon, _ = _run_monitor(0.0, block_s, monkeypatch)
    assert mon.stall_stamp_to_post, "no stall recorded"
    lag, s2p = max(mon.stall_stamp_to_post, key=lambda p: p[0])
    # A heartbeat posted at most one tick (plus a sleep's granularity) after
    # the block began waits out the rest of it.
    assert block_s * 1000 - 80 <= lag <= block_s * 1000 + 100, (lag, block_s)
    assert s2p < 20, (lag, s2p)
    assert mon.max_lag_ms == pytest.approx(lag)


@pytest.mark.parametrize("block_s", [0.7, 0.9])
def test_a_blocked_loop_and_a_slow_snapshot_is_a_stall_of_the_block_not_the_sum(block_s, monkeypatch):
    """Both at once: a 0.35 s snapshot while the loop is blocked. The stall
    is the block; stamped before the snapshot, it was up to block + 350 ms."""
    snapshot_s = 0.35
    mon, _ = _run_monitor(snapshot_s, block_s, monkeypatch)
    assert mon.stall_stamp_to_post, "no stall recorded"
    lag, s2p = max(mon.stall_stamp_to_post, key=lambda p: p[0])
    assert 250 <= lag <= block_s * 1000 + 100, (lag, block_s)
    assert lag < (block_s + snapshot_s) * 1000 - 150, (lag, block_s)
    assert s2p < 20, (lag, s2p)
    assert mon.max_snapshot_ms >= snapshot_s * 1000 * 0.9
