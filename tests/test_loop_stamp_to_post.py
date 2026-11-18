"""STAMP TO POST: the watchdog keeps how long each heartbeat waited between
its stamp and its post to the loop (advisor 07:28Z 18 Nov).

#435 measured it: stamped before the CPU snapshot (/proc reads), every
heartbeat's lag carried the snapshot's cost. PROPOSAL (outcome A): the
heartbeat is stamped at the post, after the snapshot. These tests check the
two sides: a slow snapshot with an idle loop is no longer a stall, and a
blocked loop still is, with a fast snapshot and with a slow one. The
stamp-to-post figure itself stays diagnostic: nothing gated reads it.

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


@pytest.mark.parametrize("snapshot_s", [0.35, 0.6])
def test_a_slow_snapshot_with_an_idle_loop_is_not_a_stall(snapshot_s, monkeypatch):
    """PROPOSAL (outcome A): the heartbeat is stamped at the post, so the
    monitor's own snapshot -- however slow -- is never charged to the loop.
    The loop here is idle the whole time: no stall, and every lag is small."""
    calls: list = []

    def snapshot():
        calls.append(1)
        time.sleep(snapshot_s)
        return 0.0, {}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", snapshot)
    monkeypatch.setattr(loop_watchdog, "host_cpu", lambda: {})
    loop = asyncio.new_event_loop()
    mon = LagMonitor(loop, threshold_ms=250, tick_ms=20, log=False)

    async def run():
        # Long enough for several slow snapshots to have run.
        await asyncio.sleep(snapshot_s * 4)

    mon.start()
    try:
        loop.run_until_complete(run())
    finally:
        mon.stop()
        loop.close()

    assert len(calls) >= 2, "the slow snapshot never ran"
    assert not mon.stalls, mon.stalls
    assert mon.max_lag_ms < 150, mon.max_lag_ms
    assert mon.max_stamp_to_post_ms < 50, mon.max_stamp_to_post_ms


@pytest.mark.parametrize("slow_snapshot", [True, False])
def test_a_blocked_loop_is_still_a_stall_whatever_the_snapshot_costs(slow_snapshot, monkeypatch):
    """The loop itself blocked past the limit is a stall, with an instant
    snapshot and with a slow one; its stamp-to-post stays near zero."""
    snapshot_s = 0.35 if slow_snapshot else 0.0

    def snapshot():
        if snapshot_s:
            time.sleep(snapshot_s)
        return 0.0, {}

    monkeypatch.setattr(loop_watchdog, "cpu_snapshot", snapshot)
    monkeypatch.setattr(loop_watchdog, "host_cpu", lambda: {})
    loop = asyncio.new_event_loop()
    mon = LagMonitor(loop, threshold_ms=250, tick_ms=20, log=False)

    async def run():
        await asyncio.sleep(0.1)
        # The loop itself is blocked. 0.9 s: even if a 0.35 s snapshot starts
        # as the block starts, the heartbeat posted after it waits >= 0.55 s.
        time.sleep(0.9)
        deadline = time.monotonic() + 5
        while not mon.stall_recorded.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

    mon.start()
    try:
        loop.run_until_complete(run())
    finally:
        mon.stop()
        loop.close()

    assert mon.stall_stamp_to_post, "no stall recorded"
    lag, s2p = max(mon.stall_stamp_to_post, key=lambda p: p[0])
    assert lag >= 250, (lag, s2p)
    assert s2p < 50, (lag, s2p)
    assert mon.max_stamp_to_post_ms >= s2p
