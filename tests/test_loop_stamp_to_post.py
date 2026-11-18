"""STAMP TO POST: the watchdog keeps how long each heartbeat waited between
its stamp and its post to the loop (advisor 07:28Z 18 Nov).

The heartbeat is stamped, then the monitor takes its CPU snapshot (/proc
reads), and only then is the heartbeat posted. Whatever the snapshot costs
is counted as loop lag. These tests check that the new figure tells the two
cases apart: a slow snapshot with an idle loop (the lag is about the
stamp-to-post time) and a blocked loop with a fast snapshot (stamp-to-post is
small next to the lag). Diagnostic only: nothing gated reads it.

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


@pytest.mark.parametrize("slow_snapshot", [True, False])
def test_the_figure_tells_a_slow_snapshot_from_a_blocked_loop(slow_snapshot, monkeypatch):
    """A slow snapshot with an idle loop: the stall's lag is its stamp-to-post.
    A blocked loop with an instant snapshot: stamp-to-post is near zero."""
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
        if not slow_snapshot:
            time.sleep(0.45)  # the loop itself is blocked
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
    assert lag >= 250
    if slow_snapshot:
        assert s2p >= 300
        assert lag - s2p < 150, (lag, s2p)
    else:
        assert s2p < 50, (lag, s2p)
    assert mon.max_stamp_to_post_ms >= s2p
