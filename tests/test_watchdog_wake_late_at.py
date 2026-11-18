"""The watchdog keeps WHEN its own wake-up came late, not only how late.

A wake-up at least ``WAKE_LATE_NOTE_MS`` late is kept with its monotonic
time (``LagMonitor.wake_late_at``), so the fifteen-user report can say where
in the load window the thread could not run: once, near the first requests
(a one-time initialisation), or spread through it. Capture stays OFF here, as
in the gated measurement: the record is plain arithmetic.

Every test builds its own hold; no document text.
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
    while n < 34:
        t = time.perf_counter()
        _gil_holding_call(n)
        if time.perf_counter() - t >= seconds:
            return n
        n += 1
    return n


@pytest.mark.parametrize("hold", [True, False])
def test_a_late_wake_is_kept_with_its_time(hold):
    n = _size_for(0.45) if hold else 0
    loop = asyncio.new_event_loop()
    mon = loop_watchdog.LagMonitor(loop, threshold_ms=250, tick_ms=100, log=False)
    started = threading.Event()

    def work():
        started.set()
        time.sleep(0.03)  # begins INSIDE the monitor's tick sleep
        if hold:
            _gil_holding_call(n)

    worker = threading.Thread(target=work, name="gil-holder")
    t0 = time.monotonic()
    worker.start()
    started.wait(2)
    try:
        mon._sleep_watched(None, {})
    finally:
        worker.join()
        loop.close()
    t1 = time.monotonic()
    if hold:
        assert len(mon.wake_late_at) == 1, mon.wake_late_at
        late_ms, at = mon.wake_late_at[0]
        assert late_ms >= loop_watchdog.WAKE_LATE_NOTE_MS
        assert late_ms == round(mon.max_wake_late_ms, 1)
        assert t0 <= at <= t1
    else:
        assert mon.wake_late_at == [], mon.wake_late_at


@pytest.mark.parametrize("lates,kept", [
    ([199.9, 50.0], []),                       # under the note threshold: no time kept
    ([200.0, 640.0], [(200.0, 1.0), (640.0, 2.0)]),
])
def test_only_wake_ups_past_the_note_threshold_are_kept(lates, kept):
    mon = loop_watchdog.LagMonitor(asyncio.new_event_loop(), log=False)
    for i, late in enumerate(lates, start=1):
        mon._note_wake_late(late, float(i))
    assert mon.wake_late_at == kept


@pytest.mark.parametrize("count,expected_min", [(loop_watchdog.LIVE_KEEP + 4, 205.0), (3, 201.0)])
def test_the_record_keeps_only_the_worst(count, expected_min):
    mon = loop_watchdog.LagMonitor(asyncio.new_event_loop(), log=False)
    for i in range(count):
        mon._note_wake_late(201.0 + i, float(i))
    assert len(mon.wake_late_at) == min(count, loop_watchdog.LIVE_KEEP)
    assert min(late for late, _ in mon.wake_late_at) == expected_min
