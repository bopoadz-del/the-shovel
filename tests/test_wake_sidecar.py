"""The side-by-side control's own arithmetic, and one real side process.

tests/_wake_sidecar.py measures a SEPARATE process's wake-up lateness beside
the fifteen-user load, so a stall can be read as the runner starving every
process or as the server's own interpreter lock held without computing. Its
figures are only printed; these tests pin that the figures are right.

Every test builds its own wake-up list; no document text.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from tests import _wake_sidecar as side

_REPO = str(Path(__file__).resolve().parents[1])


@pytest.mark.parametrize("start,end,expected", [
    (10.0, 11.0, 300.0),   # the 300 ms wake-up is inside, the 900 ms one is not
    (11.5, 13.0, 900.0),   # a later window picks the later wake-up
    (20.0, 21.0, None),    # no wake-up there: not reported as on time
])
def test_window_max_takes_only_the_wake_ups_inside_the_window(start, end, expected):
    wakes = [[9.9, 700.0], [10.2, 4.0], [10.8, 300.0], [12.0, 900.0], [12.5, 2.0]]
    assert side.window_max_ms(wakes, start, end) == expected


@pytest.mark.parametrize("over,expected", [
    (100.0, [[0.8, 300.0], [2.0, 900.0]]),
    (500.0, [[2.0, 900.0]]),
])
def test_late_wake_ups_are_listed_with_their_offset_into_the_window(over, expected):
    wakes = [[9.9, 700.0], [10.2, 4.0], [10.8, 300.0], [12.0, 900.0], [12.5, 2.0]]
    assert side.late_in_window(wakes, 10.0, 13.0, over_ms=over) == expected


@pytest.mark.parametrize("overslept_s,expected_late_ms", [(0.0, 0.0), (0.35, 350.0)])
def test_each_wake_up_records_how_far_past_its_due_time_it_came(overslept_s, expected_late_ms):
    now = [100.0]
    stop = threading.Event()
    calls = []

    def sleep(s):
        now[0] += s + overslept_s
        calls.append(s)
        if len(calls) == 3:
            stop.set()

    wakes = side.tick_until(stop, tick_s=0.02, clock=lambda: now[0], sleep=sleep)
    assert [late for _, late in wakes] == [expected_late_ms] * 3
    assert calls == [0.02] * 3


@pytest.mark.parametrize("error,tail", [("", "side_wakes=4"), ("start failed: x", "side_error='start failed: x'")])
def test_the_summary_puts_the_three_figures_side_by_side(error, tail):
    wakes = [[1.0, 2.0], [2.0, 3.0], [5.0, 40.0], [6.0, 520.0]]
    line = side.summary_line(wakes, error, idle=(0.5, 2.5), load=(4.0, 7.0),
                             load_max_ms=281.0, wake_late_max_ms=546.0)
    assert line.startswith("LOOP_LAG_SIDE_BY_SIDE server_load_max=281ms server_wake_late_max=546ms "
                           "side_wake_late_max=520ms side_idle_wake_late_max=3ms ")
    assert "side_late_over_100ms_at=[[2.0, 520.0]]" in line
    assert line.endswith(tail)


def test_an_empty_window_prints_none_not_zero():
    line = side.summary_line([], "", idle=(0.0, 1.0), load=(1.0, 2.0), load_max_ms=5.0, wake_late_max_ms=6.0)
    assert "side_wake_late_max=none side_idle_wake_late_max=none" in line


def test_a_real_side_process_ticks_until_its_stdin_closes():
    import time
    sc = side.Sidecar().start(_REPO)
    t0 = time.monotonic()
    time.sleep(0.5)
    t1 = time.monotonic()
    wakes = sc.collect()
    assert sc.error == ""
    assert sc.proc.returncode == 0
    inside = [w for w in wakes if t0 <= w[0] <= t1]
    assert len(inside) >= 5, wakes[:5]
    assert all(late >= 0 for _, late in wakes)


def test_a_side_process_that_never_started_reports_no_wake_ups():
    sc = side.Sidecar()
    assert sc.collect() == []
    assert json.loads(json.dumps(sc.collect())) == []
