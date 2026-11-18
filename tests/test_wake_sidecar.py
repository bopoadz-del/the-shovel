"""The side-by-side control's own arithmetic, and one real side process.

tests/_wake_sidecar.py measures a SEPARATE process's wake-up lateness beside
the fifteen-user load, so a stall can be read as the runner starving every
process or as the server's own interpreter lock held without computing. Its
figures are only printed; these tests pin that the figures are right.

Every test builds its own wake-up list; no document text.
"""
from __future__ import annotations

import sys
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


@pytest.mark.parametrize("run_s", [1.0, 2.0])
def test_a_real_side_process_ticks_until_its_stdin_closes(run_s):
    """Counted from the side process's FIRST wake-up, not from its start: an
    interpreter that is slow to start on a loaded shard does not fail this."""
    import time
    sc = side.Sidecar().start(_REPO)
    t0 = time.monotonic()
    time.sleep(run_s)
    wakes = sc.collect()
    t1 = time.monotonic()
    assert sc.error == ""
    assert sc.proc.returncode == 0
    assert wakes, "the side process never woke inside the run"
    first, last = wakes[0][0], wakes[-1][0]
    assert t0 - 1.0 <= first <= last <= t1
    # At least a third of the 20 ms ticks that fit between the first and the
    # last wake-up (scheduling slack), and every wake-up's lateness >= 0.
    assert len(wakes) - 1 >= (last - first) / (3 * side.TICK_MS / 1000), (len(wakes), last - first)
    assert all(late >= 0 for _, late in wakes)


@pytest.mark.parametrize("earlier_error", ["", "start failed: FileNotFoundError"])
def test_a_side_process_that_never_started_reports_no_wake_ups(earlier_error):
    sc = side.Sidecar()
    sc.error = earlier_error
    assert sc.collect() == []
    assert sc.error == earlier_error


# --- STRICTLY REPORT-ONLY (advisor 00:54Z 18 Nov) ---------------------------
# Nothing the side process does may raise into, or mask, the gated test, and
# side_error carries the exception class and the return code only.

_SECRET = "/home/runner/private-path token=abc123"


def _child(code: str) -> list:
    return [sys.executable, "-c", code]


@pytest.mark.parametrize("printed,expected_error", [
    ("WAKE_SIDECAR not json at all", "collect failed: JSONDecodeError rc=0"),
    ('WAKE_SIDECAR {"wakes": 1}', "collect failed: ValueError rc=0"),
    ('WAKE_SIDECAR [[1.0, "late"]]', "collect failed: ValueError rc=0"),
])
def test_a_side_process_that_prints_garbage_is_an_error_not_a_raise(printed, expected_error):
    code = f"import sys; sys.stdin.read(); sys.stderr.write({_SECRET!r}); print({printed!r})"
    sc = side.Sidecar(_child(code)).start(_REPO)
    assert sc.collect() == []
    assert sc.error == expected_error
    assert _SECRET not in sc.error


@pytest.mark.parametrize("code,expected_error", [
    (f"import sys; sys.stderr.write({_SECRET!r}); sys.exit(3)", "no result rc=3"),
    ("import os; os._exit(0)", "no result rc=0"),
])
def test_a_side_process_that_dies_is_an_error_with_its_return_code_only(code, expected_error):
    sc = side.Sidecar(_child(code)).start(_REPO)
    assert sc.collect() == []
    assert sc.error == expected_error
    assert _SECRET not in sc.error


@pytest.mark.parametrize("argv,expected_error", [
    (["/no/such/interpreter-for-the-side-process"], "start failed: FileNotFoundError"),
    ([12345], "start failed: TypeError"),
])
def test_a_side_process_that_cannot_start_is_an_error_by_class_only(argv, expected_error):
    sc = side.Sidecar(argv).start(_REPO)
    assert sc.proc is None
    assert sc.error == expected_error
    assert sc.collect() == []


@pytest.mark.parametrize("wakes", [[["x", 1.0]], None])
def test_the_summary_line_never_raises(wakes):
    line = side.summary_line(wakes, "", idle=(0.0, 1.0), load=(1.0, 2.0), load_max_ms=5.0, wake_late_max_ms=6.0)
    assert line.startswith("LOOP_LAG_SIDE_BY_SIDE side_error='summary failed: ")
    assert line.endswith("Error'")


@pytest.mark.parametrize("wakes,expected_max", [([[1.5, 120.0], [1.6, 3.0]], 120.0), ([["x", 1.0]], None)])
def test_the_capture_pass_figures_never_raise(wakes, expected_max):
    figures = side.window_figures(wakes, (1.0, 2.0))
    assert figures["side_wake_late_max_ms"] == expected_max
    assert ("side_figures_error" in figures) is (expected_max is None)


# --- ORDER and MARKER of the capture pass (advisor 00:54Z 18 Nov) ------------
_GATED = "test_fifteen_users_never_block_the_loop_for_more_than_250_ms"
_CAPTURE = "test_the_capture_runs_over_the_fifteen_user_load_in_its_own_process"


@pytest.mark.parametrize("paths", [
    ["tests/test_the_loop_is_never_blocked_under_load.py"],
    ["tests/test_loop_stall_live_capture.py", "tests/test_the_loop_is_never_blocked_under_load.py",
     "tests/test_wake_sidecar.py"],
])
def test_the_capture_pass_is_timing_and_runs_after_the_gated_test(paths):
    """Collected exactly as CI's timing step selects (``-m timing``): the
    capture pass must be there (the marker is pinned: removing it drops the
    pass from the step) and must come AFTER the gated measurement."""
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "timing",
                           "-p", "no:cacheprovider", *paths],
                          cwd=_REPO, capture_output=True, text=True, timeout=120, check=False)
    ids = [ln.split("::")[-1] for ln in proc.stdout.splitlines() if "::" in ln]
    assert _GATED in ids and _CAPTURE in ids, proc.stdout[-2000:]
    assert ids.index(_GATED) < ids.index(_CAPTURE), ids
