"""NON-GATING: how much does search on 1, 2 and 4 threads delay the event loop?

Runs on the timing job's runner (marked ``timing``: Linux, PostgreSQL in CI)
so the figures come from the machine where main's timing gate goes red, not a
laptop. The measurement runs in a child process (tests/_search_thread_lag_pass.py)
and is printed as ``SEARCH_THREAD_LAG``: heartbeat lag p50 / p95 / max per
thread count, with the search in the calling thread and in the retrieval
worker process, plus switch interval, CPU count and Python version.

Logging only: no latency figure is asserted. The gate stays the fifteen-user
test (tests/test_the_loop_is_never_blocked_under_load.py), unchanged. The one
assertion: a child that exits cleanly printed its result (a silent exit is a
broken harness). Synthetic corpus only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests._search_thread_lag_pass import lag_summary, measure

pytestmark = [pytest.mark.serial, pytest.mark.timing]

_REPO = Path(__file__).resolve().parents[1]
#: Seeding plus 7 cells of 4 s plus worker-pool start, with a wide margin.
_PASS_TIMEOUT_S = 100


def test_search_thread_loop_lag_is_printed_from_its_own_process(tmp_path, request):
    try:
        proc = subprocess.run([sys.executable, "-m", "tests._search_thread_lag_pass", str(tmp_path)],
                              cwd=_REPO, capture_output=True, text=True, timeout=_PASS_TIMEOUT_S,
                              check=False)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc, out, err = None, exc.stdout or "", exc.stderr or ""
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        err = err.decode("utf-8", "replace") if isinstance(err, bytes) else err
    found = [ln for ln in out.splitlines() if ln.startswith("SEARCH_THREAD_LAG_RESULT ")]
    result = json.loads(found[-1].split(" ", 1)[1]) if found else None
    line = (f"SEARCH_THREAD_LAG rc={rc} " + (json.dumps(result) if result is not None
            else f"no result; stderr tail: {err[-1500:]!r}"))
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(line)
    else:
        sys.__stderr__.write(line + "\n")
    assert not (rc == 0 and result is None), line


@pytest.mark.parametrize("lags,expected", [
    ([], {"ticks": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}),
    ([1.0, 3.0, 2.0], {"ticks": 3, "p50_ms": 2.0, "p95_ms": 3.0, "max_ms": 3.0}),
    ([0.0] * 19 + [400.0], {"ticks": 20, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 400.0}),  # one in 20 is past p95
    ([0.0] * 17 + [400.0] * 3, {"ticks": 20, "p50_ms": 0.0, "p95_ms": 400.0, "max_ms": 400.0}),
])
def test_lag_summary_reports_median_tail_and_worst(lags, expected):
    assert lag_summary(lags) == expected


@pytest.mark.parametrize("threads", [0, 2])
def test_measure_runs_the_given_number_of_search_threads(threads):
    """0 threads is the idle control (no search, heartbeat only); N threads
    each run the search back to back for the window."""
    import threading
    import time

    seen = set()

    def search(_q):
        seen.add(threading.current_thread().name)
        time.sleep(0.02)

    cell = measure(search, ["q"], threads=threads, window_s=0.3)
    assert cell["threads"] == threads and cell["ticks"] >= 1
    assert len(seen) == threads
    if threads:
        assert cell["searches"] >= threads and cell["caller_cpu_ms_per_search"] is not None
    else:
        assert cell["searches"] == 0 and cell["caller_cpu_ms_per_search"] is None
