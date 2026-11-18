"""A SIDE-BY-SIDE control for the fifteen-user loop-lag measurement.

A separate process -- its own interpreter, so no interpreter lock shared with
the server under test -- that does nothing but sleep ``TICK_MS`` and record how
late each wake-up was. Read beside the server's loop lag and watchdog
wake-late for the same window:

* the side process ALSO late in that window -> the runner did not schedule
  processes at all (the host starved everything);
* the side process on time while the server's loop AND its watchdog thread
  were late -> something inside the server held the interpreter lock without
  computing.

Non-gating: nothing here is asserted by the load test; the figures are only
printed. ``time.monotonic`` is the system-wide monotonic clock, so the
parent's window bounds and this process's wake times are on one scale.

Protocol: started with ``python -m tests._wake_sidecar``; it ticks until its
stdin closes, then prints ``WAKE_SIDECAR {json}`` with every wake-up
``[monotonic_s, late_ms]`` and exits. Not collected by pytest.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

#: The side process's own sleep, the same tick the server's watchdog uses.
TICK_MS = 20
#: Wake-ups later than this are listed one by one in the printed summary.
LIST_LATE_MS = 100.0
#: Seconds the parent waits for the side process's result after closing stdin.
COLLECT_TIMEOUT_S = 10.0


def tick_until(stop: threading.Event, tick_s: float = TICK_MS / 1000,
               clock=time.monotonic, sleep=time.sleep) -> list:
    """Sleep ``tick_s`` until ``stop`` is set; ``[wake_time_s, late_ms]`` for
    each wake-up, late = how far past the requested wake time it came."""
    wakes = []
    while not stop.is_set():
        due = clock() + tick_s
        sleep(tick_s)
        now = clock()
        wakes.append([now, round(max(0.0, now - due) * 1000, 2)])
    return wakes


def window_max_ms(wakes: list, start: float, end: float) -> float | None:
    """The latest wake-up whose wake time falls inside ``[start, end]``; None
    when no wake-up falls there (an empty window is not "on time")."""
    inside = [late for t, late in wakes if start <= t <= end]
    return max(inside) if inside else None


def late_in_window(wakes: list, start: float, end: float, over_ms: float = LIST_LATE_MS) -> list:
    """``[seconds_after_start, late_ms]`` for each wake-up in the window later
    than ``over_ms``, in time order, to line up with the server's stalls."""
    return [[round(t - start, 3), late] for t, late in wakes if start <= t <= end and late > over_ms]


def _valid_wakes(parsed) -> list:
    """The parsed result if it is a list of ``[number, number]`` pairs; any
    other shape raises ``ValueError`` (reported by class, never printed)."""
    if not isinstance(parsed, list) or not all(
            isinstance(w, list) and len(w) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in w)
            for w in parsed):
        raise ValueError("not a list of [wake_time_s, late_ms] pairs")
    return parsed


class Sidecar:
    """The side process, seen from the parent. STRICTLY REPORT-ONLY: every
    failure to start, to collect or to read the result becomes ``error`` (the
    exception class and the return code, nothing else: no stderr text, no OS
    error text; the CI log is public). It never raises into, or masks, the
    measurement it sits beside (``collect`` runs in that test's ``finally``)."""

    def __init__(self, argv: list | None = None) -> None:
        self.argv = argv if argv is not None else [sys.executable, "-m", "tests._wake_sidecar"]
        self.proc = None
        self.error = ""

    def start(self, repo: str) -> Sidecar:
        try:
            self.proc = subprocess.Popen(self.argv, cwd=repo, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception as exc:  # report-only: any failure is a figure, not a raise
            self.proc = None
            self.error = f"start failed: {type(exc).__name__}"
        return self

    def _rc(self) -> str:
        rc = getattr(self.proc, "returncode", None)
        return "none" if rc is None else str(rc)

    def collect(self) -> list:
        """Close the side process's stdin and read its wake-ups; ``[]`` and
        ``error`` set on any failure."""
        if self.proc is None:
            return []
        try:
            try:
                out, _ = self.proc.communicate(input="", timeout=COLLECT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.communicate()
                self.error = f"no result before the collect timeout rc={self._rc()}"
                return []
            found = [ln for ln in (out or "").splitlines() if ln.startswith("WAKE_SIDECAR ")]
            if not found:
                self.error = f"no result rc={self._rc()}"
                return []
            return _valid_wakes(json.loads(found[-1].split(" ", 1)[1]))
        except Exception as exc:  # report-only, see the class docstring
            self.error = f"collect failed: {type(exc).__name__} rc={self._rc()}"
            try:
                if self.proc.poll() is None:
                    self.proc.kill()
            except Exception as kill_exc:  # the result is already an error; keep its class
                self.error += f" kill failed: {type(kill_exc).__name__}"
            return []


def summary_line(wakes: list, error: str, idle: tuple, load: tuple,
                 load_max_ms: float, wake_late_max_ms: float) -> str:
    """The three figures side by side: the server's loop lag, the server's
    watchdog wake-late, and the side process's wake-late, for the load window
    (and the side process's idle-window figure as its own control). Never
    raises: a failure to build the line is printed as ``side_error`` by class."""
    def fig(v: float | None) -> str:
        return "none" if v is None else f"{v:.0f}ms"
    try:
        side_load, side_idle = window_max_ms(wakes, *load), window_max_ms(wakes, *idle)
        return (f"LOOP_LAG_SIDE_BY_SIDE server_load_max={load_max_ms:.0f}ms "
                f"server_wake_late_max={wake_late_max_ms:.0f}ms "
                f"side_wake_late_max={fig(side_load)} side_idle_wake_late_max={fig(side_idle)} "
                f"side_late_over_{LIST_LATE_MS:.0f}ms_at={json.dumps(late_in_window(wakes, *load))} "
                f"side_wakes={len(wakes)}" + (f" side_error={error!r}" if error else ""))
    except Exception as exc:  # report-only: the gated assertion must still run
        return f"LOOP_LAG_SIDE_BY_SIDE side_error='summary failed: {type(exc).__name__}'"


def window_figures(wakes: list, window: tuple) -> dict:
    """The side process's figures for one window, for the capture pass's JSON
    result. Never raises (report-only), like ``summary_line``."""
    try:
        return {"side_wake_late_max_ms": window_max_ms(wakes, *window),
                "side_late_over_100ms_at": late_in_window(wakes, *window)}
    except Exception as exc:
        return {"side_wake_late_max_ms": None, "side_late_over_100ms_at": [],
                "side_figures_error": f"figures failed: {type(exc).__name__}"}


def main() -> None:
    stop = threading.Event()

    def wait_for_eof() -> None:
        sys.stdin.read()
        stop.set()

    threading.Thread(target=wait_for_eof, name="sidecar-stdin", daemon=True).start()
    wakes = tick_until(stop)
    print("WAKE_SIDECAR " + json.dumps(wakes), flush=True)


if __name__ == "__main__":
    main()
