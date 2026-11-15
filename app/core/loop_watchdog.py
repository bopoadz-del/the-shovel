"""Name the code that freezes the event loop.

Live 2026-10-07, 15 users holding 3-turn conversations: the I/O-free
``/livez`` waited up to 10.9 s. Something in a chat turn ran on the event loop
(or held the GIL from another thread long enough to starve it), and nothing in
the logs said what.

A daemon thread posts a heartbeat to the loop every ``LOOP_WATCHDOG_TICK_MS``
(default 50). When a heartbeat is still unserved ``LOOP_WATCHDOG_MS`` later
(default 250), it samples the stacks -- the loop thread's first, then every
other thread's, because a loop that is idle in ``select`` but late is being
starved of the GIL by a thread, and that thread is the culprit. When the
heartbeat finally runs, one ``LOOP_STALL lag=<ms>`` line is logged with the
sample. Code locations only (file, line, function), never local variables, so
no document text or client figures reach the logs. ``LOOP_WATCHDOG_MS=0``
turns it off.

``LagMonitor`` is the same measurement without logging, for tests.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import traceback
from typing import Optional

_LOG = logging.getLogger(__name__)
_DEFAULT_MS = 250.0
_DEFAULT_TICK_MS = 50.0
_MAX_FRAMES = 25
_MIN_LOG_GAP_S = 2.0

_started: Optional["LagMonitor"] = None


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _stack(frame, max_frames: int = _MAX_FRAMES) -> str:
    lines = [f"{fs.filename}:{fs.lineno} {fs.name}" for fs in traceback.extract_stack(frame)]
    return " <- ".join(reversed(lines[-max_frames:]))


def sample_stacks(loop_ident: Optional[int]) -> str:
    """The loop thread's stack, then every other thread's (innermost first).

    When the loop thread id is not yet resolved (its ``call_soon`` to note it
    can itself be stuck behind the very stall we are sampling), fall back to the
    main thread, which is where the event loop runs. The loop stack is therefore
    always captured, so the worst-stall sample is never empty."""
    names = {t.ident: t.name for t in threading.enumerate()}
    me = threading.get_ident()
    frames = sys._current_frames()
    target = loop_ident if loop_ident in frames else threading.main_thread().ident
    parts = []
    if target in frames:
        parts.append(f"[loop] {_stack(frames[target])}")
    for ident, frame in frames.items():
        if ident in (me, target):
            continue
        parts.append(f"[{names.get(ident, ident)}] {_stack(frame, 8)}")
    return " || ".join(parts)


#: Linux exposes each thread's CPU time; elsewhere the CPU delta is omitted.
_TASK_DIR = "/proc/self/task"
#: Per-thread CPU is snapshotted at most this often (each read costs a file
#: per thread), so a stall's delta spans the stall plus at most this much.
_CPU_SNAPSHOT_S = 0.1


def thread_cpu_ms(task_dir: str = _TASK_DIR) -> dict:
    """CPU milliseconds (user + system) used so far by each thread, keyed by the
    thread's native id; empty where the OS gives no per-thread figures."""
    try:
        tids = os.listdir(task_dir)
    except OSError as exc:
        _LOG.debug("per-thread CPU unavailable (%s): stalls carry no cpu figures", exc)
        return {}
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    out = {}
    for tid in tids:
        try:
            with open(os.path.join(task_dir, tid, "stat"), "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            out[int(tid)] = (int(fields[11]) + int(fields[12])) * 1000.0 / tick
        except (OSError, ValueError, IndexError):
            continue
    return out


def busiest_threads(before: dict, after: dict, names: dict, top: int = 4) -> str:
    """The threads that burned the most CPU between two snapshots, by name.

    A loop that is late while parked in ``select`` is waiting for the GIL;
    the thread that burned CPU across that window held it. Threads that used
    no CPU are left out."""
    deltas = sorted(((after[t] - before.get(t, 0.0), t) for t in after), reverse=True)
    parts = [f"{names.get(t, t)}={d:.0f}ms" for d, t in deltas[:top] if d > 0]
    return ("cpu " + " ".join(parts)) if parts else ""


def heartbeat_lag_ms(posted: float, served_at: Optional[float], now: float) -> float:
    """How late the LOOP served a heartbeat posted at ``posted``.

    ``served_at`` is stamped by the heartbeat callback itself, on the loop
    thread, so the figure is the loop's lateness only. The watchdog thread
    must re-take the GIL to wake from its wait; under load that wake-up can
    itself be hundreds of ms late, and timing the heartbeat at the watchdog's
    wake-up charged that delay to the loop (a lag above the threshold with no
    stall recorded, because the heartbeat had in fact been served in time).
    A heartbeat not yet served (``served_at`` None) is late by ``now - posted``.
    """
    end = served_at if served_at is not None else now
    return max(0.0, (end - posted) * 1000)


class LagMonitor:
    """Measures how late the loop serves a heartbeat; optionally logs stalls."""

    def __init__(self, loop: asyncio.AbstractEventLoop, threshold_ms: float = _DEFAULT_MS,
                 tick_ms: float = _DEFAULT_TICK_MS, log: bool = True) -> None:
        self.loop, self.threshold, self.tick = loop, threshold_ms / 1000, tick_ms / 1000
        self.log = log
        self.max_lag_ms = 0.0
        self.stalls: list[tuple[float, str]] = []
        # The loop runs on the main thread; _note_loop_thread refines this once
        # it is served. Seeding it (never None) means a stall sampled before
        # that call still resolves the loop stack instead of dropping it.
        self._loop_ident: Optional[int] = threading.main_thread().ident
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="loop-watchdog", daemon=True)
        self._last_log = 0.0
        #: Set each time a stall is recorded. Lets a waiter observe detection
        #: deterministically (event, not a wall-clock guess) instead of racing
        #: the sampler thread -- see tests/test_offload_and_loop_watchdog.
        self.stall_recorded = threading.Event()
        self._cpu_at = 0.0
        self._cpu_before: dict = {}

    def start(self) -> "LagMonitor":
        self.loop.call_soon_threadsafe(self._note_loop_thread)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _note_loop_thread(self) -> None:
        self._loop_ident = threading.get_ident()

    def _run(self) -> None:
        while not self._stop.is_set():
            served = threading.Event()
            served_at: list[float] = []

            def _serve(served=served, served_at=served_at) -> None:
                served_at.append(time.monotonic())
                served.set()

            posted = time.monotonic()
            if posted - self._cpu_at >= _CPU_SNAPSHOT_S:
                self._cpu_before, self._cpu_at = thread_cpu_ms(), posted
            try:
                self.loop.call_soon_threadsafe(_serve)
            except RuntimeError:
                _LOG.info("loop watchdog stopped: the event loop is closed")
                self._stop.set()
                break
            sample = ""
            stalled = not served.wait(self.threshold)
            if stalled:
                # Sample now and keep re-sampling while the stall persists, so
                # the stack we keep is the loop caught in the act (the deepest,
                # most recent frames), not just the instant the threshold passed.
                sample = sample_stacks(self._loop_ident)
                while not served.wait(0.05):
                    if self._stop.is_set():
                        return
                    snap = sample_stacks(self._loop_ident)
                    if snap:
                        sample = snap
            lag_ms = heartbeat_lag_ms(posted, served_at[0] if served_at else None,
                                      time.monotonic())
            self.max_lag_ms = max(self.max_lag_ms, lag_ms)
            # The loop's own lateness decides: a heartbeat served past the
            # threshold is a stall even if this thread woke too late to sample
            # it during the stall (it then samples below).
            stalled = stalled or lag_ms >= self.threshold * 1000
            if stalled:
                # A detected stall is ALWAYS recorded with its stack. sample is
                # never empty (sample_stacks falls back to the main thread), but
                # resample once more as a belt-and-braces guard so the worst
                # stall can never carry an empty trace.
                if not sample:
                    sample = sample_stacks(self._loop_ident)
                # Which threads burned CPU across the stall: the GIL holder
                # when the loop itself was idle-but-late. Leads the sample, so
                # a truncated message still names it.
                names = {t.native_id: t.name for t in threading.enumerate()}
                cpu = busiest_threads(self._cpu_before, thread_cpu_ms(), names)
                if cpu:
                    sample = f"{cpu} :: {sample}"
                self.stalls.append((lag_ms, sample))
                now = time.monotonic()
                if self.log and now - self._last_log >= _MIN_LOG_GAP_S:
                    self._last_log = now
                    _LOG.warning("LOOP_STALL lag=%dms %s", lag_ms, sample)
                # Recorded and (if logging) logged: signal last so a waiter that
                # sees the event also sees both.
                self.stall_recorded.set()
            self._stop.wait(self.tick)


def start(loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[LagMonitor]:
    """Start the process-wide watchdog once (from the running loop's lifespan)."""
    global _started
    threshold = _env_float("LOOP_WATCHDOG_MS", _DEFAULT_MS)
    if threshold <= 0 or _started is not None:
        return _started
    loop = loop or asyncio.get_running_loop()
    _started = LagMonitor(loop, threshold, _env_float("LOOP_WATCHDOG_TICK_MS", _DEFAULT_TICK_MS)).start()
    _LOG.info("loop watchdog on: logs LOOP_STALL when the event loop is %dms late", threshold)
    return _started
