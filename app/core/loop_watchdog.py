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
    """The loop thread's stack, then every other thread's (innermost first)."""
    names = {t.ident: t.name for t in threading.enumerate()}
    me = threading.get_ident()
    frames = sys._current_frames()
    parts = []
    if loop_ident in frames:
        parts.append(f"[loop] {_stack(frames[loop_ident])}")
    for ident, frame in frames.items():
        if ident in (me, loop_ident):
            continue
        parts.append(f"[{names.get(ident, ident)}] {_stack(frame, 8)}")
    return " || ".join(parts)


class LagMonitor:
    """Measures how late the loop serves a heartbeat; optionally logs stalls."""

    def __init__(self, loop: asyncio.AbstractEventLoop, threshold_ms: float = _DEFAULT_MS,
                 tick_ms: float = _DEFAULT_TICK_MS, log: bool = True) -> None:
        self.loop, self.threshold, self.tick = loop, threshold_ms / 1000, tick_ms / 1000
        self.log = log
        self.max_lag_ms = 0.0
        self.stalls: list[tuple[float, str]] = []
        self._loop_ident: Optional[int] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="loop-watchdog", daemon=True)
        self._last_log = 0.0

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
            posted = time.monotonic()
            try:
                self.loop.call_soon_threadsafe(served.set)
            except RuntimeError:
                _LOG.info("loop watchdog stopped: the event loop is closed")
                self._stop.set()
                break
            sample = ""
            if not served.wait(self.threshold):
                sample = sample_stacks(self._loop_ident)
                while not served.wait(0.05):
                    if self._stop.is_set():
                        return
            lag_ms = (time.monotonic() - posted) * 1000
            self.max_lag_ms = max(self.max_lag_ms, lag_ms)
            if sample:
                self.stalls.append((lag_ms, sample))
                now = time.monotonic()
                if self.log and now - self._last_log >= _MIN_LOG_GAP_S:
                    self._last_log = now
                    _LOG.warning("LOOP_STALL lag=%dms %s", lag_ms, sample)
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
