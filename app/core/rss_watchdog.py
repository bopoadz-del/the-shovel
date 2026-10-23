"""Name the code that is eating the memory, before the kernel kills us.

Live 21 Sep 2026, 18:16 UAE-4: RSS sat at 773 MB, then one chat turn took the
web instance past its 2 GiB limit in ~40 s and Render OOM-killed it. The last
log line was the final LLM call; nothing after it said which function was
allocating. An OOM kill leaves no traceback.

A daemon thread samples RSS every RSS_WATCHDOG_INTERVAL_S seconds. Each time
RSS crosses RSS_WATCHDOG_MB (default 1200), and again every
RSS_WATCHDOG_STEP_MB above that, it logs the stack of every thread. Code
locations only -- file, line, function -- never local variables, so no
document text or client figures reach the logs. A thread that is allocating
in Python bytecode yields the GIL every switch interval, so the sampler runs
even while the event loop is starved. RSS_WATCHDOG_MB=0 turns it off.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
from typing import Callable, Optional

_LOG = logging.getLogger(__name__)
_DEFAULT_MB = 1200
_DEFAULT_STEP_MB = 300
_DEFAULT_INTERVAL_S = 0.25
_MAX_FRAMES = 30
_STATM = "/proc/self/statm"

_started: Optional[threading.Thread] = None


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def current_rss_mb() -> Optional[float]:
    """Resident set size of this process in MB; None off Linux (no /proc)."""
    if not os.path.exists(_STATM):
        return None
    with open(_STATM, encoding="ascii") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


def format_thread_stacks(max_frames: int = _MAX_FRAMES) -> str:
    """Stacks of all threads, innermost frame last. Locations only."""
    names = {t.ident: t.name for t in threading.enumerate()}
    me = threading.get_ident()
    out = []
    for ident, frame in sys._current_frames().items():
        if ident == me:
            continue
        frames = traceback.extract_stack(frame)[-max_frames:]
        out.append(f"--- thread {names.get(ident, ident)}")
        out.extend(f"  {f.filename}:{f.lineno} in {f.name}" for f in frames)
    return "\n".join(out)


class RssWatchdog:
    def __init__(
        self,
        threshold_mb: float,
        step_mb: float = _DEFAULT_STEP_MB,
        read_rss: Callable[[], Optional[float]] = current_rss_mb,
        log: Callable[[str], None] = _LOG.warning,
    ) -> None:
        self.threshold_mb = threshold_mb
        self.step_mb = max(step_mb, 1.0)
        self._read_rss = read_rss
        self._log = log
        self._next_mb = threshold_mb

    def check(self) -> bool:
        """Sample once; log the stacks if RSS reached the next mark."""
        rss = self._read_rss()
        if rss is None:
            return False
        if rss < self.threshold_mb:
            # Back under the threshold: re-arm, so the next spike is named too.
            self._next_mb = self.threshold_mb
            return False
        if rss < self._next_mb:
            return False
        self._log(
            f"RSS WATCHDOG rss={rss:.0f}MB threshold={self.threshold_mb:.0f}MB "
            f"-- thread stacks follow\n{format_thread_stacks()}"
        )
        while self._next_mb <= rss:
            self._next_mb += self.step_mb
        return True


def start() -> Optional[threading.Thread]:
    """Start the process-wide watchdog once. No-op when disabled or unsupported."""
    global _started
    threshold = _env_float("RSS_WATCHDOG_MB", _DEFAULT_MB)
    if threshold <= 0 or _started is not None or current_rss_mb() is None:
        return _started
    dog = RssWatchdog(threshold, _env_float("RSS_WATCHDOG_STEP_MB", _DEFAULT_STEP_MB))
    interval = max(_env_float("RSS_WATCHDOG_INTERVAL_S", _DEFAULT_INTERVAL_S), 0.05)
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            try:
                dog.check()
            except Exception:  # the watchdog must never take the server down
                _LOG.warning("rss watchdog check failed", exc_info=True)

    _started = threading.Thread(target=loop, name="rss-watchdog", daemon=True)
    _started.start()
    _LOG.info("rss watchdog armed at %.0f MB", threshold)
    return _started
