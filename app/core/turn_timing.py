"""Where a turn's time goes, from the moment the request arrived.

The old TIMING lines started the clock after retrieval and pre-dispatch, so on
live a turn read "2.9 s" while the user waited 18.9 s for the first token. The
chat endpoint marks arrival first thing; every stage then logs

    TIMING turn stage=<name> dur=<seconds> at=<seconds since arrival>

(queue, retrieval, predispatch, postprocess, first_token, ...), and the model
calls' existing ``cum=`` figures count from arrival too. Lines are written
when ``AGENT_TIMING_LOG`` is on, like the other TIMING lines.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_LOG = logging.getLogger("app.agents.runtime")
_STATE: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar("turn_timing", default=None)


def _on() -> bool:
    return (os.getenv("AGENT_TIMING_LOG") or "").strip().lower() in ("1", "true", "yes", "on")


# The stages a turn spends before its first token, in the order they run.
# ``llm_first_token`` is the model request up to its first token: the time
# from the last completed stage before it to the first token.
PRE_FIRST_TOKEN_STAGES = ("auth", "route", "hat_classification", "retrieval", "llm_first_token")


def mark_arrival(at: Optional[float] = None) -> None:
    """Start this turn's clock (call first thing in the request handler).

    ``at`` is an earlier ``time.monotonic()`` reading -- the moment the
    request reached the endpoint's dependencies, before authentication."""
    now = time.monotonic()
    t0 = now if at is None else min(float(at), now)
    _STATE.set({"t0": t0, "last": now, "first_token": False, "stages": {}})


def record(name: str, dur: float) -> None:
    """Add ``dur`` seconds to stage ``name`` for this turn (no log line).
    With no turn started, nothing is stored."""
    st = _STATE.get()
    if st is None:
        return
    stages = st["stages"]
    stages[name] = stages.get(name, 0.0) + max(0.0, float(dur))


def snapshot() -> dict:
    """This turn's pre-first-token stage timings in seconds: every stage in
    ``PRE_FIRST_TOKEN_STAGES`` (0.0 when it did not run) plus any other stage
    timed so far, and ``since_arrival``."""
    st = _STATE.get() or {}
    stages = st.get("stages") or {}
    out = dict.fromkeys(PRE_FIRST_TOKEN_STAGES, 0.0)
    out.update({k: round(v, 4) for k, v in stages.items()})
    at = since_arrival()
    out["since_arrival"] = round(at, 4) if at is not None else 0.0
    return out


def arrived_at() -> Optional[float]:
    st = _STATE.get()
    return st["t0"] if st else None


def since_arrival() -> Optional[float]:
    st = _STATE.get()
    return time.monotonic() - st["t0"] if st else None


def _rss_mb() -> int:
    """Resident memory in MB (Linux /proc; psutil elsewhere; -1 if neither)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss // (1024 * 1024))
    except Exception:  # noqa: BLE001 — a missing reading is reported as -1
        _LOG.debug("rss unreadable", exc_info=True)
        return -1


_WRITER = None


def set_writer(writer) -> None:
    """Route stage lines through ``writer(fmt, *args)`` -- the runtime's
    ``_timing_log``, so every TIMING line is written (and carries rss) the
    same way."""
    global _WRITER
    _WRITER = writer


def log_stage(name: str, dur: float) -> None:
    """One TIMING line; like every TIMING line it carries the process's rss.
    The duration is also recorded for :func:`snapshot`."""
    record(name, dur)
    if not _on():
        return
    at = since_arrival()
    at_txt = f"{at:.2f}s" if at is not None else "n/a"
    if _WRITER is not None:
        _WRITER("TIMING turn stage=%s dur=%.2fs at=%s", name, dur, at_txt)
        return
    _LOG.warning("TIMING turn stage=%s dur=%.2fs at=%s rss=%dMB", name, dur, at_txt, _rss_mb())


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a block (it may contain awaits) and log it as one stage."""
    t = time.monotonic()
    try:
        yield
    finally:
        now = time.monotonic()
        st = _STATE.get()
        if st is not None:
            st["last"] = now
        log_stage(name, now - t)


def timed(name: str):
    """Decorator: time every call of a function as stage ``name``. Uses
    functools.wraps, so the function's own source and signature stay visible."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def recorded(name: str):
    """Decorator: add every call's duration to stage ``name`` for
    :func:`snapshot`, without a TIMING line of its own (sync or async)."""
    import inspect

    def deco(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                t = time.monotonic()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    record(name, time.monotonic() - t)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t = time.monotonic()
            try:
                return fn(*args, **kwargs)
            finally:
                record(name, time.monotonic() - t)
        return wrapper
    return deco


def mark(name: str) -> None:
    """Log the time since the previous stage or mark as stage ``name``."""
    st = _STATE.get()
    if st is None:
        return
    now = time.monotonic()
    dur, st["last"] = now - st["last"], now
    log_stage(name, dur)


def first_token() -> None:
    """Log the first token of this turn's answer (once)."""
    st = _STATE.get()
    if st is None or st["first_token"]:
        return
    st["first_token"] = True
    record("llm_first_token", time.monotonic() - st["last"])
    log_stage("first_token", 0.0)
