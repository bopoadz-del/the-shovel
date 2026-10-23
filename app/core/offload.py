"""Run a chat turn's synchronous work off the event loop, with a bounded queue.

Live 2026-10-07, 15 users at once: /livez waited up to 10.9 s. A turn's
database reads and writes (conversation store, document names, coverage
counts) and its answer post-processing ran synchronously inside the async chat
path, so each one froze the single event loop for every user.

``await off_loop(fn, *args, **kw)`` runs ``fn`` in a dedicated thread pool.
The pool has ``TURN_IO_WORKERS`` threads (default: the database pool size,
``DB_POOL_SIZE``), so offloaded work never asks for more connections than
exist; when every worker is busy, further calls wait in the pool's queue
instead of freezing the server.

Context variables behave as if ``fn`` ran inline: ``fn`` sees the caller's
context (the caller-role gate, the turn's provenance record), and values it
sets are copied back to the caller when it returns.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")
_MISSING = object()
_POOL: Optional[ThreadPoolExecutor] = None
_LOG = logging.getLogger(__name__)
#: A call that waited longer than this for a worker is logged (OFFLOAD_QUEUE):
#: the pool, not the work, was the delay.
_QUEUE_WARN_S = float(os.getenv("OFFLOAD_QUEUE_WARN_MS", "250")) / 1000
_last_queue_log = 0.0
#: Pools for work that must never queue ahead of a turn's database reads:
#: calculator runs (CPU) and background jobs (hydration). Sized by their own
#: settings; the turn pool keeps the database pool's size.
_NAMED_POOLS: Dict[str, ThreadPoolExecutor] = {}
_NAMED_SIZES = {"calc": ("CALC_WORKERS", 4), "background": ("BACKGROUND_WORKERS", 2)}


def _workers() -> int:
    for name, default in (("TURN_IO_WORKERS", None), ("DB_POOL_SIZE", "10")):
        raw = (os.getenv(name) or "").strip() or default
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                continue
    return 10


def pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=_workers(), thread_name_prefix="turn-io")
    return _POOL


def named_pool(name: str) -> ThreadPoolExecutor:
    """The dedicated pool for ``name`` ("calc", "background")."""
    if name not in _NAMED_POOLS:
        env, default = _NAMED_SIZES[name]
        try:
            size = max(1, int((os.getenv(env) or "").strip() or default))
        except ValueError:
            size = default
        _NAMED_POOLS[name] = ThreadPoolExecutor(max_workers=size, thread_name_prefix=f"{name}-work")
    return _NAMED_POOLS[name]


def _timed(fn: Callable[..., T], label: str, executor: ThreadPoolExecutor) -> Callable[..., T]:
    submitted = time.monotonic()

    def run(*args: Any, **kwargs: Any) -> T:
        global _last_queue_log
        waited = time.monotonic() - submitted
        if waited > _QUEUE_WARN_S and time.monotonic() - _last_queue_log > 5:
            _last_queue_log = time.monotonic()
            queued = getattr(getattr(executor, "_work_queue", None), "qsize", lambda: -1)()
            _LOG.warning("OFFLOAD_QUEUE pool=%s waited_ms=%d still_queued=%d fn=%s", label, waited * 1000,
                         queued, getattr(fn, "__qualname__", repr(fn)))
        return fn(*args, **kwargs)
    return run


async def _run_in(executor: ThreadPoolExecutor, label: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor, functools.partial(ctx.run, _timed(fn, label, executor), *args, **kwargs))
    for var, value in ctx.items():
        if var.get(_MISSING) is not value:
            var.set(value)
    return result


async def off_loop(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """``fn(*args, **kwargs)`` in the turn pool; context variables round-trip."""
    return await _run_in(pool(), "turn-io", fn, *args, **kwargs)


async def off_loop_in(name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """``fn`` in the dedicated ``name`` pool ("calc", "background"): CPU work
    and background jobs never queue ahead of a turn's database reads."""
    return await _run_in(named_pool(name), name, fn, *args, **kwargs)
