"""At most N chat turns run at once; the rest wait a bounded time, then get a 503.

The server has one event loop and a fixed retrieval worker pool. Letting every
arriving turn start at once only stretches all of them and starves the loop;
queueing them unboundedly hides overload until clients time out. A turn takes a
slot before its stream starts; with none free it waits up to
``CHAT_TURN_QUEUE_WAIT_SECONDS`` (default 60) and is then refused with a clear
503 and ``Retry-After``, before any stream has begun. The slot is released when
the stream ends, however it ends.

``CHAT_MAX_CONCURRENT_TURNS`` (default 8) slots.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Callable, Optional

from fastapi import HTTPException

BUSY_MESSAGE = "The assistant is busy with other questions. Please try again in a minute."
_SEM: Optional[asyncio.Semaphore] = None
_SEM_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def _semaphore() -> asyncio.Semaphore:
    global _SEM, _SEM_LOOP
    loop = asyncio.get_running_loop()
    if _SEM is None or _SEM_LOOP is not loop:
        _SEM = asyncio.Semaphore(max(1, int(_env_number("CHAT_MAX_CONCURRENT_TURNS", 8))))
        _SEM_LOOP = loop
    return _SEM


async def acquire() -> Callable[[], None]:
    """Take a turn slot or raise HTTP 503. Returns the release function."""
    sem = _semaphore()
    wait = max(0.0, _env_number("CHAT_TURN_QUEUE_WAIT_SECONDS", 60))
    try:
        await asyncio.wait_for(sem.acquire(), timeout=wait)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail=BUSY_MESSAGE,
                            headers={"Retry-After": "30"}) from None
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            sem.release()
    return release


async def hold_until_done(stream: AsyncIterator, release: Callable[[], None]) -> AsyncIterator:
    """Yield ``stream``; release the slot when it ends, fails or is abandoned."""
    try:
        async for item in stream:
            yield item
    finally:
        release()
