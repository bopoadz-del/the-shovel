from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import redis
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_async_client: Optional[aioredis.Redis] = None
_sync_client: Optional[redis.Redis] = None


async def get_redis_client() -> Optional[aioredis.Redis]:
    global _async_client
    if _async_client is not None:
        return _async_client
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        _async_client = aioredis.from_url(redis_url, decode_responses=True)
        await _async_client.ping()
    except Exception as exc:
        logger.warning("Redis async client unavailable (%s); fallbacks active", exc)
        _async_client = None
    return _async_client


def get_sync_redis_client() -> Optional[redis.Redis]:
    global _sync_client
    if _sync_client is not None:
        return _sync_client
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    try:
        _sync_client = redis.from_url(redis_url, decode_responses=True)
        _sync_client.ping()
    except Exception as exc:
        logger.warning("Redis sync client unavailable (%s); fallbacks active", exc)
        _sync_client = None
    return _sync_client


async def close_redis_client() -> None:
    global _async_client, _sync_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None


async def redis_health() -> dict[str, Any]:
    client = await get_redis_client()
    if client is None:
        return {"connected": False, "latency_ms": None}
    start = time.perf_counter()
    try:
        await client.ping()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"connected": True, "latency_ms": latency_ms}
    except Exception:
        return {"connected": False, "latency_ms": None}


def reset_for_tests() -> None:
    global _async_client, _sync_client
    _async_client = None
    _sync_client = None
