from __future__ import annotations
import json
import logging
from typing import Any, Optional

from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


async def cache_get(key: str) -> Optional[Any]:
    client = await get_redis_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        return json.loads(raw) if raw is not None else None
    except Exception:
        logger.warning("cache_get failed for key %r", key, exc_info=True)
        return None


async def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    client = await get_redis_client()
    if client is None:
        return False
    try:
        await client.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception:
        logger.warning("cache_set failed for key %r", key, exc_info=True)
        return False


async def cache_delete(key: str) -> bool:
    client = await get_redis_client()
    if client is None:
        return False
    try:
        return bool(await client.delete(key))
    except Exception:
        logger.warning("cache_delete failed for key %r", key, exc_info=True)
        return False
