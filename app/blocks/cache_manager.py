"""Cache Manager Block - Redis wrapper with in-memory fallback."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
from app.core.redis_client import get_redis_client
from app.core.universal_base import UniversalBlock


logger = logging.getLogger(__name__)


def compose_scoped_key(
    raw_key: str,
    *,
    project_id: str | None = None,
    contract_id: str | None = None,
    source_class: str | None = None,
) -> str:
    """Cache key that cannot collide across project, contract, or class.

    #443-via-cache: an unscoped ``delay-damages`` key returns another
    year's contract. The scoped form embeds all three isolation axes.
    """
    return "|".join((
        f"p={(project_id or '').strip()}",
        f"c={(contract_id or '').strip()}",
        f"class={(source_class or '').strip()}",
        raw_key or "",
    ))


class CacheManagerBlock(UniversalBlock):
    """Key-value cache with Redis support and local fallback."""

    name = "cache_manager"
    version = "1.0.0"
    description = "Redis wrapper with get/set/delete/stats actions"
    layer = 0
    tags = ["infrastructure", "cache", "redis"]
    requires = []

    default_config = {
        "default_ttl": 3600,
        "max_local_entries": 10000,
    }

    ui_schema = {
        "input": {
            "type": "json",
            "accept": None,
            "placeholder": '{"action": "get", "key": "my-key"}',
            "multiline": False
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "value", "type": "json", "label": "Cached Value"}
            ]
        },
        "quick_actions": [
            {"icon": "📥", "label": "Get Value", "prompt": '{"action":"get","key":"my-key"}'},
            {"icon": "📤", "label": "Set Value", "prompt": '{"action":"set","key":"my-key","value":"my-value","ttl":3600}'},
            {"icon": "🗑️", "label": "Clear Cache", "prompt": '{"action":"clear"}'}
        ]
    }

    def __init__(self, hal_block=None, config=None):
        super().__init__(hal_block, config)
        self._local_cache: Dict[str, Dict] = {}

    async def _redis(self):
        return await get_redis_client()

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Route to appropriate cache action."""
        params = params or {}
        action = params.get("action") or (input_data.get("action") if isinstance(input_data, dict) else "stats")
        handlers = {
            "get": self.get,
            "set": self.set,
            "delete": self.delete,
            "exists": self.exists,
            "flush": self.flush,
            "stats": self.stats,
            "health_check": self.health_check,
        }
        handler = handlers.get(action)
        if not handler:
            return {"status": "error", "error": f"Unknown action: {action}"}
        return await handler(input_data, params)

    async def get(self, input_data: Any, params: Dict) -> Dict:
        """Retrieve value by key."""
        key = self._resolve_key(input_data, params)
        if not key:
            return {"status": "error", "error": "No key provided"}

        redis = await self._redis()
        if redis:
            try:
                raw = await redis.get(key)
                if raw is None:
                    return {"status": "success", "found": False, "key": key}
                return {"status": "success", "found": True, "key": key, "value": json.loads(raw)}
            except Exception as exc:
                logger.warning("Redis cache operation failed, falling back to local dict: %s", exc)

        entry = self._local_cache.get(key)
        if entry is None or entry.get("expires", float("inf")) < time.time():
            return {"status": "success", "found": False, "key": key}
        return {"status": "success", "found": True, "key": key, "value": entry["value"]}

    async def set(self, input_data: Any, params: Dict) -> Dict:
        """Store value by key with optional TTL."""
        key = self._resolve_key(input_data, params)
        if not key:
            return {"status": "error", "error": "No key provided"}

        value = params.get("value") or (input_data.get("value") if isinstance(input_data, dict) else None)
        ttl = params.get("ttl", self.config.get("default_ttl", 3600))

        redis = await self._redis()
        if redis:
            try:
                await redis.setex(key, ttl, json.dumps(value))
                return {"status": "success", "action": "set", "key": key, "ttl": ttl}
            except Exception as exc:
                logger.warning("Redis cache operation failed, falling back to local dict: %s", exc)

        # Enforce local limit
        if len(self._local_cache) >= self.config.get("max_local_entries", 10000):
            self._evict_oldest()

        self._local_cache[key] = {"value": value, "expires": time.time() + ttl}
        return {"status": "success", "action": "set", "key": key, "ttl": ttl}

    async def delete(self, input_data: Any, params: Dict) -> Dict:
        """Remove key from cache."""
        key = self._resolve_key(input_data, params)
        if not key:
            return {"status": "error", "error": "No key provided"}

        redis = await self._redis()
        if redis:
            try:
                deleted = await redis.delete(key)
                return {"status": "success", "deleted": bool(deleted), "key": key}
            except Exception as exc:
                logger.warning("Redis cache operation failed, falling back to local dict: %s", exc)

        existed = key in self._local_cache
        self._local_cache.pop(key, None)
        return {"status": "success", "deleted": existed, "key": key}

    async def exists(self, input_data: Any, params: Dict) -> Dict:
        """Check if key exists."""
        key = self._resolve_key(input_data, params)
        if not key:
            return {"status": "error", "error": "No key provided"}

        redis = await self._redis()
        if redis:
            try:
                return {"status": "success", "exists": bool(await redis.exists(key)), "key": key}
            except Exception as exc:
                logger.warning("Redis cache operation failed, falling back to local dict: %s", exc)

        entry = self._local_cache.get(key)
        exists = entry is not None and entry.get("expires", float("inf")) >= time.time()
        return {"status": "success", "exists": exists, "key": key}

    async def flush(self, input_data: Any = None, params: Dict = None) -> Dict:
        """Clear all cached entries."""
        redis = await self._redis()
        if redis:
            try:
                await redis.flushdb()
                return {"status": "success", "action": "flush"}
            except Exception as exc:
                logger.warning("Redis cache operation failed, falling back to local dict: %s", exc)

        count = len(self._local_cache)
        self._local_cache.clear()
        return {"status": "success", "action": "flush", "local_entries_cleared": count}

    async def stats(self, input_data: Any = None, params: Dict = None) -> Dict:
        """Return cache statistics."""
        redis = await self._redis()
        if redis:
            try:
                info = await redis.info()
                return {
                    "status": "success",
                    "backend": "redis",
                    "keys": await redis.dbsize(),
                    "used_memory_human": info.get("used_memory_human", "unknown")
                }
            except Exception as exc:
                logger.warning("Redis cache operation failed, falling back to local dict: %s", exc)

        # Clean expired local entries
        now = time.time()
        valid = {k: v for k, v in self._local_cache.items() if v.get("expires", float("inf")) >= now}
        self._local_cache = valid
        return {
            "status": "success",
            "backend": "local",
            "entries": len(self._local_cache)
        }

    async def health_check(self, input_data: Any = None, params: Dict = None) -> Dict:
        """Health check for cache manager."""
        client = await self._redis()
        return {
            "status": "success",
            "block": self.name,
            "version": self.version,
            "redis_connected": client is not None
        }

    def _resolve_key(self, input_data: Any, params: Dict) -> Optional[str]:
        raw = params.get("key") or (input_data.get("key") if isinstance(input_data, dict) else None)
        if not raw:
            return None
        blob = params if params else {}
        if isinstance(input_data, dict):
            blob = {**input_data, **blob}
        project_id = blob.get("project_id") or blob.get("projectId")
        contract_id = blob.get("contract_id") or blob.get("contractId")
        source_class = blob.get("source_class") or blob.get("sourceClass")
        if project_id or contract_id or source_class:
            return compose_scoped_key(
                str(raw),
                project_id=project_id,
                contract_id=contract_id,
                source_class=source_class,
            )
        return str(raw)

    def _evict_oldest(self):
        if self._local_cache:
            oldest = min(self._local_cache, key=lambda k: self._local_cache[k]["expires"])
            self._local_cache.pop(oldest, None)

    def get_actions(self) -> Dict[str, Any]:
        """Return all public methods for block registry."""
        return {
            "get": self.get,
            "set": self.set,
            "delete": self.delete,
            "exists": self.exists,
            "flush": self.flush,
            "stats": self.stats,
            "health_check": self.health_check,
        }
