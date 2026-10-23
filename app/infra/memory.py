from app.infra.lego_base import LegoBlock
from typing import Dict, Any, Optional, List
import time
import asyncio
from collections import OrderedDict

class MemoryBlock(LegoBlock):
    """
    High-Speed Memory Cache Block - TTL, LRU eviction, session storage
    Acts as Redis alternative for edge/local deployments
    """
    
    name = "memory"
    version = "1.0.0"
    requires = ["config"]
    layer = 1  # Security/Session layer
    tags = ["security", "cache", "infrastructure"]
    default_config = {
        "max_size": 10000,
        "default_ttl": 3600,
        "cleanup_interval": 300
    }
    
    def __init__(self, hal_block, config: Dict[str, Any]):
        super().__init__(hal_block, config)
        self.cache = {}  # key -> {value, expiry, access_count}
        self.access_order = OrderedDict()  # LRU tracking
        self.max_size = config.get("max_size", 10000)  # Max items
        self.default_ttl = config.get("default_ttl", 3600)  # 1 hour default
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.cleanup_task = None
    
    async def initialize(self):
        """Start background cleanup"""
        self.cleanup_task = asyncio.create_task(self._cleanup_expired())
        print(f"[memory] Memory Block ready (max: {self.max_size}, TTL: {self.default_ttl}s)")
        return True
    
    async def execute(self, input_data: Dict) -> Dict:
        """Cache operations: get, set, delete, flush"""
        action = input_data.get("action")
        
        if action == "get":
            return await self._get(input_data.get("key"))
        elif action == "set":
            return await self._set(
                input_data.get("key"), 
                input_data.get("value"),
                input_data.get("ttl", self.default_ttl)
            )
        elif action == "delete":
            return await self._delete(input_data.get("key"))
        elif action == "exists":
            return {"exists": input_data.get("key") in self.cache}
        elif action == "flush":
            return await self._flush()
        elif action == "stats":
            return self._get_stats()
        elif action == "keys":
            return {"keys": list(self.cache.keys())}
        
        return {"error": f"Unknown action: {action}"}
    
    async def _get(self, key: str) -> Dict:
        """Get value with LRU update"""
        if key in self.cache:
            item = self.cache[key]
            
            # Check TTL
            if time.time() > item["expiry"]:
                del self.cache[key]
                if key in self.access_order:
                    del self.access_order[key]
                self.stats["misses"] += 1
                return {"value": None, "hit": False, "reason": "expired"}
            
            # Update access order (LRU)
            if key in self.access_order:
                del self.access_order[key]
            self.access_order[key] = None
            
            self.stats["hits"] += 1
            item["access_count"] += 1
            
            return {"value": item["value"], "hit": True, "ttl_remaining": item["expiry"] - time.time()}
        
        self.stats["misses"] += 1
        return {"value": None, "hit": False}
    
    async def _set(self, key: str, value: Any, ttl: int) -> Dict:
        """Set value with TTL"""
        # Eviction if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            await self._evict_lru()
        
        expiry = time.time() + ttl if ttl > 0 else float('inf')
        
        self.cache[key] = {
            "value": value,
            "expiry": expiry,
            "created": time.time(),
            "access_count": 0
        }
        
        # Update access order
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = None
        
        return {"stored": True, "key": key, "ttl": ttl}
    
    async def _delete(self, key: str) -> Dict:
        """Delete key"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                del self.access_order[key]
            return {"deleted": True}
        return {"deleted": False, "reason": "not_found"}
    
    async def _flush(self) -> Dict:
        """Clear all"""
        count = len(self.cache)
        self.cache.clear()
        self.access_order.clear()
        return {"flushed": True, "count": count}
    
    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_order:
            return
        
        # Get oldest item
        oldest_key = next(iter(self.access_order))
        del self.cache[oldest_key]
        del self.access_order[oldest_key]
        self.stats["evictions"] += 1
    
    async def _cleanup_expired(self):
        """Background task: remove expired keys every 60s"""
        while True:
            await asyncio.sleep(60)
            current_time = time.time()
            expired = [k for k, v in self.cache.items() if current_time > v["expiry"]]
            for k in expired:
                del self.cache[k]
                if k in self.access_order:
                    del self.access_order[k]
    
    def _get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate_percent": round(hit_rate, 2),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "memory_items": len(self.cache)
        }
    
    def health(self) -> Dict:
        h = super().health()
        h.update(self._get_stats())
        h["utilization_percent"] = round(len(self.cache) / self.max_size * 100, 2)
        return h
