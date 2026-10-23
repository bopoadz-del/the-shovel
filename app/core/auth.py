"""API Key authentication and usage tracking for Cerebrum Blocks."""

import os
import time
import hashlib
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


#: The environment variable that holds the development API key. There is no
#: built-in value: a key in the source would be a credential anyone can read.
DEV_KEY_ENV = "CEREBRUM_DEV_KEY"
_DEV_ENVS = {"dev", "development", "local", "test", "testing"}


def is_dev_environment() -> bool:
    import sys

    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "production")).strip().lower()
    return env in _DEV_ENVS or "pytest" in sys.modules


def dev_key() -> Optional[str]:
    """The development API key: the value of ``CEREBRUM_DEV_KEY`` in a
    development environment, None anywhere else or when it is unset (then no
    dev key is accepted -- fail closed)."""
    if not is_dev_environment():
        return None
    value = os.getenv(DEV_KEY_ENV, "").strip()
    return value or None


class APIKeyAuth:
    """Simple API key authentication with usage tracking."""
    
    def __init__(self):
        self.security = HTTPBearer(auto_error=False)
        # In production, use a database. For MVP, env-based keys.
        self._keys = self._load_keys()
        self._usage: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    def _is_dev_environment() -> bool:
        return is_dev_environment()

    def _load_keys(self) -> Dict[str, Dict]:
        keys = {}

        master = os.getenv("CEREBRUM_MASTER_KEY")
        if master:
            keys[master] = {
                "user": "master",
                "tier": "unlimited",
                "role": "admin",
                "rate_limit": float('inf'),
                "created_at": time.time()
            }

        for k, v in os.environ.items():
            if k.startswith("CEREBRUM_API_KEY_") and v:
                keys[v] = {
                    "user": k.replace("CEREBRUM_API_KEY_", "").lower(),
                    "tier": "standard",
                    "rate_limit": 1000,
                    "created_at": time.time()
                }

        return keys


    def validate_key(self, credentials: Optional[HTTPAuthorizationCredentials]) -> Dict[str, Any]:
        if not credentials:
            raise HTTPException(status_code=401, detail="API key required.")

        key = credentials.credentials

        # The development key: read from the environment at every check, valid
        # only in a development environment, and only when it is set.
        dev = dev_key()
        if dev is not None and key == dev:
            return {"user": "dev", "tier": "unlimited", "valid": True}

        if key not in self._keys:
            if dev is None and self._is_dev_environment():
                raise HTTPException(status_code=401, detail=(
                    f"Invalid API key. No development key is configured: set {DEV_KEY_ENV} "
                    "to use one in this environment."))
            raise HTTPException(status_code=401, detail="Invalid API key")

        key_data = self._keys[key].copy()
        key_data["valid"] = True
        self._track_usage(key)

        if self._is_rate_limited(key, key_data.get("rate_limit", 100)):
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")

        return key_data
    
    def _track_usage(self, key: str):
        """Track API usage."""
        now = time.time()
        hour_key = int(now / 3600)
        
        if key not in self._usage:
            self._usage[key] = {}
        
        if hour_key not in self._usage[key]:
            self._usage[key] = {hour_key: 0}
        
        self._usage[key][hour_key] += 1
    
    def _is_rate_limited(self, key: str, limit: int) -> bool:
        """Check if key is rate limited."""
        if limit == float('inf'):
            return False
        
        now = time.time()
        hour_key = int(now / 3600)
        
        usage = self._usage.get(key, {})
        current_hour_usage = usage.get(hour_key, 0)
        
        return current_hour_usage > limit
    
    def get_usage(self, key: str) -> Dict[str, Any]:
        """Get usage stats for a key."""
        now = time.time()
        hour_key = int(now / 3600)
        
        usage = self._usage.get(key, {})
        current_hour = usage.get(hour_key, 0)
        
        return {
            "requests_this_hour": current_hour,
            "rate_limit": self._keys.get(key, {}).get("rate_limit", 100),
            "tier": self._keys.get(key, {}).get("tier", "free")
        }


# Global auth instance
auth = APIKeyAuth()
