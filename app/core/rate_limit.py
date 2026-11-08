"""Per-caller rate limiting with optional Redis-backed sliding window.

A fixed 60-second sliding window per identity. This covers EVERY request,
including JWT-authenticated sessions — the per-API-key limiter in
app/core/auth.py only ever saw legacy API keys, so logged-in user traffic
(the bulk of the API) was completely unthrottled.

Controlled by RATE_LIMIT_PER_MINUTE (default 300; 0 or negative disables it).

When REDIS_URL is set and reachable, limits are shared across uvicorn workers
via a Redis sorted-set sliding window. Otherwise falls back to the in-process
limiter (fine for single-worker dev).

Anonymous identities are ``ip:<socket peer>`` unless ``ClientAddressMiddleware``
has bound the request and ``TRUSTED_PROXY_COUNT`` plus ``TRUSTED_PROXY_CIDRS``
both name a real proxy tier. Only then is the bucket the forwarded client.
See ``app.core.client_address``.

A Redis error while that shared limiter is selected fails open for ordinary
routes, so a cache blip does not take the API down. Credential routes
(login, registration, password reset, and ``/v1/auth``) fail closed: the
request is denied rather than served with no shared limit.
"""

import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional
import logging

logger = logging.getLogger(__name__)

from app.core.client_address import (
    bound_forwarded_for,
    bound_peer,
    credential_route_fail_closed,
    forwarded_client_host,
    forwarding_context_bound,
)
from app.core.redis_client import get_sync_redis_client

_WINDOW_SECONDS = 60.0
_lock = threading.Lock()
_buckets: Dict[str, Deque[float]] = {}

# Set by init_rate_limiter(); None until first check or explicit init.
_use_redis: Optional[bool] = None
_redis_limiter: Optional["RedisRateLimiter"] = None

_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
    return 0
end
redis.call('ZADD', key, now, tostring(now))
redis.call('EXPIRE', key, math.ceil(window) + 1)
return 1
"""


def _limit() -> int:
    """Requests allowed per identity per minute (read at call time)."""
    try:
        return int(os.getenv("RATE_LIMIT_PER_MINUTE", "300"))
    except (TypeError, ValueError):
        return 300


def _in_memory_check_and_record(identity: str) -> bool:
    limit = _limit()
    if limit <= 0:
        return True
    now = time.time()
    cutoff = now - _WINDOW_SECONDS
    with _lock:
        bucket = _buckets.get(identity)
        if bucket is None:
            bucket = _buckets[identity] = deque()
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        if len(_buckets) > 5000:
            stale = [k for k, b in _buckets.items() if not b or b[-1] < cutoff]
            for k in stale:
                _buckets.pop(k, None)
        return True


class RedisRateLimiter:
    """Shared sliding-window limiter backed by a Redis sorted set."""

    _PREFIX = "ratelimit:"

    def __init__(self):
        self._client = None
        self._script = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        client = get_sync_redis_client()
        if client is None:
            return False
        self._client = client
        self._script = self._client.register_script(_SLIDING_WINDOW_LUA)
        return True

    def ping(self) -> None:
        if not self._ensure_client():
            raise ConnectionError("Redis unavailable")
        self._client.ping()

    def check_and_record(self, identity: str, *, fail_closed: bool = False) -> bool:
        """Record ``identity`` and report whether it is still under the limit.

        ``fail_closed`` denies the request when Redis cannot answer. Credential
        routes pass True; every other caller keeps the historical fail-open.
        """
        limit = _limit()
        if limit <= 0:
            return True
        if not self._ensure_client():
            if fail_closed:
                logger.warning(
                    "rate limit redis client missing; denying credential route"
                )
                return False
            return _in_memory_check_and_record(identity)
        key = f"{self._PREFIX}{identity}"
        try:
            allowed = self._script(
                keys=[key],
                args=[time.time(), _WINDOW_SECONDS, limit],
            )
            return bool(allowed)
        except Exception:
            if fail_closed:
                logger.warning(
                    "rate limit redis error; denying credential route",
                    exc_info=True,
                )
                return False
            # Ordinary traffic stays up during a cache blip. Credential
            # routes do not: an open limiter there is an unthrottled login.
            return True


def init_rate_limiter() -> str:
    """Select and warm the active backend. Returns a short label for logging."""
    global _use_redis, _redis_limiter
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            limiter = RedisRateLimiter()
            limiter.ping()
            _redis_limiter = limiter
            _use_redis = True
            return "redis"
        except Exception:
            logger.warning(
                "swallowed %s in init_rate_limiter() — continuing",
                "Exception", exc_info=True,
            )
    _redis_limiter = None
    _use_redis = False
    return "in-memory"


def _ensure_backend() -> None:
    if _use_redis is None:
        init_rate_limiter()


def reset_for_tests() -> None:
    """Reset backend selection and in-memory buckets (tests only)."""
    global _use_redis, _redis_limiter
    _use_redis = None
    _redis_limiter = None
    _buckets.clear()


def resolve_rate_limit_identity(identity: str) -> str:
    """Apply trusted-proxy forwarding to an anonymous ``ip:`` identity.

    Authenticated identities (``user:`` / ``key:``) are returned unchanged.
    With no request binding, the identity is returned unchanged so direct
    callers keep the bucket they named.
    """
    if not isinstance(identity, str) or not identity.startswith("ip:"):
        return identity
    if not forwarding_context_bound():
        return identity
    host = forwarded_client_host(bound_peer(), bound_forwarded_for())
    return "ip:" + host


def check_and_record(identity: str) -> bool:
    """Return True if a request from ``identity`` is within its per-minute
    limit — recording it — or False if the identity is over the limit.

    Credential routes deny the request when the shared Redis limiter cannot
    answer. Other routes keep a fail-open on that error.
    """
    identity = resolve_rate_limit_identity(identity)
    fail_closed = credential_route_fail_closed()
    _ensure_backend()
    if _use_redis and _redis_limiter is not None:
        return _redis_limiter.check_and_record(identity, fail_closed=fail_closed)
    return _in_memory_check_and_record(identity)
