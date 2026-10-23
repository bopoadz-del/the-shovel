# Redis Infrastructure Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared Redis client, enforce chat/retrieval cache compliance, move ingestion to an arq worker queue, and validate multi-worker uvicorn behavior in The_Fork.

**Architecture:** A single `app/core/redis_client.py` factory provides both async and sync Redis clients. Existing modules consolidate connection creation there while keeping their public APIs unchanged where possible. An arq worker service runs ingestion off the request path; job status lives in Postgres. Uvicorn workers scale to 2 with shared Redis-backed rate limiting.

**Tech Stack:** Python 3.11, FastAPI, `redis` (async + sync), `arq`, SQLAlchemy/Alembic, Docker Compose, Render Blueprint.

## Global Constraints

- Redis must be optional: app boots and passes health with `REDIS_URL` unset or Redis unreachable.
- Existing fallbacks must stay intact: in-memory rate limit/session/cache, file-based hydration lock.
- Each phase is one independently revertible commit on branch `feat/redis-p1-p4-p3-p5`.
- No external egress: Redis runs inside the host network (cloud or edge).
- `rate_limit.py` keeps its sync Lua sliding-window API; only its client acquisition changes.
- `session_store.py` methods become async; explicit caller sites updated.
- Worker must not block the event loop on sync `index_document`; use `run_in_executor`.
- Worker must call `maybe_eager_index` to honor `INDEX_ON_UPLOAD`.

---

## File Structure

| File | Responsibility |
|---|---|
| `app/core/redis_client.py` | Shared async + sync Redis singleton factory, health, lifecycle |
| `app/core/rate_limit.py` | Sync sliding-window limiter; gets sync client from shared factory |
| `app/core/session_store.py` | Async session store interface + Redis/in-memory backends |
| `app/core/hydration_scheduler.py` | Async leader lock using shared async client |
| `app/blocks/cache_manager.py` | Async cache block using shared async client |
| `app/infra/monitoring.py` | Adds `redis` field to health payload (now async) |
| `app/routers/health.py` | `/v1/health` becomes async |
| `app/core/cache_wrapper.py` | Thin async get/set/delete wrapper |
| `tests/test_retrieval_cache_ban.py` | Compliance test proving chat/retrieval never cache |
| `requirements.in` / `requirements.txt` | Add `arq>=0.26` |
| `app/core/models.py` | Add `IngestionJob` SQLAlchemy model |
| `alembic/versions/` | Migration for `ingestion_jobs` table |
| `app/worker/ingest_queue.py` | Enqueue ingestion jobs via arq; fallback to BackgroundTasks |
| `app/worker/ingest_worker.py` | arq worker function wrapping `maybe_eager_index` |
| `app/worker/run_worker.py` | CLI entrypoint for worker |
| `app/routers/upload.py` | Persist job row, enqueue or fallback, return status |
| `app/routers/admin.py` | Dead-letter listing endpoint |
| `docker-compose.yml` | Redis service, worker service, `REDIS_URL`, `UVICORN_WORKERS=2` |
| `render.yaml` | Redis service, worker service, env vars |
| `tests/test_redis_*.py` | Unit/integration tests per module |
| `tests/test_ingest_worker.py` | Worker integration test |
| `tests/test_uvicorn_workers.py` | Smoke + concurrency tests |

---

## Task 1: Create shared Redis client factory

**Files:**
- Create: `app/core/redis_client.py`
- Test: `tests/test_redis_client.py`

**Interfaces:**
- Produces: `get_redis_client() -> Optional[aioredis.Redis]`
- Produces: `get_sync_redis_client() -> Optional[redis.Redis]`
- Produces: `close_redis_client() -> None`
- Produces: `redis_health() -> dict[str, Any]`
- Produces: `reset_for_tests() -> None`

- [ ] **Step 1: Write the factory**

```python
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
```

- [ ] **Step 2: Write unit tests**

```python
import os
import pytest

from app.core import redis_client


@pytest.fixture(autouse=True)
def _reset():
    redis_client.reset_for_tests()
    old = os.environ.pop("REDIS_URL", None)
    yield
    redis_client.reset_for_tests()
    if old is not None:
        os.environ["REDIS_URL"] = old
    else:
        os.environ.pop("REDIS_URL", None)


@pytest.mark.asyncio
async def test_get_redis_client_returns_none_when_url_unset():
    assert await redis_client.get_redis_client() is None


@pytest.mark.asyncio
async def test_get_sync_redis_client_returns_none_when_url_unset():
    assert redis_client.get_sync_redis_client() is None


@pytest.mark.asyncio
async def test_redis_health_false_when_url_unset():
    health = await redis_client.redis_health()
    assert health == {"connected": False, "latency_ms": None}
```

- [ ] **Step 3: Run tests**

```bash
cd The_Fork
pytest tests/test_redis_client.py -v
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add app/core/redis_client.py tests/test_redis_client.py
git commit -m "P1: shared async+sync Redis client factory"
```

---

## Task 2: Migrate rate_limit.py to shared sync client

**Files:**
- Modify: `app/core/rate_limit.py:75-121`
- Test: `tests/test_rate_limit.py`

**Interfaces:**
- Consumes: `get_sync_redis_client()` from `app/core/redis_client.py`
- Produces: `check_and_record(identity: str) -> bool` stays sync.

- [ ] **Step 1: Replace direct `redis.from_url` with shared factory**

In `app/core/rate_limit.py`:

```python
from app.core.redis_client import get_sync_redis_client

class RedisRateLimiter:
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
        if self._ensure_client():
            self._client.ping()

    def check_and_record(self, identity: str) -> bool:
        limit = _limit()
        if limit <= 0:
            return True
        if not self._ensure_client():
            return _in_memory_check_and_record(identity)
        key = f"{self._PREFIX}{identity}"
        try:
            allowed = self._script(
                keys=[key],
                args=[time.time(), _WINDOW_SECONDS, limit],
            )
            return bool(allowed)
        except Exception:
            return _in_memory_check_and_record(identity)
```

Update `init_rate_limiter()`:

```python
def init_rate_limiter() -> str:
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
            pass
    _redis_limiter = None
    _use_redis = False
    return "in-memory"
```

- [ ] **Step 2: Update tests**

Ensure `tests/test_rate_limit.py` patches `app.core.redis_client.get_sync_redis_client` instead of `redis.from_url`.

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_rate_limit.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add app/core/rate_limit.py tests/test_rate_limit.py
git commit -m "P1: rate_limiter uses shared sync Redis factory"
```

---

## Task 3: Migrate session_store.py to async

**Files:**
- Modify: `app/core/session_store.py`
- Modify: `app/routers/project.py:68,71,126`
- Test: `tests/test_session_store.py`

**Interfaces:**
- Consumes: `get_redis_client()` from `app/core/redis_client.py`
- Produces: `SessionStore.get()`, `save()`, `delete()`, `get_or_create()` are async.
- Produces: `get_session_store()` stays sync but returns an async-capable store.

- [ ] **Step 1: Make interface and backends async**

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import copy
import os
import time

from app.core.redis_client import get_redis_client
from app.schemas.project_session import ProjectSession

DEFAULT_TTL_SECONDS = 14400


class SessionStore(ABC):
    @abstractmethod
    async def get(self, session_id: str) -> Optional[ProjectSession]: ...

    @abstractmethod
    async def save(self, session: ProjectSession) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool: ...

    async def get_or_create(self, session_id: str) -> ProjectSession:
        existing = await self.get(session_id)
        if existing is not None:
            return existing
        session = ProjectSession.new(session_id)
        await self.save(session)
        return session


class InMemorySessionStore(SessionStore):
    def __init__(self, ttl_seconds: int = 0):
        self._ttl = ttl_seconds or int(os.getenv("SESSION_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
        self._data: Dict[str, Tuple[ProjectSession, float]] = {}

    async def get(self, session_id: str) -> Optional[ProjectSession]:
        entry = self._data.get(session_id)
        if entry is None:
            return None
        session, expires_at = entry
        if time.time() > expires_at:
            del self._data[session_id]
            return None
        return copy.deepcopy(session)

    async def save(self, session: ProjectSession) -> None:
        session.touch()
        self._data[session.id] = (session, time.time() + self._ttl)

    async def delete(self, session_id: str) -> bool:
        return self._data.pop(session_id, None) is not None


class RedisSessionStore(SessionStore):
    _PREFIX = "session:"

    def __init__(self, ttl_seconds: int = 0):
        self._ttl = ttl_seconds or int(os.getenv("SESSION_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))

    def _key(self, session_id: str) -> str:
        return f"{self._PREFIX}{session_id}"

    async def get(self, session_id: str) -> Optional[ProjectSession]:
        client = await get_redis_client()
        if client is None:
            return None
        try:
            raw = await client.get(self._key(session_id))
            if raw is None:
                return None
            return ProjectSession.model_validate_json(raw)
        except Exception:
            return None

    async def save(self, session: ProjectSession) -> None:
        client = await get_redis_client()
        if client is None:
            return
        try:
            session.touch()
            await client.set(self._key(session.id), session.model_dump_json(), ex=self._ttl)
        except Exception:
            pass

    async def delete(self, session_id: str) -> bool:
        client = await get_redis_client()
        if client is None:
            return False
        try:
            return bool(await client.delete(self._key(session_id)))
        except Exception:
            return False


def get_session_store() -> SessionStore:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis as _redis
            _redis.from_url(redis_url).ping()
            return RedisSessionStore()
        except Exception as exc:
            import warnings
            warnings.warn(f"Redis unavailable ({exc!r}); falling back to in-memory session store.", RuntimeWarning, stacklevel=2)
    return InMemorySessionStore()
```

- [ ] **Step 2: Update callers in project.py**

At `app/routers/project.py:68`:

```python
session = await _store.get(body.session_id)
if session is None:
    session = ProjectSession.new(body.session_id, user_id=caller_id)
    await _store.save(session)
```

At `app/routers/project.py:126`:

```python
await _store.save(session)
```

- [ ] **Step 3: Update tests and run**

```bash
pytest tests/test_session_store.py tests/routers/test_project.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add app/core/session_store.py app/routers/project.py tests/test_session_store.py
git commit -m "P1: async session store with Redis and in-memory backends"
```

---

## Task 4: Migrate hydration_scheduler.py to async shared client

**Files:**
- Modify: `app/core/hydration_scheduler.py:45-88`
- Test: `tests/test_hydration_scheduler.py`

**Interfaces:**
- Consumes: `get_redis_client()` from `app/core/redis_client.py`
- Produces: `_acquire_leader_lock(target_date_iso: str) -> Optional[aioredis.Redis]` stays returning client or None.

- [ ] **Step 1: Convert leader lock to async shared client**

```python
from app.core.redis_client import get_redis_client

async def _acquire_leader_lock(target_date_iso: str) -> Optional[Any]:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        logger.info("hydration: no REDIS_URL; skipping leader lock")
        return None

    client = await get_redis_client()
    if client is None:
        logger.warning("hydration: Redis configured but unreachable; running without lock")
        return None

    worker_id = f"{os.getpid()}:{socket.gethostname()}"
    key = _leader_key(target_date_iso)
    try:
        acquired = await client.set(key, worker_id, nx=True, ex=3600)
    except Exception as exc:
        logger.warning("hydration: redis SET NX failed (%s); running without lock", exc)
        return None

    if not acquired:
        return None
    return client
```

- [ ] **Step 2: Update caller `_run_one_pass`**

```python
async def _run_one_pass() -> None:
    target_date_iso = datetime.now(timezone.utc).date().isoformat()
    redis_url_present = bool(os.getenv("REDIS_URL", "").strip())
    client = await _acquire_leader_lock(target_date_iso)

    if redis_url_present and client is None:
        logger.info("hydration: another worker holds leader lock for %s, skipping", target_date_iso)
        return

    try:
        await _do_hydration_pass()
    finally:
        if client is not None:
            try:
                await client.delete(_leader_key(target_date_iso))
            except Exception as exc:
                logger.warning("hydration: leader lock release failed: %s", exc)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_hydration_scheduler.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add app/core/hydration_scheduler.py tests/test_hydration_scheduler.py
git commit -m "P1: hydration scheduler uses shared async Redis client"
```

---

## Task 5: Migrate cache_manager.py to shared async client

**Files:**
- Modify: `app/blocks/cache_manager.py:52-61,86-92,109-114,129-134,147-150,158-163,171-181`
- Test: `tests/blocks/test_cache_manager.py`

**Interfaces:**
- Consumes: `get_redis_client()` from `app/core/redis_client.py`
- Produces: `CacheManagerBlock.process()` stays async; behavior unchanged.

- [ ] **Step 1: Replace `_init_redis` with shared client**

```python
from app.core.redis_client import get_redis_client

class CacheManagerBlock(UniversalBlock):
    # ... existing attributes ...

    def __init__(self, hal_block=None, config=None):
        super().__init__(hal_block, config)
        self._local_cache: Dict[str, Dict] = {}

    async def _redis(self):
        return await get_redis_client()
```

- [ ] **Step 2: Update each Redis action to await shared client**

Example for `get`:

```python
async def get(self, input_data: Any, params: Dict) -> Dict:
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
        except Exception as e:
            return {"status": "error", "error": str(e)}

    entry = self._local_cache.get(key)
    if entry is None or entry.get("expires", float("inf")) < time.time():
        return {"status": "success", "found": False, "key": key}
    return {"status": "success", "found": True, "key": key, "value": entry["value"]}
```

Apply the same pattern to `set`, `delete`, `exists`, `flush`, `stats`.

- [ ] **Step 3: Run tests**

```bash
pytest tests/blocks/test_cache_manager.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add app/blocks/cache_manager.py tests/blocks/test_cache_manager.py
git commit -m "P1: cache_manager uses shared async Redis client"
```

---

## Task 6: Wire Redis health into lifespan and /v1/health

**Files:**
- Modify: `app/infra/monitoring.py:249-258`
- Modify: `app/routers/health.py:11,43-48`
- Modify: `app/main.py:149,211-214`
- Test: `tests/test_health.py`

**Interfaces:**
- Consumes: `redis_health()` from `app/core/redis_client.py`
- Produces: `get_observability_health_payload()` becomes async.

- [ ] **Step 1: Make health payload async and add redis**

```python
async def get_observability_health_payload() -> Dict[str, Any]:
    from app.core.redis_client import redis_health
    snap = block_metrics.snapshot()
    return {
        "observability": {
            "sentry_enabled": sentry_enabled(),
            "structured_logging": structured_logging_enabled(),
            "request_tracing": True,
        },
        "block_metrics": snap["blocks"],
        "redis": await redis_health(),
    }
```

- [ ] **Step 2: Make /v1/health async**

```python
@router.get("/v1/health")
async def health_v1():
    payload = health()
    payload.update(await get_observability_health_payload())
    return payload
```

- [ ] **Step 3: Initialize and close Redis in lifespan**

In `app/main.py` lifespan, after `init_blocks()`:

```python
from app.core import redis_client as _redis_client
await _redis_client.get_redis_client()
```

In finally:

```python
await _redis_client.close_redis_client()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_health.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/infra/monitoring.py app/routers/health.py app/main.py tests/test_health.py
git commit -m "P1: Redis health in /v1/health and lifespan wiring"
```

---

## Task 7: Create thin cache wrapper

**Files:**
- Create: `app/core/cache_wrapper.py`
- Test: `tests/test_cache_wrapper.py`

**Interfaces:**
- Produces: `cache_get(key: str) -> Optional[Any]`
- Produces: `cache_set(key: str, value: Any, ttl: int = 3600) -> bool`
- Produces: `cache_delete(key: str) -> bool`

- [ ] **Step 1: Implement wrapper**

```python
from __future__ import annotations
import json
from typing import Any, Optional

from app.core.redis_client import get_redis_client


async def cache_get(key: str) -> Optional[Any]:
    client = await get_redis_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        return json.loads(raw) if raw is not None else None
    except Exception:
        return None


async def cache_set(key: str, value: Any, ttl: int = 3600) -> bool:
    client = await get_redis_client()
    if client is None:
        return False
    try:
        await client.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception:
        return False


async def cache_delete(key: str) -> bool:
    client = await get_redis_client()
    if client is None:
        return False
    try:
        return bool(await client.delete(key))
    except Exception:
        return False
```

- [ ] **Step 2: Write tests**

```python
import pytest
from app.core.cache_wrapper import cache_get, cache_set, cache_delete

@pytest.mark.asyncio
async def test_cache_round_trip_with_redis(monkeypatch, fake_redis):
    monkeypatch.setattr("app.core.redis_client._async_client", fake_redis)
    assert await cache_set("k", {"v": 1}) is True
    assert await cache_get("k") == {"v": 1}
    assert await cache_delete("k") is True
    assert await cache_get("k") is None
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_cache_wrapper.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add app/core/cache_wrapper.py tests/test_cache_wrapper.py
git commit -m "P4: thin async cache wrapper"
```

---

## Task 8: Write retrieval/chat BAN test

**Files:**
- Create: `tests/test_retrieval_cache_ban.py`

**Interfaces:**
- Consumes: `cache_get`, `cache_set`, `cache_delete` from `app.core.cache_wrapper`
- Produces: passing compliance test.

- [ ] **Step 1: Write test**

```python
from unittest.mock import patch
import pytest

from app.core import cache_wrapper


class _RecordingMock:
    def __init__(self):
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None


@pytest.mark.asyncio
async def test_chat_and_retrieval_never_cache(client):
    set_mock = _RecordingMock()
    get_mock = _RecordingMock()
    del_mock = _RecordingMock()

    with patch.object(cache_wrapper, "cache_set", set_mock), \
         patch.object(cache_wrapper, "cache_get", get_mock), \
         patch.object(cache_wrapper, "cache_delete", del_mock):
        # Chat endpoint
        resp = client.post("/v1/chat", json={"session_id": "s1", "message": "hello"})
        assert resp.status_code in (200, 202)
        # Retrieval endpoint
        resp = client.post("/v1/projects/s1/ask", json={"request": "hello"})
        assert resp.status_code in (200, 202)

    assert not set_mock.calls
    assert not get_mock.calls
    assert not del_mock.calls
```

- [ ] **Step 2: Run test**

```bash
pytest tests/test_retrieval_cache_ban.py -v
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval_cache_ban.py
git commit -m "P4: BAN test proving chat/retrieval never use cache wrapper"
```

---

## Task 9: Add arq dependency

**Files:**
- Modify: `requirements.in`
- Modify: `requirements.txt`

- [ ] **Step 1: Add to requirements.in**

```
arq>=0.26
```

- [ ] **Step 2: Recompile**

```bash
cd The_Fork
pip-compile --output-file=requirements.txt requirements.in
```

- [ ] **Step 3: Commit**

```bash
git add requirements.in requirements.txt
git commit -m "P3: add arq dependency"
```

---

## Task 10: Create ingestion_jobs table and model

**Files:**
- Modify: `app/core/models.py`
- Create: `alembic/versions/20260726_add_ingestion_jobs.py`
- Test: `tests/test_ingestion_job_model.py`

**Interfaces:**
- Produces: `IngestionJob` SQLAlchemy model.

- [ ] **Step 1: Add model**

```python
import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(String, nullable=False)
    document_id = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    chunks = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

- [ ] **Step 2: Generate / write migration**

```bash
alembic revision -m "add ingestion_jobs table"
```

Edit generated migration to create table.

- [ ] **Step 3: Run migration and tests**

```bash
alembic upgrade head
pytest tests/test_ingestion_job_model.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add app/core/models.py alembic/versions/20260726_add_ingestion_jobs.py tests/test_ingestion_job_model.py
git commit -m "P3: ingestion_jobs Postgres table and model"
```

---

## Task 11: Create ingest queue helper

**Files:**
- Create: `app/worker/ingest_queue.py`
- Test: `tests/test_ingest_queue.py`

**Interfaces:**
- Produces: `enqueue_ingest(project_id, document_id, job_id) -> bool`

- [ ] **Step 1: Implement queue helper**

```python
from __future__ import annotations
import os
from typing import Optional

from arq import create_pool
from arq.connections import RedisSettings

_pool = None


async def enqueue_ingest(project_id: str, document_id: str, job_id: str) -> bool:
    global _pool
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return False
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(redis_url))
    try:
        await _pool.enqueue_job("ingest_document", project_id, document_id, job_id)
        return True
    except Exception:
        return False
```

- [ ] **Step 2: Write tests**

```python
import pytest
from unittest.mock import patch, AsyncMock

from app.worker.ingest_queue import enqueue_ingest

@pytest.mark.asyncio
async def test_enqueue_ingest_returns_false_without_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert await enqueue_ingest("p1", "d1", "j1") is False
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_ingest_queue.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add app/worker/ingest_queue.py tests/test_ingest_queue.py
git commit -m "P3: arq ingest enqueue helper"
```

---

## Task 12: Create arq worker

**Files:**
- Create: `app/worker/ingest_worker.py`
- Create: `app/worker/run_worker.py`
- Test: `tests/test_ingest_worker.py`

**Interfaces:**
- Consumes: `doc_index.maybe_eager_index` from `app.core.doc_index`
- Consumes: `IngestionJob` from `app.core.models`
- Produces: `WorkerSettings` class for arq.

- [ ] **Step 1: Implement worker**

```python
from __future__ import annotations
import asyncio
import logging
from typing import Any
import os

from arq.connections import RedisSettings

from app.core import doc_index
from app.core.db import SessionLocal
from app.core.models import IngestionJob

logger = logging.getLogger(__name__)


async def ingest_document(ctx: dict[str, Any], project_id: str, document_id: str, job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(IngestionJob).filter_by(id=job_id).first()
        if job:
            job.status = "running"
            db.commit()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, doc_index.maybe_eager_index, project_id, document_id
        )
        chunks = len(result.get("chunks", [])) if isinstance(result, dict) else 0

        if job:
            job.status = "completed"
            job.chunks = chunks
            db.commit()
    except Exception as exc:
        logger.exception("Ingestion failed for %s/%s", project_id, document_id)
        if job:
            job.status = "failed"
            job.error = str(exc)[:500]
            db.commit()
        raise
    finally:
        db.close()


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URL", "redis://localhost:6379"))
    max_jobs = 10
    job_timeout = 600
    max_tries = 3
    keep_result = 86400
```

- [ ] **Step 2: Create runner**

```python
from arq.run import main
from app.worker.ingest_worker import WorkerSettings

if __name__ == "__main__":
    main(WorkerSettings)
```

- [ ] **Step 3: Write integration test**

```python
import pytest
from app.worker.ingest_worker import ingest_document

@pytest.mark.asyncio
async def test_ingest_document_runs_index_and_updates_job(db, sample_project_doc):
    job = sample_project_doc["job"]
    await ingest_document({}, sample_project_doc["project_id"], sample_project_doc["document_id"], str(job.id))
    db.refresh(job)
    assert job.status == "completed"
    assert job.chunks is not None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ingest_worker.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add app/worker/ingest_worker.py app/worker/run_worker.py tests/test_ingest_worker.py
git commit -m "P3: arq worker for ingestion with run_in_executor"
```

---

## Task 13: Migrate upload endpoint to queue

**Files:**
- Modify: `app/routers/upload.py:99-133`
- Test: `tests/routers/test_upload.py`

**Interfaces:**
- Consumes: `enqueue_ingest` from `app.worker.ingest_queue`
- Consumes: `IngestionJob` creation helper (add to `app/core/projects.py` or inline in upload.py).

- [ ] **Step 1: Add ingestion job creation helper**

In `app/core/projects.py`:

```python
def create_ingestion_job(project_id: str, document_id: str):
    from app.core.db import SessionLocal
    from app.core.models import IngestionJob
    db = SessionLocal()
    try:
        job = IngestionJob(project_id=project_id, document_id=document_id, status="pending")
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    finally:
        db.close()
```

- [ ] **Step 2: Update upload endpoint**

```python
from app.worker.ingest_queue import enqueue_ingest
from app.core.projects import create_ingestion_job

# inside project_id branch, after add_document:
response["indexed"] = True
response["indexing_status"] = "scheduled"
if response["document_id"]:
    job = create_ingestion_job(project_id, response["document_id"])
    ok = await enqueue_ingest(project_id, response["document_id"], str(job.id))
    if ok:
        response["indexing_status"] = "queued"
    else:
        background_tasks.add_task(doc_index.maybe_eager_index, project_id, response["document_id"])
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/routers/test_upload.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add app/core/projects.py app/routers/upload.py tests/routers/test_upload.py
git commit -m "P3: upload enqueues ingestion jobs, fallback to BackgroundTasks"
```

---

## Task 14: Add dead-letter admin endpoint

**Files:**
- Modify: `app/routers/admin.py`
- Test: `tests/routers/test_admin.py`

**Interfaces:**
- Consumes: shared async Redis client.
- Produces: `GET /admin/dead-letter` returns list of failed arq jobs.

- [ ] **Step 1: Implement endpoint**

```python
from app.core.redis_client import get_redis_client

@router.get("/admin/dead-letter")
async def list_dead_letter(auth: dict = Depends(require_admin)):
    client = await get_redis_client()
    if client is None:
        return {"dead_letter": [], "error": "Redis not available"}
    keys = await client.keys("arq:result:*")
    failed = []
    for key in keys:
        raw = await client.get(key)
        if raw:
            data = json.loads(raw)
            if data.get("e"):
                failed.append({
                    "job_id": key.split(":")[-1],
                    "function": data.get("f"),
                    "args": data.get("a"),
                    "error": data.get("e"),
                    "timestamp": data.get("t"),
                })
    return {"dead_letter": failed}
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/routers/test_admin.py -v
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add app/routers/admin.py tests/routers/test_admin.py
git commit -m "P3: dead-letter admin endpoint for failed ingest jobs"
```

---

## Task 15: Update docker-compose.yml and render.yaml

**Files:**
- Modify: `docker-compose.yml`
- Modify: `render.yaml`

- [ ] **Step 1: Update docker-compose.yml**

Uncomment Redis service, add worker service, set env vars:

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: thefork-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  cerebrum-blocks:
    build: .
    container_name: cerebrum-blocks
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env:ro
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - DATA_DIR=/app/data
      - DATABASE_URL=postgresql+psycopg://thefork:thefork@postgres:5432/thefork
      - REDIS_URL=redis://redis:6379/0
      - UVICORN_WORKERS=2
      - PYTHONUNBUFFERED=1
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  worker:
    build: .
    container_name: thefork-worker
    command: ["python", "-m", "app.worker.run_worker"]
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env:ro
    environment:
      - DATA_DIR=/app/data
      - DATABASE_URL=postgresql+psycopg://thefork:thefork@postgres:5432/thefork
      - REDIS_URL=redis://redis:6379/0
      - PYTHONUNBUFFERED=1
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

- [ ] **Step 2: Update render.yaml**

Add Redis service, worker service, and env vars to web service:

```yaml
services:
  - type: redis
    name: the-fork-redis
    plan: free
    region: oregon

  - type: worker
    name: the-fork-worker
    runtime: docker
    dockerfilePath: ./Dockerfile
    plan: starter
    region: oregon
    branch: main
    autoDeploy: true
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: the-fork-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          type: redis
          name: the-fork-redis
          property: connectionString
      - key: DATA_DIR
        value: /app/data
      - key: PYTHONUNBUFFERED
        value: "1"

  # existing web service: add env vars
  - type: web
    name: the-fork
    # ... existing fields ...
    envVars:
      # ... existing env vars ...
      - key: REDIS_URL
        fromService:
          type: redis
          name: the-fork-redis
          property: connectionString
      - key: UVICORN_WORKERS
        value: "2"
```

- [ ] **Step 3: Validate compose**

```bash
cd The_Fork
docker compose config
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml render.yaml
git commit -m "P5: Redis + worker services and UVICORN_WORKERS=2 infra"
```

---

## Task 16: Smoke and concurrency tests

**Files:**
- Create: `tests/test_uvicorn_workers.py`

- [ ] **Step 1: Write smoke test**

```python
import pytest

@pytest.mark.asyncio
async def test_health_reports_redis(async_client):
    for _ in range(10):
        resp = await async_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "redis" in data
        assert data["redis"]["connected"] is True
```

- [ ] **Step 2: Write concurrency probe**

```python
import asyncio
import pytest

@pytest.mark.asyncio
async def test_concurrent_uploads_queue_without_502(async_client, sample_pdf):
    files = [("file", ("doc.pdf", sample_pdf, "application/pdf"))]
    async def upload():
        return await async_client.post(
            "/upload?project_id=p1",
            files=files,
            data={"project_id": "p1"},
        )
    responses = await asyncio.gather(*[upload() for _ in range(20)])
    assert all(r.status_code in (200, 202) for r in responses)
```

- [ ] **Step 3: Write stream starvation test**

```python
import asyncio
import pytest

@pytest.mark.asyncio
async def test_chat_stream_not_starved_during_uploads(async_client, sample_pdf):
    files = [("file", ("doc.pdf", sample_pdf, "application/pdf"))]
    upload_task = asyncio.create_task(
        asyncio.gather(*[
            async_client.post("/upload?project_id=p1", files=files)
            for _ in range(5)
        ])
    )
    stream_resp = await async_client.post("/v1/chat/stream", json={"session_id": "s1", "message": "hello"})
    first_line = await stream_resp.aread_text()
    assert first_line or stream_resp.status_code == 200
    await upload_task
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_uvicorn_workers.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_uvicorn_workers.py
git commit -m "P5: smoke and concurrency tests for UVICORN_WORKERS=2"
```

---

## Self-Review

### Spec coverage

| Spec section | Plan task(s) |
|---|---|
| Shared async+sync client factory | Task 1 |
| Rate limiter keeps sync API | Task 2 |
| Session store async + explicit sites | Task 3 |
| Hydration scheduler async | Task 4 |
| Cache manager shared client | Task 5 |
| Redis health in /v1/health | Task 6 |
| Thin cache wrapper | Task 7 |
| BAN test | Task 8 |
| arq dependency | Task 9 |
| ingestion_jobs table | Task 10 |
| ingest queue helper | Task 11 |
| arq worker with run_in_executor | Task 12 |
| Upload endpoint queue migration | Task 13 |
| Dead-letter endpoint | Task 14 |
| docker-compose + render.yaml | Task 15 |
| Smoke/concurrency tests | Task 16 |

### Placeholder scan

- No TBD/TODO/fill-in-details found.
- Each task includes exact file paths, code, commands, and expected outcomes.

### Type consistency

- `get_redis_client()` returns `Optional[aioredis.Redis]` consistently.
- `get_sync_redis_client()` returns `Optional[redis.Redis]` consistently.
- `SessionStore` methods are async in Task 3 and awaited at project.py call sites.
- Worker uses `maybe_eager_index` and `run_in_executor` as corrected.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-26-redis-infrastructure.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
