# Redis Infrastructure Build — Design Spec

**Repo:** The_Fork  
**Date:** 2026-07-26  
**Scope:** P1 CONSOLIDATE → P4 PIN → P3 BUILD → P5 FLIP  
**Deployment target:** Cloud **and** Edge (profile-agnostic, URL-driven)

---

## 1. Goal

Introduce a single shared async Redis client, enforce retrieval/chat caching compliance via tests, move document ingestion from inline `BackgroundTasks` into an arq-backed worker queue, and validate multi-worker uvicorn behavior. Each phase is independently revertible.

---

## 2. Non-goals

- No change to external LLM provider logic.
- No new cloud-only managed services required; Redis URL is env-driven so the same code runs on Render (cloud) and a self-hosted Docker stack (edge/on-prem).
- No removal of existing fallbacks.

---

## 3. Architectural overview

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI app                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ rate_limit  │  │ session_store│  │   hydration_scheduler│ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         └─────────────────┴────────────────────┘            │
│                              │                              │
│                  app/core/redis_client.py                   │
│                  (single async Redis singleton)             │
│                              │                              │
└──────────────────────────────┼──────────────────────────────┘
                               │ REDIS_URL
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       Redis (queue + state)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              app/worker/ingest_worker.py (arq)              │
│              ──► doc_index.index_document()                 │
│              ──► ingestion_jobs rows in Postgres            │
└─────────────────────────────────────────────────────────────┘
```

All four existing Redis consumers stop creating their own `redis.from_url()` pools. They import the shared async client and fall back to in-memory/local behavior on any Redis transport or auth error.

---

## 4. Deployment profiles

The implementation is deployment-profile neutral:

| Profile | Redis location | Workers |
|---|---|---|
| Local dev | Optional `redis` container, or unset `REDIS_URL` | 1 (default) |
| Cloud (Render) | Render managed Redis or Docker Redis service | `UVICORN_WORKERS=2` + standalone `worker` service |
| Edge / on-prem | Local Docker Redis only | `UVICORN_WORKERS=2` + standalone `worker` service |

`REDIS_URL` is the only switch. If unset, Redis-backed features gracefully degrade to existing fallbacks.

---

## 5. Phase designs

### P1 CONSOLIDATE — shared async Redis client

#### 5.1.1 New shared client: `app/core/redis_client.py`

Vends both async and sync clients from one factory so existing sync consumers can consolidate without API rewrites.

```python
from __future__ import annotations
import os
import time
from typing import Any, Optional

import redis
import redis.asyncio as aioredis

_async_client: Optional[aioredis.Redis] = None
_sync_client: Optional[redis.Redis] = None

async def get_redis_client() -> Optional[aioredis.Redis]:
    """Return the shared async Redis client, or None if not configured/unreachable."""
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
        import logging
        logging.getLogger(__name__).warning("Redis unavailable (%s); fallbacks active", exc)
        _async_client = None
    return _async_client

def get_sync_redis_client() -> Optional[redis.Redis]:
    """Return the shared sync Redis client, or None if not configured/unreachable."""
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
        import logging
        logging.getLogger(__name__).warning("Redis unavailable (%s); fallbacks active", exc)
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

#### 5.1.2 Lifespan wiring

In `app/main.py` lifespan:

```python
from app.core import redis_client as _redis_client

# After init_blocks / init_db / seeding
await _redis_client.get_redis_client()
# ... existing startup ...
try:
    yield
finally:
    await _redis_client.close_redis_client()
    await hydration_scheduler.stop()
```

#### 5.1.3 Module migrations

| Module | Current | Change | Fallback preserved? |
|---|---|---|---|
| `hydration_scheduler.py` | sync `redis.from_url`; `_acquire_leader_lock` is sync | `_acquire_leader_lock` becomes `async`; uses `await redis.set(..., nx=True, ex=3600)`. Returns `None` for three cases: (1) `REDIS_URL` unset → run; (2) Redis unreachable → run (fallback); (3) lock already held → skip. Caller disambiguates by checking `REDIS_URL` and whether the shared client was reachable. | Yes — if Redis down, pass runs unconditionally |
| `rate_limit.py` | sync `RedisRateLimiter` with Lua script | **Keep sync API.** `RedisRateLimiter` obtains its client from `redis_client.get_sync_redis_client()` instead of calling `redis.from_url` directly. The Lua sliding window and `check_and_record()` stay sync; `main.py` rate-limit middleware keeps calling it synchronously. | Yes — Redis hiccup returns `True` (fail open) or falls back to in-memory |
| `session_store.py` | sync `SessionStore` interface | Interface methods become `async` (`get`, `save`, `delete`, `get_or_create`). `RedisSessionStore` uses shared async client. `InMemorySessionStore` keeps sync internals but exposes async API. Explicit call sites updated: `app/routers/project.py:68,71,126`. Other consumers discovered during build also gain `await`. | Yes — Redis unreachable → in-memory store |
| `cache_manager.py` | sync local Redis client in async block | Uses shared async client; local dict fallback on Redis error | Yes — local dict unchanged |

#### 5.1.4 Health payload

Make `app/infra/monitoring.py` `get_observability_health_payload()` async and add:

```python
from app.core.redis_client import redis_health

payload["redis"] = await redis_health()
```

Make `/v1/health` in `app/routers/health.py` async so it can await `get_observability_health_payload()`. `/health` may stay sync (legacy) and skip the Redis latency field, or also be made async.

---

### P4 PIN — cache wrapper + BAN test

#### 5.2.1 Thin cache wrapper: `app/core/cache_wrapper.py`

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

#### 5.2.2 Retrieval/chat BAN test: `tests/test_retrieval_cache_ban.py`

- Patches `app.core.cache_wrapper.cache_get`, `cache_set`, and `cache_delete` with recording mocks that still return plausible values.
- Calls the main chat completion endpoint and the retrieval/search endpoint with identical prompts/contexts.
- Asserts:
  1. `cache_set` is never called from within the chat or retrieval router modules.
  2. `cache_get` is never called from within the chat or retrieval router modules.
  3. No Redis `set`/`setex` is invoked by any module in the call stack of those endpoints.
- This turns “we don’t cache chat/retrieval by default” into “the chat/retrieval code paths cannot call the cache without breaking a test”.

---

### P3 BUILD — arq worker + ingestion queue

#### 5.3.1 Dependencies

Add to `requirements.in`:

```
arq>=0.26
```

Re-run `pip-compile --output-file=requirements.txt requirements.in`.

#### 5.3.2 Postgres job table

New table `ingestion_jobs`:

```sql
CREATE TABLE ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','queued','running','completed','failed')),
    chunks INTEGER,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

SQLAlchemy model added to `app/core/models.py`; Alembic migration generated.

#### 5.3.3 Worker: `app/worker/ingest_worker.py`

```python
from __future__ import annotations
import asyncio
import logging
from typing import Any

from arq import create_pool
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

        # index_document is sync and CPU/IO-heavy (PDF/OCR/embedding). Run it in
        # the default executor so one ingest cannot block the worker event loop
        # and stall other queued jobs.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, doc_index.maybe_eager_index, project_id, document_id
        )
        # maybe_eager_index returns None when INDEX_ON_UPLOAD is disabled, so
        # chunks may legitimately be zero; only count them when indexing ran.
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

Dead-letter visibility: arq stores job results (including exceptions) in Redis for `keep_result` seconds. An admin endpoint `/admin/dead-letter` queries Redis `arq:result:*` keys with non-null `e` (exception) fields and returns `(job_id, function, args, error, timestamp)` for admin users.

#### 5.3.4 Upload flow change

In `app/routers/upload.py`, replace:

```python
background_tasks.add_task(doc_index.maybe_eager_index, project_id, document_id)
```

with:

```python
from app.worker.ingest_queue import enqueue_ingest

job = projects_store.create_ingestion_job(project_id, document_id)
ok = await enqueue_ingest(project_id, document_id, str(job.id))
response["indexing_status"] = "queued" if ok else "error: queue unreachable"
```

New `app/worker/ingest_queue.py`:

```python
import os
from arq import create_pool
from arq.connections import RedisSettings

_pool = None

async def enqueue_ingest(project_id: str, document_id: str, job_id: str) -> bool:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return False
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(redis_url))
    try:
        await _pool.enqueue_job("ingest_document", project_id, document_id, job_id)
        return True
    except Exception:
        return False
```

If `REDIS_URL` is unset or enqueue fails, the upload endpoint falls back to the existing `BackgroundTasks.add_task(doc_index.maybe_eager_index, ...)` path and reports `indexing_status: "scheduled"` instead of `"queued"`.

#### 5.3.5 Worker runner

`app/worker/run_worker.py`:

```python
from arq.run import main
from app.worker.ingest_worker import WorkerSettings

if __name__ == "__main__":
    main(WorkerSettings)
```

---

### P5 FLIP — `UVICORN_WORKERS=2`

#### 5.4.1 Configuration

- Dockerfile CMD or compose environment sets `UVICORN_WORKERS=2`.
- Render `render.yaml` sets `UVICORN_WORKERS=2`.
- Rate-limiting relies on Redis when workers > 1; health check advertises whether Redis is connected.

#### 5.4.2 Validation

1. **Smoke 10:** sequential `GET /health` × 10; assert all 200, `redis.connected == True` when Redis is up.
2. **Concurrency probe:** 20 parallel uploads to the same `project_id`; assert all return 200/202, no 502s, and within 60s all `ingestion_jobs` rows are `completed` with `chunks > 0`.
3. **Stream starvation:** run a long `/chat/stream` while uploads are queueing; assert first token arrives within 5s and stream completes.

---

## 6. Error handling and fallbacks

| Scenario | Behavior |
|---|---|
| Redis down / unreachable | All four modules use in-memory fallback; health reports `connected: false` |
| Redis auth fails | Treated as unavailable; fallbacks active |
| Worker crashes mid-ingestion | `ingestion_jobs` row stays `failed` with error; arq retries 3× then dead-letters |
| Redis URL unset | App boots normally; rate/session/cache use local fallback; upload uses `BackgroundTasks` |
| Multi-worker without Redis | Rate limit becomes per-process; health warns operator |

---

## 7. Testing strategy

- **Unit:** mock async Redis client for each migrated module (`tests/test_redis_client.py`, `tests/test_rate_limit.py`, `tests/test_session_store.py`, `tests/test_cache_manager.py`).
- **Integration:** `pytest` spins up Redis via `docker compose up redis` or pytest-docker.
- **Boot tests:**
  - `docker compose up` **without** Redis commented in → `/v1/health` passes.
  - `docker compose up` **with** Redis commented out → `/v1/health` passes.
- **Compliance:** `tests/test_retrieval_cache_ban.py` runs in CI.
- **Worker test:** enqueue a job, run worker, assert `ingestion_jobs.status == 'completed'` and `chunks > 0`.

---

## 8. Infra changes

### 8.1 `docker-compose.yml`

Uncomment Redis service, add `worker` service, add `REDIS_URL` env:

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
    # ... existing ...
    environment:
      - REDIS_URL=redis://redis:6379/0
      - UVICORN_WORKERS=2
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  worker:
    build: .
    container_name: thefork-worker
    command: ["python", "-m", "app.worker.run_worker"]
    environment:
      - DATABASE_URL=postgresql+psycopg://thefork:thefork@postgres:5432/thefork
      - REDIS_URL=redis://redis:6379/0
      - DATA_DIR=/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 8.2 `render.yaml`

Add:

```yaml
  - type: redis
    name: the-fork-redis
    plan: free
    region: oregon
    ipAllowList: [] # private only

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
```

Add `REDIS_URL` and `UVICORN_WORKERS=2` to the existing web service envVars.

---

## 9. EGRESS ledger

Add one row per phase to `docs/EGRESS_LEDGER.md`:

| Phase | Component | Traffic direction | Justification |
|---|---|---|---|
| P1 | Shared async Redis client | Internal only | Consolidates existing internal Redis usage; no external traffic |
| P4 | Cache wrapper + BAN test | Internal only | Wrapper only touches Redis; BAN test is local assertion |
| P3 | arq worker + ingestion queue | Internal only | Queue state in Redis; job status in Postgres; no external traffic |
| P5 | UVICORN_WORKERS=2 + smoke/concurrency | Internal only | Process scaling; Redis coordination remains internal |

---

## 10. Revertibility and commit plan

Branch: `feat/redis-p1-p4-p3-p5`

| Commit | Phase | Revert command |
|---|---|---|
| `P1: consolidate async Redis client` | P1 | `git revert <p1-sha>` |
| `P4: cache wrapper and retrieval BAN test` | P4 | `git revert <p4-sha>` |
| `P3: arq worker queue for ingestion` | P3 | `git revert <p3-sha>` |
| `P5: uvicorn workers=2 + smoke tests` | P5 | `git revert <p5-sha>` |

Each commit passes CI and boot-tests independently.

---

## 11. Open questions resolved

1. **P2 omitted** — intentional; sequence is P1 → P4 → P3 → P5.
2. **Sync vs async** — migrate all four modules to async Redis; sync call sites are rare and already run inside the async app.
3. **U1** — upload/ingestion 502 user story; closed by queueing ingestion off the request path.
4. **Cloud vs edge** — design supports both via `REDIS_URL`; local dev works without Redis.

---

## 12. Success criteria

- [ ] `/v1/health` includes `redis: {connected, latency_ms}`.
- [ ] `docker compose up` passes health with and without Redis.
- [ ] `test_retrieval_cache_ban.py` passes.
- [ ] Upload with `project_id` returns `indexing_status: "queued"` and creates a `pending`/`queued` `ingestion_jobs` row.
- [ ] Worker processes the job and the row becomes `completed`; when `INDEX_ON_UPLOAD` is enabled, `chunks > 0`.
- [ ] `UVICORN_WORKERS=2` smoke and concurrency tests pass.
- [ ] Each phase is one revertible commit.
