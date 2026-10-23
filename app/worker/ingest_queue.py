from __future__ import annotations
import os

from arq import create_pool
from arq.connections import RedisSettings

_pool = None


def _worker_declared() -> bool:
    """True only when an ingest worker is DECLARED to be draining the queue.

    Redis being configured does not imply a worker exists. Redis is also used
    for sessions and rate limiting, so it stayed in render.yaml when the arq
    worker was removed on 2026-08-08 — and every subsequent upload was enqueued
    to a queue nobody served, leaving documents at chunk_count 0 permanently.
    Callers fall back to inline indexing on False, so the safe default is False:
    a missing worker must degrade to slower ingestion, never to no ingestion.
    """
    return os.getenv("INGEST_WORKER_ENABLED", "").strip().lower() in {"1", "true", "yes"}


async def enqueue_ingest(project_id: str, document_id: str, job_id: str) -> bool:
    global _pool
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url or not _worker_declared():
        return False
    try:
        if _pool is None:
            _pool = await create_pool(RedisSettings.from_dsn(redis_url))
        await _pool.enqueue_job("ingest_document", project_id, document_id, job_id)
        return True
    except Exception:
        return False
