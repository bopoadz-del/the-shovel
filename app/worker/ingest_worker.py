from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from arq.connections import RedisSettings

from app.core import doc_index
from app.core.db import SessionLocal
from app.core.models import IngestionJob

logger = logging.getLogger(__name__)


async def ingest_document(
    ctx: dict[str, Any], project_id: str, document_id: str, job_id: str
) -> None:
    """arq worker function that indexes a single document and updates its job.

    Runs the synchronous ``doc_index.maybe_eager_index`` in the default executor
    so the arq event loop is not blocked by CPU/IO-heavy extraction/chunking.
    Honours the ``INDEX_ON_UPLOAD`` gate: when indexing is disabled the helper
    returns ``None`` and the job is recorded as completed with ``chunks=0``.

    On any failure the job row is marked ``failed`` with a short error message
    and the exception is re-raised so arq can retry / dead-letter.
    """
    db = SessionLocal()
    job: IngestionJob | None = None
    try:
        job_id_uuid = uuid.UUID(job_id)
        job = db.query(IngestionJob).filter_by(id=job_id_uuid).first()
        if job is not None:
            job.status = "running"
            db.commit()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, doc_index.maybe_eager_index, project_id, document_id
        )
        chunks = len(result.get("chunks", [])) if isinstance(result, dict) else 0

        if job is not None:
            job.status = "completed"
            job.chunks = chunks
            db.commit()
    except Exception as exc:
        logger.exception("Ingestion failed for %s/%s", project_id, document_id)
        if job is not None:
            job.status = "failed"
            job.error = str(exc)[:500]
            db.commit()
        raise
    finally:
        db.close()


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(
        os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    max_jobs = 10
    job_timeout = 600
    max_tries = 3
    keep_result = 86400
