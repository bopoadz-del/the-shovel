"""Tests for the arq ingest_document worker."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from app.core.db import SessionLocal, engine
from app.core.models import IngestionJob
from app.worker.ingest_worker import ingest_document


@pytest.fixture(scope="module", autouse=True)
def _ensure_ingestion_jobs_table() -> None:
    """Create the ingestion_jobs table once for this module."""
    IngestionJob.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def db() -> Session:
    """Yield a fresh SQLAlchemy session for a test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_project_doc(db: Session) -> dict[str, Any]:
    """Create a pending IngestionJob and return its identifiers."""
    job = IngestionJob(
        id=uuid.uuid4(),
        project_id=f"proj_{uuid.uuid4().hex[:8]}",
        document_id=f"doc_{uuid.uuid4().hex[:8]}",
        status="pending",
    )
    db.add(job)
    db.commit()
    return {
        "project_id": job.project_id,
        "document_id": job.document_id,
        "job": job,
    }


@pytest.mark.asyncio
async def test_ingest_document_runs_index_and_updates_job(
    db: Session, sample_project_doc: dict[str, Any]
) -> None:
    """When maybe_eager_index returns a dict, the job is completed with chunks."""
    job = sample_project_doc["job"]
    with patch(
        "app.worker.ingest_worker.doc_index.maybe_eager_index",
        return_value={"chunks": ["chunk1", "chunk2", "chunk3"]},
    ):
        await ingest_document(
            {}, sample_project_doc["project_id"], sample_project_doc["document_id"], str(job.id)
        )

    db.refresh(job)
    assert job.status == "completed"
    assert job.chunks == 3
    assert job.error is None


@pytest.mark.asyncio
async def test_ingest_document_completes_with_zero_chunks_when_indexing_disabled(
    db: Session, sample_project_doc: dict[str, Any]
) -> None:
    """When INDEX_ON_UPLOAD disables indexing, the job completes with chunks=0."""
    job = sample_project_doc["job"]
    with patch(
        "app.worker.ingest_worker.doc_index.maybe_eager_index",
        return_value=None,
    ):
        await ingest_document(
            {}, sample_project_doc["project_id"], sample_project_doc["document_id"], str(job.id)
        )

    db.refresh(job)
    assert job.status == "completed"
    assert job.chunks == 0
    assert job.error is None


@pytest.mark.asyncio
async def test_ingest_document_marks_job_failed_on_index_error(
    db: Session, sample_project_doc: dict[str, Any]
) -> None:
    """When maybe_eager_index raises, the job is failed and the exception propagates."""
    job = sample_project_doc["job"]
    with patch(
        "app.worker.ingest_worker.doc_index.maybe_eager_index",
        side_effect=RuntimeError("extraction failed"),
    ):
        with pytest.raises(RuntimeError, match="extraction failed"):
            await ingest_document(
                {}, sample_project_doc["project_id"], sample_project_doc["document_id"], str(job.id)
            )

    db.refresh(job)
    assert job.status == "failed"
    assert job.error is not None
    assert "extraction failed" in job.error
