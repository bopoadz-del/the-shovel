"""Tests for the IngestionJob model and ingestion_jobs table."""

from __future__ import annotations

import uuid

from app.core.db import engine
from app.core.models import IngestionJob
from sqlalchemy.orm import Session


def test_ingestion_job_table_exists_and_can_be_queried():
    """The IngestionJob model maps to a table that can be created and queried."""
    IngestionJob.__table__.create(bind=engine, checkfirst=True)

    with Session(engine) as session:
        # Default id and status are applied by the model/server defaults.
        default_job = IngestionJob(project_id="proj_default", document_id="doc_default")
        session.add(default_job)
        session.commit()
        assert default_job.id is not None
        assert isinstance(default_job.id, uuid.UUID)
        assert default_job.status == "pending"
        assert default_job.created_at is not None
        assert default_job.updated_at is not None

        job_id = uuid.uuid4()
        job = IngestionJob(
            id=job_id,
            project_id="proj_123",
            document_id="doc_456",
            status="pending",
            chunks=None,
            error=None,
        )
        session.add(job)
        session.commit()

        result = session.get(IngestionJob, job_id)
        assert result is not None
        assert result.project_id == "proj_123"
        assert result.document_id == "doc_456"
        assert result.status == "pending"
        assert result.chunks is None
        assert result.error is None
        assert result.created_at is not None
        assert result.updated_at is not None

        # updated_at should refresh on a real column change.
        original_updated_at = result.updated_at
        result.status = "completed"
        session.commit()
        session.refresh(result)
        assert result.status == "completed"
        assert result.updated_at is not None
        assert result.updated_at >= original_updated_at

        # Clean up so the isolated test does not leak rows.
        session.delete(result)
        session.delete(default_job)
        session.commit()
