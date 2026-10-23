"""Upload endpoint queue integration tests.

Covers the P3 migration of /upload to enqueue ingestion jobs via
``app.worker.ingest_queue`` with a BackgroundTasks fallback.
"""

import io
from typing import Any, Callable, List, Tuple

import pytest
from fastapi import BackgroundTasks, UploadFile
from sqlalchemy import delete, select

from app.core import doc_index
from app.core import projects as projects_store
from app.core.db import SessionLocal
from app.core.models import IngestionJob
from app.routers import upload as upload_module
from app.routers.upload import upload_v1


class BackgroundTasksSpy:
    """Minimal stand-in for FastAPI BackgroundTasks that records added tasks."""

    def __init__(self) -> None:
        self.tasks: List[Tuple[Callable, Tuple[Any, ...], dict]] = []

    def add_task(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self.tasks.append((func, args, kwargs))


def _make_file(name: str = "note.txt", content: bytes = b"hello upload") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content))


@pytest.fixture(autouse=True)
def _isolate_db() -> None:
    """Ensure the unified schema (including ingestion_jobs) exists for each test."""
    projects_store._initialized = False  # noqa: SLF001
    projects_store.init_db()
    yield
    with SessionLocal() as db:
        db.execute(delete(IngestionJob))
        db.commit()
    projects_store._initialized = False  # noqa: SLF001


@pytest.fixture
def project_pid() -> str:
    return "proj123"


@pytest.fixture
def uploaded_doc() -> dict:
    return {"id": "doc456"}


@pytest.fixture(autouse=True)
def _stub_projects(monkeypatch, project_pid: str, uploaded_doc: dict) -> None:
    """Provide a project and document so the project_id branch runs."""

    def fake_get_project(pid: str, user_id: str | None = None) -> dict | None:
        if pid == project_pid:
            return {"id": pid}
        return None

    def fake_add_document(**_kwargs: Any) -> dict:
        return uploaded_doc

    monkeypatch.setattr(projects_store, "get_project", fake_get_project)
    monkeypatch.setattr(projects_store, "add_document", fake_add_document)


@pytest.mark.asyncio
async def test_upload_enqueues_job_when_redis_configured(
    monkeypatch, project_pid: str, uploaded_doc: dict
) -> None:
    """When enqueue_ingest returns True, status becomes queued and a row is persisted."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    calls: List[Tuple[str, str, str]] = []

    async def fake_enqueue_ingest(pid: str, did: str, job_id: str) -> bool:
        calls.append((pid, did, job_id))
        return True

    monkeypatch.setattr(upload_module, "enqueue_ingest", fake_enqueue_ingest)

    spy = BackgroundTasksSpy()
    response = await upload_v1(
        file=_make_file(),
        project_id=project_pid,
        background_tasks=spy,
        auth={"user_id": "user1", "role": "admin"},
    )

    assert response["indexed"] is True
    assert response["indexing_status"] == "queued"
    assert response["document_id"] == uploaded_doc["id"]
    assert "stored_as" in response
    assert "file_path" not in response

    # Verify the ingestion job row was created.
    with SessionLocal() as db:
        job = db.execute(
            select(IngestionJob).where(
                IngestionJob.project_id == project_pid,
                IngestionJob.document_id == uploaded_doc["id"],
            )
        ).scalar_one_or_none()
    assert job is not None
    assert job.status == "pending"

    # enqueue_ingest must be called with the exact positional args.
    assert calls == [(project_pid, uploaded_doc["id"], str(job.id))]

    # Queue success means no BackgroundTasks fallback.
    assert spy.tasks == []


@pytest.mark.asyncio
async def test_upload_falls_back_to_background_task_when_enqueue_fails(
    monkeypatch, project_pid: str, uploaded_doc: dict
) -> None:
    """When REDIS_URL is unset, enqueue fails and maybe_eager_index is scheduled."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    scheduled: List[Tuple[str, str]] = []

    def capture_maybe_eager_index(pid: str, did: str) -> None:
        scheduled.append((pid, did))

    monkeypatch.setattr(doc_index, "maybe_eager_index", capture_maybe_eager_index)

    spy = BackgroundTasksSpy()
    response = await upload_v1(
        file=_make_file(),
        project_id=project_pid,
        background_tasks=spy,
        auth={"user_id": "user1", "role": "admin"},
    )

    assert response["indexed"] is True
    assert response["indexing_status"] == "scheduled"
    assert response["document_id"] == uploaded_doc["id"]

    # The fallback task was scheduled, not executed inline.
    assert len(spy.tasks) == 1
    func, args, kwargs = spy.tasks[0]
    assert func is doc_index.maybe_eager_index
    assert args == (project_pid, uploaded_doc["id"])
    assert kwargs == {}
    assert scheduled == []


@pytest.mark.asyncio
async def test_upload_falls_back_when_enqueue_returns_false(
    monkeypatch, project_pid: str, uploaded_doc: dict
) -> None:
    """When REDIS_URL is set but enqueue_ingest returns False, fall back to BackgroundTasks."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    async def fake_enqueue_ingest(_pid: str, _did: str, _job_id: str) -> bool:
        return False

    monkeypatch.setattr(upload_module, "enqueue_ingest", fake_enqueue_ingest)

    spy = BackgroundTasksSpy()
    response = await upload_v1(
        file=_make_file(),
        project_id=project_pid,
        background_tasks=spy,
        auth={"user_id": "user1", "role": "admin"},
    )

    assert response["indexed"] is True
    assert response["indexing_status"] == "scheduled"
    assert response["document_id"] == uploaded_doc["id"]

    assert len(spy.tasks) == 1
    func, args, kwargs = spy.tasks[0]
    assert func is doc_index.maybe_eager_index
    assert args == (project_pid, uploaded_doc["id"])
    assert kwargs == {}


@pytest.mark.asyncio
async def test_upload_without_project_id_preserves_existing_behavior() -> None:
    """The non-project branch is unchanged."""
    response = await upload_v1(
        file=_make_file(),
        project_id=None,
        background_tasks=None,
        auth={"user_id": "user1"},
    )

    assert response["indexed"] is False
    assert response["indexing_status"] == "no_project_id"
    assert "document_id" not in response


@pytest.mark.asyncio
async def test_upload_project_not_found_preserves_existing_behavior(
    monkeypatch, project_pid: str
) -> None:
    """A missing project keeps the old skip response."""
    monkeypatch.setattr(projects_store, "get_project", lambda pid, user_id=None: None)

    response = await upload_v1(
        file=_make_file(),
        project_id=project_pid,
        background_tasks=None,
        auth={"user_id": "user1"},
    )

    assert response["indexed"] is False
    assert response["indexing_status"] == "skipped_project_not_found"
