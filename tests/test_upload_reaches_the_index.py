"""An upload must reach the INDEX. Being queued is not being indexed.

Live-fire campaign, run 1 (2026-08-14): a document uploaded to the production
service returned HTTP 200 with ``indexed: true, indexing_status: "queued"`` and
then sat at ``chunk_count = 0`` forever. Asking about its contents produced
"I could not confirm this reference in the indexed project sources".

Mechanism: the arq ingest worker was removed from render.yaml on 2026-08-08 to
halve the compute bill, on the stated assumption that inline ingestion in the
web request would take over. It never did. ``upload_v1`` only falls back to
inline ``maybe_eager_index`` when ``enqueue_ingest()`` returns False, and that
only happened when Redis was unreachable — but Redis was deliberately KEPT for
sessions and rate limiting. So the enqueue always succeeded, the fallback never
ran, and nothing drained the queue.

The defect in one line: **"Redis is configured" was used as a proxy for "a
worker exists".** These are different facts, and the fence below keeps them
different. The existing tests in tests/test_ingest_queue.py check what
``enqueue_ingest`` RETURNS; these check where an uploaded document ENDS UP,
which is the property that actually broke.

Each test patches ``create_pool`` so Redis appears REACHABLE. That is what makes
the first test red on the old code: with a working queue the old branch takes
the enqueue path and schedules no inline work. A test that let Redis fail would
pass on the broken code and prove nothing.
"""

import io
from typing import Any, Callable, List, Tuple
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile
from sqlalchemy import delete

from app.core import doc_index
from app.core import projects as projects_store
from app.core.db import SessionLocal
from app.core.models import IngestionJob
from app.routers.upload import upload_v1
from app.worker import ingest_queue
from app.worker.ingest_queue import enqueue_ingest


class BackgroundTasksSpy:
    def __init__(self) -> None:
        self.tasks: List[Tuple[Callable, Tuple[Any, ...], dict]] = []

    def add_task(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        self.tasks.append((func, args, kwargs))


PID = "proj-campaign"
DOC = {"id": "doc-campaign"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch) -> None:
    projects_store._initialized = False  # noqa: SLF001
    projects_store.init_db()
    ingest_queue._pool = None  # noqa: SLF001
    # Stub the project store exactly as tests/routers/test_upload.py does: a
    # real Project row needs a real User row (FK), and the identity of the
    # project is irrelevant to the property under test. What is NOT stubbed is
    # ``enqueue_ingest`` — its real decision is the thing being fenced.
    monkeypatch.setattr(
        projects_store, "get_project",
        lambda pid, user_id=None: {"id": pid} if pid == PID else None,
    )
    monkeypatch.setattr(projects_store, "add_document", lambda **_kw: DOC)
    yield
    with SessionLocal() as db:
        db.execute(delete(IngestionJob))
        db.commit()
    projects_store._initialized = False  # noqa: SLF001
    ingest_queue._pool = None  # noqa: SLF001


def _reachable_redis():
    """Patch a Redis that ACCEPTS jobs — the condition that hid the bug."""
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=None)
    return patch(
        "app.worker.ingest_queue.create_pool",
        new_callable=AsyncMock,
        return_value=pool,
    ), pool


def _eager_tasks(spy: BackgroundTasksSpy):
    return [t for t in spy.tasks if t[0] is doc_index.maybe_eager_index]


@pytest.mark.asyncio
async def test_upload_indexes_inline_when_no_worker_is_declared(monkeypatch):
    """RED on the old code: a reachable Redis meant nothing indexed the file."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.delenv("INGEST_WORKER_ENABLED", raising=False)
    spy = BackgroundTasksSpy()

    ctx, pool = _reachable_redis()
    with ctx:
        resp = await upload_v1(
            UploadFile(filename="spec.txt", file=io.BytesIO(b"torque 337 N.m")),
            PID, spy, {"user_id": "u1", "role": "admin"},
        )

    # The document must be handed to something that will actually index it.
    assert _eager_tasks(spy), (
        "no worker is declared, so the upload MUST index inline; instead the "
        f"document was left at indexing_status={resp.get('indexing_status')!r} "
        "with nothing to drain the queue"
    )
    # And it must not have been dropped on a queue nobody serves.
    pool.enqueue_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_uses_the_queue_when_a_worker_is_declared(monkeypatch):
    """The other direction: declaring a worker must still use the queue.

    Without this, a fix could simply always-inline and silently break a real
    worker deployment while the first test stayed green.
    """
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("INGEST_WORKER_ENABLED", "true")
    spy = BackgroundTasksSpy()

    ctx, pool = _reachable_redis()
    with ctx:
        resp = await upload_v1(
            UploadFile(filename="spec.txt", file=io.BytesIO(b"torque 337 N.m")),
            PID, spy, {"user_id": "u1", "role": "admin"},
        )

    pool.enqueue_job.assert_awaited_once()
    assert resp.get("indexing_status") == "queued"
    assert not _eager_tasks(spy), "must not double-index when a worker will drain it"


@pytest.mark.asyncio
async def test_reachable_redis_alone_never_implies_a_worker(monkeypatch):
    """The unit-level statement of the same fact, at the seam that got it wrong."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.delenv("INGEST_WORKER_ENABLED", raising=False)
    ctx, pool = _reachable_redis()
    with ctx:
        assert await enqueue_ingest("p1", "d1", "j1") is False
    pool.enqueue_job.assert_not_awaited()
