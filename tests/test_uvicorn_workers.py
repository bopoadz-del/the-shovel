"""Smoke and concurrency tests for the multi-worker Redis build (P5).

These are integration-level tests. They run against the FastAPI app with the
synchronous TestClient, which drives the ASGI lifespan and therefore exercises
the same startup/shutdown hooks as production. Tests that require Redis are
skipped when REDIS_URL is not configured or Redis is unreachable. Tests that
require a chat backend are skipped when the chat block is not available.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.blocks import BLOCK_REGISTRY
from app.core.db import SessionLocal
from app.core.models import IngestionJob
from app.dependencies import require_api_key, require_user
from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Sync TestClient that drives the ASGI lifespan with auth overridden."""
    app.dependency_overrides[require_api_key] = lambda: {
        "user": "test@example.com",
        "tier": "pro",
        "valid": True,
        "user_id": "test-user",
        "role": "admin",
        "email": "test@example.com",
        "auth_method": "bearer",
    }
    app.dependency_overrides[require_user] = lambda: {
        "user_id": "test-user",
        "role": "admin",
        "email": "test@example.com",
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf() -> io.BytesIO:
    """Minimal valid PDF bytes for upload tests."""
    return io.BytesIO(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n105\n%%EOF\n"
    )


def _redis_configured() -> bool:
    return bool(os.getenv("REDIS_URL", "").strip())


def test_health_reports_redis_structure(client: TestClient):
    """/v1/health always contains a redis block; connected state reflects reality."""
    resp = client.get("/v1/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "redis" in data
    assert "connected" in data["redis"]
    assert "latency_ms" in data["redis"]


def test_health_reports_redis_connected_with_redis(client: TestClient):
    """When Redis is reachable, /v1/health reports connected True."""
    if not _redis_configured():
        pytest.skip("REDIS_URL not configured")

    resp = client.get("/v1/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["redis"]["connected"] is True


def test_concurrent_uploads_queue_without_502(client: TestClient, sample_pdf: io.BytesIO):
    """Many concurrent uploads return 200/202 and never 502."""
    files = {"file": ("doc.pdf", sample_pdf, "application/pdf")}

    def upload():
        return client.post(
            "/upload?project_id=p1",
            files=files,
            data={"project_id": "p1"},
        )

    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(lambda _: upload(), range(5)))

    assert all(r.status_code in (200, 202) for r in responses)


def test_chat_stream_not_starved_during_uploads(client: TestClient, sample_pdf: io.BytesIO):
    """Upload traffic does not prevent the chat stream endpoint from responding."""
    if "chat" not in BLOCK_REGISTRY:
        pytest.skip("chat block not registered")

    upload_files = {"file": ("doc.pdf", sample_pdf, "application/pdf")}

    with ThreadPoolExecutor(max_workers=3) as pool:
        upload_futures = [pool.submit(client.post, "/upload?project_id=p1", files=upload_files) for _ in range(3)]
        stream_resp = client.post("/chat/stream", json={"session_id": "s1", "message": "hello"})
        upload_results = [f.result() for f in upload_futures]

    assert stream_resp.status_code in (200, 202)
    assert all(r.status_code in (200, 202) for r in upload_results)


@pytest.fixture(autouse=True)
def _cleanup_ingestion_jobs():
    """Remove ingestion job rows created by upload tests."""
    yield
    try:
        with SessionLocal() as db:
            db.execute(delete(IngestionJob))
            db.commit()
    except Exception:
        pass
