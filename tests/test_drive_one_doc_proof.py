"""Admin Drive proof endpoints (Task T4) — mocked Drive + store.

These tests prove the shape of /v1/admin/drive/download-proof and
/v1/admin/drive/ingest-proof without needing live Google credentials.
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.dependencies import require_api_key
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _admin_auth():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "system",
        "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated DATA_DIR + fresh projects DB for each test."""
    from app.core import projects as projects_mod

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    return tmp_path


def _monkeypatch_download(monkeypatch, content: bytes):
    """Patch gdrive_service.download_file_bytes to return ``content``."""
    import app.core.gdrive_service as gdrive_service

    monkeypatch.setattr(
        gdrive_service, "download_file_bytes", lambda _fid: (content, None)
    )


def test_download_proof_ok(client, fresh_db, monkeypatch):
    from app.core import projects as projects_mod

    proj = projects_mod.create_project("Drive Proof")
    payload = b"hello drive proof"
    _monkeypatch_download(monkeypatch, payload)

    resp = client.post(
        "/v1/admin/drive/download-proof",
        json={"project_id": proj["id"], "file_id": "FILE123"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["file_id"] == "FILE123"
    assert body["bytes"] == len(payload)
    assert body["sha256"] == hashlib.sha256(payload).hexdigest()


def test_ingest_proof_ok(client, fresh_db, monkeypatch):
    from app.core import doc_index, projects as projects_mod

    proj = projects_mod.create_project("Drive Ingest")
    payload = b"ingest me"
    _monkeypatch_download(monkeypatch, payload)

    def fake_index_document(pid, did, chunker="default", **_k):
        return {"status": "ok", "project_id": pid, "total_chunks": 3}

    monkeypatch.setattr(doc_index, "index_document", fake_index_document)

    resp = client.post(
        "/v1/admin/drive/ingest-proof",
        json={"project_id": proj["id"], "file_id": "INGEST99"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["file_id"] == "INGEST99"
    assert body["chunk_count"] == 3
    assert "document_id" in body


def test_ingest_proof_zero_chunk(client, fresh_db, monkeypatch):
    from app.core import doc_index, projects as projects_mod

    proj = projects_mod.create_project("Drive Zero")
    payload = b"unparseable"
    _monkeypatch_download(monkeypatch, payload)

    def fake_index_document(pid, did, chunker="default", **_k):
        return {
            "status": "error",
            "error": "ZERO_CHUNK",
            "banner": "ERROR: document produced 0 chunks",
            "project_id": pid,
            "document_id": did,
        }

    monkeypatch.setattr(doc_index, "index_document", fake_index_document)

    resp = client.post(
        "/v1/admin/drive/ingest-proof",
        json={"project_id": proj["id"], "file_id": "ZERO99"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "ZERO_CHUNK"
    assert "0 chunks" in body["banner"]
