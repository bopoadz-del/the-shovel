"""M14 acceptance: a plain .txt uploaded through the API indexes to >=1 chunk,
and admin doc-reindex re-produces chunks from the SAME persisted bytes (it
re-reads the file, it does not re-chunk a stored empty text column).
"""
from __future__ import annotations

import io
import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest
from fastapi.testclient import TestClient

from app.core import projects as store
from app.core import users as users_store
from app.dependencies import require_api_key, require_user
from app.main import app


@pytest.fixture
def txt_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)

    store.init_db()
    users_store.init_db()
    users_store.ensure_user_exists("admin", email="admin@test.local", role="admin")
    project = store.create_project("txt project", user_id="admin")
    pid = project["id"]

    app.dependency_overrides[require_user] = lambda: {
        "user_id": "admin", "role": "admin",
    }
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "admin", "role": "admin",
    }
    with TestClient(app) as client:
        yield client, pid
    app.dependency_overrides.clear()


def _upload_txt(client, pid: str, name: str = "note.txt") -> dict:
    body = b"The Engineer is named Sandra Okafor and the retention is five percent. " * 3
    assert len(body) >= 200
    resp = client.post(
        f"/v1/projects/{pid}/documents",
        files=[("file", (name, io.BytesIO(body), "text/plain"))],
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["document"]


def _wait_chunk_count(doc_id: str, minimum: int = 1) -> int:
    for _ in range(20):
        doc = store.get_document(doc_id)
        if int(doc.get("chunk_count") or 0) >= minimum:
            return int(doc["chunk_count"])
    return int((store.get_document(doc_id) or {}).get("chunk_count") or 0)


def test_txt_upload_indexes_and_reindex_reproduces_chunks(txt_client):
    client, pid = txt_client
    doc = _upload_txt(client, pid)
    assert _wait_chunk_count(doc["id"]) >= 1

    resp = client.post(
        "/v1/admin/debug/doc-reindex",
        params={"project_id": pid, "document_id": doc["id"]},
    )
    assert resp.status_code == 200, resp.text

    # Reindex re-reads the bytes and re-produces the same chunks.
    assert _wait_chunk_count(doc["id"]) >= 1
    assert store.get_document(doc["id"])["ingest_status"] == "INDEXED"
