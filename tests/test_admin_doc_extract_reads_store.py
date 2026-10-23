"""The admin doc-extract diagnostic reads chunk text from chunks_v2.

It was the one caller that read ``entry["chunks"]`` out of the doc_index
blob. Since 0018 no row carries chunk text, so the diagnostic must show what
retrieval sees: the vector store's rows for that document.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from app.core import file_crypto
from app.core import projects as projects_mod
from app.dependencies import require_api_key
from app.main import app
from tests.conftest import _postgres_test_mode

pytestmark = pytest.mark.skipif(
    _postgres_test_mode(),
    reason="SQLite-backed doc_index storage tests",
)


@pytest.fixture(autouse=True)
def _admin_override():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-admin", "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


def test_admin_diagnostic_reads_chunks_from_chunks_v2(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    from app.core import doc_index
    importlib.reload(doc_index)
    from app.core.rag import vector_store as _vs
    _vs.reset_store_cache()

    pid = projects_mod.create_project("Diag Project")["id"]
    body = ("Retaining wall drainage shall use free-draining granular fill. " * 30).encode()
    path = str(tmp_path / "wall.txt")
    file_crypto.write_document(path, body)
    doc = projects_mod.add_document(pid, "wall.txt", file_path=path, size=len(body))

    result = doc_index.index_document(pid, doc["id"])
    assert result["chunks"] and result.get("rag_indexed", 0) == len(result["chunks"]), result
    # Nothing in the index carries the text the diagnostic is about to show.
    assert "chunks" not in doc_index._load_index(pid)["documents"][0]

    with TestClient(app) as client:
        r = client.get(
            "/v1/admin/debug/doc-extract",
            params={"project_id": pid, "document_id": doc["id"]},
        )
    assert r.status_code == 200, r.text
    body_json = r.json()
    assert body_json["indexed_chunk_count"] == len(result["chunks"])
    previews = [p["snippet"] for p in body_json["indexed_chunks_preview"]]
    assert previews and all("granular fill" in s for s in previews)
