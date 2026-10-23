"""POST /v1/admin/knowledge/documents -- the admin path into general knowledge.

A reference work (a code, a standard, a contract-form guide) belongs in the
general-knowledge layer, not in anyone's project: the admin uploads it here, it
is stored in the configured general-knowledge project with admin provenance
(never ``user_upload``, which the layered RAG files under the uploader's own
session layer), and recorded as pending: the ingest task indexes it with the
platform's own pipeline and embedder, never the live web process.

Synthetic file names and text only.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.main import app

GK = "gk_admin_test"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    from app.core.projects import init_db
    init_db()
    indexed = []
    monkeypatch.setattr("app.core.doc_index.maybe_eager_index",
                        lambda pid, did: indexed.append((pid, did)))
    monkeypatch.setattr("app.core.doc_index.index_document",
                        lambda pid, did, *a, **k: indexed.append((pid, did)))
    with TestClient(app) as c:
        c.indexed = indexed
        yield c
    app.dependency_overrides.clear()


def _as(role):
    from app.dependencies import require_api_key
    app.dependency_overrides[require_api_key] = lambda: {"user_id": "admin-x", "role": role}


def _post(client, name, data, ctype="text/plain"):
    return client.post("/v1/admin/knowledge/documents", files={"file": (name, data, ctype)})


def test_admin_upload_lands_in_general_knowledge_with_admin_provenance(client):
    from app.core import projects as store
    _as("admin")
    body = b"Section 4.2 Synthetic Loads Handbook: wind pressure on walls is 1.2 kPa."
    r = _post(client, "synthetic_loads_handbook.txt", body)
    assert r.status_code == 201, r.text
    doc = r.json()["document"]
    assert doc["project_id"] == GK
    row = store.get_document(doc["id"])
    assert (row.get("metadata") or {}).get("provenance") == "admin_knowledge"
    # Stored and pending; extraction is the ingest task's job, not this request's.
    assert r.json()["status"] == "pending"
    assert (row.get("metadata") or {}).get("indexing", {}).get("status") == "pending"
    assert row["ingest_status"] == "UNVERIFIED"
    assert (GK, doc["id"]) not in client.indexed


def test_same_content_twice_is_one_document(client):
    from app.core import projects as store
    _as("admin")
    body = b"Synthetic Clause 9.9: the notice period is 14 days."
    first = _post(client, "synthetic_form_guide.txt", body).json()["document"]["id"]
    again = _post(client, "synthetic_form_guide.txt", body)
    assert again.status_code == 200
    assert again.json()["document"]["id"] == first
    assert sum(1 for d in store.list_documents(GK) if d["id"] == first) == 1


def test_non_admin_is_refused(client):
    _as("user")
    assert _post(client, "synthetic.txt", b"text").status_code == 403


def test_compressed_and_non_text_files_are_refused(client):
    _as("admin")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    assert _post(client, "bundle.zip", buf.getvalue(), "application/zip").status_code == 415
    assert _post(client, "photo.jpg", b"\xff\xd8\xff", "image/jpeg").status_code == 415
