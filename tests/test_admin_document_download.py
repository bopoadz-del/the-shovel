"""PR #156 - admin document-download endpoint contract guards.

Verifies GET /v1/admin/debug/document-download:
  * gates on admin role (403 for non-admin)
  * 404 for unknown document id
  * 400 when the document belongs to a different project
  * 404 when the stored file is missing from disk
  * 422 when the stored file cannot be decrypted/read
  * 200 happy path: returns the original plaintext bytes with an
    attachment Content-Disposition and a guessed media type
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import file_crypto
from app.core.db import SessionLocal
from app.core.models import Document, Project

PROJECT_ID = "dl_proof_project"
OTHER_PROJECT_ID = "dl_other_project"


def _ensure_schema():
    from app.core.projects import init_db as init_projects_db
    init_projects_db()


@pytest.fixture(autouse=True)
def _stub_admin():
    from app.dependencies import require_api_key as real_require_api_key

    def fake_admin():
        return {"user_id": "test-admin", "role": "admin"}

    def fake_user():
        return {"user_id": "test-user", "role": "user"}

    yield {"admin": fake_admin, "user": fake_user, "real": real_require_api_key}
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _override(identity):
    from app.dependencies import require_api_key
    app.dependency_overrides[require_api_key] = identity


def _seed_document(file_path, original_name="report.pdf", project_id=PROJECT_ID):
    """Insert a project + document row pointing at file_path; returns doc id."""
    _ensure_schema()
    doc_id = str(uuid.uuid4())[:8]
    with SessionLocal() as session:
        for pid in (PROJECT_ID, OTHER_PROJECT_ID):
            if session.get(Project, pid) is None:
                session.add(Project(
                    id=pid, name=pid, user_id="system",
                    created_at="2026-07-09T00:00:00Z",
                    status="active",
                ))
        session.flush()
        session.add(Document(
            id=doc_id, project_id=project_id,
            original_name=original_name,
            file_path=file_path,
            doc_type="document", doc_role="other",
            size=0, uploaded_at="2026-07-09T00:00:00Z",
        ))
        session.commit()
    return doc_id


def test_non_admin_gets_403(client, _stub_admin):
    _override(_stub_admin["user"])
    resp = client.get(
        "/v1/admin/debug/document-download",
        params={"project_id": PROJECT_ID, "document_id": "whatever"},
    )
    assert resp.status_code == 403


def test_unknown_document_404(client, _stub_admin):
    _override(_stub_admin["admin"])
    resp = client.get(
        "/v1/admin/debug/document-download",
        params={"project_id": PROJECT_ID, "document_id": "no-such-doc"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Document not found"


def test_wrong_project_400(client, _stub_admin, tmp_path):
    payload = b"wrong project bytes"
    path = str(tmp_path / "doc.bin")
    file_crypto.write_document(path, payload)
    doc_id = _seed_document(path, project_id=OTHER_PROJECT_ID)

    _override(_stub_admin["admin"])
    resp = client.get(
        "/v1/admin/debug/document-download",
        params={"project_id": PROJECT_ID, "document_id": doc_id},
    )
    assert resp.status_code == 400
    assert "does not belong" in resp.json()["detail"]


def test_missing_file_404(client, _stub_admin, tmp_path):
    path = str(tmp_path / "gone.bin")  # never written
    doc_id = _seed_document(path)

    _override(_stub_admin["admin"])
    resp = client.get(
        "/v1/admin/debug/document-download",
        params={"project_id": PROJECT_ID, "document_id": doc_id},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Document file is not available"


def test_unreadable_file_422(client, _stub_admin, tmp_path, monkeypatch):
    path = str(tmp_path / "corrupt.bin")
    file_crypto.write_document(path, b"data")
    doc_id = _seed_document(path)

    def boom(_path):
        raise ValueError("decryption key mismatch")

    monkeypatch.setattr(file_crypto, "read_document", boom)

    _override(_stub_admin["admin"])
    resp = client.get(
        "/v1/admin/debug/document-download",
        params={"project_id": PROJECT_ID, "document_id": doc_id},
    )
    assert resp.status_code == 422
    assert "Could not read document" in resp.json()["detail"]


def test_download_happy_path(client, _stub_admin, tmp_path):
    payload = b"%PDF-1.4 fake pdf payload for the download proof"
    path = str(tmp_path / "stored.bin")
    file_crypto.write_document(path, payload)
    doc_id = _seed_document(path, original_name="site report.pdf")

    _override(_stub_admin["admin"])
    resp = client.get(
        "/v1/admin/debug/document-download",
        params={"project_id": PROJECT_ID, "document_id": doc_id},
    )
    assert resp.status_code == 200
    assert resp.content == payload
    assert resp.headers["content-type"].startswith("application/pdf")
    assert 'attachment; filename="site report.pdf"' in resp.headers["content-disposition"]


def test_unknown_extension_falls_back_to_octet_stream(client, _stub_admin, tmp_path):
    payload = b"raw bytes with no known extension"
    path = str(tmp_path / "stored2.bin")
    file_crypto.write_document(path, payload)
    doc_id = _seed_document(path, original_name="blob.zzz_unknown")

    _override(_stub_admin["admin"])
    resp = client.get(
        "/v1/admin/debug/document-download",
        params={"project_id": PROJECT_ID, "document_id": doc_id},
    )
    assert resp.status_code == 200
    assert resp.content == payload
    assert resp.headers["content-type"].startswith("application/octet-stream")
