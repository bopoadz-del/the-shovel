"""Recorded document sizes must describe the bytes that actually exist.

THE DEFECT THIS PINS
--------------------
Every document in the pilot Master Corpus rendered as "0 B" in the UI. The
cause was not the UI: ``add_document(size: int = 0)`` let any caller register a
document whose recorded size was a fabrication, and the bulk-insert migration
endpoint did exactly that for the whole corpus (the source
``rag_backfill_*.json`` carry a real ``size`` per file — the field was dropped
in the payload transform).

Why 400 test files did not catch it: the upload tests called ``upload_v1()`` as
a plain Python function with the document store monkeypatched to return
``{"id": "doc456"}``. Nothing wrote a byte to disk, nothing read a listing
back, and NO test anywhere asserted the invariant that makes a corpus
trustworthy — that a listed size equals the bytes on disk.

These tests cross that boundary: real multipart HTTP in, real file on disk,
real listing out.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest
from fastapi.testclient import TestClient

from app.core import file_crypto
from app.core import projects as store
from app.dependencies import require_api_key
from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client, name="Size Integrity"):
    r = client.post("/v1/projects", json={"name": name}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


# ── the end-to-end invariant ────────────────────────────────────────────────

def test_uploaded_document_reports_its_real_byte_count_end_to_end(client):
    """Upload real bytes over HTTP; the listing must report exactly that many.

    This is the assertion whose absence let the whole corpus read "0 B".
    """
    proj = _new_project(client, "Byte Count")
    payload = b"%PDF-1.4 " + b"x" * 4_321
    r = client.post(
        f"/v1/projects/{proj['id']}/documents",
        files={"file": ("spec.pdf", payload, "application/pdf")},
        headers=H,
    )
    assert r.status_code == 201, r.text

    listing = client.get(f"/v1/projects/{proj['id']}/documents", headers=H)
    assert listing.status_code == 200, listing.text
    docs = listing.json()
    docs = docs.get("documents", docs) if isinstance(docs, dict) else docs
    uploaded = [d for d in docs if d.get("original_name") == "spec.pdf"]
    assert uploaded, f"uploaded document missing from listing: {docs}"

    assert uploaded[0]["size"] == len(payload), (
        f"listing reports {uploaded[0]['size']} bytes for a {len(payload)}-byte "
        f"upload — this is the '0 B corpus' defect"
    )


def test_listed_size_matches_the_bytes_on_disk(client):
    """The recorded size must agree with the stored file, not just with itself."""
    proj = _new_project(client, "Disk Agreement")
    payload = b"%PDF-1.4 " + b"y" * 2_048
    r = client.post(
        f"/v1/projects/{proj['id']}/documents",
        files={"file": ("drawing.pdf", payload, "application/pdf")},
        headers=H,
    )
    assert r.status_code == 201, r.text
    doc_id = r.json()["document"]["id"]

    doc = store.get_document(doc_id)
    assert doc is not None
    on_disk = file_crypto.plaintext_size(doc["file_path"])
    assert doc["size"] == on_disk == len(payload)


# ── the store-level guard ───────────────────────────────────────────────────

def test_add_document_measures_size_from_disk_when_caller_omits_it(tmp_path):
    """A caller that forgets `size` must NOT silently register a 0-byte lie."""
    path = tmp_path / "measured.txt"
    body = b"z" * 777
    file_crypto.write_document(str(path), body)

    resolved = store.resolve_document_size(0, str(path))
    assert resolved == len(body), (
        "size was not recovered from disk — a forgetful caller still registers 0 B"
    )


def test_add_document_keeps_an_explicit_size_over_the_measured_one(tmp_path):
    """An explicit plaintext size wins: it is correct even when the file on
    disk is encrypted (and therefore larger)."""
    path = tmp_path / "explicit.txt"
    file_crypto.write_document(str(path), b"abc")
    assert store.resolve_document_size(999, str(path)) == 999


def test_metadata_only_row_is_allowed_but_reports_zero(tmp_path):
    """A row referencing no file legitimately has no size — it must not crash,
    and must not invent one."""
    assert store.resolve_document_size(0, None) == 0
    assert store.resolve_document_size(0, str(tmp_path / "does-not-exist")) == 0


def test_encrypted_files_report_plaintext_size_not_ciphertext(tmp_path, monkeypatch):
    """Encryption at rest inflates the file ~33%. Recording the ciphertext
    length would make every size wrong the moment encryption is enabled."""
    from cryptography.fernet import Fernet

    monkeypatch.setenv("DATA_ENCRYPTION_KEY", Fernet.generate_key().decode())
    path = tmp_path / "secret.txt"
    body = b"q" * 5_000
    file_crypto.write_document(str(path), body)

    assert os.path.getsize(str(path)) > len(body), "test premise: ciphertext is larger"
    assert file_crypto.plaintext_size(str(path)) == len(body)


# ── audit + repair for corpora already damaged ──────────────────────────────

def test_audit_and_repair_fix_an_already_damaged_corpus(client, tmp_path):
    """The fix must also recover the rows the old code already wrote."""
    proj = _new_project(client, "Repair Me")
    payload = b"%PDF-1.4 " + b"w" * 1_500
    r = client.post(
        f"/v1/projects/{proj['id']}/documents",
        files={"file": ("damaged.pdf", payload, "application/pdf")},
        headers=H,
    )
    assert r.status_code == 201, r.text
    doc_id = r.json()["document"]["id"]

    # Recreate the damage the pre-fix code left behind: a real file on disk,
    # a recorded size of 0.
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        session.get(Document, doc_id).size = 0
        session.commit()

    audit = store.audit_document_sizes(proj["id"])
    assert any(e["id"] == doc_id for e in audit["zero_with_file"]), (
        f"audit failed to flag a 0-byte row that has a real file: {audit}"
    )

    dry = store.repair_document_sizes(proj["id"], dry_run=True)
    assert dry["repaired"] == 0 and dry["repairable"] >= 1
    assert store.get_document(doc_id)["size"] == 0, "dry run must not write"

    result = store.repair_document_sizes(proj["id"])
    assert result["repaired"] >= 1
    assert store.get_document(doc_id)["size"] == len(payload)

    after = store.audit_document_sizes(proj["id"])
    assert not any(e["id"] == doc_id for e in after["zero_with_file"])


# ── the migration endpoint that actually produced the 0 B corpus ────────────

@pytest.fixture
def admin_client():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "size-admin", "role": "admin",
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _bulk_payload(size: int):
    return {
        "projects": [{"id": "sz_corpus", "name": "Size Corpus"}],
        "documents": [{
            "id": "sz_doc_1", "project_id": "sz_corpus",
            "original_name": "prc-601_tendering.pdf", "size": size,
        }],
        "chunks": [],
    }


def test_bulk_insert_refuses_sizeless_documents(admin_client):
    """THE defect reproduction. The pilot corpus was loaded through this
    endpoint with `size` dropped in the payload transform, so all ~227
    documents registered as 0 bytes and rendered as '0 B' forever."""
    r = admin_client.post("/v1/admin/corpus/bulk-insert", json=_bulk_payload(0))

    assert r.status_code == 400, r.text
    body = r.text.lower()
    assert "size" in body and "sz_doc_1" in r.text
    assert store.get_document("sz_doc_1") is None, (
        "the rejected payload still wrote a row — rejection must be whole"
    )


def test_bulk_insert_accepts_documents_carrying_their_size(admin_client):
    r = admin_client.post("/v1/admin/corpus/bulk-insert", json=_bulk_payload(345_970_034))
    assert r.status_code == 200, r.text
    assert r.json()["counts"]["documents"] == 1
    assert store.get_document("sz_doc_1")["size"] == 345_970_034


def test_metadata_only_rows_require_an_explicit_opt_in(admin_client):
    """Registering a document with no size stays POSSIBLE — an operator may
    genuinely index a corpus whose files live elsewhere — but it must be a
    decision, not a default."""
    payload = _bulk_payload(0)
    payload["allow_missing_size"] = True

    r = admin_client.post("/v1/admin/corpus/bulk-insert", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["documents_without_size"] == 1
