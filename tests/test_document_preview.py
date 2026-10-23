"""Tests for the inline document-preview endpoint.

Covers the render-friendly JSON shapes returned by
``GET /v1/projects/{pid}/documents/{did}/preview`` — an xlsx workbook renders
as a table, an unpreviewable-but-allowed extension reports 'unsupported', and a
malformed spreadsheet returns 422 (never 500).
"""

import io
import json
import os

import openpyxl
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import PW_10

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client, name="Preview Project"):
    r = client.post("/v1/projects", json={"name": name}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def _upload(client, pid, filename, content, content_type):
    files = {"file": (filename, content, content_type)}
    r = client.post(f"/v1/projects/{pid}/documents", files=files, headers=H)
    assert r.status_code == 201, r.text
    return r.json()["document"]


def _xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "BOQ"
    ws.append(["Item", "Qty", "Rate"])
    ws.append(["Concrete", 100, 350])
    ws.append(["Rebar", 50, 4200])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xlsx_previews_as_table(client):
    proj = _new_project(client)
    doc = _upload(
        client, proj["id"], "boq.xlsx", _xlsx_bytes(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "table"
    assert len(body["sheets"]) == 1
    sheet = body["sheets"][0]
    assert sheet["name"] == "BOQ"
    assert sheet["rows"][0] == ["Item", "Qty", "Rate"]
    assert sheet["rows"][1] == ["Concrete", "100", "350"]


def test_unsupported_extension_reports_unsupported(client):
    proj = _new_project(client)
    # .ifc is accepted by the upload route but has no preview renderer.
    doc = _upload(
        client, proj["id"], "model.ifc", b"ISO-10303-21;\nHEADER;",
        "application/octet-stream",
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "unsupported"
    assert body["ext"] == ".ifc"


def test_malformed_spreadsheet_returns_422(client):
    proj = _new_project(client)
    # A .xlsx extension over non-workbook bytes must 422, not 500.
    doc = _upload(
        client, proj["id"], "broken.xlsx", b"not a real spreadsheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H
    )
    assert r.status_code == 422, r.text


def test_txt_previews_as_text(client):
    proj = _new_project(client)
    doc = _upload(
        client, proj["id"], "notes.txt", b"Line one\nLine two", "text/plain",
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "text"
    assert "Line one" in body["text"]


def test_preview_missing_document_404(client):
    proj = _new_project(client)
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/nope1234/preview", headers=H
    )
    assert r.status_code == 404


def test_preview_cited_corpus_doc_from_workspace(client, monkeypatch):
    """Opening a Sources cite fetches preview via the *workspace* project
    id plus the cited doc id. That doc often lives on Master Corpus / GK,
    not on the workspace — the fetch must still succeed."""
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    doc = _upload(
        client, corpus["id"], "cited_spec.txt",
        b"Clause 8.7 Delay Damages are 0.1 percent per day.",
        "text/plain",
    )
    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "text"
    assert "Delay Damages" in body["text"]


def test_preview_foreign_private_doc_still_404(client):
    """A document the workspace cannot cite stays 404 — no cross-project leak."""
    workspace = _new_project(client, "Preview Workspace")
    other = _new_project(client, "Private Other")
    doc = _upload(
        client, other["id"], "secret.txt", b"not a cited source", "text/plain",
    )
    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 404


def test_preview_drive_backed_corpus_doc_missing_local_file(
    client, monkeypatch, tmp_path,
):
    """Citeable Master Corpus / GK docs keep no local copy (P1B deletes its
    transient copy after indexing). Preview must follow the row's
    ``drive_file_id``, not 404 on a stale ``file_path`` / size 0."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
    doc = store.add_document(
        project_id=corpus["id"],
        original_name="AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf",
        file_path=str(tmp_path / "gone-after-index.pdf"),
        size=0,
        metadata={"drive_file_id": "drive-deadbeef"},
    )
    assert doc["size"] == 0
    assert not os.path.exists(doc["file_path"] or "")

    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    drive_calls: list[str] = []

    def _fake_download(fid: str):
        drive_calls.append(fid)
        return (pdf, None) if fid == "drive-deadbeef" else (None, "nope")

    monkeypatch.setattr("app.core.gdrive_service.download_file_bytes", _fake_download)

    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "pdf"
    assert drive_calls == ["drive-deadbeef"]

    raw = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview/raw",
        headers=H,
    )
    assert raw.status_code == 200, raw.text
    assert raw.content == pdf

    refreshed = store.get_document(doc["id"])
    assert refreshed is not None
    assert refreshed["size"] == len(pdf)
    assert r.json()["has_file"] is True
    assert r.json()["size"] == len(pdf)


def test_preview_foreign_private_doc_does_not_fetch_drive(
    client, monkeypatch, tmp_path,
):
    """Ownership fails closed before any remote hydrate — no blob leak."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    other = _new_project(client, "Private Other")
    doc = store.add_document(
        project_id=other["id"],
        original_name="secret.pdf",
        file_path=str(tmp_path / "secret-missing.pdf"),
        size=0,
        metadata={"drive_file_id": "drive-secret"},
    )
    called: list[str] = []
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: called.append(fid) or (b"%PDF-1.4 secret", None),
    )
    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 404
    assert called == []


def test_preview_zero_byte_file_clear_404(client, tmp_path):
    """A real 0-byte file must not 500 — surface an empty-file state."""
    from app.core import projects as store

    proj = _new_project(client)
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    doc = store.add_document(
        project_id=proj["id"],
        original_name="empty.txt",
        file_path=str(empty),
        size=0,
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H
    )
    assert r.status_code == 404
    assert r.status_code != 500
    detail = r.json()["detail"]
    assert "empty" in detail.lower() or "0 bytes" in detail.lower()


def test_preview_missing_blob_clear_404(client, tmp_path):
    """No local file and no Drive pointer — honest unavailable, not 500."""
    from app.core import projects as store

    proj = _new_project(client)
    doc = store.add_document(
        project_id=proj["id"],
        original_name="vanished.txt",
        file_path=str(tmp_path / "never-written.txt"),
        size=0,
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H
    )
    assert r.status_code == 404
    assert r.status_code != 500
    assert "not available" in r.json()["detail"].lower()


def test_preview_drive_source_when_local_copy_gone(
    client, monkeypatch, tmp_path,
):
    """P1B deletes its transient copy after indexing — the Drive id is the
    source of the original."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    doc = store.add_document(
        project_id=corpus["id"],
        original_name="cited_from_drive.txt",
        file_path=str(tmp_path / "deleted-after-index.txt"),
        size=0,
        metadata={"drive_file_id": "drive-file-123"},
    )
    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (
            (b"Clause from Drive original", None)
            if fid == "drive-file-123"
            else (None, "nope")
        ),
    )
    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "text"
    assert "Drive original" in r.json()["text"]


def _p1b_meta(**extra):
    """Exact keys ``scripts/p1b_ingest_drive_server.py`` writes to metadata."""
    meta = {
        "drive_file_id": "1ExampleDriveFileId001xxxxxxxxxxx",
        "drive_path": (
            "Master Folder/the client project/Contract Docs/Contractor/"
            "Contract docs SIGNED/AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf"
        ),
        "source": "p1b_server_drive_reingestion",
        "ingestion_run_id": "run-test",
        "mimeType": "application/pdf",
        "content_sha256": "abc123",
    }
    meta.update(extra)
    return meta


def test_extract_p1b_keys_including_string_and_aliases():
    """Live rows carry the Drive id as a dict key, inside JSONB-as-string,
    or under the camelCase alias — preview must read all three. The Drive id
    is the only pointer: the platform keeps no archive of originals."""
    from app.core.projects import extract_document_source_pointers

    p1b = extract_document_source_pointers({"metadata": _p1b_meta()})
    assert p1b == {"drive_file_id": "1ExampleDriveFileId001xxxxxxxxxxx"}

    as_string = extract_document_source_pointers(
        {"metadata": json.dumps(_p1b_meta())},
    )
    assert as_string == p1b

    camel = extract_document_source_pointers(
        {"metadata": {"driveFileId": "drive-camel"}},
    )
    assert camel == {"drive_file_id": "drive-camel"}

    legacy = extract_document_source_pointers(
        {"metadata": {"r2_object_key": "proj/k.pdf", "r2_bucket": "b"}},
    )
    assert legacy == {"drive_file_id": ""}


def test_preview_p1b_string_metadata_drive_hydrate(
    client, monkeypatch, tmp_path,
):
    """JSONB-as-string metadata (the audit script already special-cases this)
    must still find drive_file_id and return 200."""
    from app.core import projects as store
    from app.core.db import SessionLocal
    from app.core.models import Document

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
    doc = store.add_document(
        project_id=corpus["id"],
        original_name="AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf",
        file_path=str(tmp_path / "gone-after-index.pdf"),
        size=0,
        metadata=_p1b_meta(),
    )
    with SessionLocal() as session:
        row = session.get(Document, doc["id"])
        assert row is not None
        row.metadata_ = json.dumps(_p1b_meta())
        session.commit()

    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (pdf, None) if fid == _p1b_meta()["drive_file_id"] else (None, "nope"),
    )

    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "pdf"
    assert body["has_file"] is True
    assert body["size"] == len(pdf)

    listing = client.get(
        f"/v1/projects/{corpus['id']}/documents", headers=H,
    )
    assert listing.status_code == 200, listing.text
    listed = next(d for d in listing.json()["documents"] if d["id"] == doc["id"])
    assert listed["has_file"] is True
    assert listed["has_remote_source"] is True
    assert listed["size"] == len(pdf)


def test_preview_rag_render_row_opens_from_its_drive_id(
    client, monkeypatch, tmp_path,
):
    """rag_render bulk ingest stores drive_file_id + size=0 and a stale
    Windows path. The Drive id is used directly — one download, that id."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    drive_id = "1ExampleDriveFileId001xxxxxxxxxxx"
    name = "AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf"
    pdf = b"%PDF-1.4 from the Drive id\n"
    doc = store.add_document(
        project_id=corpus["id"],
        original_name=name,
        file_path=str(tmp_path / "stale-windows-path.pdf"),
        size=0,
        metadata={
            "drive_file_id": drive_id,
            "source_path": "X:\\\\Example Drive\\\\Master Folder\\\\" + name,
            "source": "rag_backfill_client_clean_all",
        },
    )
    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    seen: list[str] = []

    def _fake_download(fid: str):
        seen.append(fid)
        return (pdf, None) if fid == drive_id else (None, "nope")

    monkeypatch.setattr("app.core.gdrive_service.download_file_bytes", _fake_download)

    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert seen == [drive_id]
    assert r.json()["kind"] == "pdf"


def test_preview_drive_not_configured_clear_404(
    client, monkeypatch, tmp_path,
):
    """Drive credentials missing must not look like a missing document."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    proj = _new_project(client)
    doc = store.add_document(
        project_id=proj["id"],
        original_name="AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf",
        file_path=str(tmp_path / "deleted-local.pdf"),
        size=0,
        metadata=_p1b_meta(),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (None, "service account unavailable"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.get_file_metadata",
        lambda fid: (None, "service account unavailable"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_public_file_bytes",
        lambda fid: (None, "public Drive download is not world-readable"),
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H,
    )
    assert r.status_code == 404
    assert r.status_code != 500
    detail = r.json()["detail"].lower()
    assert "not available" in detail
    assert "service account unavailable" in detail


def test_preview_drive_fetch_failed_clear_404(
    client, monkeypatch, tmp_path,
):
    """Drive id present, download failed — say so, do not hide behind generic missing."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    proj = _new_project(client)
    doc = store.add_document(
        project_id=proj["id"],
        original_name="cited.pdf",
        file_path=str(tmp_path / "gone.pdf"),
        size=0,
        metadata=_p1b_meta(),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (None, "Drive download returned 404"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.get_file_metadata",
        lambda fid: (None, "Drive files.get returned 404"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_public_file_bytes",
        lambda fid: (None, "public Drive download returned 404"),
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H,
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "not available" in detail.lower()
    assert "returned 404" in detail


def test_list_documents_has_file_for_remote_pointer(client, tmp_path):
    """Nav must not treat size=0 + Drive pointer as 'no file' forever."""
    from app.core import projects as store

    proj = _new_project(client)
    doc = store.add_document(
        project_id=proj["id"],
        original_name="AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf",
        file_path=str(tmp_path / "stale.pdf"),
        size=0,
        metadata=_p1b_meta(),
    )
    r = client.get(f"/v1/projects/{proj['id']}/documents", headers=H)
    assert r.status_code == 200, r.text
    listed = next(d for d in r.json()["documents"] if d["id"] == doc["id"])
    assert listed["size"] == 0
    assert listed["has_remote_source"] is True
    assert listed["has_file"] is True


def test_preview_rag_backfill_stub_resolves_drive_by_filename(
    client, monkeypatch, tmp_path,
):
    """Live ocr9demo shape: size=0, G:\\ path, source=rag_backfill_client_clean_all,
    drive_file_id null. Filename lookup + download must
    200 and persist the Drive id."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    name = "AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf"
    drive_id = "1ExampleDriveFileId001xxxxxxxxxxx"
    pdf = b"%PDF-1.4 from Drive filename resolve\n"
    doc = store.add_document(
        project_id=corpus["id"],
        original_name=name,
        file_path=(
            r"X:\Example Drive\Master Folder\the client project\Contract Docs"
            r"\Contractor\Contract docs SIGNED\\" + name
        ),
        size=0,
        metadata={
            "source": "rag_backfill_client_clean_all",
            "drive_file_id": None,
            "source_path": (
                r"X:\Example Drive\Master Folder\the client project\Contract Docs"
                r"\Contractor\Contract docs SIGNED\\" + name
            ),
            "ext": ".pdf",
        },
    )
    assert doc.get("has_remote_source") is False
    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    lookups: list[str] = []

    def _lookup(filename: str):
        lookups.append(filename)
        return (drive_id, None) if filename == name else (None, "nope")

    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name", _lookup,
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (pdf, None) if fid == drive_id else (None, "wrong id"),
    )

    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "pdf"
    assert lookups == [name]

    refreshed = store.get_document(doc["id"])
    assert refreshed is not None
    assert (refreshed.get("metadata") or {}).get("drive_file_id") == drive_id
    assert refreshed["size"] == len(pdf)


def _stub_and_twins(store, tmp_path, name, twin_ids):
    """A cited rag_backfill stub (no Drive id) plus P1B rows of the same name."""
    from app.core import users

    users.ensure_user_exists("twin-owner")
    stub_proj = store.create_project(name="Stub corpus", user_id="twin-owner")["id"]
    twin_proj = store.create_project(name="Drive corpus", user_id="twin-owner")["id"]
    stub = store.add_document(
        project_id=stub_proj, original_name=name,
        file_path="X:\Example Drive\\" + name, size=0,
        metadata={"source": "rag_backfill_client_clean_all", "drive_file_id": None},
    )
    for fid in twin_ids:
        store.add_document(
            project_id=twin_proj, original_name=name,
            file_path=str(tmp_path / f"gone-{fid}.pdf"), size=0,
            metadata={"source": "p1b_server_drive_reingestion", "drive_file_id": fid},
        )
    return stub


def test_cited_stub_opens_from_its_ledger_twins_drive_id(client, monkeypatch, tmp_path):
    """Live 2026-10-04: a chat cited a rag_backfill stub, whose row has no
    Drive id, and the Drive name search found nothing -- 404. The same
    original was ingested from Drive into another row: its id is the source."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    pdf = b"%PDF-1.4 conditions of contract"
    stub = _stub_and_twins(store, tmp_path, "Vol 1 - Conditions of Contract.pdf", ["drive-twin-1"])
    searched: list[str] = []
    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda name: searched.append(name) or (None, "not found"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (pdf, None) if fid == "drive-twin-1" else (None, "wrong id"),
    )

    path, status = store.materialize_document_file(store.get_document(stub["id"]))

    assert status == "ok", status
    assert open(path, "rb").read() == pdf
    assert searched == []  # the ledger answered; no Drive-wide name search
    meta = store.get_document(stub["id"])["metadata"]
    assert meta["drive_file_id"] == "drive-twin-1"
    assert meta["drive_resolved_from"] == "ledger_twin"


def test_ambiguous_ledger_twins_are_not_guessed(client, monkeypatch, tmp_path):
    """Two different originals share the name: no guess from the ledger."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    stub = _stub_and_twins(store, tmp_path, "Drawing register.pdf", ["drive-a", "drive-b"])
    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda name: (None, f"no Drive file named {name}"),
    )
    downloads: list[str] = []
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: downloads.append(fid) or (b"%PDF-1.4", None),
    )

    path, status = store.materialize_document_file(store.get_document(stub["id"]))

    assert path is None and "no Google Drive file id" in status
    assert downloads == []
    assert not (store.get_document(stub["id"])["metadata"] or {}).get("drive_file_id")


def test_preview_rag_backfill_stub_unresolved_clear_404(
    client, monkeypatch, tmp_path,
):
    """Unlinkable stub stays an honest 404 — no invented bytes."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    proj = _new_project(client)
    name = "AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf"
    doc = store.add_document(
        project_id=proj["id"],
        original_name=name,
        file_path=r"X:\Example Drive\Master Folder\\" + name,
        size=0,
        metadata={
            "source": "rag_backfill_client_clean_all",
            "drive_file_id": None,
        },
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda filename: (None, f"no Drive file named {filename}"),
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H,
    )
    assert r.status_code == 404
    assert r.status_code != 500
    detail = r.json()["detail"]
    assert "not available" in detail.lower()
    assert "no Google Drive file id" in detail
    assert name in detail


def test_preview_user_upload_local_path_still_200(client):
    """A real on-disk upload must not go through Drive filename resolve."""
    proj = _new_project(client)
    doc = _upload(client, proj["id"], "site-note.txt", b"hello from disk", "text/plain")
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "text"
    assert "hello from disk" in r.json()["text"]


SIGNED_VOLUME_NAME = "AB-2023-101 - Infrastructure Package 1- vol 1-Executed.pdf"
SIGNED_VOLUME_DRIVE_ID = "1ExampleDriveFileId001xxxxxxxxxxx"


def _rag_backfill_stub(store, project_id, tmp_path, *, drive_file_id=None):
    return store.add_document(
        project_id=project_id,
        original_name=SIGNED_VOLUME_NAME,
        file_path=(
            r"X:\Example Drive\Master Folder\the client project\Contract Docs"
            r"\Contractor\Contract docs SIGNED\\" + SIGNED_VOLUME_NAME
        ),
        size=0,
        metadata={
            "source": "rag_backfill_client_clean_all",
            "drive_file_id": drive_file_id,
            "source_path": (
                r"X:\Example Drive\Master Folder\the client project\Contract Docs"
                r"\Contractor\Contract docs SIGNED\\" + SIGNED_VOLUME_NAME
            ),
            "ext": ".pdf",
        },
    )


def test_preview_drive_id_public_download_when_sa_list_and_media_fail(
    client, monkeypatch, tmp_path,
):
    """ocr9demo after PATCH: name-search still blind, SA media fails,
    public anyone-with-link download hydrates and persists the id + size."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    pdf = b"%PDF-1.4 public anyone-with-link\n"
    doc = _rag_backfill_stub(
        store, corpus["id"], tmp_path, drive_file_id=SIGNED_VOLUME_DRIVE_ID,
    )
    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    lookups: list[str] = []
    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda filename: lookups.append(filename) or (None, f"no Drive file named {filename}"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.get_file_metadata",
        lambda fid: (
            ({"id": fid, "name": SIGNED_VOLUME_NAME, "mimeType": "application/pdf"}, None)
            if fid == SIGNED_VOLUME_DRIVE_ID
            else (None, "Drive files.get returned 404")
        ),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (None, "Drive download returned 403"),
    )
    public_hits: list[str] = []

    def _public(fid: str):
        public_hits.append(fid)
        return (pdf, None) if fid == SIGNED_VOLUME_DRIVE_ID else (None, "wrong id")

    monkeypatch.setattr(
        "app.core.gdrive_service.download_public_file_bytes", _public,
    )

    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "pdf"
    assert r.json()["has_file"] is True
    assert r.json()["size"] == len(pdf)
    assert lookups == []  # id already on the row — skip name-search
    assert public_hits == [SIGNED_VOLUME_DRIVE_ID]

    refreshed = store.get_document(doc["id"])
    assert refreshed is not None
    assert (refreshed.get("metadata") or {}).get("drive_file_id") == SIGNED_VOLUME_DRIVE_ID
    assert refreshed["size"] == len(pdf)


def test_preview_name_search_miss_files_get_media_hit(
    client, monkeypatch, tmp_path,
):
    """SA cannot list-by-name, but files.get + media works once id is set."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    workspace = _new_project(client, "Preview Workspace")
    corpus = _new_project(client, "Citeable Corpus")
    pdf = b"%PDF-1.4 from files.get media\n"
    doc = _rag_backfill_stub(
        store, corpus["id"], tmp_path, drive_file_id=SIGNED_VOLUME_DRIVE_ID,
    )
    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda filename: (None, f"no Drive file named {filename}"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.get_file_metadata",
        lambda fid: ({"id": fid, "name": SIGNED_VOLUME_NAME}, None),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (pdf, None) if fid == SIGNED_VOLUME_DRIVE_ID else (None, "wrong id"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_public_file_bytes",
        lambda fid: (_ for _ in ()).throw(AssertionError("public download must not run")),
    )
    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/preview",
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "pdf"


def test_preview_private_drive_file_still_404(
    client, monkeypatch, tmp_path,
):
    """A Drive id that is not anyone-with-link must not leak bytes."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    proj = _new_project(client)
    private_id = "1PrivateFileIdNotSharedXXXXXX"
    doc = _rag_backfill_stub(
        store, proj["id"], tmp_path, drive_file_id=private_id,
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda filename: (None, f"no Drive file named {filename}"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.get_file_metadata",
        lambda fid: (None, "Drive files.get returned 404"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (None, "Drive download returned 404"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_public_file_bytes",
        lambda fid: (None, "public Drive download is not world-readable"),
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H,
    )
    assert r.status_code == 404
    assert r.status_code != 500
    detail = r.json()["detail"].lower()
    assert "not available" in detail
    assert "not world-readable" in detail


def test_patch_drive_file_id_owner_then_preview_hydrates(
    client, monkeypatch, tmp_path,
):
    """Owner PATCH seeds the id; name-search miss + public download 200."""
    from app.core import projects as store

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    proj = _new_project(client)
    pdf = b"%PDF-1.4 after owner patch\n"
    doc = _rag_backfill_stub(store, proj["id"], tmp_path, drive_file_id=None)
    assert (doc.get("metadata") or {}).get("drive_file_id") in (None, "")

    patched = client.patch(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}",
        json={"drive_file_id": SIGNED_VOLUME_DRIVE_ID},
        headers=H,
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert (body.get("metadata") or {}).get("drive_file_id") == SIGNED_VOLUME_DRIVE_ID
    assert body["has_remote_source"] is True

    monkeypatch.setattr(
        "app.core.gdrive_service.find_file_id_by_exact_name",
        lambda filename: (None, f"no Drive file named {filename}"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.get_file_metadata",
        lambda fid: (None, "service account unavailable"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_file_bytes",
        lambda fid: (None, "service account unavailable"),
    )
    monkeypatch.setattr(
        "app.core.gdrive_service.download_public_file_bytes",
        lambda fid: (pdf, None) if fid == SIGNED_VOLUME_DRIVE_ID else (None, "wrong id"),
    )
    r = client.get(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "pdf"
    refreshed = store.get_document(doc["id"])
    assert (refreshed.get("metadata") or {}).get("drive_file_id") == SIGNED_VOLUME_DRIVE_ID
    assert refreshed["size"] == len(pdf)


def _register(client, email):
    r = client.post("/v1/users/register", json={"email": email, "password": PW_10})
    assert r.status_code in (201, 409), r.text
    r = client.post("/v1/users/login", json={"email": email, "password": PW_10})
    assert r.status_code == 200, r.text
    return {"token": r.json()["token"], "id": r.json()["user"]["id"]}


def _promote_admin(uid):
    from app.core.db import SessionLocal
    from app.core.models import User

    with SessionLocal() as db:
        db.get(User, uid).role = "admin"
        db.commit()


def test_patch_drive_file_id_stranger_unauthorized(client, tmp_path):
    """Stranger cannot seed a Drive id — 404 on private, 403 on shared."""
    from app.core import projects as store
    from app.core.db import SessionLocal
    from app.core.models import Project

    owner = _register(client, "patch-owner@example.com")
    stranger = _register(client, "patch-stranger@example.com")
    owner_h = {"Authorization": f"Bearer {owner['token']}"}
    stranger_h = {"Authorization": f"Bearer {stranger['token']}"}

    private = client.post(
        "/v1/projects", json={"name": "Owner private"}, headers=owner_h,
    )
    assert private.status_code == 201, private.text
    private_pid = private.json()["id"]
    private_doc = _rag_backfill_stub(store, private_pid, tmp_path)

    denied_private = client.patch(
        f"/v1/projects/{private_pid}/documents/{private_doc['id']}",
        json={"drive_file_id": SIGNED_VOLUME_DRIVE_ID},
        headers=stranger_h,
    )
    assert denied_private.status_code in (403, 404), denied_private.text
    assert (store.get_document(private_doc["id"]).get("metadata") or {}).get(
        "drive_file_id"
    ) in (None, "")

    shared = client.post(
        "/v1/projects", json={"name": "Owner shared"}, headers=owner_h,
    )
    assert shared.status_code == 201, shared.text
    shared_pid = shared.json()["id"]
    with SessionLocal() as db:
        row = db.get(Project, shared_pid)
        row.origin = "admin_drive_approved"
        row.is_approved = True
        db.commit()
    shared_doc = _rag_backfill_stub(store, shared_pid, tmp_path)
    # Stranger can open the shared project but must not mutate the pointer.
    opened = client.get(f"/v1/projects/{shared_pid}", headers=stranger_h)
    assert opened.status_code == 200, opened.text
    denied_shared = client.patch(
        f"/v1/projects/{shared_pid}/documents/{shared_doc['id']}",
        json={"drive_file_id": SIGNED_VOLUME_DRIVE_ID},
        headers=stranger_h,
    )
    assert denied_shared.status_code == 403, denied_shared.text
    assert "owner" in denied_shared.json()["detail"].lower()
    assert (store.get_document(shared_doc["id"]).get("metadata") or {}).get(
        "drive_file_id"
    ) in (None, "")


def test_patch_drive_file_id_admin_ok(client, tmp_path):
    """Admin can seed drive_file_id on a project they do not own."""
    from app.core import projects as store

    owner = _register(client, "patch-admin-owner@example.com")
    admin = _register(client, "patch-admin@example.com")
    _promote_admin(admin["id"])
    owner_h = {"Authorization": f"Bearer {owner['token']}"}
    admin_h = {"Authorization": f"Bearer {admin['token']}"}

    proj = client.post(
        "/v1/projects", json={"name": "Admin seed target"}, headers=owner_h,
    )
    assert proj.status_code == 201, proj.text
    pid = proj.json()["id"]
    doc = _rag_backfill_stub(store, pid, tmp_path)
    r = client.patch(
        f"/v1/projects/{pid}/documents/{doc['id']}",
        json={"drive_file_id": SIGNED_VOLUME_DRIVE_ID},
        headers=admin_h,
    )
    assert r.status_code == 200, r.text
    assert (r.json().get("metadata") or {}).get("drive_file_id") == SIGNED_VOLUME_DRIVE_ID


def test_patch_drive_file_id_rejects_url(client, tmp_path):
    from app.core import projects as store

    proj = _new_project(client)
    doc = _rag_backfill_stub(store, proj["id"], tmp_path)
    r = client.patch(
        f"/v1/projects/{proj['id']}/documents/{doc['id']}",
        json={"drive_file_id": "https://drive.google.com/file/d/abc/view"},
        headers=H,
    )
    assert r.status_code == 422


def test_download_public_file_bytes_handles_confirm_token(monkeypatch):
    """Google's virus-scan HTML must be retried with confirm= before failing."""
    from app.core import gdrive_service

    pdf = b"%PDF-1.4 after confirm\n"
    html = (
        '<form action="https://drive.usercontent.google.com/download">'
        '<input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="11111111-1111-1111-1111-111111111111">'
        "</form>"
    )

    class _Resp:
        def __init__(self, status_code, content, content_type, url="https://drive.google.com/uc"):
            self.status_code = status_code
            self.content = content
            self.headers = {"content-type": content_type}
            self.url = url
            self.cookies = {}

    class _Client:
        def __init__(self, *a, **k):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            self.calls.append((url, dict(params or {})))
            if (params or {}).get("confirm") == "t":
                return _Resp(200, pdf, "application/pdf")
            return _Resp(200, html.encode(), "text/html")

    fake = _Client()
    monkeypatch.setattr(
        "httpx.Client", lambda *a, **k: fake,
    )
    blob, err = gdrive_service.download_public_file_bytes(SIGNED_VOLUME_DRIVE_ID)
    assert err is None
    assert blob == pdf
    assert any(c[1].get("confirm") == "t" for c in fake.calls)


def test_download_public_file_bytes_private_html_fails_closed(monkeypatch):
    from app.core import gdrive_service

    class _Resp:
        status_code = 200
        content = b"<html>You need access. Request access from the owner.</html>"
        headers = {"content-type": "text/html"}
        url = "https://drive.google.com/file/d/1PrivateFileIdNotSharedXXXXXX/view"
        cookies = {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return _Resp()

    monkeypatch.setattr("httpx.Client", lambda *a, **k: _Client())
    blob, err = gdrive_service.download_public_file_bytes("1PrivateFileIdNotSharedXXXXXX")
    assert blob is None
    assert err == "public Drive download is not world-readable"


def test_is_valid_drive_file_id():
    from app.core.gdrive_service import is_valid_drive_file_id

    assert is_valid_drive_file_id(SIGNED_VOLUME_DRIVE_ID)
    assert not is_valid_drive_file_id("")
    assert not is_valid_drive_file_id("short")
    assert not is_valid_drive_file_id("https://drive.google.com/file/d/abc/view")
    assert not is_valid_drive_file_id("../../etc/passwd")
