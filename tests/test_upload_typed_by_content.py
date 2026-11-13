"""An upload is typed by what its bytes prove, not by its name (S-U2).

A phone's share sheet names a shared document after its sender, in the
sender's script, with an emoji and no extension. The name told the server
nothing, so the upload was refused as type ''. These tests build their own
bytes and names: the content proves the type, the user's name is kept, and
the error for an unprovable file names the actual cause.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.core import upload_limits as ul
from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00\x10JFIF\x00" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _ooxml(folder: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr(f"{folder}/document.xml", "<x/>")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("WARM_MODELS_BLOCKING", "1")
    try:
        with TestClient(app) as c:
            yield c
    finally:
        mp.undo()


def _project(client, name):
    r = client.post("/v1/projects", json={"name": name, "client": "ACME"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.mark.parametrize(
    "data,proven",
    [
        (PDF, ".pdf"),
        (b"\xef\xbb\xbf  " + PDF, ".pdf"),
        (JPEG, ".jpg"),
        (PNG, ".png"),
        (b"RIFF\x10\x00\x00\x00WEBPVP8 ", ".webp"),
        (b"II*\x00" + b"\x00" * 8, ".tif"),
        (_ooxml("word"), ".docx"),
        (_ooxml("xl"), ".xlsx"),
        (b"PK\x03\x04 not really a zip", None),
        (b"item,qty\nrebar,12\n", None),
        (b"", None),
    ],
)
def test_sniff_extension_reads_the_bytes(data, proven):
    stream = io.BytesIO(data)
    assert ul.sniff_extension(stream) == proven
    assert stream.tell() == 0


@pytest.mark.parametrize(
    "name,data,typed",
    [
        ("Document from \u0634\u0627\u062f\u064a Chadi \U0001F1F8\U0001F1E6", PDF,
         "Document from \u0634\u0627\u062f\u064a Chadi \U0001F1F8\U0001F1E6.pdf"),
        ("Report v1.2", _ooxml("word"), "Report v1.2.docx"),
        ("minutes.txt", PDF, "minutes.pdf"),
        ("photo.jpeg", JPEG, "photo.jpeg"),
        ("scan.TIFF", b"MM\x00*" + b"\x00" * 8, "scan.TIFF"),
        ("boq.csv", b"item,qty\n", "boq.csv"),
        ("notes", b"plain words", "notes"),
    ],
)
def test_typed_upload_name_keeps_the_users_name(name, data, typed):
    assert ul.typed_upload_name(name, io.BytesIO(data)) == typed


@pytest.mark.parametrize(
    "display,multipart,expected",
    [
        ("Document from \u0634\u0627\u062f\u064a", "upload.bin", "Document from \u0634\u0627\u062f\u064a"),
        ("  ", "C:\\phone\\shared\\spec.pdf", "spec.pdf"),
        (None, "dir/sub/plan.docx", "plan.docx"),
        (None, None, ""),
    ],
)
def test_upload_display_name_prefers_the_form_field(display, multipart, expected):
    assert ul.upload_display_name(display, multipart) == expected


@pytest.mark.parametrize("ext,expected", [(".dwg", ".dwg"), ("", "no extension")])
def test_describe_ext_names_a_missing_extension(ext, expected):
    assert ul.describe_ext(ext).startswith(expected)


@pytest.mark.parametrize(
    "multipart_name,display_name,data,stored",
    [
        # The share-sheet name travels as the multipart filename itself.
        ("Document from \u0634\u0627\u062f\u064a Chadi \U0001F1F8\U0001F1E6", None, PDF,
         "Document from \u0634\u0627\u062f\u064a Chadi \U0001F1F8\U0001F1E6.pdf"),
        # The client sends an ASCII stand-in and the real name as a field.
        ("Document_from_Chadi", "Document from \u0634\u0627\u062f\u064a Chadi \U0001F1F8\U0001F1E6",
         _ooxml("word"),
         "Document from \u0634\u0627\u062f\u064a Chadi \U0001F1F8\U0001F1E6.docx"),
        # A known extension the bytes contradict is corrected.
        ("site minutes.txt", None, PDF, "site minutes.pdf"),
    ],
)
def test_document_upload_is_typed_by_content(client, multipart_name, display_name, data, stored):
    pid = _project(client, "Typed by content")
    form = {"display_name": display_name} if display_name else {}
    r = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": (multipart_name, data, "application/octet-stream")},
        data=form,
        headers=H,
    )
    assert r.status_code == 201, r.text
    assert r.json()["document"]["original_name"] == stored


@pytest.mark.parametrize(
    "name,data,detail",
    [
        ("Shared item", b"\x00\x01\x02 opaque", "no extension, and the content is not a recognised format"),
        ("Shared photo", JPEG, "'.jpg' is not ingestible"),
    ],
)
def test_unprovable_or_excluded_upload_names_the_cause(client, name, data, detail):
    pid = _project(client, "Refused by content")
    r = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": (name, data, "application/octet-stream")},
        headers=H,
    )
    assert r.status_code == ul.UNSUPPORTED_TYPE_STATUS, r.text
    assert detail in r.json()["detail"]


@pytest.mark.parametrize(
    "name,data",
    [("Document from \u0634\u0627\u062f\u064a", PDF), ("big.pdf", PDF)],
)
def test_oversize_upload_gets_the_size_message(client, monkeypatch, name, data):
    pid = _project(client, "Oversize")
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "64")
    r = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": (name, data + b"\x00" * 128, "application/octet-stream")},
        headers=H,
    )
    assert r.status_code == 413, r.text
    assert "File too large" in r.json()["detail"]


@pytest.mark.parametrize(
    "name,data,detail",
    [
        ("Shared from phone", PDF, "Not an image: .pdf"),
        ("Shared from phone", b"opaque bytes", "Not an image: no extension"),
    ],
)
def test_photo_route_types_by_content(client, name, data, detail):
    r = client.post(
        "/v1/chat/analyze-photo",
        files={"file": (name, data, "application/octet-stream")},
        headers=H,
    )
    assert r.status_code == 400, r.text
    assert r.json()["detail"].startswith(detail)
