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
from app.routers import chat_photos, projects
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00\x10JFIF\x00" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _ooxml(folder: str) -> bytes:
    # A fixed entry timestamp keeps the bytes (and so the parametrize ids)
    # identical across xdist workers; ZipFile.writestr(str, ...) stamps "now".
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for arc, body in (("[Content_Types].xml", "<Types/>"), (f"{folder}/document.xml", "<x/>")):
            z.writestr(zipfile.ZipInfo(arc, date_time=(2020, 1, 1, 0, 0, 0)), body)
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


def _route_path(endpoint):
    """The routed path of ``endpoint``, read from the app, never typed here."""
    for route in app.routes:
        routed = getattr(route, "endpoint", None)
        if routed is not None and (routed.__module__, routed.__qualname__) == (
            endpoint.__module__, endpoint.__qualname__,
        ):
            return route.path
    raise AssertionError(f"{endpoint.__qualname__} is not routed")


def _documents_path(pid):
    return _route_path(projects.add_document).format(project_id=pid)


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
        _documents_path(pid),
        files={"file": (multipart_name, data, "application/octet-stream")},
        data=form,
        headers=H,
    )
    assert r.status_code == 201, r.text
    assert r.json()["document"]["original_name"] == stored


@pytest.mark.parametrize("display_name", ["..", "folder/"])
def test_display_name_with_no_file_part_is_refused(client, display_name):
    # The user's name wins over the multipart name, so a display name that is
    # only a path or a dot-segment must be refused as a filename, not typed.
    pid = _project(client, "Invalid display name")
    r = client.post(
        _documents_path(pid),
        files={"file": ("spec.pdf", PDF, "application/pdf")},
        data={"display_name": display_name},
        headers=H,
    )
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "Invalid filename"


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
        _documents_path(pid),
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
        _documents_path(pid),
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


class _CountingImageBlock:
    """Stands in for the image block: reports a fixed person count."""

    def __init__(self, persons):
        self.persons = persons
        self.paths = []

    async def execute(self, inputs, params):
        self.paths.append(inputs["file_path"])
        return {"result": {"summary_by_class": {"person": self.persons}, "safety_qaqc": []}}


@pytest.mark.parametrize(
    "multipart_name,display_name,data,persons,typed,observation",
    [
        ("Photo from شادي \U0001F1F8\U0001F1E6", None, JPEG, 2,
         "Photo from شادي \U0001F1F8\U0001F1E6.jpg", "2 people visible"),
        ("Photo_from_Chadi", "Photo from شادي", PNG, 1,
         "Photo from شادي.png", "1 person visible"),
    ],
)
def test_extensionless_photo_is_analysed_by_content(
    client, monkeypatch, multipart_name, display_name, data, persons, typed, observation,
):
    block = _CountingImageBlock(persons)
    monkeypatch.setattr("app.dependencies.get_block_instance", lambda name: block)
    form = {"display_name": display_name} if display_name else {}
    r = client.post(
        _route_path(chat_photos.analyze_chat_photo),
        files={"file": (multipart_name, data, "application/octet-stream")},
        data=form,
        headers=H,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == typed
    assert body["observations"] == [observation]
    assert body["person_count"] == persons
    assert len(block.paths) == 1 and block.paths[0].endswith(typed[-4:])
