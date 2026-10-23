"""HTTP redline detection endpoint — Stream D, Part 2.

POST /v1/projects/{project_id}/documents/{document_id}/redlines

Tests:
  - Marked-up drawing PDF → has_markup true, total_regions >= 1, red region detected
  - Clean all-white drawing PDF → has_markup false
  - Nonexistent document id → 404
  - Cross-tenant: user B calls user A's project/document → 404
  - Non-PDF document (.txt, or a pre-rule image row) → 400
"""

import io
import uuid

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import PW_06

# Run-unique suffix so parallel test runs don't collide on user emails.
_RUN = uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _user_token(client, email: str) -> str:
    """Register + login a user; return the JWT token string."""
    client.post("/v1/users/register",
                json={"email": email, "password": PW_06})
    r = client.post("/v1/users/login",
                    json={"email": email, "password": PW_06})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _create_project(client, headers, name="Redline Test Project") -> str:
    r = client.post("/v1/projects", json={"name": name}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_png_bytes(width: int, height: int, bg=(255, 255, 255),
                    patch_color=None, patch_rect=None) -> bytes:
    """Build a PNG image in memory. Optionally paint a coloured patch.

    patch_color: (R, G, B) tuple
    patch_rect:  (x0, y0, x1, y1) inclusive pixel range
    """
    img = Image.new("RGB", (width, height), bg)
    if patch_color and patch_rect:
        x0, y0, x1, y1 = patch_rect
        for y in range(y0, y1):
            for x in range(x0, x1):
                img.putpixel((x, y), patch_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _as_drawing_pdf(png_bytes: bytes) -> bytes:
    """The same image as a one-page PDF -- how a scanned or exported drawing
    reaches a project. Images are not project documents (text formats only,
    docs/INGEST_EXCLUSION_RULE.md); redline runs on the drawing PDF."""
    buf = io.BytesIO()
    Image.open(io.BytesIO(png_bytes)).convert("RGB").save(buf, format="PDF")
    return buf.getvalue()


def _upload_file(client, headers, project_id: str, filename: str,
                 file_bytes: bytes, content_type: str = "application/pdf") -> str:
    """Upload a file to a project; return the document id."""
    files = {"file": (filename, file_bytes, content_type)}
    r = client.post(f"/v1/projects/{project_id}/documents",
                    files=files, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["document"]["id"]


# ── tests ────────────────────────────────────────────────────────────────────

def test_redline_on_marked_image(client):
    """An image with a solid red patch → has_markup true, red region present."""
    tok = _user_token(client, f"redline-marked-{_RUN}@x.com")
    h = _headers(tok)
    pid = _create_project(client, h, "Marked Image Project")

    # 200x200 white image with a 30x30 red square starting at (10, 10).
    # 30x30 = 900 px >> _MIN_REGION_AREA=40; coverage 900/40000=0.0225 >> 0.001.
    img_bytes = _make_png_bytes(
        200, 200,
        bg=(255, 255, 255),
        patch_color=(255, 0, 0),
        patch_rect=(10, 10, 40, 40),
    )
    doc_id = _upload_file(client, h, pid, "marked.pdf", _as_drawing_pdf(img_bytes))

    r = client.post(f"/v1/projects/{pid}/documents/{doc_id}/redlines",
                    headers=h)
    assert r.status_code == 200, r.text
    body = r.json()

    # Top-level shape
    assert body["project_id"] == pid
    assert body["document_id"] == doc_id
    assert body["filename"] == "marked.pdf"
    assert body["has_markup"] is True
    assert body["total_regions"] >= 1

    # pages array
    assert len(body["pages"]) == 1
    page = body["pages"][0]
    assert page["page"] == 1
    assert page["has_markup"] is True
    assert page["coverage"] > 0.0

    # At least one red region
    red_regions = [rg for rg in page["regions"]
                   if rg["dominant_colour"] == "red"]
    assert len(red_regions) >= 1


def test_redline_on_clean_image(client):
    """A plain white image → has_markup false, zero regions."""
    tok = _user_token(client, f"redline-clean-{_RUN}@x.com")
    h = _headers(tok)
    pid = _create_project(client, h, "Clean Image Project")

    img_bytes = _make_png_bytes(200, 200, bg=(255, 255, 255))
    doc_id = _upload_file(client, h, pid, "clean.pdf", _as_drawing_pdf(img_bytes))

    r = client.post(f"/v1/projects/{pid}/documents/{doc_id}/redlines",
                    headers=h)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["has_markup"] is False
    assert body["total_regions"] == 0
    assert len(body["pages"]) == 1
    assert body["pages"][0]["has_markup"] is False


def test_redline_404_missing_document(client):
    """A nonexistent document id → 404."""
    tok = _user_token(client, f"redline-missingdoc-{_RUN}@x.com")
    h = _headers(tok)
    pid = _create_project(client, h, "404 Doc Project")

    r = client.post(f"/v1/projects/{pid}/documents/nonexistent-doc-id/redlines",
                    headers=h)
    assert r.status_code == 404


def test_redline_cross_tenant_404(client):
    """User B cannot access user A's project/document — gets 404."""
    tok_a = _user_token(client, f"redline-xtenant-a-{_RUN}@x.com")
    tok_b = _user_token(client, f"redline-xtenant-b-{_RUN}@x.com")
    h_a = _headers(tok_a)
    h_b = _headers(tok_b)

    pid = _create_project(client, h_a, "Redline Owner Project")
    img_bytes = _make_png_bytes(200, 200)
    doc_id = _upload_file(client, h_a, pid, "alice.pdf", _as_drawing_pdf(img_bytes))

    # User B tries to access user A's project/document
    r = client.post(f"/v1/projects/{pid}/documents/{doc_id}/redlines",
                    headers=h_b)
    assert r.status_code == 404


def test_redline_rejects_non_image(client):
    """A .txt document → 400 (redline detection needs PDF or image)."""
    tok = _user_token(client, f"redline-txt-{_RUN}@x.com")
    h = _headers(tok)
    pid = _create_project(client, h, "Text Doc Project")

    files = {"file": ("notes.txt", b"This is a plain text document.", "text/plain")}
    r = client.post(f"/v1/projects/{pid}/documents", files=files, headers=h)
    assert r.status_code == 201, r.text
    doc_id = r.json()["document"]["id"]

    r = client.post(f"/v1/projects/{pid}/documents/{doc_id}/redlines",
                    headers=h)
    assert r.status_code == 400


def test_a_bare_image_is_not_a_project_document(client):
    """Redline input is a drawing PDF; a bare image is refused at upload."""
    tok = _user_token(client, f"redline-png-{_RUN}@x.com")
    h = _headers(tok)
    pid = _create_project(client, h, "Bare Image Project")
    r = client.post(f"/v1/projects/{pid}/documents",
                    files={"file": ("marked.png", _make_png_bytes(20, 20), "image/png")},
                    headers=h)
    assert r.status_code == 415, r.text


def test_redline_does_not_analyse_an_image_document(client, tmp_path):
    """Images are never project documents, so redline has no image branch:
    a pre-rule image row gets a clear 400 and is not analysed. Redline runs
    on the PDF drawing; a photo is chat question context."""
    import os

    from app.core import file_crypto, projects as store

    h = _headers(_user_token(client, f"redline-legacy-{_RUN}@x.com"))
    pid = _create_project(client, h, "Legacy Image Project")
    p = str(tmp_path / "marked.png")
    file_crypto.write_document(p, _make_png_bytes(
        200, 200, patch_color=(255, 0, 0), patch_rect=(10, 10, 40, 40)))
    doc = store.add_document(pid, "marked.png", file_path=p, size=os.path.getsize(p))
    r = client.post(f"/v1/projects/{pid}/documents/{doc['id']}/redlines", headers=h)
    assert r.status_code == 400, r.text
    assert "needs a PDF drawing" in r.json()["detail"]
