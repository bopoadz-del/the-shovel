"""A cited document can always be opened and downloaded.

Two failures seen in the UI pass (2026-10-08):

* 3 of 5 cited Master Corpus sources had no stored original, so the preview
  404'd ("no Google Drive file id") even though the platform indexed their
  text and answered from it. The preview now falls back to that indexed text,
  marked ``indexed_only`` with a note saying so.
* The preview told users to "use the document list to download it", but there
  was no download endpoint at all. ``GET .../documents/{id}/download`` now
  returns the original bytes, or the indexed text as
  ``"<name> (indexed text).txt"`` when no original is stored. A document from
  another project's layer keeps its filename scrubbed, as in the Sources panel.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client, name):
    r = client.post("/v1/projects", json={"name": name}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def _upload(client, pid, filename, content, content_type="text/plain"):
    files = {"file": (filename, content, content_type)}
    r = client.post(f"/v1/projects/{pid}/documents", files=files, headers=H)
    assert r.status_code == 201, r.text
    return r.json()["document"]


def _stored_without_file(pid, name, tmp_path):
    from app.core import projects as store

    return store.add_document(
        project_id=pid,
        original_name=name,
        file_path=str(tmp_path / "never-written.bin"),
        size=0,
    )


def _indexed(monkeypatch, texts):
    """``texts``: {doc_id: indexed text}. Stands in for the chunk store."""
    monkeypatch.setattr(
        "app.routers.projects._indexed_text",
        lambda owner, doc_id: texts.get(doc_id, ""),
    )


# -- preview falls back to the indexed text ------------------------------------------


def test_preview_without_original_shows_indexed_text(client, monkeypatch, tmp_path):
    proj = _new_project(client, "Indexed Only")
    doc = _stored_without_file(proj["id"], "programme.pdf", tmp_path)
    _indexed(monkeypatch, {doc["id"]: "Activity A1 starts on day 3.\n\nFloat is 4 days."})
    r = client.get(f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "text"
    assert body["indexed_only"] is True
    assert body["has_file"] is False
    assert "Float is 4 days." in body["text"]
    assert "original" in body["note"].lower()


def test_preview_without_original_or_index_still_404(client, monkeypatch, tmp_path):
    proj = _new_project(client, "Nothing Stored")
    doc = _stored_without_file(proj["id"], "gone.pdf", tmp_path)
    _indexed(monkeypatch, {})
    r = client.get(f"/v1/projects/{proj['id']}/documents/{doc['id']}/preview", headers=H)
    assert r.status_code == 404
    assert "not available" in r.json()["detail"].lower()


# -- download ----------------------------------------------------------------------


def test_download_returns_the_original_bytes(client):
    proj = _new_project(client, "Download Own")
    doc = _upload(client, proj["id"], "site_diary.txt", b"Pour 3 complete at 14:00.")
    r = client.get(f"/v1/projects/{proj['id']}/documents/{doc['id']}/download", headers=H)
    assert r.status_code == 200, r.text
    assert r.content == b"Pour 3 complete at 14:00."
    disposition = r.headers["content-disposition"]
    assert "attachment" in disposition and "site_diary.txt" in disposition


def test_download_without_original_returns_indexed_text(client, monkeypatch, tmp_path):
    proj = _new_project(client, "Download Indexed")
    doc = _stored_without_file(proj["id"], "programme.pdf", tmp_path)
    _indexed(monkeypatch, {doc["id"]: "Float is 4 days."})
    r = client.get(f"/v1/projects/{proj['id']}/documents/{doc['id']}/download", headers=H)
    assert r.status_code == 200, r.text
    assert r.text == "Float is 4 days."
    assert r.headers["content-type"].startswith("text/plain")
    assert "programme.pdf (indexed text).txt" in r.headers["content-disposition"]


def test_download_without_original_or_index_is_404(client, monkeypatch, tmp_path):
    proj = _new_project(client, "Download Nothing")
    doc = _stored_without_file(proj["id"], "gone.pdf", tmp_path)
    _indexed(monkeypatch, {})
    r = client.get(f"/v1/projects/{proj['id']}/documents/{doc['id']}/download", headers=H)
    assert r.status_code == 404


def test_download_of_a_foreign_private_document_is_404(client):
    workspace = _new_project(client, "Download Workspace")
    other = _new_project(client, "Download Private")
    doc = _upload(client, other["id"], "private.txt", b"not yours")
    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/download", headers=H
    )
    assert r.status_code == 404


def test_download_of_a_cited_document_from_another_layer_keeps_its_stored_name(
    client, monkeypatch
):
    """A cited shared document downloads under its stored name: the shared
    layer is cleaned at source, so no serve-time list rewrites it."""
    workspace = _new_project(client, "Download Cite Workspace")
    corpus = _new_project(client, "Download Cite Corpus")
    doc = _upload(client, corpus["id"], "shared_method_statement.txt", b"Method.")
    monkeypatch.setattr(
        "app.routers.projects._preview_citeable_owner_ids",
        lambda pid: {pid, corpus["id"]},
    )
    r = client.get(
        f"/v1/projects/{workspace['id']}/documents/{doc['id']}/download", headers=H
    )
    assert r.status_code == 200, r.text
    assert "shared_method_statement.txt" in r.headers["content-disposition"]
