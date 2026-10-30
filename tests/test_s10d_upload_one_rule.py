"""S-10d: /v1/upload-limits publishes exactly what the routes accept; an
unaccepted format is 415 with or without a project; HTML follows one rule."""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest
from fastapi.testclient import TestClient

import app.routers.health
import app.routers.projects
import app.routers.upload
from app.core import upload_limits
from app.core.upload_limits import _ingestible_extensions
from app.core.ingest_status import TEXT_BEARING_EXTS

from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def _upload(c, name, pid=None):
    data = {"project_id": pid} if pid else {}
    return c.post("/upload", files={"file": (name, b"<p>body</p>", "text/plain")}, data=data, headers=H)


@pytest.mark.parametrize("key,accepted", [
    ("allowed_extensions", lambda: upload_limits.ALLOWED_UPLOAD_EXTENSIONS),
    ("project_document_extensions", lambda: TEXT_BEARING_EXTS),
])
def test_published_lists_equal_the_accepted_sets(client, key, accepted):
    """upload_limits() publishes each list the routes enforce."""
    body = client.get("/v1/upload-limits").json()
    assert set(body[key]) == set(accepted())
    assert set(TEXT_BEARING_EXTS) <= set(body["allowed_extensions"])
    assert body["unsupported_type_status"] == upload_limits.UNSUPPORTED_TYPE_STATUS


@pytest.mark.parametrize("ext", [".html", ".rtf"])
def test_ingestible_extensions_are_the_ingest_rule(ext):
    assert ext in _ingestible_extensions()
    assert _ingestible_extensions() == frozenset(TEXT_BEARING_EXTS)


@pytest.mark.parametrize("name", [".", ".."])
def test_add_document_refuses_a_dot_filename(client, name):
    pid = client.post("/v1/projects", json={"name": "s10d-dot"}, headers=H).json()["id"]
    r = client.post(f"/v1/projects/{pid}/documents", files={"file": (name, b"x", "text/plain")}, headers=H)
    assert r.status_code == 400, r.text


@pytest.mark.parametrize("cap,status", [(4, 413), (10_000, 200)])
def test_upload_v1_refuses_a_file_over_the_cap(client, monkeypatch, cap, status):
    monkeypatch.setattr(upload_limits, "max_upload_bytes", lambda: cap)
    assert _upload(client, "note.txt").status_code == status


@pytest.mark.parametrize("name", ["thing.exe", "thing.zzz"])
def test_upload_v1_checks_size_before_type(client, monkeypatch, name):
    """The size cap is the first rule: an oversized file is 413 whatever its type."""
    monkeypatch.setattr(upload_limits, "max_upload_bytes", lambda: 4)
    assert _upload(client, name).status_code == 413


@pytest.mark.parametrize("with_project", [False, True])
def test_unaccepted_format_is_415_either_way(client, with_project):
    pid = client.post("/v1/projects", json={"name": "s10d"}, headers=H).json()["id"] if with_project else None
    r = _upload(client, "thing.exe", pid)
    assert r.status_code == 415, r.text


@pytest.mark.parametrize("name", ["page.html", "page.htm"])
@pytest.mark.parametrize("with_project", [False, True])
def test_html_one_rule(client, name, with_project):
    pid = client.post("/v1/projects", json={"name": "s10d-h"}, headers=H).json()["id"] if with_project else None
    assert _upload(client, name, pid).status_code == 200
