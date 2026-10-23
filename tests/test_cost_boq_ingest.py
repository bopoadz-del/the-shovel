"""The cost-BOQ export must ALSO persist the generated workbook back into the
project and queue it for indexing, so the generated BOQ lands in the project's
RAG corpus and chat can answer from it -- while still returning the download.

Mirrors the upload path (app/routers/projects.py::add_document): write the
bytes, store the document row, schedule doc_index.maybe_eager_index.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import projects as store
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}

_XLSX_MEDIA = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

SAMPLE_CATEGORIES = [
    {"name": "Site Works", "items": [
        {"item_no": "A.1", "description": "Site clearing", "unit": "Lot", "qty": 1, "rate": 120000},
        {"item_no": "A.2", "description": "Topsoil stripping", "unit": "sqm", "qty": 15000, "rate": 15},
    ]},
    {"name": "Substructure", "items": [
        {"item_no": "B.1", "description": "Bored piles", "unit": "nos", "qty": 120, "rate": 8500},
    ]},
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client, name="Cost BOQ Ingest Project"):
    r = client.post("/v1/projects", json={"name": name, "client": "ACME"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.usefixtures("dev_key_is_admin")
def test_cost_boq_export_persists_document_and_schedules_index(client, monkeypatch):
    """A cost-boq export with categories must: (a) return a valid xlsx download,
    (b) add a new document to the project, and (c) schedule eager indexing."""
    proj = _new_project(client)
    pid = proj["id"]

    before = {d["id"] for d in store.list_documents(pid)}

    scheduled: list[tuple[str, str]] = []

    # Patch the symbol the router resolved at import time so the scheduled
    # background task records its call instead of doing real indexing work.
    import app.routers.exports as exports

    def _record(project_id, document_id):
        scheduled.append((project_id, document_id))

    monkeypatch.setattr(exports.doc_index, "maybe_eager_index", _record)

    r = client.post(
        f"/v1/projects/{pid}/export/cost-boq",
        json={"categories": SAMPLE_CATEGORIES},
        headers=H,
    )
    assert r.status_code == 200, r.text
    # Still a valid xlsx download (existing contract preserved).
    assert r.headers["content-type"] == _XLSX_MEDIA
    assert r.content[:2] == b"PK"  # xlsx is a zip

    # A new document landed in the project.
    after = store.list_documents(pid)
    new_docs = [d for d in after if d["id"] not in before]
    assert len(new_docs) == 1, new_docs
    assert "Cost BOQ (generated)" in new_docs[0]["original_name"]

    # Eager indexing was scheduled for exactly that new document.
    assert scheduled == [(pid, new_docs[0]["id"])]


def test_cost_boq_export_ingest_false_skips_persistence(client, monkeypatch):
    """With ingest=false the export still downloads but persists nothing."""
    proj = _new_project(client, "No Ingest Project")
    pid = proj["id"]

    before = {d["id"] for d in store.list_documents(pid)}

    scheduled: list[tuple[str, str]] = []
    import app.routers.exports as exports
    monkeypatch.setattr(
        exports.doc_index, "maybe_eager_index",
        lambda project_id, document_id: scheduled.append((project_id, document_id)),
    )

    r = client.post(
        f"/v1/projects/{pid}/export/cost-boq",
        json={"categories": SAMPLE_CATEGORIES, "ingest": False},
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == _XLSX_MEDIA

    after = {d["id"] for d in store.list_documents(pid)}
    assert after == before  # no new document
    assert scheduled == []  # no indexing scheduled


def test_cost_boq_ingest_failure_does_not_break_download(client, monkeypatch):
    """A RAG-ingest failure must never 500 the export -- the download wins."""
    proj = _new_project(client, "Ingest Failure Project")
    pid = proj["id"]

    import app.routers.exports as exports

    def _boom(*a, **k):
        raise RuntimeError("simulated store failure")

    monkeypatch.setattr(exports.projects_store, "add_document", _boom)

    r = client.post(
        f"/v1/projects/{pid}/export/cost-boq",
        json={"categories": SAMPLE_CATEGORIES},
        headers=H,
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == _XLSX_MEDIA
    assert r.content[:2] == b"PK"
