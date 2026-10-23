"""The Documents panel states the reason already stored on the row.

A file a normal user uploaded is kept for that user's session and is not
added to the project knowledge base. ``chunk_count == 0`` on that row is
the skipped index, not a failed extract. The panel must show the stored
reason. It must not invent an extraction failure for every unindexed file.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import privileges
from app.core import projects as store
from app.core.ingest_status import EXTRACT_FAILED, ZERO_CHUNK
from app.dependencies import require_api_key, require_user
from app.main import app
from tests._fixture_values import STUB_KEY_01

USER = {"user_id": "panel-status-user", "role": "user"}
NO_KEY = {"Authorization": "Bearer " + STUB_KEY_01}
_INVENTED_EXTRACT_FAILURE = "Extraction returned no usable text"
_PANEL = Path("frontend/src/pages/ProjectWorkspace.tsx")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def as_user():
    app.dependency_overrides[require_user] = lambda: dict(USER)
    app.dependency_overrides[require_api_key] = lambda: dict(USER)
    yield
    app.dependency_overrides.pop(require_user, None)
    app.dependency_overrides.pop(require_api_key, None)


@pytest.fixture
def indexed(monkeypatch):
    from app.core import doc_index

    calls: list = []
    monkeypatch.setattr(doc_index, "maybe_eager_index", lambda pid, did: calls.append((pid, did)))
    return calls


def _project() -> str:
    from app.core import users

    users.ensure_user_exists(USER["user_id"])
    return store.create_project(name="Panel status", user_id=USER["user_id"])["id"]


def _doc(body: dict, doc_id: str) -> dict:
    return next(d for d in body["documents"] if d["id"] == doc_id)


def test_user_upload_panel_states_the_stored_skip_not_an_extraction_failure(
    client, as_user, indexed,
):
    pid = _project()
    uploaded = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": ("pour-note.txt", b"The slab pour is scheduled for level three.", "text/plain")},
        headers=NO_KEY,
    )
    assert uploaded.status_code == 201, uploaded.text
    doc_id = uploaded.json()["document"]["id"]
    assert indexed == []

    listed = client.get(f"/v1/projects/{pid}", headers=NO_KEY)
    assert listed.status_code == 200, listed.text
    doc = _doc(listed.json(), doc_id)

    assert doc.get("chunk_count") == 0
    status = doc.get("panel_status") or {}
    assert status.get("detail") == privileges.PROJECT_RAG_ADMIN_ONLY_DETAIL
    assert _INVENTED_EXTRACT_FAILURE not in (status.get("detail") or "")
    # The row the panel appends on upload carries the same stored reason.
    posted = (uploaded.json().get("document") or {}).get("panel_status") or {}
    assert posted.get("detail") == privileges.PROJECT_RAG_ADMIN_ONLY_DETAIL


def test_stored_extraction_failure_is_stated_as_itself(client, as_user, indexed):
    """A row the indexer classified as an empty extract keeps that reason.

    It is not relabelled with the admin-only skip, and a skip is not
    relabelled as this failure.
    """
    pid = _project()
    row = store.add_document(pid, "blank-scan.pdf", size=120)
    store.stamp_document_index(
        row["id"],
        chunk_count=0,
        ingest_status=ZERO_CHUNK,
        ingest_status_reason="empty extract",
    )

    listed = client.get(f"/v1/projects/{pid}", headers=NO_KEY)
    doc = _doc(listed.json(), row["id"])
    status = doc.get("panel_status") or {}
    assert status.get("detail") == "empty extract"
    assert status.get("detail") != privileges.PROJECT_RAG_ADMIN_ONLY_DETAIL
    assert status.get("status") == ZERO_CHUNK

    failed = store.add_document(pid, "unreadable.docx", size=80)
    store.stamp_document_index(
        failed["id"],
        chunk_count=0,
        ingest_status=EXTRACT_FAILED,
        ingest_status_reason=None,
    )
    listed = client.get(f"/v1/projects/{pid}", headers=NO_KEY)
    failed_doc = _doc(listed.json(), failed["id"])
    failed_status = failed_doc.get("panel_status") or {}
    assert failed_status.get("status") == EXTRACT_FAILED
    assert _INVENTED_EXTRACT_FAILURE in (failed_status.get("detail") or "")
    assert privileges.PROJECT_RAG_ADMIN_ONLY_DETAIL not in (failed_status.get("detail") or "")


def test_searchable_document_has_no_not_indexed_note(client, as_user, monkeypatch):
    pid = _project()
    row = store.add_document(pid, "measured-boq.xlsx", size=400)
    # The ledger still says unverified. The live chunk count is what the
    # panel shows, so the note follows that count.
    store.stamp_document_index(row["id"], chunk_count=0, ingest_status="UNVERIFIED")

    class _Counts:
        def count(self, _project_id):
            return 4

        def count_by_doc(self, _project_id):
            return {row["id"]: 4}

    monkeypatch.setattr("app.core.rag.embeddings.get_embedder", lambda: type("E", (), {"dim": 8})())
    monkeypatch.setattr("app.core.rag.vector_store.get_store", lambda dim=None: _Counts())

    listed = client.get(f"/v1/projects/{pid}", headers=NO_KEY)
    doc = _doc(listed.json(), row["id"])
    assert doc.get("chunk_count") == 4
    assert not doc.get("panel_status")


def test_documents_panel_renders_the_stored_reason():
    src = _PANEL.read_text(encoding="utf-8")
    assert _INVENTED_EXTRACT_FAILURE not in src
    assert "panel_status" in src
    assert "doc.chunk_count === 0" not in src
