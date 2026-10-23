"""POST /v1/admin/documents/{old_id}/supersede -- retire, never delete.

``projects.supersede_document`` already existed and is the one safe,
reversible way to replace a badly-indexed document: it hides the old row from
retrieval and points it at its replacement. It was only reachable through
``add_document(reingest_of=...)``, which no route exposes, so from outside the
process the choices were a hand-written UPDATE against production or a
delete.

Found when three documents carried a false Contract Data label and could not
be re-indexed in place (the server does not retain the source bytes of
archive-ingested documents, so doc-reindex returns ZERO_CHUNK).
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import projects as store
from app.dependencies import require_api_key
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def as_admin():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-admin", "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def as_plain_user():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "test-user", "role": "user",
    }
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def two_docs():
    pid = f"proj_sup_{uuid.uuid4().hex[:8]}"
    store.create_project("supersede fixture", user_id="system", project_id=pid)
    old = store.add_document(pid, "minutes_mislabelled.pdf", size=1)
    new = store.add_document(pid, "minutes_corrected.pdf", size=1)
    yield pid, old["id"], new["id"]
    store.archive_project(pid)


def test_the_old_document_is_hidden_and_points_at_its_replacement(
    client, as_admin, two_docs
):
    _pid, old_id, new_id = two_docs

    resp = client.post(f"/v1/admin/documents/{old_id}/supersede?new_id={new_id}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "superseded"
    assert body["retrieval_visible"] is False
    assert body["superseded_by"] == new_id

    old = store.get_document(old_id)
    assert old is not None, "supersede must never delete the row"
    assert old["retrieval_visible"] is False
    assert old["superseded_by"] == new_id


def test_the_replacement_is_left_visible(client, as_admin, two_docs):
    _pid, old_id, new_id = two_docs
    client.post(f"/v1/admin/documents/{old_id}/supersede?new_id={new_id}")

    assert store.get_document(new_id)["retrieval_visible"] is True


def test_a_plain_user_cannot_retire_a_document(client, as_plain_user, two_docs):
    _pid, old_id, new_id = two_docs

    resp = client.post(f"/v1/admin/documents/{old_id}/supersede?new_id={new_id}")

    assert resp.status_code in (401, 403), resp.text
    assert store.get_document(old_id)["retrieval_visible"] is True


def test_a_document_cannot_be_retired_in_favour_of_another_projects(
    client, as_admin, two_docs
):
    """The guard rail that matters: without it this endpoint could make one
    project's document stand in for another's."""
    _pid, old_id, _new_id = two_docs
    other = f"proj_sup_other_{uuid.uuid4().hex[:8]}"
    store.create_project("other project", user_id="system", project_id=other)
    foreign = store.add_document(other, "foreign.pdf", size=1)
    try:
        resp = client.post(
            f"/v1/admin/documents/{old_id}/supersede?new_id={foreign['id']}"
        )
        assert resp.status_code == 400, resp.text
        assert "different projects" in resp.text
        assert store.get_document(old_id)["retrieval_visible"] is True
    finally:
        store.archive_project(other)


def test_a_document_cannot_supersede_itself(client, as_admin, two_docs):
    _pid, old_id, _new_id = two_docs
    resp = client.post(f"/v1/admin/documents/{old_id}/supersede?new_id={old_id}")

    assert resp.status_code == 400, resp.text
    assert store.get_document(old_id)["retrieval_visible"] is True


@pytest.mark.parametrize("which", ["old", "new"])
def test_an_unknown_document_is_404_and_changes_nothing(
    client, as_admin, two_docs, which
):
    _pid, old_id, new_id = two_docs
    ghost = "nope0000"
    url = (f"/v1/admin/documents/{ghost}/supersede?new_id={new_id}"
           if which == "old"
           else f"/v1/admin/documents/{old_id}/supersede?new_id={ghost}")

    assert client.post(url).status_code == 404
    assert store.get_document(old_id)["retrieval_visible"] is True


def test_superseding_twice_is_harmless(client, as_admin, two_docs):
    _pid, old_id, new_id = two_docs
    first = client.post(f"/v1/admin/documents/{old_id}/supersede?new_id={new_id}")
    second = client.post(f"/v1/admin/documents/{old_id}/supersede?new_id={new_id}")

    assert first.status_code == second.status_code == 200
    assert store.get_document(old_id)["superseded_by"] == new_id
