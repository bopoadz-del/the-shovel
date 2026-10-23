"""Uploading to the Master Corpus must work.

THE BUG THIS PINS
-----------------
``MASTER_CORPUS_PROJECT_ID`` is a VIRTUAL alias. It is injected into listings on
the fly and deliberately has NO row in ``projects`` (see
``_is_master_corpus_alias``). Every reader resolves it to the backing corpus —
``get_project``, ``list_documents``, ``count_documents`` all open with
``_master_corpus_source(project_id) or project_id``.

``add_document`` did not, and it is a WRITER. So an upload to the Master Corpus:

  1. passed the permission check — the READER resolved the alias and found the
     backing corpus, so the caller was correctly authorised; then
  2. tried to insert a Document whose ``project_id`` foreign key pointed at an
     id with no project behind it:

         sqlite3.IntegrityError: FOREIGN KEY constraint failed

That exception was unhandled, so the request died without producing a response
— and a request that dies without a response is reported by the browser as
``TypeError: Failed to fetch``, with no status, because there is no response to
read a status from.

So uploading to the Master Corpus was impossible, and reported itself in the
least diagnosable way available: a bare network error, indistinguishable from
the server being unreachable, on a project the UI presents as first-class and
offers an upload button for.
"""
from __future__ import annotations

import io
import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest
from fastapi.testclient import TestClient

from app.core import projects as store
from app.core import users as users_store
from app.dependencies import require_user
from app.main import app


@pytest.fixture
def master_corpus(tmp_path, monkeypatch):
    """The production shape: a backing corpus owned by SOMEONE ELSE, surfaced
    to every user through the alias."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)

    store.init_db()
    users_store.init_db()
    users_store.ensure_user_exists("corpus_owner", email="owner@test.local")
    users_store.ensure_user_exists("pilot_user", email="pilot@test.local")
    store.create_project(
        "Drive Corpus", user_id="corpus_owner",
        project_id=store.MASTER_CORPUS_SOURCE_PROJECT_ID,
    )

    app.dependency_overrides[require_user] = lambda: {
        "user_id": "pilot_user", "role": "user",
    }
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _upload(client, project_id: str, name: str = "spec.txt", body: bytes = b"C40/50"):
    return client.post(
        f"/v1/projects/{project_id}/documents",
        files=[("file", (name, io.BytesIO(body), "text/plain"))],
    )


def test_uploading_to_the_master_corpus_succeeds(master_corpus):
    """Previously raised FOREIGN KEY constraint failed and killed the request."""
    response = _upload(master_corpus, store.MASTER_CORPUS_PROJECT_ID)

    assert response.status_code == 201, (
        f"upload to the Master Corpus failed: {response.status_code} "
        f"{response.text[:300]}"
    )


def test_the_uploaded_document_appears_in_the_master_corpus(master_corpus):
    """Written under the alias it would be invisible: readers resolve to the
    backing corpus, so a row stored under the alias is in neither view."""
    _upload(master_corpus, store.MASTER_CORPUS_PROJECT_ID, name="uploaded.txt")

    names = [
        d["original_name"]
        for d in store.list_documents(store.MASTER_CORPUS_PROJECT_ID)
    ]
    assert "uploaded.txt" in names


def test_the_document_is_stored_under_the_backing_corpus_not_the_alias(
    master_corpus,
):
    """The invariant behind the fix: writes go to the real project row."""
    _upload(master_corpus, store.MASTER_CORPUS_PROJECT_ID, name="backing.txt")

    backing = [
        d["original_name"]
        for d in store.list_documents(store.MASTER_CORPUS_SOURCE_PROJECT_ID)
    ]
    assert "backing.txt" in backing


def test_the_recorded_size_is_the_real_byte_count(master_corpus):
    """A Master-Corpus upload must not reintroduce the 0 B corpus."""
    body = b"concrete grade C40/50 for suspended slabs"
    _upload(master_corpus, store.MASTER_CORPUS_PROJECT_ID, name="sized.txt", body=body)

    doc = next(
        d for d in store.list_documents(store.MASTER_CORPUS_PROJECT_ID)
        if d["original_name"] == "sized.txt"
    )
    assert doc["size"] == len(body)


def test_storage_project_id_resolves_only_the_alias():
    """Ordinary projects must pass through untouched."""
    assert (
        store.storage_project_id(store.MASTER_CORPUS_PROJECT_ID)
        == store.MASTER_CORPUS_SOURCE_PROJECT_ID
    )
    assert store.storage_project_id("some_ordinary_project") == "some_ordinary_project"


def test_an_ordinary_project_upload_is_unaffected(master_corpus):
    """The fix must not change the normal path."""
    proj = store.create_project("Ordinary", user_id="pilot_user")
    response = _upload(master_corpus, proj["id"], name="normal.txt")

    assert response.status_code == 201
    names = [d["original_name"] for d in store.list_documents(proj["id"])]
    assert "normal.txt" in names
