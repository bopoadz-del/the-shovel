"""Every document the retriever can CITE must be one the model can OPEN.

THE DEFECT THIS PINS
--------------------
Retrieval merges three corpora into a turn's context: the active project, the
configured general-knowledge projects, and (for a thin project) the
Master-Corpus fallback. Each contributes chunks carrying a ``[doc_id=...]``
marker the model is told to cite.

``fetch_document`` resolved ids against ``list_documents(project_id)`` — the
ACTIVE project only. So the model could be handed an excerpt, cite it, then be
told the document does not exist when it tried to read it:

    "The fetch_document call failed because ksa_saudi_building_code.md is a
     cross-project general-knowledge reference, not a file stored in this
     active project's document register."

A citation the model cannot resolve is a citation the USER cannot verify. The
fetch scope has to equal the retrieval scope, or the provenance the product
sells is decorative.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest

from app.agents.runtime import _fetch_document_content
from app.core import projects as store

ACTIVE = "active_project_fetch_scope"
GK = "gk_corpus_fetch_scope"


@pytest.fixture
def corpora(monkeypatch, tmp_path):
    """One document in the active project, one in a general-knowledge corpus
    that retrieval is configured to merge in."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", "")

    from app.core import users as users_store

    store.init_db()
    users_store.init_db()
    users_store.ensure_user_exists("u1", email="u1@test.local")
    users_store.ensure_user_exists("system", email="system@test.local")
    store.create_project("Active Project", user_id="u1", project_id=ACTIVE)
    store.create_project("GK Corpus", user_id="system", project_id=GK)

    own_path = tmp_path / "project_spec.txt"
    own_path.write_text("Concrete grade C40/50 for suspended slabs.")
    own = store.add_document(
        project_id=ACTIVE, original_name="project_spec.txt",
        stored_as="project_spec.txt", file_path=str(own_path),
    )

    gk_path = tmp_path / "ksa_saudi_building_code.md"
    gk_path.write_text(
        "# Saudi Building Code\nSBC 304 covers concrete structures."
    )
    gk = store.add_document(
        project_id=GK, original_name="ksa_saudi_building_code.md",
        stored_as="ksa_saudi_building_code.md", file_path=str(gk_path),
    )
    return {"own": own, "gk": gk}


def test_a_general_knowledge_document_is_fetchable_by_id(corpora):
    """The exact live failure: a GK reference the retriever can cite, that
    fetch_document reported as absent."""
    content, doc, err = _fetch_document_content(ACTIVE, corpora["gk"]["id"], "")

    assert err is None, f"a citable general-knowledge document was unreadable: {err}"
    assert doc["original_name"] == "ksa_saudi_building_code.md"
    assert "SBC 304" in content["text"]


def test_a_general_knowledge_document_is_fetchable_by_name(corpora):
    content, doc, err = _fetch_document_content(
        ACTIVE, "", "ksa_saudi_building_code.md",
    )
    assert err is None, err
    assert "Saudi Building Code" in content["text"]


def test_the_active_project_still_resolves_first(corpora):
    """Widening the scope must not change which document an ordinary lookup
    finds."""
    content, doc, err = _fetch_document_content(ACTIVE, corpora["own"]["id"], "")
    assert err is None, err
    assert doc["original_name"] == "project_spec.txt"
    assert "C40/50" in content["text"]


def test_a_genuinely_absent_document_still_errors_honestly(corpora):
    """The scope widened; it did not become permissive. An id in no corpus
    must still fail rather than resolve to something approximate."""
    _content, _doc, err = _fetch_document_content(ACTIVE, "no-such-id", "")
    assert err and "no document" in err.lower()


def test_scope_matches_retrieval_scope(corpora):
    """The invariant, stated directly: anything the retriever may merge into
    context must be reachable by the fetch tool."""
    from app.core.rag import retriever as rag

    assert GK in rag._general_knowledge_project_ids()
    _content, _doc, err = _fetch_document_content(ACTIVE, corpora["gk"]["id"], "")
    assert err is None


def test_general_knowledge_lookup_failure_does_not_break_the_tool(
    corpora, monkeypatch
):
    """If the GK corpora cannot be resolved, the active project must still
    work — a degraded reference shelf is not a reason to fail a normal read."""
    def _boom():
        raise RuntimeError("gk config unavailable")

    monkeypatch.setattr(
        "app.core.rag.retriever._general_knowledge_project_ids", _boom,
    )
    content, doc, err = _fetch_document_content(ACTIVE, corpora["own"]["id"], "")
    assert err is None, err
    assert doc["original_name"] == "project_spec.txt"
