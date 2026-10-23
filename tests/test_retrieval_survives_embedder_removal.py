"""Deleting the embedder must not take search down with it.

THE ARCHITECTURAL CONTRACT
--------------------------
This platform is built out of blocks precisely so a component can be removed
without damaging the rest. Retrieval violated that: the embedding model was a
load-bearing dependency of the ENTIRE search surface.

``retrieve_with_filter`` opened with ``if not available(): return [], 0``, and
``get_store`` constructed an embedder just to read the table width. So with the
model absent, ``search_project_documents`` returned ZERO hits over a corpus
whose text was fully indexed and sitting in the chunks table — matchable by
BM25, which needs no embedder at all. Losing the model lost keyword search too,
for no technical reason.

What the embedder is actually needed for is narrow:

  * the VECTOR leg of hybrid ranking, and
  * WRITING new chunks (the embedding column is NOT NULL).

Reading existing chunks lexically needs none of it. These tests hold that line:
remove the model and semantic ranking degrades, while keyword retrieval over an
already-indexed corpus keeps working.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest

from app.core import projects as projects_mod


@pytest.fixture
def indexed_corpus(tmp_path, monkeypatch):
    """A project indexed normally (embedder present), as production already is."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")

    from app.core import doc_index, file_crypto, users as users_store
    from app.core.rag import vector_store as vs

    vs.reset_store_cache()
    projects_mod.init_db()
    users_store.init_db()
    users_store.ensure_user_exists("u1", email="u1@test.local")

    proj = projects_mod.create_project("Embedder Removal", user_id="u1")
    body = (
        b"SECTION 03300 CAST-IN-PLACE CONCRETE. Concrete for suspended slabs "
        b"shall be grade C40/50 with 25 mm nominal cover to reinforcement."
    )
    path = str(tmp_path / "spec.txt")
    file_crypto.write_document(path, body)
    doc = projects_mod.add_document(
        proj["id"], "spec.txt", file_path=path, size=len(body),
    )
    result = doc_index.index_document(proj["id"], doc["id"])
    assert result["rag_indexed"] > 0, "fixture failed to index with an embedder"
    return {"pid": proj["id"], "doc_id": doc["id"]}


@pytest.fixture
def embedder_deleted(monkeypatch):
    """Simulate the embedding model being GONE — no backend importable.

    Patches the import probes rather than ``available()`` itself, so the code
    under test reaches its real decision the same way production would.
    """
    from app.core.rag import embeddings

    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "some-model-that-is-not-installed")
    monkeypatch.setattr(embeddings, "_has_sentence_transformers", lambda: False)
    monkeypatch.setattr(embeddings, "_has_model2vec", lambda: False)
    embeddings.reset_embedder_cache()

    from app.core.rag import retriever as rag

    assert rag.available() is False, "fixture did not actually remove the embedder"
    return rag


def test_keyword_retrieval_survives_the_embedder_being_removed(
    indexed_corpus, embedder_deleted
):
    """The load-bearing assertion. Previously returned [] unconditionally."""
    chunks, _noise = embedder_deleted.retrieve_with_filter(
        "concrete grade suspended slabs", indexed_corpus["pid"], k=5,
    )

    assert chunks, (
        "removing the embedder returned zero chunks over a corpus whose text is "
        "fully indexed — search is still coupled to the embedding model"
    )
    assert any("C40/50" in c.text for c in chunks)


@pytest.mark.asyncio
async def test_the_search_tool_still_finds_the_document(
    indexed_corpus, embedder_deleted
):
    """End-to-end through the tool the agent actually calls."""
    from app.core import doc_index

    hits = await doc_index.search_project_documents(
        indexed_corpus["pid"], "concrete grade suspended slabs",
    )
    assert hits, "search_project_documents went to zero hits without an embedder"
    assert hits[0]["document_id"] == indexed_corpus["doc_id"]


def test_a_nonsense_query_still_returns_nothing(indexed_corpus, embedder_deleted):
    """Degrading must not mean matching everything — precision has to survive
    too, or the fallback is worse than the failure."""
    chunks, _noise = embedder_deleted.retrieve_with_filter(
        "helicopter avionics certification", indexed_corpus["pid"], k=5,
    )
    assert not chunks


def test_an_empty_query_is_still_empty(indexed_corpus, embedder_deleted):
    assert embedder_deleted.retrieve_with_filter("", indexed_corpus["pid"], k=5) == ([], 0)


def test_the_fallback_never_raises_when_the_store_is_unreachable(
    embedder_deleted, monkeypatch
):
    """This IS the degradation path; it must not become a new way to fail."""
    from app.core.rag import retriever as rag

    def _boom(*_a, **_kw):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(rag, "get_lexical_store", _boom)
    assert rag.retrieve_with_filter("anything", "some_project", k=5) == ([], 0)


def test_indexing_reports_that_it_could_not_embed(indexed_corpus, embedder_deleted):
    """Writing new chunks genuinely DOES need the embedder (the embedding column
    is NOT NULL). That limit must be reported, not hidden — an unembeddable
    document is keyword-searchable only once something else indexes its text."""
    from app.core import doc_index

    result = doc_index.index_document(
        indexed_corpus["pid"], indexed_corpus["doc_id"],
    )
    assert result.get("rag_error") == "embedding stack unavailable"


def test_a_lexical_store_opens_without_constructing_an_embedder(
    indexed_corpus, embedder_deleted
):
    """The specific coupling that caused the outage: get_store() built an
    embedder purely to read the table width."""
    from app.core.rag.vector_store import get_lexical_store

    store = get_lexical_store()
    assert store.count(indexed_corpus["pid"]) > 0
