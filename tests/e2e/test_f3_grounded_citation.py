"""F3 — ingest a construction document, ask about it, get a REAL citation.

The primary the client capability, and the one that has to be flawless:
a specification goes in, a question comes in, and the answer is traceable to
the document it came from. Not "a plausible answer" — a cited one.

SCOPE, stated so nobody mistakes what is proven here: this drives the
retrieval and citation layer end to end with a deterministic embedder. It
does NOT call an LLM. That is deliberate — a test that needs a live model key
cannot run in CI, and a non-deterministic answer cannot assert on a citation.
What the LLM does with the context is covered by the grounding tests; what
this proves is the part underneath, which is where a broken citation actually
comes from: the chunk must survive ingest, come back for a real question, and
carry a marker that resolves to the source document.

If this passes and a client still gets an uncited answer, the fault is in the
model prompt, not the plumbing — and that is a useful thing to be able to say.
"""
from __future__ import annotations

import re

import pytest

from app.core.rag import vector_store as vs
from app.core.rag.inject import format_chunks_as_system_message


SPEC_TEXT = (
    "SECTION 03300 CAST-IN-PLACE CONCRETE. "
    "Concrete for suspended slabs shall be grade C40/50 with a minimum "
    "cement content of 380 kg/m3. Nominal cover to reinforcement in slabs "
    "shall be 25 mm. Slump at point of placement shall not exceed 120 mm."
)
UNRELATED_TEXT = (
    "SECTION 26050 ELECTRICAL. Cable trays shall be hot-dip galvanised to "
    "BS EN ISO 1461 and supported at maximum 1.5 m centres."
)


@pytest.fixture
def ingested(monkeypatch, tmp_path):
    """A project with one real specification section ingested into it."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    # No general-knowledge merge: this test is about the PROJECT's own
    # document being citable, and a GK hit would muddy what is being proven.
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    vs.reset_store_cache()

    from app.core.rag.embeddings import get_embedder, reset_embedder_cache
    reset_embedder_cache()
    emb = get_embedder()
    store = vs.get_store(dim=emb.dim)

    project_id = "client_tower"
    store.upsert_chunks(
        project_id=project_id,
        doc_id="spec-03300-concrete.pdf",
        chunks=[SPEC_TEXT],
        embeddings=emb.encode([SPEC_TEXT]),
    )
    store.upsert_chunks(
        project_id=project_id,
        doc_id="spec-26050-electrical.pdf",
        chunks=[UNRELATED_TEXT],
        embeddings=emb.encode([UNRELATED_TEXT]),
    )
    return store, emb, project_id


def _retrieve(project_id, question):
    from app.core.rag.retriever import retrieve_with_filter
    chunks, _noise = retrieve_with_filter(question, project_id, k=5)
    return chunks


def test_the_ingested_specification_comes_back_for_a_question_about_it(ingested):
    """Ingest -> retrieve. If this fails, nothing downstream can be cited."""
    _store, _emb, project_id = ingested

    chunks = _retrieve(project_id, "what is the concrete grade for suspended slabs")

    assert chunks, "the ingested specification was not retrievable at all"
    assert any(c.doc_id == "spec-03300-concrete.pdf" for c in chunks), (
        f"retrieved {[c.doc_id for c in chunks]} — not the concrete spec"
    )


def test_the_citation_marker_resolves_to_the_ingested_document(ingested):
    """The load-bearing assertion: the context handed to the model carries a
    doc_id that names a document that was actually ingested.

    A citation that does not resolve is worse than no citation — it reads as
    provenance while being decoration.
    """
    _store, _emb, project_id = ingested

    chunks = _retrieve(project_id, "minimum cement content for suspended slabs")
    system_message = format_chunks_as_system_message(chunks, total_candidates=len(chunks))
    content = system_message["content"]

    cited = set(re.findall(r"\[doc_id=([^\s\]]+)", content))
    assert cited, f"no citation markers emitted:\n{content[:400]}"

    ingested_docs = {"spec-03300-concrete.pdf", "spec-26050-electrical.pdf"}
    dangling = cited - ingested_docs
    assert not dangling, f"cited documents that were never ingested: {dangling}"
    assert "spec-03300-concrete.pdf" in cited


def test_the_cited_chunk_actually_contains_the_answer(ingested):
    """A citation pointing at the right file but the wrong passage is still
    a false citation. The figure the question asks for has to be IN the text
    that was cited."""
    _store, _emb, project_id = ingested

    chunks = _retrieve(project_id, "nominal cover to reinforcement in slabs")
    system_message = format_chunks_as_system_message(chunks, total_candidates=len(chunks))

    assert "25 mm" in system_message["content"], (
        "the retrieved context does not contain the cover figure it was "
        "retrieved for — the model would have to invent it"
    )


def test_the_context_tells_the_model_not_to_invent_missing_details(ingested):
    """The no-fabrication contract is carried IN the context, every time.
    If this instruction goes missing the model is free to fill gaps."""
    _store, _emb, project_id = ingested

    chunks = _retrieve(project_id, "concrete grade")
    content = format_chunks_as_system_message(chunks, len(chunks))["content"]

    lowered = content.lower()
    assert "not in this context" in lowered or "don't have it" in lowered, (
        "the retrieval context no longer instructs the model to decline "
        "rather than invent:\n" + content[:400]
    )


def test_an_empty_retrieval_produces_no_citations_rather_than_a_stub(ingested):
    """Nothing retrieved must mean nothing cited. An empty context that still
    carries a marker would let a fabricated answer look sourced."""
    message = format_chunks_as_system_message([], total_candidates=0)

    assert message["content"] == ""
    assert "doc_id=" not in message["content"]
