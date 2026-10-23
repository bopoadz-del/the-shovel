"""Retrieval must not report a corpus-wide absence it cannot see.

THE INCIDENT THIS REPRODUCES (2026-08-17)
-----------------------------------------
Turn 5, the assistant quoted from the project's structural general notes:

    "SBC 304 — Saudi Building Code, Concrete Structural"  (doc cd44bdd0, chunk 285)

Turn 8, asked "Saudi buiding code", it retrieved five unrelated excerpts
(cable ladders, LV single-line diagrams, fire alarm, water pipelines, road
alignment) and answered:

    "The five chunks supplied in this turn's reference context do not mention
     the Saudi Building Code at all."

Two independent defects, both pinned here:

* RECALL — the phrase is three ordinary words, so ``extract_query_identifiers``
  extracted nothing and no lexical pass ran at all; the turn was pure cosine
  over a 227-document corpus at k=5, and the user's typo ("buiding") degraded
  the lexical half of the hybrid search too.
* OVERCLAIM — "not in these five excerpts" was reported to the user as a fact
  about the corpus. Retrieved excerpts are evidence of presence, never of
  absence.

Why the existing e2e missed it: ``test_f3_grounded_citation`` retrieves k=5
over a TWO-chunk corpus. Retrieval over two chunks that returns five cannot
miss anything, so no recall defect is expressible in it.

These tests use the deterministic fake embedder, whose vectors are hashes and
therefore carry no semantic similarity at all. That is the point: it reproduces
a cosine miss exactly, so the only thing that can recover the document is the
lexical rescue.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest

from app.core.rag import vector_store as vs
from app.core.rag.inject import format_chunks_as_system_message

PROJECT = "acme_infra_pack_1"

# The chunk the assistant had already quoted from, two turns earlier.
SBC_CHUNK = (
    "STRUCTURAL GENERAL NOTES - PUMP STATIONS. Design codes and standards: "
    "SBC 304 Saudi Building Code, Concrete Structural. ACI 318-14 Building "
    "Code Requirements for Structural Concrete and Commentary. IBC-2009 "
    "International Building Code. ACI 350 Code Requirements for Environmental "
    "Engineering Concrete Structures. SBC 305 for concrete masonry blocks."
)

# The kinds of excerpt the failing turn retrieved INSTEAD. The corpus is built
# at a realistic size (60+ documents competing for 5 slots) on purpose: at
# 2-6 documents a k=5 retrieval returns essentially everything, so a recall
# defect cannot be expressed at all — which is exactly why the pre-existing e2e
# passed all the way through this incident. Every noise chunk also contains the
# word "code", so no single common term can isolate the target either.
NOISE_TOPICS = [
    "cable ladders hot-dip galvanised supported at 1.5 m centres",
    "LV single line diagram, distribution board 400 A with 36 kA fault level",
    "fire alarm addressable detectors, sounder loop with end of line resistor",
    "water pipeline ductile iron DN 300 class K9 with thrust blocks at bends",
    "road alignment table: chainage, easting, northing and design level",
    "drainage manhole invert levels and cover slab reinforcement",
    "earthworks compaction to modified proctor density in layers",
    "asphalt wearing course binder and aggregate grading envelope",
    "street lighting poles, luminaires and photocell control circuit",
    "telecom duct bank, chambers and draw pit construction details",
]
NOISE_DOC_COUNT = 60


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", "")
    vs.reset_store_cache()

    from app.core.rag.embeddings import get_embedder, reset_embedder_cache

    reset_embedder_cache()
    emb = get_embedder()
    store = vs.get_store(dim=emb.dim)

    store.upsert_chunks(
        project_id=PROJECT,
        doc_id="structural-general-notes.pdf",
        chunks=[SBC_CHUNK],
        embeddings=emb.encode([SBC_CHUNK]),
    )
    for i in range(NOISE_DOC_COUNT):
        text = (
            f"DRAWING NOTE {i}. {NOISE_TOPICS[i % len(NOISE_TOPICS)]}. All works "
            f"shall comply with the applicable code and specification clause {i}."
        )
        store.upsert_chunks(
            project_id=PROJECT, doc_id=f"noise-{i}.pdf", chunks=[text],
            embeddings=emb.encode([text]),
        )
    return store


def _retrieve(question, k=5):
    from app.core.rag.retriever import retrieve_with_filter

    chunks, _noise = retrieve_with_filter(question, PROJECT, k=k)
    return chunks


# ── recall ──────────────────────────────────────────────────────────────────

def test_the_document_that_names_the_code_is_retrieved_despite_the_typo(corpus):
    """The verbatim failing query. 'buiding' is misspelt; the other two terms
    still co-occur in exactly one document, and that is enough."""
    chunks = _retrieve("Saudi buiding code")

    docs = [c.doc_id for c in chunks]
    assert "structural-general-notes.pdf" in docs, (
        f"the document containing 'Saudi Building Code' was not retrieved; "
        f"got {docs} — this is the live miss that produced 'the corpus does "
        f"not mention the Saudi Building Code at all'"
    )


def test_correctly_spelled_query_also_finds_it(corpus):
    chunks = _retrieve("what does the Saudi Building Code require")
    assert "structural-general-notes.pdf" in [c.doc_id for c in chunks]


def test_a_single_common_term_does_not_drag_in_the_whole_corpus(corpus):
    """The rescue matches on term CO-OCCURRENCE. A query sharing one ordinary
    word with every document must not return everything — that would trade a
    recall bug for a precision bug."""
    chunks = _retrieve("code")
    assert len(chunks) <= 5


def test_an_unrelated_question_still_reaches_its_own_document(corpus):
    """The rescue must not distort an ordinary lookup: a question about the
    pipeline drawing still returns a pipeline chunk, not the structural notes."""
    chunks = _retrieve("ductile iron pipe DN 300 class K9 thrust blocks")
    assert chunks, "an ordinary lookup returned nothing at all"
    assert any("ductile iron" in (c.text or "").lower() for c in chunks), (
        f"retrieved {[c.doc_id for c in chunks]} — none is the pipeline note"
    )


# ── absence scope ───────────────────────────────────────────────────────────

def test_the_context_forbids_claiming_corpus_wide_absence(corpus):
    """The load-bearing prompt contract. The model may report what it did not
    find in the excerpts; it may NOT report what the project does not contain,
    because a top-K sample cannot support that claim."""
    chunks = _retrieve("Saudi buiding code")
    content = format_chunks_as_system_message(chunks, len(chunks))["content"].lower()

    assert "not in the retrieved excerpts" in content, (
        "the context no longer scopes absence to the retrieved excerpts"
    )
    for forbidden in ("not mentioned anywhere", "does not apply"):
        assert forbidden in content, (
            f"the context no longer names {forbidden!r} as a forbidden claim"
        )
    assert "sample" in content, (
        "the context no longer tells the model it is seeing a SAMPLE of the "
        "corpus rather than the whole of it"
    )


def test_the_no_fabrication_contract_still_holds(corpus):
    """Pre-existing guarantee — the new absence rule must not displace it."""
    chunks = _retrieve("Saudi buiding code")
    content = format_chunks_as_system_message(chunks, len(chunks))["content"].lower()
    assert "not in this context" in content or "don't have it" in content


def test_an_empty_retrieval_still_emits_no_context_at_all(corpus):
    """Nothing retrieved must mean nothing cited — including no instruction
    block that could read as evidence."""
    message = format_chunks_as_system_message([], total_candidates=0)
    assert message["content"] == ""
