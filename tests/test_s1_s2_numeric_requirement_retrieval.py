"""S1/S2 numeric requirements must come back from Other Documents.

Live SET5 on 7c0b255, fresh chats, project Master Corpus, 0/6:

* S1 — minimum concrete cover to reinforcement for foundations. Retrieval
  returned only Vol 2 Specification chunks (cross-reference "specified
  minimum concrete cover" / short cover as a rejection reason, Table 3-1
  C32/40). The durability sentence that states the cover in millimetres
  sits in Vol 5 Other Documents and never entered the retrieved set.
* S2 — compaction required under road pavement. Retrieval returned only
  Vol 2 Specification chunks ("all soils shall be properly compacted",
  BS 1377). The sub-grade sentence (percent of maximum dry density and a
  CBR figure) sits in Vol 5 Other Documents and never entered the set.

These tests run ``retrieve_with_filter`` on a real store. The ranking is
not mocked. Fixture ids and filenames are prefixed ``FIXTURE-e-20260925-``.
Synthetic text only.
"""
from __future__ import annotations

import pytest

PID = "FIXTURE-e-20260925-project"

S1_ASK = (
    "Per the project specification, what is the minimum concrete cover "
    "to reinforcement for foundations?"
)
S2_ASK = (
    "Per the project specification, what compaction is required under "
    "road pavement?"
)
BACKFILL_ASK = (
    "Per the project specification, to what degree must structural "
    "backfill under foundations be compacted?"
)

VOL5_COVER = "FIXTURE-e-20260925-vol5-other-documents-4"
VOL5_COMPACT = "FIXTURE-e-20260925-vol5-other-documents-2"
SPEC_EARTH = "FIXTURE-e-20260925-particular-specification-earthworks"
MOBILIZATION = "FIXTURE-e-20260925-site-office-mobilization"

# Mirrors the durability sentence. The figure is written the way the
# corpus writes it (digits glued to the unit). Not a product-code constant.
COVER_TEXT = (
    "Referring to Table A.3 of BS 8500-1:2006 the nominal cover should be "
    "50mm for foundations cast against blinding and 75mm for foundations "
    "casted against soil. Exposure class XC1, sulfate class DS-1, "
    "concrete class RC20/25."
)
# Mirrors the geotech sub-grade sentence: words and digits, and it says
# sub-grade / embankment rather than road pavement.
COMPACT_TEXT = (
    "At least twenty (20) cm of material placed on the embankment to form "
    "the sub-grade layer shall meet the requirement of class A-1-a, A-1-b "
    "or A-2-4(0) when compacted to ninety five percent (95%) of maximum "
    "dry density to get a minimum CBR value of 25."
)
SPEC_COVER_REJECT = (
    "Concrete shall be rejected where the specified minimum concrete cover "
    "to reinforcement is not achieved. Short cover is a cause for rejection "
    "of the foundation element and the concrete shall be broken out."
)
SPEC_TABLE = (
    "Table 3-1 Concrete for foundations. Grade C32/40(F). Characteristic "
    "compressive strength 40 MPa. Minimum cement content and maximum "
    "water/cement ratio shall be as specified for the exposure class."
)
SPEC_COMPACT = (
    "All soils shall be properly compacted, and where required a compaction "
    "test shall be carried out in accordance with BS 1377 Part 9. The "
    "method of compaction shall be approved by the Engineer."
)
SPEC_98_TEXT = (
    "Compaction of the backfill to minimum 98% of maximum dry density of "
    "the modified proctor test under foundations."
)
MOB_95_TEXT = (
    "Site office platform density tests should not be less than 95% "
    "Maximum Dry Density."
)

NAMES = {
    VOL5_COVER: "FIXTURE-e-20260925-vol5-other-documents-4-of-5.pdf",
    VOL5_COMPACT: "FIXTURE-e-20260925-vol5-other-documents-2-of-5.pdf",
    "FIXTURE-e-20260925-vol2-specification-4":
        "FIXTURE-e-20260925-vol2-specification-4-of-9.pdf",
    "FIXTURE-e-20260925-vol2-specification-2":
        "FIXTURE-e-20260925-vol2-specification-2-of-9.pdf",
    "FIXTURE-e-20260925-vol2-specification-3":
        "FIXTURE-e-20260925-vol2-specification-3-of-9.pdf",
    "FIXTURE-e-20260925-vol2-specification-8":
        "FIXTURE-e-20260925-vol2-specification-8-of-9.pdf",
    SPEC_EARTH: "FIXTURE-e-20260925-particular-specification-earthworks.pdf",
    MOBILIZATION: "FIXTURE-e-20260925-site-office-mobilization.pdf",
}

TEXTS = {
    VOL5_COVER: [COVER_TEXT],
    VOL5_COMPACT: [COMPACT_TEXT],
    "FIXTURE-e-20260925-vol2-specification-4": [SPEC_COVER_REJECT, SPEC_TABLE],
    "FIXTURE-e-20260925-vol2-specification-2": [SPEC_COMPACT],
    "FIXTURE-e-20260925-vol2-specification-3": [
        SPEC_COMPACT + " Clause 3.2 earthworks pavement formation.",
    ],
    "FIXTURE-e-20260925-vol2-specification-8": [
        SPEC_COMPACT + " Roads and hardstanding.",
    ],
    SPEC_EARTH: [SPEC_98_TEXT],
    MOBILIZATION: [MOB_95_TEXT],
}

# Enough Vol 2 Specification neighbours that the durability chunk and the
# sub-grade chunk fall outside the hybrid lexical fetch. Each pad repeats
# the query's own words and states no figure.
for _i in range(55):
    _doc = f"FIXTURE-e-20260925-vol2-specification-pad-{_i}"
    NAMES[_doc] = f"FIXTURE-e-20260925-vol2-specification-pad-{_i}.pdf"
    TEXTS[_doc] = [
        (
            f"Section {_i}. The specified minimum concrete cover to "
            f"reinforcement for foundations and walls shall be inspected. "
            f"Short cover is grounds for rejection. Concrete grade and "
            f"reinforcement placing shall follow the specification. "
            f"Item {_i} of the project specification."
        ),
        (
            f"Section {_i} roads. All soils shall be properly compacted "
            f"under road pavement. Compaction testing to BS 1377. The "
            f"pavement formation shall be approved. Item {_i}."
        ),
    ]


def _fmt(chunks) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"doc={c.doc_id} chunk={c.chunk_id} score={c.score} "
            f"name={getattr(c, 'source_name', '')!r} text={(c.text or '')[:90]!r}"
        )
    return "\n".join(lines) or "(empty)"


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    from app.core.rag import embeddings as emb
    from app.core.rag import vector_store as vs

    emb.reset_embedder_cache()
    vs.reset_store_cache()
    from app.core.rag import retriever as ret
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store

    embedder = Embedder(model_name="fake")
    store = get_store(dim=embedder.dim)
    for doc_id, texts in TEXTS.items():
        store.upsert_chunks(PID, doc_id, texts, embedder.encode(texts))
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda doc_id: NAMES.get(doc_id, ""))
    yield ret
    emb.reset_embedder_cache()
    vs.reset_store_cache()


def test_s1_cover_figure_is_in_the_retrieved_set(corpus):
    """The millimetre cover sentence must be retrieved, not only the
    specification chunks that mention cover as a rejection reason."""
    chunks, _noise = corpus.retrieve_with_filter(S1_ASK, PID, k=5)
    ids = [c.doc_id for c in chunks]
    assert VOL5_COVER in ids, (
        "Vol 5 Other Documents cover chunk is missing from the retrieved set:\n"
        + _fmt(chunks)
    )
    hit = next(c for c in chunks if c.doc_id == VOL5_COVER)
    assert "75mm" in (hit.text or "").replace(" ", "")


def test_s2_subgrade_compaction_figure_is_in_the_retrieved_set(corpus):
    """The sub-grade maximum-dry-density / CBR sentence must be retrieved,
    not only the specification chunks that say properly compacted."""
    chunks, _noise = corpus.retrieve_with_filter(S2_ASK, PID, k=5)
    ids = [c.doc_id for c in chunks]
    assert VOL5_COMPACT in ids, (
        "Vol 5 Other Documents sub-grade chunk is missing from the retrieved set:\n"
        + _fmt(chunks)
    )
    hit = next(c for c in chunks if c.doc_id == VOL5_COMPACT)
    text = hit.text or ""
    assert "95%" in text
    assert "CBR" in text
    # Foundation-backfill's own percent is a different element. It must not
    # take the slot the sub-grade sentence just earned.
    if SPEC_EARTH in ids:
        spec = next(c for c in chunks if c.doc_id == SPEC_EARTH)
        assert (hit.score or 0) > (spec.score or 0), _fmt(chunks)


def test_kill_switch_restores_specification_only_retrieval(corpus, monkeypatch):
    """Flag off is the old path: the figure-bearing chunks stay out."""
    monkeypatch.setenv("RETRIEVAL_NUMERIC_REQUIREMENT_BOOST", "0")
    cover, _ = corpus.retrieve_with_filter(S1_ASK, PID, k=5)
    compact, _ = corpus.retrieve_with_filter(S2_ASK, PID, k=5)
    assert VOL5_COVER not in [c.doc_id for c in cover], _fmt(cover)
    assert VOL5_COMPACT not in [c.doc_id for c in compact], _fmt(compact)
    assert cover, "flag off must still return the specification chunks"
    assert compact


def test_an_ordinary_question_is_not_a_quantity_ask():
    from app.core.rag.retriever import asked_quantity_kinds

    assert asked_quantity_kinds(
        "What is the Defects Notification Period under this contract?"
    ) == frozenset()
    assert asked_quantity_kinds("Who signed the cover letter?") == frozenset()
    assert "length_mm" in asked_quantity_kinds(S1_ASK)
    assert "compaction" in asked_quantity_kinds(S2_ASK)
    assert "compaction" in asked_quantity_kinds(BACKFILL_ASK)


def test_a_qualitative_clause_is_not_a_stated_figure():
    from app.core.rag.retriever import chunk_states_asked_quantity

    assert chunk_states_asked_quantity(SPEC_COVER_REJECT, frozenset({"length_mm"})) is False
    assert chunk_states_asked_quantity(COVER_TEXT, frozenset({"length_mm"})) is True
    assert chunk_states_asked_quantity(SPEC_COMPACT, frozenset({"compaction"})) is False
    assert chunk_states_asked_quantity(COMPACT_TEXT, frozenset({"compaction"})) is True
    assert chunk_states_asked_quantity(SPEC_98_TEXT, frozenset({"compaction"})) is True
    from app.core.rag.retriever import compaction_subject_agrees

    assert compaction_subject_agrees(S2_ASK, COMPACT_TEXT) is True
    assert compaction_subject_agrees(S2_ASK, SPEC_98_TEXT) is False
    assert compaction_subject_agrees(BACKFILL_ASK, SPEC_98_TEXT) is True
    assert compaction_subject_agrees(BACKFILL_ASK, COMPACT_TEXT) is False


def test_foundation_backfill_still_retrieves_the_specification_figure(corpus):
    """Another lane: structural backfill under foundations stays on the
    specification's own compaction figure, ahead of a site-office note."""
    chunks, _noise = corpus.retrieve_with_filter(BACKFILL_ASK, PID, k=5)
    ids = [c.doc_id for c in chunks]
    assert SPEC_EARTH in ids, (
        "foundation backfill specification chunk dropped:\n" + _fmt(chunks)
    )
    spec = next(c for c in chunks if c.doc_id == SPEC_EARTH)
    if MOBILIZATION in ids:
        mob = next(c for c in chunks if c.doc_id == MOBILIZATION)
        assert (spec.score or 0) > (mob.score or 0), _fmt(chunks)
    if VOL5_COMPACT in ids:
        other = next(c for c in chunks if c.doc_id == VOL5_COMPACT)
        assert (spec.score or 0) > (other.score or 0), _fmt(chunks)
