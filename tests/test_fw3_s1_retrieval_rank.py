"""S1 pre-answer ranking: the specification-class lift must not bury the cover.

Live probe (d708b5d, POST /v1/rag/search, k=50) for

    Per the project specification, what is the minimum concrete cover
    to reinforcement for foundations?

put Vol 5 Other Documents (4 of 6) chunk 92 at rank 22 (score ~3.26).
Ranks 1–12 were Vol 2 Specification chunks at a flat ~3.70–3.77. Chat
injection (``rag_inject`` → ``retrieve_with_filter``, k=5) never passed
chunk 92 to the model.

The flat gap is the specification filename lift stacked on a cover
detector that treats any "cover" plus any millimetre in the same chunk
as a stated cover length. Bollard bands and raised-floor panels then
outrank the durability sentence ("nominal cover … 50mm … blinding …
75mm … casted against soil").

Signed and unsigned copies of that same sentence also take two slots.

Fixtures are prefixed ``FIXTURE-e-20260926-``. Clause text is a short
mirror of the quoted sentences only. No live ids in the assertions.
"""
from __future__ import annotations

import pytest

PID = "FIXTURE-e-20260926-project"

S1_ASK = (
    "Per the project specification, what is the minimum concrete cover "
    "to reinforcement for foundations?"
)
S2_ASK = (
    "Per the project specification, what compaction is required under "
    "road pavement?"
)
P1A_ASK = (
    "Per the project specification, to what degree must structural "
    "backfill under foundations be compacted?"
)
S4_ASK = "Which contract governs this project?"

# Chunk-92 equivalent. "nominal cover" / "cast against" / "casted against",
# not the question's "minimum cover to reinforcement".
COVER_BODY = (
    "In accordance with BS8500-1-2006, a site exposure class of (XC1) is "
    "considered representative for the concrete elements. Referring to "
    "Table A.3 of the nominal cover should be 50mm for foundations cast "
    "against blinding and 75mm for foundations casted against soil. "
    "Furthermore, the minimum designated concrete is RC20/25."
)
UNSIGNED = "FIXTURE-e-20260926-vol5-other-4-unsigned"
SIGNED = "FIXTURE-e-20260926-vol5-other-4-signed"
UNSIGNED_NAME = (
    "FIXTURE-e-20260926-vol5-other-documents-4-of-5-not-signed.pdf"
)
SIGNED_NAME = (
    "FIXTURE-e-20260926-vol5-other-documents-4-of-5-signed.pdf"
)

# Far enough from "concrete cover" that a proximity check does not treat
# the bollard / panel dimension as the cover length. The loose detector
# still fires: the word cover and a millimetre share the chunk.
_PAD = (
    "Bar crossings shall be secured with tying wire and turned down "
    "into the work. "
) * 4

def _spec_distractor(i: int) -> str:
    return (
        f"Section {i}. Removable bollards shall be painted in alternating "
        f"bands of 200 mm. Raised floor panels shall be 600 mm square. "
        f"{_PAD}"
        f"The specified minimum concrete cover to reinforcement for "
        f"foundations shall be inspected. Short cover is grounds for "
        f"rejection. Item {i} of the project specification."
    )


SPEC_IDS = [f"FIXTURE-e-20260926-vol2-specification-{i}" for i in range(6)]

S2_DOC = "FIXTURE-e-20260926-vol5-other-2"
S2_NAME = (
    "FIXTURE-e-20260926-vol5-other-documents-2-of-5-rsm-15492-rev0.pdf"
)
S2_TEXT = (
    "RSM 15492-Rev0. At least twenty (20) cm of material placed on the "
    "embankment to form the sub-grade layer shall meet class A-1-a when "
    "compacted to ninety five percent (95%) of maximum dry density to get "
    "a minimum CBR value of 25."
)
P1A_DOC = "FIXTURE-e-20260926-particular-specification-earthworks"
P1A_NAME = "FIXTURE-e-20260926-particular-specification-earthworks.pdf"
P1A_TEXT = (
    "Compaction of the backfill to minimum 98% of maximum dry density of "
    "the modified proctor test under foundations."
)
MOB_DOC = "FIXTURE-e-20260926-site-office-mobilization"
MOB_NAME = "FIXTURE-e-20260926-site-office-mobilization.pdf"
MOB_TEXT = (
    "Site office platform density tests should not be less than 95% "
    "Maximum Dry Density."
)
QUAL_SPEC = "FIXTURE-e-20260926-vol2-specification-qualitative"
QUAL_NAME = "FIXTURE-e-20260926-vol2-specification-qualitative.pdf"
QUAL_TEXT = (
    "All soils shall be properly compacted under road pavement. Compaction "
    "testing shall follow BS 1377. The pavement formation shall be approved."
)


def _fmt(chunks) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"doc={c.doc_id} score={c.score} "
            f"name={getattr(c, 'source_name', '')!r} "
            f"text={(c.text or '')[:80]!r}"
        )
    return "\n".join(lines) or "(empty)"


def _cover_copies() -> dict:
    return {
        UNSIGNED: (
            "[source: FIXTURE-e-20260926 Contract docs UNSIGNED "
            "vol5-other-documents-4-of-5.pdf]\n" + COVER_BODY
        ),
        SIGNED: (
            "[source: FIXTURE-e-20260926 Contract docs SIGNED "
            "vol5-other-documents-4-of-5.pdf]\n" + COVER_BODY
        ),
    }


@pytest.fixture
def rank_corpus(tmp_path, monkeypatch):
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

    names = {
        UNSIGNED: UNSIGNED_NAME,
        SIGNED: SIGNED_NAME,
    }
    texts = _cover_copies()
    for i, doc_id in enumerate(SPEC_IDS):
        names[doc_id] = (
            f"FIXTURE-e-20260926-vol2-specification-{i}-of-9.pdf"
        )
        texts[doc_id] = _spec_distractor(i)
    # Fillers so a duplicate copy is what crowds the last slot, not an
    # empty result. No millimetre, so they are not cover figures.
    for i in range(4):
        doc_id = f"FIXTURE-e-20260926-spec-filler-{i}"
        names[doc_id] = f"FIXTURE-e-20260926-vol2-specification-filler-{i}.pdf"
        texts[doc_id] = (
            f"Filler {i}. Concrete for foundations shall follow the "
            f"specification. Reinforcement placing shall be inspected. "
            f"Item {i}."
        )

    embedder = Embedder(model_name="fake")
    store = get_store(dim=embedder.dim)
    for doc_id, text in texts.items():
        store.upsert_chunks(PID, doc_id, [text], embedder.encode([text]))
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda doc_id: names.get(doc_id, ""))
    yield ret
    emb.reset_embedder_cache()
    vs.reset_store_cache()


@pytest.fixture
def keep_corpus(tmp_path, monkeypatch):
    """P1a and S2 clauses that must stay inside the passed top-k."""
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

    names = {
        S2_DOC: S2_NAME,
        P1A_DOC: P1A_NAME,
        MOB_DOC: MOB_NAME,
        QUAL_SPEC: QUAL_NAME,
    }
    texts = {
        S2_DOC: S2_TEXT,
        P1A_DOC: P1A_TEXT,
        MOB_DOC: MOB_TEXT,
        QUAL_SPEC: QUAL_TEXT,
    }
    for i in range(8):
        doc_id = f"FIXTURE-e-20260926-spec-pad-{i}"
        names[doc_id] = f"FIXTURE-e-20260926-vol2-specification-pad-{i}.pdf"
        texts[doc_id] = (
            f"Pad {i}. All soils shall be properly compacted under road "
            f"pavement. The specified minimum concrete cover to reinforcement "
            f"for foundations shall be inspected. Item {i}."
        )
    embedder = Embedder(model_name="fake")
    store = get_store(dim=embedder.dim)
    for doc_id, text in texts.items():
        store.upsert_chunks(PID, doc_id, [text], embedder.encode([text]))
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda doc_id: names.get(doc_id, ""))
    yield ret
    emb.reset_embedder_cache()
    vs.reset_store_cache()


def _passed(ret, ask, k=5):
    chunks, _noise = ret.retrieve_with_filter(ask, PID, k=k)
    return chunks


def test_s1_cover_clause_is_in_the_passed_topk(rank_corpus):
    """The nominal-cover sentence must be one of the k=5 chunks rag_inject
    passes to the model.

    Wide k proves the clause was retrieved. A miss at k=5, with the passed
    slots filled by specification-named chunks, is the filename lift.
    """
    wide = _passed(rank_corpus, S1_ASK, k=20)
    wide_ids = [c.doc_id for c in wide]
    assert UNSIGNED in wide_ids or SIGNED in wide_ids, (
        "cover clause was not retrieved at all (recall), so this is not "
        "the specification-boost miss:\n" + _fmt(wide)
    )
    top = _passed(rank_corpus, S1_ASK, k=5)
    top_ids = [c.doc_id for c in top]
    spec_slots = [
        c for c in top
        if "specification" in (getattr(c, "source_name", "") or "").lower()
    ]
    assert UNSIGNED in top_ids or SIGNED in top_ids, (
        "nominal-cover clause is outside the passed top-5. "
        f"{len(spec_slots)} of those slots are specification-named "
        "(filename lift on chunks that mention cover and an unrelated "
        "millimetre):\n" + _fmt(top)
    )
    hit = next(c for c in top if c.doc_id in (UNSIGNED, SIGNED))
    assert "75mm" in (hit.text or "").replace(" ", "")
    # One clause, one slot. The signed copy must not consume another.
    assert not (UNSIGNED in top_ids and SIGNED in top_ids), (
        "signed and unsigned copies both occupy a passed slot:\n" + _fmt(top)
    )
    # rag_inject is the pre-answer path: these chunks are the system
    # message the model sees. Token cap must not drop the clause.
    from app.core.rag.inject import rag_inject

    msg, audit = rag_inject(
        user_message=S1_ASK,
        project_id=PID,
        conversation_id="ws-FIXTURE-e-20260926-1",
        user_id="fixture",
        agent_name="project-assistant",
    )
    assert msg is not None, audit
    injected = [c["doc_id"] for c in audit.get("chunks") or []]
    assert UNSIGNED in injected or SIGNED in injected, (
        "cover clause was ranked into top-5 but not passed to the model:\n"
        + str(injected)
    )
    assert not (UNSIGNED in injected and SIGNED in injected), injected
    assert "75mm" in msg["content"].replace(" ", "")


def test_signed_and_unsigned_copies_collapse_to_one_slot(rank_corpus):
    """Same clause body, two copies. Only one passed slot.

    Exposure-class wording does not apply the specification filename
    lift, so the copies are the hit. The body key ignores the leading
    ``[source:]`` line, which is where the two copies differ.
    """
    ask = (
        "What nominal cover does exposure class XC1 give for foundations "
        "cast against soil and against blinding?"
    )
    top = _passed(rank_corpus, ask, k=5)
    ids = [c.doc_id for c in top]
    present = [doc for doc in (UNSIGNED, SIGNED) if doc in ids]
    assert len(present) == 1, (
        "signed and unsigned copies should occupy one passed slot:\n"
        + _fmt(top)
    )
    hit = next(c for c in top if c.doc_id == present[0])
    assert "75mm" in (hit.text or "").replace(" ", "")
    bodies = [
        rank_corpus.chunk_copy_key(c.text or "")
        for c in top
        if rank_corpus.chunk_copy_key(c.text or "")
    ]
    assert len(bodies) == len(set(bodies))


def test_p1a_backfill_clause_stays_in_passed_sources(keep_corpus):
    chunks = _passed(keep_corpus, P1A_ASK, k=5)
    ids = [c.doc_id for c in chunks]
    assert P1A_DOC in ids, _fmt(chunks)
    hit = next(c for c in chunks if c.doc_id == P1A_DOC)
    assert "98%" in (hit.text or "")
    assert "modified proctor" in (hit.text or "").lower()
    if MOB_DOC in ids:
        mob = next(c for c in chunks if c.doc_id == MOB_DOC)
        assert (hit.score or 0) > (mob.score or 0), _fmt(chunks)


def test_s2_subgrade_clause_stays_in_passed_sources(keep_corpus):
    chunks = _passed(keep_corpus, S2_ASK, k=5)
    ids = [c.doc_id for c in chunks]
    assert S2_DOC in ids, _fmt(chunks)
    hit = next(c for c in chunks if c.doc_id == S2_DOC)
    text = hit.text or ""
    assert "95%" in text
    assert "CBR" in text
    assert "15492" in text


def test_kill_switch_restores_spec_lift_and_both_copies(rank_corpus, monkeypatch):
    """``RETRIEVAL_SPEC_BOOST_GUARD=0`` is the pre-fix ranking."""
    monkeypatch.setenv("RETRIEVAL_SPEC_BOOST_GUARD", "0")
    top = _passed(rank_corpus, S1_ASK, k=5)
    top_ids = [c.doc_id for c in top]
    assert UNSIGNED not in top_ids and SIGNED not in top_ids, _fmt(top)
    ask = (
        "What nominal cover does exposure class XC1 give for foundations "
        "cast against soil and against blinding?"
    )
    copies = _passed(rank_corpus, ask, k=5)
    ids = [c.doc_id for c in copies]
    assert UNSIGNED in ids and SIGNED in ids, _fmt(copies)


def test_s4_governing_contract_ask_does_not_name_specification(rank_corpus):
    """S4 is not a specification-class question. The filename lift must
    not run. Chunk 1032's PSA text is not in this fixture."""
    assert rank_corpus.source_class_named_by(S4_ASK) == ""
    assert rank_corpus.source_class_named_by(S1_ASK) == "specification"
