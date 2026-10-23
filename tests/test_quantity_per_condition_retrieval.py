"""A figure stated per condition reaches the model in the document's own words.

Asked-quantity recall (``recall_asked_quantity_chunks``) end to end. The
question says "cast directly against earth"; the drawing notes that answer it
say "in contact with the ground" / "in contact with soil" and never repeat the
question's other words. Many chunks that DO repeat the question's words (and
state no figure) crowd both retrieval legs, so neither leg pools the notes on
its own. A note about the negated condition ("not in contact with ground")
states a figure for a different question.

This replaces a soil-contact-only path with its own phrase list and kill
switch. Everything below is invented (project, notes, figures) and runs on the
fake embedder.
"""
from __future__ import annotations

import pytest

from app.core.rag.vector_store import Chunk

PID = "synthetic-condition-project"

ASK = (
    "Per the project specification, what is the minimum concrete cover for "
    "pad foundations cast directly against earth?"
)
ASK_REPHRASED = (
    "According to the specification, what cover to reinforcement is required "
    "where footings are poured against the ground?"
)
BLINDING_ASK = (
    "Per the project specification, what is the minimum concrete cover for "
    "pad foundations cast on blinding?"
)

SPEC = "spec-vol2"
NOTE_GROUND = "dwg-general-notes"
SHEET_SOIL = "dwg-retaining-sheet"
DWG_PADS = "dwg-pad-details"
NOTE_NEGATED = "dwg-slab-notes"
NOTE_BLINDING = "dwg-blinding-notes"

DOCS = {
    SPEC: ("Hillcrest Works Specification Volume 2.pdf", [
        "6.3.9 Fixing of bars. Binding wire shall be 1.2 mm soft iron. Spacers "
        "shall give the cover specified on the Drawings or as the engineer "
        "directs.",
    ]),
    NOTE_GROUND: ("Hillcrest Works Drawings general notes.pdf", [
        "8.1 LAPS TO BE STAGGERED. 8.2 COVER TO REINFORCEMENT: ALL CONCRETE IN "
        "CONTACT WITH THE GROUND 65mm, ELSEWHERE 40mm UNLESS NOTED. 8.3 BAR "
        "MARKS AS SCHEDULED.",
    ]),
    SHEET_SOIL: ("Hillcrest Works sheet 4410 retaining structure.pdf", [
        "COVER TO REINFORCEMENT (MINIMUM): FACES IN CONTACT WITH SOIL 65mm. "
        "OTHER FACES 40mm.",
    ]),
    DWG_PADS: ("Hillcrest Works Drawings pad foundation details.pdf", [
        "PAD NOTES. MINIMUM CONCRETE COVER TO REINFORCEMENT: UNDERSIDE OF PAD "
        "FOUNDATIONS CAST AGAINST EARTH : 80mm.",
    ]),
    NOTE_NEGATED: ("Hillcrest Works Drawings slab notes.pdf", [
        "SLAB NOTES. MINIMUM CONCRETE COVER FOR FACES NOT IN CONTACT WITH "
        "GROUND : 45mm TO ALL FACES.",
    ]),
    NOTE_BLINDING: ("Hillcrest Works Drawings blinding notes.pdf", [
        "BLINDING NOTES. NOMINAL COVER WHERE PADS ARE CAST ON BLINDING : 55mm.",
    ]),
}


def _fillers():
    out = []
    for n in range(1, 81):
        out.append(
            f"Hillcrest Works pour record {n}: the minimum concrete cover for pad "
            f"foundations cast directly against earth per the project "
            f"specification is to be confirmed before pour {n}."
        )
    return out


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    for var in (
        "MASTER_CORPUS_SOURCE_PROJECT_ID", "RAG_K", "MAX_RAG_TOKENS",
        "RETRIEVAL_SPEC_BOOST_GUARD", "RETRIEVAL_NUMERIC_REQUIREMENT_BOOST",
    ):
        monkeypatch.delenv(var, raising=False)
    from app.core.rag import embeddings as emb
    from app.core.rag import vector_store as vs

    emb.reset_embedder_cache()
    vs.reset_store_cache()
    from app.core.rag import retriever as ret
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    embedder = get_embedder()
    store = get_store(dim=embedder.dim)
    names = {}
    for doc_id, (name, texts) in DOCS.items():
        store.upsert_chunks(PID, doc_id, texts, embedder.encode(texts))
        names[doc_id] = name
    fillers = _fillers()
    store.upsert_chunks(PID, "pour-records", fillers, embedder.encode(fillers))
    names["pour-records"] = "Hillcrest Works pour records.pdf"
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda d: names.get(d, ""))
    yield ret
    emb.reset_embedder_cache()
    vs.reset_store_cache()


def _handed(ask):
    from app.core.rag.inject import rag_inject

    msg, audit = rag_inject(
        user_message=ask, project_id=PID, conversation_id="ws-synthetic-condition",
        user_id="fixture", agent_name="project-assistant", history=[],
    )
    return msg, [c["doc_id"] for c in audit.get("chunks") or []]


def _retrieved(ret, ask, k):
    chunks, _ = ret.retrieve_with_filter(ask, PID, k=k, operator_text=ask)
    return [c.doc_id for c in chunks]


@pytest.mark.parametrize("ask", [ASK, ASK_REPHRASED], ids=["asked", "rephrased"])
def test_the_own_words_note_reaches_the_model_beside_the_pointer(corpus, ask):
    msg, handed = _handed(ask)
    assert msg is not None
    assert NOTE_GROUND in handed, f"own-words ground-contact note: {handed}"
    # The fix adds; it does not swap out the source's pointer clause.
    assert SPEC in handed, f"pointer clause pushed out: {handed}"


@pytest.mark.parametrize("ask", [ASK, ASK_REPHRASED], ids=["asked", "rephrased"])
def test_both_own_words_chunks_leave_the_retriever(corpus, ask):
    deeper = _retrieved(corpus, ask, 10)
    for doc in (NOTE_GROUND, SHEET_SOIL):
        assert doc in deeper, f"{doc} not retrieved: {deeper}"


def test_source_scoped_quantity_questions_get_two_more_slots():
    from app.core.rag.inject import rag_retrieval_k

    assert rag_retrieval_k(ASK, 5) == 7
    assert rag_retrieval_k(
        "Per the project specification, what compaction is required under "
        "the access road?", 5,
    ) == 7
    # A quantity question that names no source, and an ordinary question.
    assert rag_retrieval_k("What is the minimum concrete cover for pads?", 5) == 5
    assert rag_retrieval_k("What is the site address?", 5) == 5


# ── the recall on its own ─────────────────────────────────────────────────

class _LexStore:
    def __init__(self):
        self.chunks = [
            Chunk(chunk_id=doc_id, project_id=PID, doc_id=doc_id, chunk_index=0,
                  text=texts[0], score=0.0)
            for doc_id, (_name, texts) in DOCS.items()
        ]

    def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
        want = [str(n).lower() for n in needles]
        return [c for c in self.chunks if all(n in c.text.lower() for n in want)][:k]


def _recalled(ret, ask):
    fused = {}
    ret.recall_asked_quantity_chunks(ask, PID, fused, _LexStore())
    return set(fused)


def test_recall_takes_synonyms_of_the_condition_and_skips_its_negation(monkeypatch):
    from app.core.rag import retriever as ret

    got = _recalled(ret, ASK)
    assert {NOTE_GROUND, SHEET_SOIL, DWG_PADS} <= got, got
    # "not in contact with ground" is a different condition.
    assert NOTE_NEGATED not in got, got


def test_a_different_condition_does_not_take_the_ground_contact_notes():
    from app.core.rag import retriever as ret

    got = _recalled(ret, BLINDING_ASK)
    assert NOTE_BLINDING in got, got
    assert not ({NOTE_GROUND, SHEET_SOIL, NOTE_NEGATED} & got), got
