"""Pointer-following: a named source that sends the asked figure elsewhere.

General mechanism (``follow_quantity_pointers`` + ``_apply_quantity_pointer_boost``
in the retriever). A question names its source ("per the project
specification") and asks for a measured quantity. The source often states no
figure: one sentence names the quantity and points at another document class
("the cover specified on the Drawings", "compacted to the density stated in
the Schedule of ..."). That clause is the source's answer, and the pointed-to
document states the figure. The clause states no figure, so the numeric fetch
rejects it and its cosine is low; this mechanism pools it from the chunk's own
structure (quantity + pointer in one sentence) and prefers the documents of
the class it points to.

It replaces a cover-only, drawings-only path that carried a kill switch. The
whole file runs on the fake embedder over an invented project; no document,
clause number or figure comes from a real corpus.
"""
from __future__ import annotations

import pytest

PID = "synthetic-pointer-project"

SPEC_COVER = "spec-vol3"
SPEC_EARTH = "spec-vol4"
DWG_BEAM = "dwg-s101"
SCHEDULE = "schedule-earthworks"

DOCS = {
    SPEC_COVER: ("Lakeside Depot Specification Volume 3 concrete.pdf", [
        "7.2.4 Bar supports. Tie wire shall be 1.6 mm annealed steel. Chairs "
        "and spacers shall hold the reinforcement to give the cover specified "
        "on the Drawings or as the engineer instructs.",
        "7.2.5 Curing. Exposed faces shall be kept wet for seven days.",
    ]),
    SPEC_EARTH: ("Lakeside Depot Specification Volume 4 earthworks.pdf", [
        "4.6 Reinstatement of service runs. Material shall be placed in 150 mm "
        "lifts, each compacted to the density stated in the Schedule of "
        "Earthworks Requirements.",
        "4.7 Topsoil. Topsoil shall be stripped and stockpiled separately.",
    ]),
    DWG_BEAM: ("Lakeside Depot DWG S-101 ground beam details.pdf", [
        "GROUND BEAM NOTES. CONCRETE COVER TO REINFORCEMENT: GROUND BEAMS CAST "
        "AGAINST EARTH : 70mm. INTERNAL FACES : 40mm.",
    ]),
    SCHEDULE: ("Lakeside Depot Schedule of Earthworks Requirements.pdf", [
        "| Material | Required density | | Trench backfill | 93% of maximum dry "
        "density | | Capping layer | 97% of maximum dry density |",
    ]),
    "memo-cover": ("Lakeside Depot site memo cover meter readings.pdf", [
        "Cover meter readings: nominal cover 45mm recorded at wall W3 during "
        "the pre-pour inspection of the ground beams.",
    ]),
    "ms-concrete": ("Lakeside Depot method statement concreting.pdf", [
        "Concrete cover to reinforcement shall be checked before every pour; "
        "typical cover 35mm on suspended slabs.",
    ]),
    "ms-haul": ("Lakeside Depot temporary works method statement.pdf", [
        "Temporary haul road fill shall be compacted to 90% of maximum dry "
        "density before plant moves onto it.",
    ]),
}

COVER_ASK = (
    "Per the project specification, what is the minimum concrete cover for "
    "ground beams cast against earth?"
)
DENSITY_ASK = (
    "Per the project specification, to what density must trench backfill be "
    "compacted?"
)


def _fillers():
    """Chunks that repeat both questions' words and state no figure.

    Enough of them to fill both retrieval legs' over-fetch, so a clause that
    words the topic differently is pooled by neither leg on its own.
    """
    out = []
    for n in range(1, 71):
        out.append(
            f"Lakeside Depot RFI {n}: confirm the minimum concrete cover for "
            f"ground beams cast against earth per the project specification "
            f"before pour {n}."
        )
        out.append(
            f"Lakeside Depot RFI {n + 100}: confirm to what density trench "
            f"backfill must be compacted per the project specification at "
            f"chainage {n}."
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
    store.upsert_chunks(PID, "rfi-log", fillers, embedder.encode(fillers))
    names["rfi-log"] = "Lakeside Depot RFI log.pdf"
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda d: names.get(d, ""))
    yield ret
    emb.reset_embedder_cache()
    vs.reset_store_cache()


def _inject(ask):
    from app.core.rag.inject import rag_inject

    msg, audit = rag_inject(
        user_message=ask,
        project_id=PID,
        conversation_id="ws-synthetic-pointer",
        user_id="fixture",
        agent_name="project-assistant",
        history=[],
    )
    return msg, [c["doc_id"] for c in audit.get("chunks") or []]


# ── end to end through rag_inject ─────────────────────────────────────────

def test_cover_deferred_to_the_drawings_reaches_the_model(corpus):
    msg, handed = _inject(COVER_ASK)
    assert msg is not None
    assert SPEC_COVER in handed, f"the specification's pointer clause: {handed}"
    assert DWG_BEAM in handed, f"the drawing it points to: {handed}"
    assert "on the Drawings" in msg["content"]
    assert "GROUND BEAMS CAST AGAINST EARTH : 70mm" in msg["content"]


def test_compaction_deferred_to_a_schedule_reaches_the_model(corpus):
    """The source points at a schedule, not the drawings, for a density."""
    msg, handed = _inject(DENSITY_ASK)
    assert msg is not None
    assert SPEC_EARTH in handed, f"the specification's pointer clause: {handed}"
    assert SCHEDULE in handed, f"the schedule it points to: {handed}"
    assert "density stated in the Schedule of" in msg["content"]
    assert "93% of maximum dry" in msg["content"]


def test_the_pointed_to_document_outranks_lookalike_figures(corpus):
    chunks, _ = corpus.retrieve_with_filter(COVER_ASK, PID, k=7, operator_text=COVER_ASK)
    order = [c.doc_id for c in chunks]
    assert DWG_BEAM in order, order
    for lookalike in ("memo-cover", "ms-concrete"):
        if lookalike in order:
            assert order.index(DWG_BEAM) < order.index(lookalike), order


def test_without_a_pointer_clause_the_ranking_is_unchanged(corpus, monkeypatch):
    """A question that names no source never takes this path."""
    ask = "What is the minimum concrete cover for ground beams cast against earth?"
    on, _ = corpus.retrieve_with_filter(ask, PID, k=5, operator_text=ask)
    monkeypatch.setattr(corpus, "follow_quantity_pointers", lambda *a, **k: {})
    monkeypatch.setattr(corpus, "_apply_quantity_pointer_boost", lambda *a, **k: None)
    off, _ = corpus.retrieve_with_filter(ask, PID, k=5, operator_text=ask)
    assert [c.chunk_id for c in on] == [c.chunk_id for c in off]


# ── detectors (no store) ──────────────────────────────────────────────────

def _ret():
    from app.core.rag import retriever as ret
    return ret


@pytest.mark.parametrize("text, kinds, target", [
    ("Spacers shall give the cover specified on the Drawings.", {"length_mm"}, "drawings"),
    ("Concrete cover to reinforcement shall be as shown on the drawings.",
     {"length_mm"}, "drawings"),
    ("Fill shall be compacted to the density stated in the Schedule of "
     "Earthworks Requirements.", {"compaction"}, "schedule"),
    ("Lighting levels for each task are as listed in Appendix C.",
     {"illuminance"}, "appendix"),
    # A manhole cover is not the concrete cover.
    ("Access hatch covers and frames shall be as shown on the Drawings.",
     {"length_mm"}, None),
    # The quantity and the pointer sit in different sentences.
    ("The specified minimum concrete cover shall be checked. Bars shall be "
     "fixed as shown on the Drawings.", {"length_mm"}, None),
    # A figure the chunk states itself is not a pointer.
    ("Fill shall be compacted to 95% of maximum dry density.", {"compaction"}, None),
])
def test_pointer_is_read_from_one_sentence(text, kinds, target):
    assert _ret().chunk_points_quantity_elsewhere(text, frozenset(kinds)) == target


@pytest.mark.parametrize("ask, expected", [
    (COVER_ASK, True),
    (DENSITY_ASK, True),
    ("What cover does the spec require for footings poured against earth?", True),
    ("Under the project specification, what lighting level applies to the "
     "loading bay?", True),
    ("What is the minimum concrete cover for ground beams?", False),
    ("Does the spec cover concrete curing?", False),
    ("What does the spec say about manhole covers in slabs?", False),
])
def test_only_a_source_scoped_quantity_question_follows_pointers(ask, expected):
    assert _ret().query_follows_source_pointers(ask) is expected


def test_cover_list_item_under_its_heading_is_a_stated_length():
    text = (
        "THE CLEAR CONCRETE COVER TO STEEL REINFORCEMENT SHALL NOT BE LESS "
        "THAN THE FOLLOWING: FOR BURIED WORK THE UNDERSIDE OF GROUND BEAMS "
        "IN CONTACT WITH EARTH : 70mm"
    )
    assert _ret().chunk_states_cover_length(text)


def test_cover_heading_does_not_reach_past_a_new_note():
    text = (
        "4. CLEAR COVER TO REINFORCEMENT SHALL BE AS SHOWN ON THE SCHEDULE "
        "OF STRUCTURAL ELEMENTS FOR EACH POUR.\n5. MARKER POSTS : 300mm"
    )
    assert not _ret().chunk_states_cover_length(text)


def test_lid_size_is_not_a_cover_length():
    text = (
        "The valve chamber shall have a concrete cover of 300mm x 300mm x "
        "12mm at finished level."
    )
    assert not _ret().chunk_states_cover_length(text)
