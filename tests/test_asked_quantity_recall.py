"""Asked-quantity recall: a figure the question asks for reaches the top-k.

General mechanism (``recall_asked_quantity_chunks`` in the retriever): a
question that asks for a measured quantity (a cover length, a degree of
compaction, an illumination level) is answered by a chunk that states a
number of that quantity next to the question's subject. Such a chunk is often
a table row or a drawing note whose wording shares little with the question,
so neither retrieval leg pools it while prose that repeats the question's
words (and states no figure) fills the slots.

Every document, figure and name below is invented. None of these questions
is answered by the retriever before the mechanism exists: each subject
("formwork erection", "bearing on rock", "granular sub-base") is outside any
phrase list the retriever used to carry.
"""
from __future__ import annotations

import pytest

from app.core.rag import retriever
from app.core.rag.vector_store import Chunk

PID = "synthetic-quantity-project"

LIGHT_ASK = (
    "Per the site safety plan, what minimum lighting level is required for "
    "formwork erection?"
)
TASK_LIGHT_TABLE = (
    "Task lighting schedule. | Task | Lux | | Formwork erection | 75 | "
    "| Pipe welding | 300 | | Storage yard | 20 |"
)
ROOM_LIGHT_TABLE = (
    "Interior lighting design. | Room | Avg Lux | | Meeting room | 400 | "
    "| Lobby | 150 | | Plant room | 200 |"
)
SAFETY_PROSE = (
    "The contractor shall light every work front so that formwork erection "
    "can proceed safely after dark, in line with the site safety plan and "
    "the minimum lighting level required by the client."
)

ROCK_ASK = (
    "What is the minimum concrete cover for pile caps poured against rock?"
)
ROCK_NOTE = "NOTE 7: COVER TO REINFORCEMENT FOR ELEMENTS BEARING ON ROCK : 60mm."
ROCK_FILLERS = [
    f"Site memo {n}: the minimum concrete cover for pile caps poured against "
    f"rock at grid line {n} is to be confirmed by the designer before casting."
    for n in range(1, 9)
]

SUBBASE_ASK = (
    "To what degree must the granular sub-base under the hardstanding be "
    "compacted?"
)
SUBBASE_CLAUSE = (
    "Clause 4.2: granular sub-base material shall be laid in 150 mm layers and "
    "compacted to not less than 97% of maximum dry density."
)
SUBBASE_PROSE = (
    "The granular sub-base under the hardstanding shall be compacted with a "
    "vibrating roller of suitable mass in accordance with good practice."
)


def _chunk(cid, text, *, score=0.0, doc=None, pid=PID, index=0):
    return Chunk(
        chunk_id=cid, project_id=pid, doc_id=doc or cid, chunk_index=index,
        text=text, score=score,
    )


def _install(monkeypatch, tmp_path, semantic, lexical):
    """Patch the store: ``semantic`` answers the vector leg, ``lexical`` the
    text-match calls. Nothing else is in the corpus."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    corpus = list({c.chunk_id: c for c in list(semantic) + list(lexical)}.values())

    def search(self, project_id, qvec, k, query_text=None):
        return [
            Chunk(**{**c.__dict__}) for c in semantic if c.project_id == project_id
        ][:k]

    def bm25_search(self, project_id, query, k=50):
        return []

    def identifier_search(self, project_id, identifiers, k=20):
        out = []
        for c in corpus:
            if c.project_id != project_id:
                continue
            toks = set(retriever.re.findall(r"[a-z0-9]+", (c.text or "").lower()))
            if any(all(t in toks for t in i.lower().split()) for i in identifiers):
                out.append(Chunk(**{**c.__dict__}))
        return out[:k]

    def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
        want = [str(n).lower() for n in needles if len(str(n)) >= 3]
        out = []
        for c in corpus:
            if c.project_id != project_id:
                continue
            if doc_ids and c.doc_id not in doc_ids:
                continue
            low = (c.text or "").lower()
            if want and all(n in low for n in want):
                out.append(Chunk(**{**c.__dict__}))
        return out[:k]

    def chunks_for_docs(self, project_id, doc_ids, **_kw):
        return [Chunk(**{**c.__dict__}) for c in corpus
                if c.project_id == project_id and c.doc_id in doc_ids]

    def chunks_following(self, project_id, anchors, n=1):
        return []

    def count(self, project_id=None):
        return len(corpus)

    from app.core.rag import vector_store as vs
    for name, fn in (
        ("search", search), ("bm25_search", bm25_search),
        ("identifier_search", identifier_search),
        ("chunks_containing_all", chunks_containing_all),
        ("chunks_for_docs", chunks_for_docs),
        ("chunks_following", chunks_following), ("count", count),
    ):
        monkeypatch.setattr(vs.VectorStore, name, fn)
    monkeypatch.setattr(retriever, "_doc_name_for_id", lambda did: f"{did}.pdf")


def _ids(chunks):
    return [c.chunk_id for c in chunks]


# ── end to end through retrieve_with_filter ───────────────────────────────

def test_lux_table_row_for_the_asked_task_reaches_top_k(monkeypatch, tmp_path):
    semantic = [
        _chunk("prose", SAFETY_PROSE, score=0.71),
        _chunk("rooms", ROOM_LIGHT_TABLE, score=0.52),
    ]
    _install(monkeypatch, tmp_path, semantic, [_chunk("tasks", TASK_LIGHT_TABLE)])
    chunks, _ = retriever.retrieve_with_filter(LIGHT_ASK, PID, k=2)
    ids = _ids(chunks)
    assert "tasks" in ids, ids
    # A lux table about other subjects is a lookalike, not the answer.
    assert ids.index("tasks") == 0, ids


def test_cover_note_in_the_documents_own_words_reaches_top_k(monkeypatch, tmp_path):
    semantic = [
        _chunk(f"memo{n}", text, score=0.80 - n * 0.01)
        for n, text in enumerate(ROCK_FILLERS)
    ]
    _install(monkeypatch, tmp_path, semantic, [_chunk("note7", ROCK_NOTE)])
    chunks, _ = retriever.retrieve_with_filter(ROCK_ASK, PID, k=5)
    assert "note7" in _ids(chunks), _ids(chunks)


def test_compaction_clause_for_the_asked_layer_reaches_top_k(monkeypatch, tmp_path):
    semantic = [_chunk("prose", SUBBASE_PROSE, score=0.74)]
    _install(monkeypatch, tmp_path, semantic, [_chunk("clause", SUBBASE_CLAUSE)])
    chunks, _ = retriever.retrieve_with_filter(SUBBASE_ASK, PID, k=1)
    assert _ids(chunks) == ["clause"], _ids(chunks)


def test_a_figure_the_corpus_does_not_state_is_not_invented(monkeypatch, tmp_path):
    semantic = [_chunk("prose", SAFETY_PROSE, score=0.71)]
    _install(monkeypatch, tmp_path, semantic, [_chunk("rooms", ROOM_LIGHT_TABLE)])
    chunks, _ = retriever.retrieve_with_filter(LIGHT_ASK, PID, k=5)
    # The only lux table is about other subjects, so nothing is pooled.
    assert _ids(chunks) == ["prose"], _ids(chunks)


def test_a_question_that_asks_no_quantity_is_untouched(monkeypatch, tmp_path):
    semantic = [_chunk("prose", SAFETY_PROSE, score=0.71)]
    _install(monkeypatch, tmp_path, semantic, [_chunk("tasks", TASK_LIGHT_TABLE)])
    chunks, _ = retriever.retrieve_with_filter(
        "Who approves the site safety plan for formwork erection?", PID, k=5,
    )
    assert "tasks" not in _ids(chunks), _ids(chunks)


# ── the mechanism on its own ──────────────────────────────────────────────

class _LexStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
        self.calls.append((project_id, tuple(needles)))
        want = [str(n).lower() for n in needles]
        return [c for c in self.chunks if c.project_id == project_id
                and all(n in (c.text or "").lower() for n in want)][:k]


def test_recall_scans_the_pids_the_semantic_leg_searched():
    gk = _chunk("tasks", TASK_LIGHT_TABLE, pid="synthetic-reference")
    store = _LexStore([gk])
    fused = {}
    added = retriever.recall_asked_quantity_chunks(
        LIGHT_ASK, PID, fused, store, extra_pids=["synthetic-reference"],
    )
    assert added == 1 and "tasks" in fused
    assert {pid for pid, _n in store.calls} == {PID, "synthetic-reference"}


def test_recall_leaves_an_already_pooled_figure_alone():
    pooled = _chunk("tasks", TASK_LIGHT_TABLE, score=0.9)
    fused = {"tasks": (pooled, 0.9, 0.0)}
    retriever.recall_asked_quantity_chunks(
        LIGHT_ASK, PID, fused, _LexStore([pooled]),
    )
    assert fused["tasks"][1:] == (0.9, 0.0)


@pytest.mark.parametrize("ask, kinds", [
    (LIGHT_ASK, {"illuminance"}),
    (ROCK_ASK, {"length_mm"}),
    (SUBBASE_ASK, {"compaction"}),
    ("Who approves the site safety plan?", set()),
    ("Please attach the cover letter for the rock survey.", set()),
])
def test_quantity_classes_are_read_from_the_question(ask, kinds):
    assert set(retriever.measured_quantity_kinds(ask)) == kinds


def test_subject_words_exclude_the_quantity_and_the_question_frame():
    terms = retriever.quantity_subject_terms(LIGHT_ASK)
    # Stems, so "erection" also meets "erected" in a document.
    assert terms == ["formwork", "erect"], terms
    for frame in ("minimum", "lighting", "level", "required", "plan"):
        assert frame not in terms, terms


def test_a_negated_subject_mention_does_not_agree():
    terms = retriever.quantity_subject_terms(ROCK_ASK)
    assert retriever.chunk_names_quantity_subject(ROCK_NOTE, terms)
    assert not retriever.chunk_names_quantity_subject(
        "COVER TO REINFORCEMENT FOR ELEMENTS NOT ON ROCK : 40mm.", terms,
    )


def test_a_lux_figure_is_a_number_with_its_unit_or_a_lux_column():
    assert retriever.chunk_states_illuminance(TASK_LIGHT_TABLE)
    assert retriever.chunk_states_illuminance("Night work areas: 40 lux minimum.")
    # Prose about lighting with no figure states no level.
    assert not retriever.chunk_states_illuminance(SAFETY_PROSE)
    # A number in a table without a lux column is not a lux figure.
    assert not retriever.chunk_states_illuminance("| Task | Crew | | Pipe welding | 4 |")


def test_a_lux_table_about_other_subjects_takes_no_figure_lift():
    kinds = retriever.measured_quantity_kinds(LIGHT_ASK)
    assert retriever.chunk_matches_quantity_question(LIGHT_ASK, TASK_LIGHT_TABLE, kinds)
    assert not retriever.chunk_matches_quantity_question(LIGHT_ASK, ROOM_LIGHT_TABLE, kinds)
