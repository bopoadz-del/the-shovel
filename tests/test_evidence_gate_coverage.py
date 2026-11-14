"""S-R10 b: the evidence loop's gate is the first pass's coverage of the
question, not the question's wording.

Each test builds its own question and first-pass text. Nothing here is a
document, a project or a tester row.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.agents import runtime
from app.core.rag import route
from app.core.rag.inject import _content_terms


def _excerpt(doc_id: str, chunk: int, body: str) -> str:
    return f"[doc_id={doc_id} chunk={chunk} score=0.800 class=project_corpus] {body}"


def _context(*excerpts: str, header: str = "REFERENCE CONTEXT — answer from below.") -> str:
    return header + "\n" + "\n\n".join(excerpts)


# --- the pure coverage check -------------------------------------------------

@pytest.mark.parametrize("question,body", [
    ("Which grade of concrete is used for the raft?",
     "The raft shall be cast in concrete grade C40; sulphate resisting cement is used."),
    ("How long is the curing period for slabs",
     "Slabs: the curing period is seven days, long enough for the membrane."),
])
def test_excerpts_holding_every_term_cover_the_question(question, body):
    terms, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, body)), content_terms=_content_terms,
    )
    assert terms
    assert missing == ()


@pytest.mark.parametrize("question,body,absent", [
    ("Which grade of concrete is used for the raft?",
     "The raft shall be cast in sulphate resisting cement.", {"grade", "concrete", "used"}),
    ("How long is the curing period for slabs",
     "Slabs shall be power floated.", {"long", "curing", "period"}),
])
def test_a_term_absent_from_the_excerpts_is_missing(question, body, absent):
    _, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, body)), content_terms=_content_terms,
    )
    assert set(missing) == absent


@pytest.mark.parametrize("question", [
    "Which grade of concrete is used for the raft?",
    "How long is the curing period for slabs",
])
def test_header_prose_and_markers_never_cover(question):
    # The question's own words sit in the platform header and in a marker's
    # file name; the excerpt body holds none of them.
    header = "REFERENCE CONTEXT for: " + question
    marker_body = "[doc_id=d1 chunk=0 score=0.800 src=" + question.replace(" ", "_") + "] unrelated"
    _, missing = route.question_coverage(
        question, header + "\n" + marker_body, content_terms=_content_terms,
    )
    terms, _ = route.question_coverage(question, "", content_terms=_content_terms)
    assert missing == terms


@pytest.mark.parametrize("question", [
    "Which grade of concrete is used for the raft?",
    "How long is the curing period for slabs",
])
def test_a_platform_note_echoing_the_question_never_covers(question):
    note = "[bill item rows: " + question + "]"
    _, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, "unrelated text\n" + note)),
        content_terms=_content_terms,
    )
    terms, _ = route.question_coverage(question, "", content_terms=_content_terms)
    assert missing == terms


@pytest.mark.parametrize("question,body", [
    ("Where are the piles shown", "Pile caps are detailed on the layout."),
    ("What is the cover to rebar", "Coverage of the rebars is checked."),
])
def test_whole_words_not_substrings(question, body):
    _, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, body)), content_terms=_content_terms,
    )
    assert missing


@pytest.mark.parametrize("question", ["what is it?", "what about it?"])
def test_a_question_with_no_content_terms_is_never_uncovered(question):
    terms, missing = route.question_coverage(
        question, _context(_excerpt("d1", 0, "anything")), content_terms=_content_terms,
    )
    assert terms == () and missing == ()


# --- the runtime gate --------------------------------------------------------

def _every_term_once(project_id, terms):
    """A store whose index holds every term in exactly one chunk: each is as
    informative as the next, so rule (c) keeps every indexed absent term."""
    return {t: 1 for t in terms}


def _run_gate(
    monkeypatch, question: str, first_pass: str, *, excluded: bool = False,
    term_chunk_counts=_every_term_once,
):
    calls: list[str] = []

    async def fake_augment(**kwargs):
        calls.append(kwargs["question"])
        return kwargs["first_pass_evidence"] + "\n" + _excerpt("d9", 1, "more")

    monkeypatch.setattr(route, "augment_figure_retrieval", fake_augment)
    monkeypatch.setattr(runtime, "_first_pass_composes_figure", lambda *a: False)
    monkeypatch.setattr(runtime, "_agentic_figure_retrieval_excluded", lambda t: excluded)
    monkeypatch.setattr(runtime, "_bill_led_rag_msg", lambda q, p, m: m)
    monkeypatch.setattr(
        "app.core.rag.vector_store.get_store",
        lambda: SimpleNamespace(
            bm25_search=lambda *a, **k: [], term_chunk_counts=term_chunk_counts,
        ),
    )
    out = asyncio.run(runtime._agentic_figure_retrieval_augment(
        agent=SimpleNamespace(), user_message=question, project_id="p",
        api_key="k", user_id="u", rag_sys_msg={"role": "system", "content": first_pass},
    ))
    return calls, out


# Interrogative shapes the old noun whitelist never let through: "which",
# "how long", no figure noun.
_UNGATED_SHAPES = [
    ("Which grade of concrete is used for the raft?",
     "Concrete grade C40 is used for the raft."),
    ("How long is the curing period for slabs",
     "Slabs: the curing period is seven days, long enough."),
]


@pytest.mark.parametrize("question,covering_body", _UNGATED_SHAPES)
def test_an_uncovered_first_pass_enters_the_loop_whatever_the_wording(
    question, covering_body, monkeypatch,
):
    first_pass = _context(_excerpt("d1", 0, "Formwork shall be plywood."))
    calls, out = _run_gate(monkeypatch, question, first_pass)
    assert calls == [question]
    assert "doc_id=d9" in out["content"]


@pytest.mark.parametrize("question,covering_body", _UNGATED_SHAPES)
def test_a_covering_first_pass_skips_the_plan_call(question, covering_body, monkeypatch):
    first_pass = _context(_excerpt("d1", 0, covering_body))
    calls, out = _run_gate(monkeypatch, question, first_pass)
    assert calls == []
    assert out["content"] == first_pass


@pytest.mark.parametrize("question,covering_body", _UNGATED_SHAPES)
def test_exclusions_still_keep_the_loop_out(question, covering_body, monkeypatch):
    first_pass = _context(_excerpt("d1", 0, "Formwork shall be plywood."))
    calls, out = _run_gate(monkeypatch, question, first_pass, excluded=True)
    assert calls == []
    assert out["content"] == first_pass


# --- rules (b) and (c): the project's own index decides --------------------
#
# Each case builds a question, a first pass and the project's chunk counts.
# The counts stand in for what ``VectorStore.term_chunk_counts`` reads from
# the lexical index; the store-level tests below read a real one.

# A first pass that holds the fact; the question carries one word the
# excerpts lack and that most of the project's chunks use.
_SCAFFOLD_ABSENT = [
    ("How many days notice does the contractor give before a pour?",
     "The Contractor shall give 2 days notice before any concrete pour.",
     {"many": 140, "days": 30, "notice": 12, "contractor": 150,
      "give": 90, "before": 160, "pour": 6}),
    ("Which supplier delivers the precast stair units to site?",
     "Precast stair units: the appointed supplier delivers them to the works.",
     {"supplier": 15, "delivers": 3, "precast": 8, "stair": 10,
      "units": 70, "site": 210}),
]

# A first pass that lacks the question's rarest indexed term.
_RAREST_ABSENT = [
    ("What is the cube strength of the blinding concrete?",
     "Concrete for the slabs shall reach the specified strength.",
     {"cube": 4, "strength": 40, "blinding": 2, "concrete": 300}),
    ("When is the waterproofing membrane inspected",
     "The membrane is laid after the screed has cured.",
     {"when": 120, "waterproofing": 5, "membrane": 9, "inspected": 25}),
]

# A first pass that lacks a term no document in the project uses.
_UNINDEXED_ABSENT = [
    ("What is the retention percentage?",
     "Retention: 5% of each interim payment until practical completion.",
     {"retention": 7, "percentage": 0}),
    ("How many lux are required in the plant room",
     "Plant room lighting required: 200 lx at floor level.",
     {"many": 0, "lux": 0, "required": 60, "plant": 9, "room": 80}),
]


def _missing(question, body, counts):
    terms, absent = route.question_coverage(
        question, _context(_excerpt("d1", 0, body)), content_terms=_content_terms,
    )
    return terms, absent, route.informative_missing_terms(terms, absent, counts)


@pytest.mark.parametrize("question,body,counts", _SCAFFOLD_ABSENT)
def test_a_common_absent_word_is_not_missing(question, body, counts):
    _, absent, (indexed, missing) = _missing(question, body, counts)
    assert absent and indexed == absent  # rule (a) and (b) hold for it
    assert missing == ()                 # rule (c) does not


@pytest.mark.parametrize("question,body,counts", _RAREST_ABSENT)
def test_the_rarest_absent_term_is_missing(question, body, counts):
    terms, _, (_, missing) = _missing(question, body, counts)
    rarest = min((t for t in terms if counts.get(t)), key=lambda t: counts[t])
    assert rarest in missing


@pytest.mark.parametrize("question,body,counts", _UNINDEXED_ABSENT)
def test_a_term_the_index_lacks_is_never_missing(question, body, counts):
    _, absent, (indexed, missing) = _missing(question, body, counts)
    unindexed = {t for t in absent if not counts.get(t)}
    assert unindexed
    assert not unindexed & set(indexed)
    assert not unindexed & set(missing)


@pytest.mark.parametrize("question,body,counts", _SCAFFOLD_ABSENT)
def test_gate_holding_the_fact_with_a_common_word_skips_the_loop(
    question, body, counts, monkeypatch,
):
    first_pass = _context(_excerpt("d1", 0, body))
    calls, out = _run_gate(
        monkeypatch, question, first_pass,
        term_chunk_counts=lambda pid, terms: {t: counts.get(t, 0) for t in terms},
    )
    assert calls == []
    assert out["content"] == first_pass


@pytest.mark.parametrize("question,body,counts", _RAREST_ABSENT)
def test_gate_lacking_the_rarest_term_runs_the_loop(
    question, body, counts, monkeypatch,
):
    first_pass = _context(_excerpt("d1", 0, body))
    calls, _ = _run_gate(
        monkeypatch, question, first_pass,
        term_chunk_counts=lambda pid, terms: {t: counts.get(t, 0) for t in terms},
    )
    assert calls == [question]


@pytest.mark.parametrize("question,body,counts", _UNINDEXED_ABSENT)
def test_gate_never_fires_on_a_term_the_index_lacks(
    question, body, counts, monkeypatch,
):
    first_pass = _context(_excerpt("d1", 0, body))
    calls, _ = _run_gate(
        monkeypatch, question, first_pass,
        term_chunk_counts=lambda pid, terms: {t: counts.get(t, 0) for t in terms},
    )
    assert calls == []


@pytest.mark.parametrize("question,covering_body", _UNGATED_SHAPES)
def test_an_unreadable_index_does_not_run_the_loop(
    question, covering_body, monkeypatch,
):
    def broken(project_id, terms):
        raise RuntimeError("index unavailable")

    first_pass = _context(_excerpt("d1", 0, "Formwork shall be plywood."))
    calls, out = _run_gate(monkeypatch, question, first_pass, term_chunk_counts=broken)
    assert calls == []
    assert out["content"] == first_pass


# --- the store reads the counts from its own lexical index ------------------

@pytest.fixture
def small_store(tmp_path):
    import numpy as np

    from app.core.rag import vector_store as vs

    vs.reset_store_cache()
    store = vs.VectorStore(db_path=str(tmp_path / "terms.db"), dim=8)
    rng = np.random.default_rng(0)

    def index(project_id, doc_id, texts):
        store.upsert_chunks(
            project_id, doc_id, texts, rng.random((len(texts), 8), dtype=np.float32),
        )

    yield store, index
    store.close()


@pytest.mark.parametrize("texts,expect", [
    (["the pour is cast", "the slab is cured", "pour the blinding"],
     {"pour": 2, "slab": 1, "blinding": 1, "membrane": 0}),
    (["Retention 5% applies", "retention is released", "interim payment due"],
     {"retention": 2, "interim": 1, "percentage": 0, "payment": 1}),
])
def test_store_counts_chunks_holding_each_term(small_store, texts, expect):
    store, index = small_store
    index("p-own", "doc-a", texts)
    index("p-other", "doc-b", [" ".join(expect)] * 3)  # another project's words
    assert store.term_chunk_counts("p-own", list(expect)) == expect
