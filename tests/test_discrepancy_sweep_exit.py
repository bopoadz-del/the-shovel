"""A2 discrepancy sweep, caller half: the exit's one lexical search adds the
passage that gives the OTHER value for the same identifier, and the conflict
statement then states both.

Every document, identifier and figure here is invented for the test. Each
decision is checked on two unrelated inputs (a cover in mm against a
standard; a strength in MPa against a table).
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from app.agents import answer_exit, discrepancy_sweep

CASES = [
    pytest.param(
        {
            "question": "What is the concrete cover for footings?",
            "draft": "The concrete cover for footings is 40 mm to BS 8500 [Structural Spec.pdf, p. 3].",
            "own_doc": "Structural Spec.pdf",
            "own": "Concrete cover for footings to BS 8500 shall be 40 mm.",
            "other_doc": "S-101 General Notes.pdf",
            "other": "General notes: concrete cover to BS 8500 shall be 50 mm for all footings.",
            "same": "Concrete cover to BS 8500 is 40 mm for footings in this package.",
            "unrelated": "Cover to ACI 318 shall be 75 mm where cast against earth.",
            "stated": "40 mm",
            "differing": "50 mm",
        },
        id="cover-mm-standard",
    ),
    pytest.param(
        {
            "question": "What grout strength do the base plates need?",
            "draft": "Base plate grout needs a strength of 45 MPa per Table 6.2 [Steel Method Statement.docx, p. 9].",
            "own_doc": "Steel Method Statement.docx",
            "own": "Base plate grout strength per Table 6.2: 45 MPa.",
            "other_doc": "Grout Schedule Rev C.xlsx",
            "other": "Table 6.2 base plate grout, characteristic strength 60 MPa.",
            "same": "Table 6.2 grout under base plates shall reach 45 MPa.",
            "unrelated": "Table 9.1 base plate grout strength 60 MPa.",
            "stated": "45 MPa",
            "differing": "60 MPa",
        },
        id="grout-mpa-table",
    ),
]


def _turn(case, project_id="proj-a"):
    turn = answer_exit.Turn(question=case["question"], project_id=project_id)
    turn.evidence.append({"doc_id": "own", "doc_name": case["own_doc"], "page": 3, "page_or_section": "p. 3",
                          "chunk_index": 1, "snippet": case["own"]})
    return turn


def _result(case):
    return {"answer": case["draft"],
            "sources": [{"doc_id": "own", "doc_name": case["own_doc"], "page_or_section": "p. 3"}]}


def _wire(monkeypatch, passages, calls, other_doc):
    """The project's lexical search returns ``passages`` (doc_id, text); the
    passage's document is named ``other_doc``."""
    def search(project_id, query, k):
        calls.append((project_id, query, k))
        return [NS(doc_id=d, chunk_index=i, page=5, text=t) for i, (d, t) in enumerate(passages)]

    monkeypatch.setattr(answer_exit, "_lexical_search", search)
    monkeypatch.setattr(answer_exit, "_document_names", lambda ids: {d: other_doc for d in ids})


@pytest.mark.parametrize("case", CASES)
def test_other_value_for_the_same_identifier_is_stated_with_its_source(case, monkeypatch):
    calls: list = []
    _wire(monkeypatch, [("other", case["other"])], calls, case["other_doc"])
    out = answer_exit.check_result(_result(case), _turn(case))
    assert len(calls) == 1 and calls[0][0] == "proj-a"
    assert case["stated"] in out["answer"] and case["differing"] in out["answer"]
    docs = {v["document"] for c in out.get("conflicts") or [] for v in c["values"]}
    assert docs == {case["own_doc"], case["other_doc"]}


@pytest.mark.parametrize("case", CASES)
def test_without_the_sweep_the_draft_states_one_value(case, monkeypatch):
    """Revert guard: the same turn with no project never searches and states
    only the drafted value."""
    calls: list = []
    _wire(monkeypatch, [("other", case["other"])], calls, case["other_doc"])
    out = answer_exit.check_result(_result(case), _turn(case, project_id=""))
    assert calls == []
    assert case["differing"] not in out["answer"] and not out.get("conflicts")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("which", ["same", "unrelated"])
def test_same_value_or_other_identifier_adds_nothing(case, which, monkeypatch):
    calls: list = []
    _wire(monkeypatch, [("other", case[which])], calls, case["other_doc"])
    turn = _turn(case)
    out = answer_exit.check_result(_result(case), turn)
    assert len(calls) == 1
    assert [r["doc_id"] for r in turn.evidence] == ["own"]
    assert not out.get("conflicts")


@pytest.mark.parametrize("case", CASES)
def test_a_passage_already_handed_is_not_added_again(case, monkeypatch):
    calls: list = []
    monkeypatch.setattr(answer_exit, "_lexical_search", lambda p, q, k: (
        calls.append(q) or [NS(doc_id="own", chunk_index=1, page=3, text=case["other"])]))
    monkeypatch.setattr(answer_exit, "_document_names", lambda ids: {"own": case["own_doc"]})
    turn = _turn(case)
    answer_exit.check_result(_result(case), turn)
    assert len(calls) == 1 and len(turn.evidence) == 1


@pytest.mark.parametrize("draft", [
    "The cover for footings is 40 mm [Structural Spec.pdf, p. 3].",  # a value, no identifier
    "Footings follow BS 8500 throughout [Structural Spec.pdf, p. 3].",  # an identifier, no value
])
def test_no_identifier_beside_a_value_means_no_search(draft):
    def search(query, k):
        raise AssertionError("searched")

    assert discrepancy_sweep.sweep_rows(draft, [], search, lambda ids: {}) == []


@pytest.mark.parametrize("case", CASES)
def test_an_unnamed_document_is_not_added(case):
    found = discrepancy_sweep.sweep_rows(
        case["draft"], [],
        lambda q, k: [NS(doc_id="x", chunk_index=0, page=None, text=case["other"])],
        lambda ids: {},
    )
    assert found == []


@pytest.mark.parametrize("case", CASES)
def test_the_added_row_has_the_handed_evidence_shape(case):
    found = discrepancy_sweep.sweep_rows(
        case["draft"], [],
        lambda q, k: [NS(doc_id="x", chunk_index=2, page=8, text=case["other"])],
        lambda ids: {"x": case["other_doc"]},
    )
    assert found == [{"doc_id": "x", "doc_name": case["other_doc"], "page": 8, "page_or_section": "p. 8",
                      "chunk_index": 2, "snippet": case["other"]}]


def test_a_failing_search_leaves_the_answer_as_drafted(monkeypatch):
    case = CASES[0].values[0]

    def boom(project_id, query, k):
        raise RuntimeError("store down")

    monkeypatch.setattr(answer_exit, "_lexical_search", boom)
    turn = _turn(case)
    out = answer_exit.check_result(_result(case), turn)
    assert case["stated"] in out["answer"] and len(turn.evidence) == 1


@pytest.mark.parametrize("asking", ["questions", "awaiting_inputs"])
def test_a_turn_that_asks_for_inputs_does_not_search(asking, monkeypatch):
    case = CASES[1].values[0]
    calls: list = []
    _wire(monkeypatch, [("other", case["other"])], calls, case["other_doc"])
    turn = _turn(case)
    getattr(turn, asking).append("contract value")
    answer_exit.check_result(_result(case), turn)
    assert calls == []


@pytest.mark.parametrize(("call", "project"), [
    (lambda chat, agent: chat(agent, "What cover?", project_id="proj-b"), "proj-b"),
    (lambda chat, agent: chat(agent, "What cover?", None, None, "proj-c"), "proj-c"),
])
def test_a_chat_turn_records_the_project_it_answers_from(call, project):
    """The turn record the exit reads carries the chat call's project, by
    name or by position, so the sweep searches that project."""
    import asyncio

    from app.agents.core import loop

    seen: list[str] = []

    class Agent:
        async def _chat_impl(self, user_message, history=None, api_key=None, project_id=None):
            seen.append(answer_exit.current_turn().project_id)
            return {"answer": "", "sources": []}

    asyncio.run(call(loop.chat, Agent()))
    assert seen == [project]


@pytest.mark.parametrize(("project", "query", "k"), [("proj-a", "bs 8500", 20), ("proj-z", "table 6.2", 7)])
def test_the_search_is_the_project_lexical_ranking(project, query, k, monkeypatch):
    """The one search is the store's BM25 ranking over the turn's project."""
    from app.core.rag import vector_store

    seen: list = []
    chunk = NS(doc_id="d", chunk_index=0, page=None, text=query)

    class Store:
        def bm25_search(self, project_id, q, k):
            seen.append((project_id, q, k))
            return [chunk]

    monkeypatch.setattr(vector_store, "get_store", lambda *a, **kw: Store())
    assert answer_exit._lexical_search(project, query, k) == [chunk]
    assert seen == [(project, query, k)]


@pytest.mark.parametrize("names", [{"d1": "Spec A.pdf"}, {"d7": "Notes Rev B.docx", "d8": "Schedule.xlsx"}])
def test_the_passage_documents_are_named_from_the_documents_table(names, monkeypatch):
    from app.core import projects

    seen: list = []
    monkeypatch.setattr(projects, "document_names", lambda ids: seen.append(list(ids)) or dict(names))
    assert answer_exit._document_names(list(names)) == names
    assert seen == [list(names)]


# ── the streamed path, alone and under the router's wrapper ─────────────────

def _stream_agent(case, monkeypatch):
    """An agent whose turn drafts ``case['draft']`` from ``case['own_doc']``,
    the way the runtime records a handed passage in the turn's evidence."""
    from app.agents.runtime import Agent

    monkeypatch.delenv("DRIVER_MODE", raising=False)
    agent = Agent(name="sweep-stream-agent", description="t", system_prompt="t", allowed_blocks=[])

    async def impl(**kwargs):
        answer_exit.current_turn().evidence.append(
            {"doc_id": "own", "doc_name": case["own_doc"], "page": 3, "page_or_section": "p. 3",
             "chunk_index": 1, "snippet": case["own"]})
        yield {"type": "start"}
        yield {"type": "token", "content": case["draft"]}
        yield {"type": "end", "content": case["draft"],
               "sources": [{"doc_id": "own", "doc_name": case["own_doc"], "page_or_section": "p. 3"}]}

    monkeypatch.setattr(agent, "_chat_stream_impl", impl)
    return agent


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("project", ["proj-s1", "proj-s2"])
def test_a_streamed_turn_sweeps_the_project_it_answers_from(case, project, monkeypatch):
    import asyncio

    calls: list = []
    _wire(monkeypatch, [("other", case["other"])], calls, case["other_doc"])
    agent = _stream_agent(case, monkeypatch)

    async def run():
        return [e async for e in agent.chat_stream(case["question"], project_id=project)]

    end = [e for e in asyncio.run(run()) if e.get("type") == "end"][-1]
    assert [c[0] for c in calls] == [project]
    assert case["stated"] in end["content"] and case["differing"] in end["content"]


@pytest.mark.parametrize("case", CASES)
def test_the_router_wrapper_does_not_search_the_same_turn_again(case, monkeypatch):
    """The router's exit runs on the record the agent's stream already
    checked: one search per turn, and the evidence gains only the passage
    that differs."""
    import asyncio
    import json

    from app.routers import exit_frames

    calls: list = []

    def search(project_id, query, k):
        calls.append(query)
        return [NS(doc_id="other", chunk_index=0, page=5, text=case["other"])]

    monkeypatch.setattr(answer_exit, "_lexical_search", search)
    monkeypatch.setattr(answer_exit, "_document_names", lambda ids: {d: case["other_doc"] for d in ids})
    agent = _stream_agent(case, monkeypatch)

    async def frames():
        async for e in agent.chat_stream(case["question"], project_id="proj-r"):
            yield f"data: {json.dumps(e)}\n\n"

    async def run():
        out = [f async for f in exit_frames.with_answer_exit(frames(), case["question"])]
        return out, answer_exit.current_turn()

    out, turn = asyncio.run(run())
    end = json.loads(out[-1][len("data:"):])
    assert len(calls) == 1 and turn.swept
    assert sorted(r["doc_id"] for r in turn.evidence) == ["other", "own"]
    assert case["stated"] in end["content"] and case["differing"] in end["content"]


@pytest.mark.parametrize("case", CASES)
def test_a_turn_is_swept_once(case, monkeypatch):
    """A second exit pass on the same turn (the router's wrapper after the
    agent's stream) never searches again, even for text that still states
    the identifier beside its value."""
    calls: list = []
    _wire(monkeypatch, [("other", case["other"])], calls, case["other_doc"])
    turn = _turn(case)
    answer_exit.check_result(_result(case), turn)
    answer_exit.check_result(_result(case), turn)
    assert len(calls) == 1
    assert sorted(r["doc_id"] for r in turn.evidence) == ["other", "own"]


# ── a cited document's own number is where a value is written ──────────────

LABELS = [
    pytest.param("S-101 General Notes.pdf", "The cover is 40 mm to BS 8500 [S-101 General Notes.pdf, p. 2].",
                 "bs 8500", "s-101", "Cover at the S-101 slab edge shall be 65 mm.", id="sheet-mm"),
    pytest.param("A-205 Grout Plan.pdf", "Grout under the plates is 45 MPa per Table 6.2 [A-205 Grout Plan.pdf].",
                 "table 6.2", "a-205", "A-205 pocket grout is 30 MPa.", id="plan-mpa"),
]


@pytest.mark.parametrize(("doc", "draft", "subject", "label", "label_only"), LABELS)
def test_a_handed_document_number_is_not_searched_or_matched(doc, draft, subject, label, label_only):
    queries: list = []
    found = discrepancy_sweep.sweep_rows(
        draft, [{"doc_id": "own", "doc_name": doc, "chunk_index": 0}],
        lambda q, k: queries.append(q) or [NS(doc_id="x", chunk_index=0, page=1, text=label_only)],
        lambda ids: {"x": "Other.pdf"},
    )
    assert queries == [subject] and found == []


@pytest.mark.parametrize(("doc", "draft", "subject", "label", "label_only"), LABELS)
def test_the_same_number_is_searched_when_no_handed_document_bears_it(doc, draft, subject, label, label_only):
    """Must NOT fire: the number is a subject when no handed document is named by it."""
    queries: list = []
    found = discrepancy_sweep.sweep_rows(
        draft, [{"doc_id": "own", "doc_name": "Unrelated Report.pdf", "chunk_index": 0}],
        lambda q, k: queries.append(q) or [NS(doc_id="x", chunk_index=0, page=1, text=label_only)],
        lambda ids: {"x": "Other.pdf"},
    )
    assert queries == [f"{subject} {label}"]
    assert [r["doc_id"] for r in found] == ["x"]


@pytest.mark.parametrize("draft", [
    "The cover is 40 mm [S-101 General Notes.pdf, p. 2].",
    "Plates take 45 MPa grout [A-205 Grout Plan.pdf].",
])
def test_a_value_beside_only_a_handed_document_number_means_no_search(draft):
    def search(query, k):
        raise AssertionError("searched")

    handed = [{"doc_id": "own", "doc_name": n} for n in ("S-101 General Notes.pdf", "A-205 Grout Plan.pdf")]
    assert discrepancy_sweep.sweep_rows(draft, handed, search, lambda ids: {}) == []


# ── every stated identifier and sentence, the row's name and page ───────────

TWO = [
    pytest.param("The cover is 40 mm to BS 8500. Base plate grout is 45 MPa per Table 6.2.",
                 "bs 8500 table 6.2", id="cover-then-grout"),
    pytest.param("Base plate grout is 45 MPa per Table 6.2. The cover is 40 mm to BS 8500.",
                 "table 6.2 bs 8500", id="grout-then-cover"),
]
_PASSAGES = [NS(doc_id="c", chunk_index=0, page=4, text="Cover to BS 8500 shall be 50 mm."),
             NS(doc_id="g", chunk_index=0, page=6, text="Table 6.2 grout strength 60 MPa.")]


@pytest.mark.parametrize(("draft", "query"), TWO)
def test_every_stated_identifier_is_searched_and_every_sentence_checked(draft, query):
    queries: list = []
    found = discrepancy_sweep.sweep_rows(draft, [], lambda q, k: queries.append(q) or list(_PASSAGES),
                                         lambda ids: {"c": "Cover Notes.pdf", "g": "Grout Notes.pdf"})
    assert queries == [query]
    assert sorted(r["doc_id"] for r in found) == ["c", "g"]


@pytest.mark.parametrize("case", CASES)
def test_the_search_reads_twenty_passages(case, monkeypatch):
    calls: list = []
    _wire(monkeypatch, [("other", case["other"])], calls, case["other_doc"])
    answer_exit.check_result(_result(case), _turn(case))
    assert [c[2] for c in calls] == [20]


@pytest.mark.parametrize(("stored", "shown"), [
    ("C:/proj/docs/Notes Rev B.pdf", "Notes Rev B.pdf"),
    ("uploads/2026/site/Grout Schedule.xlsx", "Grout Schedule.xlsx"),
])
def test_a_document_stored_under_a_path_is_named_by_its_base_name(stored, shown, monkeypatch):
    case = CASES[0].values[0]
    _wire(monkeypatch, [("other", case["other"])], [], stored)
    turn = _turn(case)
    answer_exit.check_result(_result(case), turn)
    assert [r["doc_name"] for r in turn.evidence] == [case["own_doc"], shown]


@pytest.mark.parametrize("stored", ['""', "''"])
def test_a_name_that_reads_as_nothing_is_not_added(stored, monkeypatch):
    case = CASES[1].values[0]
    _wire(monkeypatch, [("other", case["other"])], [], stored)
    turn = _turn(case)
    answer_exit.check_result(_result(case), turn)
    assert [r["doc_id"] for r in turn.evidence] == ["own"]


@pytest.mark.parametrize("page", [True, False, "7"])
def test_a_page_that_is_not_a_number_is_not_shown(page):
    case = CASES[0].values[0]
    found = discrepancy_sweep.sweep_rows(
        case["draft"], [], lambda q, k: [NS(doc_id="x", chunk_index=0, page=page, text=case["other"])],
        lambda ids: {"x": "Other.pdf"},
    )
    assert [(r["page"], r["page_or_section"]) for r in found] == [(None, "")]
