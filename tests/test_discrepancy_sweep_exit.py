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
