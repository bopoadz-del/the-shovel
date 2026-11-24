"""The discrepancy sweep's caller half (STREAM A2): after a drafted answer
states a typed value beside a distinctive identifier, the exit runs ONE
lexical search for that identifier; a passage the turn was not handed that
states a different figure for the same identified thing joins the evidence,
and the existing conflict statement reports every value with its document.

Every test builds its own invented documents; the search is replaced by a
recorder so no index is needed. Each decision has two inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_exit as ae
from app.agents.core import loop as agent_loop
from app.agents.discrepancy_sweep import sweep_targets


@pytest.fixture(autouse=True)
def _sweep_on(monkeypatch):
    monkeypatch.setenv("DISCREPANCY_SWEEP", "1")


# The user's question for each invented project's draft. The conflict
# statement reads it to tell which values answer the asked thing (C2).
_QUESTIONS = {
    "Harbour": "What is the nominal cover to the retaining wall?",
    "Ridgeway": "What is the deck slab thickness?",
}


def _question(answer):
    return next(q for project, q in _QUESTIONS.items() if project in answer)


def _turn(handed, project_id="proj-1", question=""):
    turn = ae.Turn(question=question, project_id=project_id)
    for doc_id, name, chunk, text in handed:
        turn.evidence.append({"doc_id": doc_id, "doc_name": name, "page": None,
                              "page_or_section": "", "chunk_index": chunk, "snippet": text})
    return turn


def _run(monkeypatch, answer, handed, found, project_id="proj-1", fail=False):
    calls: list[tuple[str, str]] = []

    def search(pid, query):
        calls.append((pid, query))
        if fail:
            raise RuntimeError("index unavailable")
        return list(found)

    monkeypatch.setattr(ae, "_sweep_search", search)
    turn = _turn(handed, project_id, _question(answer))
    rows = [{"doc_id": d, "doc_name": n} for d, n, _c, _t in handed]
    out = ae.check_result({"answer": answer, "sources": rows}, turn)
    return out, calls, turn


# Same identified thing, a different figure in a document the turn was not
# handed: the answer states both values with both documents.
_DIFFER = [
    ("The nominal cover to the retaining wall per Table 7.3 is 50 mm [Harbour Spec.pdf].",
     [("d1", "Harbour Spec.pdf", 3, "Table 7.3: nominal cover to the retaining wall 50 mm.")],
     [("d2", "Harbour Wall Drawing S-201.pdf", 7, None,
       "Table 7.3: nominal cover to the retaining wall 40 mm.")],
     {("Harbour Spec.pdf", "50 mm"), ("Harbour Wall Drawing S-201.pdf", "40 mm")}),
    ("Under Clause 12.4 the deck slab thickness is 250 mm [Ridgeway Specification.pdf].",
     [("e1", "Ridgeway Specification.pdf", 1, "Clause 12.4: the deck slab thickness is 250 mm.")],
     [("e2", "Ridgeway Deck Section B-02.pdf", 4, 2,
       "Clause 12.4: the deck slab thickness is 300 mm.")],
     {("Ridgeway Specification.pdf", "250 mm"), ("Ridgeway Deck Section B-02.pdf", "300 mm")}),
]


@pytest.mark.parametrize("answer, handed, found, pairs", _DIFFER)
def test_a_differing_value_elsewhere_is_stated_beside_the_drafted_one(monkeypatch, answer, handed, found, pairs):
    out, calls, turn = _run(monkeypatch, answer, handed, found)
    assert len(calls) == 1
    assert "differ" in out["answer"]
    conflicts = out.get("conflicts") or []
    assert len(conflicts) == 1
    assert {(v["document"], v["value"]) for v in conflicts[0]["values"]} == pairs
    assert any(e["doc_id"] == found[0][0] for e in turn.evidence)


@pytest.mark.parametrize("answer, handed, found, pairs", _DIFFER)
def test_without_the_sweep_the_drafted_value_stands_alone(monkeypatch, answer, handed, found, pairs):
    monkeypatch.setenv("DISCREPANCY_SWEEP", "0")
    out, calls, _ = _run(monkeypatch, answer, handed, found)
    assert calls == []
    assert "differ" not in out["answer"]
    assert not out.get("conflicts")


# The same identifier, but a different structure: never a conflict (owner
# 14:58Z hard negative). The passage may join the evidence; the conflict
# statement decides it is another thing.
_OTHER_THING = [
    ("The nominal cover to the retaining wall per Table 7.3 is 50 mm [Harbour Spec.pdf].",
     [("d1", "Harbour Spec.pdf", 3, "Table 7.3: nominal cover to the retaining wall 50 mm.")],
     [("d2", "Harbour Kiosk Drawing A-110.pdf", 7, None, "Table 7.3: nominal cover to the kiosk slab 40 mm.")]),
    ("Under Clause 12.4 the deck slab thickness is 250 mm [Ridgeway Specification.pdf].",
     [("e1", "Ridgeway Specification.pdf", 1, "Clause 12.4: the deck slab thickness is 250 mm.")],
     [("e2", "Ridgeway Plant Room A-07.pdf", 4, None, "Clause 12.4: the plant room plinth height is 150 mm.")]),
]


@pytest.mark.parametrize("answer, handed, found", _OTHER_THING)
def test_a_different_structure_sharing_the_identifier_is_not_a_conflict(monkeypatch, answer, handed, found):
    out, calls, _ = _run(monkeypatch, answer, handed, found)
    assert len(calls) == 1
    assert "differ" not in out["answer"]
    assert not out.get("conflicts")


# The same figure (or a rounding of it) elsewhere is no difference: nothing joins.
_SAME = [
    ("The nominal cover to the retaining wall per Table 7.3 is 50 mm [Harbour Spec.pdf].",
     [("d1", "Harbour Spec.pdf", 3, "Table 7.3: nominal cover to the retaining wall 50 mm.")],
     [("d2", "Harbour Wall Drawing S-201.pdf", 7, None, "Table 7.3: nominal cover to the retaining wall 50 mm.")]),
    ("Under Clause 12.4 the deck slab thickness is 250 mm [Ridgeway Specification.pdf].",
     [("e1", "Ridgeway Specification.pdf", 1, "Clause 12.4: the deck slab thickness is 250 mm.")],
     [("e2", "Ridgeway Deck Section B-02.pdf", 4, None, "Clause 12.4: deck slab thickness 250 mm.")]),
]


@pytest.mark.parametrize("answer, handed, found", _SAME)
def test_the_same_figure_elsewhere_joins_nothing(monkeypatch, answer, handed, found):
    out, calls, turn = _run(monkeypatch, answer, handed, found)
    assert len(calls) == 1
    assert len(turn.evidence) == len(handed)
    assert not out.get("conflicts")


# The search runs only when the draft states a typed value beside an identifier.
@pytest.mark.parametrize("answer", [
    "The retaining wall cover is 50 mm [Harbour Spec.pdf].",  # a value, no identifier
    "Table 7.3 sets the cover for the retaining wall [Harbour Spec.pdf].",  # an identifier, no value
])
def test_no_value_beside_an_identifier_runs_no_search(monkeypatch, answer):
    handed = [("d1", "Harbour Spec.pdf", 3, "Table 7.3: nominal cover to the retaining wall 50 mm.")]
    _out, calls, _ = _run(monkeypatch, answer, handed, [])
    assert calls == []


@pytest.mark.parametrize("project_id", ["", None])
def test_a_turn_with_no_project_runs_no_search(monkeypatch, project_id):
    answer, handed, found, _pairs = _DIFFER[0]
    _out, calls, _ = _run(monkeypatch, answer, handed, found, project_id=project_id or "")
    assert calls == []


@pytest.mark.parametrize("case, project_id, query", [
    (0, "proj-31", "table 7.3"),
    (1, "proj-77", "clause 12.4"),
])
def test_the_search_is_run_on_the_turns_project_with_the_identifiers(monkeypatch, case, project_id, query):
    answer, handed, found, _pairs = _DIFFER[case]
    _out, calls, _ = _run(monkeypatch, answer, handed, found, project_id=project_id)
    assert calls == [(project_id, query)]


@pytest.mark.parametrize("case", [0, 1])
def test_a_passage_already_handed_is_not_added_twice(monkeypatch, case):
    answer, handed, _found, _pairs = _DIFFER[case]
    again = [(handed[0][0], handed[0][1], handed[0][2], None, handed[0][3].replace("50", "40").replace("250", "300"))]
    _out, calls, turn = _run(monkeypatch, answer, handed, again)
    assert len(calls) == 1
    assert len(turn.evidence) == 1


@pytest.mark.parametrize("case", [0, 1])
def test_an_unnamed_document_does_not_join(monkeypatch, case):
    answer, handed, found, _pairs = _DIFFER[case]
    nameless = [(found[0][0], "", found[0][2], None, found[0][4])]
    out, _calls, turn = _run(monkeypatch, answer, handed, nameless)
    assert len(turn.evidence) == 1
    assert not out.get("conflicts")


@pytest.mark.parametrize("case", [0, 1])
def test_a_failed_search_leaves_the_answer_as_drafted(monkeypatch, case):
    answer, handed, found, _pairs = _DIFFER[case]
    out, calls, turn = _run(monkeypatch, answer, handed, found, fail=True)
    assert len(calls) == 1
    assert "differ" not in out["answer"]
    assert len(turn.evidence) == 1


@pytest.mark.parametrize("case", [0, 1])
def test_the_exit_checking_the_same_answer_twice_searches_once(monkeypatch, case):
    answer, handed, found, _pairs = _DIFFER[case]
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(ae, "_sweep_search", lambda pid, q: calls.append((pid, q)) or list(found))
    turn = _turn(handed)
    assert ae.sweep_evidence(answer, turn) == 1
    assert ae.sweep_evidence(answer, turn) == 0
    assert len(calls) == 1


@pytest.mark.parametrize("case", [0, 1])
def test_an_answer_asking_for_inputs_is_not_swept(monkeypatch, case):
    answer, handed, found, _pairs = _DIFFER[case]
    calls: list = []
    monkeypatch.setattr(ae, "_sweep_search", lambda pid, q: calls.append(q) or list(found))
    turn = _turn(handed)
    turn.awaiting_inputs.append("contract_value")
    ae.check_result({"answer": answer, "sources": []}, turn)
    assert calls == []


# ── the pure targets reader ────────────────────────────────────────────────

@pytest.mark.parametrize("draft, stated, query", [
    ("The cover to Table 7.3 is 50 mm. It is shown on the drawing.",
     ["The cover to Table 7.3 is 50 mm."], "table 7.3"),
    ("Per BS 8500 the cover is 40 mm.\nUnder Clause 4.2 the bar spacing is 200 mm.",
     ["Per BS 8500 the cover is 40 mm.", "Under Clause 4.2 the bar spacing is 200 mm."], "bs 8500 clause 4.2"),
])
def test_targets_are_the_sentences_stating_a_value_beside_an_identifier(draft, stated, query):
    assert sweep_targets(draft) == (stated, query)


@pytest.mark.parametrize("draft", [
    "Depth 5 m is the excavation.",  # the only 'designation' is the figure itself
    "Excavate 3 m below the slab.",
])
def test_a_figure_is_never_its_own_identifier(draft):
    assert sweep_targets(draft) == ([], "")


# ── the turn learns its project ────────────────────────────────────────────

class _Agent:
    async def _chat_impl(self, user_message, history=None, api_key=None, project_id=None):
        return {}


@pytest.mark.parametrize("args, kwargs, expected", [
    (("hello",), {"project_id": "proj-9"}, "proj-9"),
    (("hello", None, None, "proj-4"), {}, "proj-4"),
    (("hello",), {}, ""),
])
def test_a_chat_call_names_its_project_by_name_or_position(args, kwargs, expected):
    assert agent_loop._chat_arg(_Agent(), args, kwargs, "project_id") == expected
