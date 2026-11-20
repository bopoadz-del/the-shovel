"""G-7 (#370 grader FAIL): the turn's question reaches answer_contract.apply
from the product call sites, and the conflict sentence is built from its
parts, never patched with string replaces afterwards.

Every document name, figure and question here is invented for this file.
Each answer below names no thing of its own ("It is 450 mm."), so only the
user's question can say which evidence values answer it: a conflict is
flagged when the question arrives and not when it is lost.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import answer_contract, answer_exit
from app.agents.answer_contract import _conflict_lines, _Markers
from app.routers.exit_frames import with_answer_exit


def _row(doc: str, text: str) -> dict:
    return {"doc_name": doc, "page_or_section": "p. 1", "snippet": text}


_CASES = [
    ("What is the transfer slab depth?", "It is 450 mm.",
     [_row("Brookline Structural Notes.pdf", "Transfer slab depth 450 mm."),
      _row("Brookline Slab Drawing S-210.pdf", "Transfer slab depth 500 mm.")],
     {("Brookline Structural Notes.pdf", "450 mm"), ("Brookline Slab Drawing S-210.pdf", "500 mm")}),
    # A kind word and the words that run straight into it ("curing period")
    # name only the kind (#453); the thing here is the podium deck.
    ("What is the curing period of the podium deck?", "That is 21 days.",
     [_row("Millbrook Specification.pdf", "Podium deck curing period 21 days."),
      _row("Millbrook Method Statement.pdf", "Podium deck curing period 28 days.")],
     {("Millbrook Specification.pdf", "21 days"), ("Millbrook Method Statement.pdf", "28 days")}),
]


def _frame(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _end_through_router(question: str, answer: str, rows: list[dict], *, agent_opens: str | None = None) -> dict:
    """The end frame as the real router wrapper lets it leave."""

    async def frames():
        if agent_opens is not None:
            # The agent loop opens its own record in the same context.
            answer_exit.open_turn(agent_opens)
        yield _frame({"type": "token", "content": answer})
        yield _frame({"type": "end", "content": answer, "sources": rows})

    out = [f async for f in with_answer_exit(frames(), question)]
    return json.loads(out[-1][len("data: "):])


def _pairs(conflict: dict) -> set[tuple[str, str]]:
    return {(v["document"], v["value"]) for v in conflict["values"]}


@pytest.mark.parametrize("question, answer, rows, pairs", _CASES)
def test_router_question_reaches_the_contract(question, answer, rows, pairs):
    end = asyncio.run(_end_through_router(question, answer, rows))
    assert len(end["conflicts"]) == 1
    assert _pairs(end["conflicts"][0]) == pairs
    assert "The documents differ on this:" in end["content"]


@pytest.mark.parametrize("question, answer, rows, pairs", _CASES)
def test_agent_record_question_reaches_the_contract(question, answer, rows, pairs):
    # The agent's own record carries the question; the router passes none.
    end = asyncio.run(_end_through_router("", answer, rows, agent_opens=question))
    assert _pairs(end["conflicts"][0]) == pairs


@pytest.mark.parametrize("question, answer, rows, pairs", _CASES)
def test_router_question_fills_an_agent_record_opened_without_one(question, answer, rows, pairs):
    end = asyncio.run(_end_through_router(question, answer, rows, agent_opens=""))
    assert _pairs(end["conflicts"][0]) == pairs


@pytest.mark.parametrize("question, answer, rows, pairs", _CASES)
def test_without_the_question_the_same_turn_flags_nothing(question, answer, rows, pairs):
    # Same answer and evidence: the question alone carries the matching words.
    end = asyncio.run(_end_through_router("", answer, rows))
    assert "conflicts" not in end
    other = asyncio.run(_end_through_router("Who signs the site diary?", answer, rows))
    assert "conflicts" not in other


@pytest.mark.parametrize("question, answer, rows, pairs", _CASES)
def test_whole_turn_result_carries_the_question(question, answer, rows, pairs):
    turn, token = answer_exit.begin_turn(question)
    try:
        fixed = answer_exit.check_result({"answer": answer, "sources": rows}, turn)
    finally:
        answer_exit.reset_turn(token)
    assert _pairs(fixed["conflicts"][0]) == pairs
    bare, token = answer_exit.begin_turn()
    try:
        plain = answer_exit.check_result({"answer": answer, "sources": rows}, bare)
    finally:
        answer_exit.reset_turn(token)
    assert "conflicts" not in plain


# ── the sentence is joined from its parts, nothing edited afterwards ────────

def _found(gval: str, other: str, kinds: tuple[str, str], basis: str) -> list[dict]:
    a = _row("Harbourside Particular Conditions.pdf", f"Value {gval}.")
    b = _row("Harbourside Drawing C-101.pdf", f"Value {other}.")
    return [{"values": [{"value": gval, "document": a["doc_name"], "row": a},
                        {"value": other, "document": b["doc_name"], "row": b}],
             "governs": a["doc_name"], "basis_document": basis,
             "governing_kind": kinds[0], "lower_kind": kinds[1]}], [a, b]


@pytest.mark.parametrize("gval, other, kinds", [
    ("12  m", "14 m", ("Particular  Conditions", "Drawings")),
    ("grade C40 .", "grade C35", ("Contract Data", "Specification  Part 2")),
])
def test_values_and_kinds_are_quoted_exactly(gval, other, kinds):
    found, rows = _found(gval, other, kinds, "Harbourside Particular Conditions.pdf")
    lines, _ = _conflict_lines(found, _Markers(rows))
    assert f" {gval} governs: the stated order of precedence puts the {kinds[0]} above the {kinds[1]} [1]." in lines[0]


@pytest.mark.parametrize("basis", ["Harbourside Conditions Index.pdf", "Unlisted Order Of Precedence.pdf"])
def test_basis_outside_the_evidence_ends_without_a_marker(basis):
    found, rows = _found("12 m", "14 m", ("Particular Conditions", "Drawings"), basis)
    lines, structured = _conflict_lines(found, _Markers(rows))
    assert lines[0].endswith("puts the Particular Conditions above the Drawings.")
    assert structured[0]["basis_document"] == basis


@pytest.mark.parametrize("question, answer, rows, pairs", _CASES)
def test_contract_apply_takes_the_question_directly(question, answer, rows, pairs):
    assert _pairs(answer_contract.apply(answer, sources=rows, question=question).conflicts[0]) == pairs
    assert answer_contract.apply(answer, sources=rows).conflicts == []
