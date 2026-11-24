"""A closing request that opens on a purpose, or on the reader's condition
before the writer's offer, is padding like any other closing request.

Rule 2's closing-request reader (``_drop_closing_request``) let a request
through when it opened on a lead-in other than a condition about the
documents: a purpose ("To get the rate, point me at the priced bill and I'll
extend it") or the reader's condition before the writer's own offer ("If you
tell me which bill it sits under, I'll pull that section"). Both only ask the
reader for a next step, so both now go when they close the answer with no
figure and no marker.

What stays, on both sides of each decision: a purpose lead-in before an
instruction that is the answer itself (no writer named), and a condition on
the reader whose main clause states a fact rather than the writer's offer.

Every input here is written by the test; none is a project file or an exam
question.
"""
from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.answer_contract import _drop_closing_request

_ANSWER = "The basement walls are waterproofed with a bentonite sheet membrane."


# ── a purpose lead-in before a request to the writer: the paragraph goes ────

@pytest.mark.parametrize("closing", [
    "To get the rate, point me at the priced bill that carries the membrane line and I'll extend it.",
    "To confirm the lap detail, upload the manufacturer's installation guide and I'll read it off.",
])
def test_purpose_lead_in_request_goes(closing: str) -> None:
    assert _drop_closing_request(f"{_ANSWER}\n\n{closing}") == _ANSWER


# ── the reader's condition before the writer's offer: the paragraph goes ────

@pytest.mark.parametrize("closing", [
    "If you can tell me which bill or building the membrane sits under, I'll pull that section.",
    "Once you share the approved submittal, I can check it against the specification.",
])
def test_reader_condition_then_writer_offer_goes(closing: str) -> None:
    assert _drop_closing_request(f"{_ANSWER}\n\n{closing}") == _ANSWER


# ── a purpose lead-in before the answer's own instruction: it stays ─────────

@pytest.mark.parametrize("instruction", [
    "To claim the extension, send the notice to the Engineer before the period runs out.",
    "To release the retention, submit the defects certificate to the Employer's representative.",
])
def test_purpose_lead_in_instruction_stays(instruction: str) -> None:
    text = f"{_ANSWER}\n\n{instruction}"
    assert _drop_closing_request(text) == text


# ── the reader's condition before a stated fact: it stays ───────────────────

@pytest.mark.parametrize("statement", [
    "If you open the valve chamber, the meter sits on the left-hand wall.",
    "When you reach the roof level, the membrane turns up behind the parapet coping.",
])
def test_reader_condition_then_fact_stays(statement: str) -> None:
    text = f"{_ANSWER}\n\n{statement}"
    assert _drop_closing_request(text) == text


# ── through the answer exit, both shapes go and the answer stays ────────────

@pytest.mark.parametrize("closing", [
    "To get the billed area, point me at the finishes bill and I'll pull the line.",
    "If you tell me which revision applies, I will read the line from it.",
])
def test_apply_drops_lead_in_request(closing: str) -> None:
    out = answer_contract.apply(f"{_ANSWER}\n\n{closing}")
    assert out.text == _ANSWER


@pytest.mark.parametrize("closing", [
    "To get the billed area, point me at the finishes bill and I'll pull the line.",
    "If you tell me which revision applies, I will read the line from it.",
])
def test_asking_turn_keeps_lead_in_request(closing: str) -> None:
    text = f"{_ANSWER}\n\n{closing}"
    assert answer_contract.apply(text, asking=True).text == text
