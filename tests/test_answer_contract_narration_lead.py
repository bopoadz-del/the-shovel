"""Narration of the search never opens an answer, on any kind of turn.

Rule 1 moved a figure line to the top only on a turn that computed, so a
document answer (lookup, list, not-found, chain, comparison) could still open
on the writer's own search: "I searched the ...", "I reviewed the ...". The
contract now reads the first sentence of every answer:

* a figure-free narration lead that is process only goes, so the sentence
  after it opens the answer;
* a lead with a figure, a lead that reports something absent (the refusal
  itself, rule 5), a lead that introduces a list, and a lead that is the
  whole answer stay.

Every input here is written by the test; none is a project file or an exam
question.
"""
from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.answer_contract import _narration_first


def _first_line(text: str) -> str:
    return next(ln.strip() for ln in text.split("\n") if ln.strip())


# ── a narration lead that is process only: it goes ──────────────────────────

@pytest.mark.parametrize("text, first", [
    ("I reviewed the door schedule. Fire doors are FD60 throughout.", "Fire doors are FD60 throughout."),
    ("I went through the finishes register. The lobby floor is honed granite.",
     "The lobby floor is honed granite."),
])
def test_process_lead_on_same_line_goes(text, first):
    out = _narration_first(text)
    assert _first_line(out) == first
    assert "I reviewed" not in out and "I went through" not in out


@pytest.mark.parametrize("text, first", [
    ("I searched the specification.\nThe screed is a bonded sand-cement screed.",
     "The screed is a bonded sand-cement screed."),
    ("I scanned the method statements.\n\nThe crane is a luffing-jib tower crane.",
     "The crane is a luffing-jib tower crane."),
])
def test_process_lead_alone_on_its_line_goes(text, first):
    out = _narration_first(text)
    assert _first_line(out) == first
    assert "I searched" not in out and "I scanned" not in out


# ── leads that stay ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "I searched the bill and the topsoil strip is 4,200 m2.\nIt is item 2.1.",
    "I checked the drawing: the beam depth is 600 mm.",
])
def test_narration_with_a_figure_stays(text):
    assert _narration_first(text) == text


@pytest.mark.parametrize("text", [
    "I searched the drainage schedule for a gully spacing and did not find one.\nIt lists only covers.",
    "I checked both schedules and neither returns a painted area. Provide the finishes take-off.",
])
def test_lead_that_reports_an_absence_stays(text):
    assert _narration_first(text) == text


@pytest.mark.parametrize("text", [
    "I checked the specification and the cover is not less than the code minimum.\nSee clause B.",
    "I reviewed the contract and the retention is not released at handover.\nIt is released at the end of defects.",
])
def test_negative_lead_stays(text):
    assert _narration_first(text) == text


@pytest.mark.parametrize("text", [
    "I reviewed the door schedule and found these ironmongery sets:\n- Set A\n- Set B",
    "I searched the register for open items:\n- Glazing sample\n- Paint mock-up",
])
def test_lead_that_introduces_a_list_stays(text):
    assert _narration_first(text) == text


@pytest.mark.parametrize("text", [
    "I reviewed the specification sections on render.",
    "I searched the landscape package.",
])
def test_lead_that_is_the_whole_answer_stays(text):
    assert _narration_first(text) == text


@pytest.mark.parametrize("text", [
    "Icing on the roof is handled by heat trace.\nI searched the spec for more.",
    "The kerbs are precast. I reviewed the bill for their type.",
])
def test_lead_that_is_not_narration_stays(text):
    assert _narration_first(text) == text


# ── through the contract ────────────────────────────────────────────────────

@pytest.mark.parametrize("text, first", [
    ("I searched the soils report.\nThe made ground is a clayey fill over chalk.",
     "The made ground is a clayey fill over chalk."),
    ("I reviewed the lighting layout. The car park uses LED bulkheads.", "The car park uses LED bulkheads."),
])
def test_apply_opens_a_document_answer_on_the_finding(text, first):
    out = answer_contract.apply(text, computed=False).text
    assert _first_line(out) == first
    # Idempotent: a second pass changes nothing.
    assert answer_contract.apply(out, computed=False).text == out


@pytest.mark.parametrize("text", [
    "I searched the soils report.\nThe made ground is a clayey fill over chalk.",
    "I reviewed the lighting layout. The car park uses LED bulkheads.",
])
def test_apply_leaves_an_asking_turn_alone(text):
    assert answer_contract.apply(text, computed=False, asking=True).text == text
