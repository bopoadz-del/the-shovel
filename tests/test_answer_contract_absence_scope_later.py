"""Rule 5, the opening paragraph: an absence stated first, its scope stated later.

An answer may open on the absence ("No X is billed in the bill") and state
the scope of its search in a later sentence ("I searched the bill for X and
found none"). That later sentence is an absence in the answer's own scope
form, so the negated opening sentence is the same absence stated first: it
leads, and the rest moves word for word to the details.

The widening is held on both sides. The opening sentence must be negated,
and a later sentence must be an absence in scope form (a negation and "I
searched"). A search scope alone, an absence alone, or a "No." that numbers
something leaves the paragraph whole. Every input is invented here; each
decision is checked from both sides with at least two inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract

_DOC_ROW = {"doc_name": "Works register.pdf", "page_or_section": "p. 9"}


def _apply(text: str, **kw):
    kw.setdefault("lead", True)
    kw.setdefault("sources", [dict(_DOC_ROW)])
    return answer_contract.apply(text, **kw)


def _words(text: str) -> list[str]:
    return text.replace("\n", " ").split()


# ── the scope in a later paragraph: the negated opening sentence leads ──────

@pytest.mark.parametrize("first,rest,scope", [
    ("No 40 mm copper riser is billed in the plumbing bill.",
     "The bill prices risers in 22 mm and 28 mm only, at 310 m and 145 m. "
     "The riser diagram shows the same two sizes.",
     "I searched the bill for 40 mm copper and for risers by size, and found no 40 mm line."),
    ("The landscaping bill does not price any irrigation timer.",
     "Irrigation is billed as drip line, valves and a controller cabinet, at 2,400 m, "
     "18 nr and 1 nr. The planting plan names no timer either.",
     "I searched the landscaping bill and the planting plan for a timer and there was none."),
])
def test_negated_opening_leads_when_a_later_paragraph_states_the_scope(first, rest, scope):
    text = first + " " + rest + "\n\n" + scope
    done = _apply(text)
    assert done.text == first
    assert done.details is not None
    assert done.details.startswith(rest)
    assert sorted(_words(done.text + " " + done.details)) == sorted(_words(text))


@pytest.mark.parametrize("text,first", [
    ("No fire damper is scheduled for the plant room. The damper schedule lists 14 "
     "dampers on floors 1 to 3. I searched the damper schedule and the plant room "
     "drawing and found no plant room damper.",
     "No fire damper is scheduled for the plant room."),
    ("The door schedule does not give a hardware set for door D-07. Doors D-01 to D-06 "
     "carry set H2. I searched the door schedule for D-07 and it is not listed.",
     "The door schedule does not give a hardware set for door D-07."),
    # the stop of "No." before a figure does not end the opening sentence
    ("No 40 mm riser is billed in Bill No. 7 of the plumbing works. Bill No. 7 "
     "prices 22 mm risers only. I searched Bill No. 7 and found no 40 mm line.",
     "No 40 mm riser is billed in Bill No. 7 of the plumbing works."),
    ("No pump is shown on Sheet No. 3 of the drainage set. The run falls at 1:60. "
     "I searched Sheet No. 3 and there is no pump.",
     "No pump is shown on Sheet No. 3 of the drainage set."),
])
def test_negated_opening_leads_when_a_later_sentence_states_the_scope(text, first):
    done = _apply(text)
    assert done.text == first
    assert sorted(_words(done.text + " " + done.details)) == sorted(_words(text))


@pytest.mark.parametrize("text,first,marker", [
    ("No 40 mm copper riser is billed in the plumbing bill. The bill prices 22 mm "
     "and 28 mm risers [1].\n\nI searched the bill and found no 40 mm line.",
     "No 40 mm copper riser is billed in the plumbing bill.", "[1]"),
    ("The landscaping bill does not price an irrigation timer. Irrigation is billed "
     "as drip line and valves [1].\n\nI searched the bill and there is no timer.",
     "The landscaping bill does not price an irrigation timer.", "[1]"),
])
def test_marker_of_a_moved_sentence_rides_on_the_lead(text, first, marker):
    done = _apply(text)
    assert done.text == first + " " + marker


# ── the other side: each condition alone leaves the paragraph whole ─────────

@pytest.mark.parametrize("text", [
    # negated opening, a later search scope, but that scope reports no absence
    "No change was made to the slab grade. The slab is C35 at 250 mm.\n\n"
    "I searched the structural notes and the slab schedule; both give C35.",
    "The roof is not insulated above the plant deck. The deck is a 200 mm slab "
    "with a 1:80 fall.\n\nI searched the roof plan and the section A-A for the build-up.",
])
def test_scope_without_an_absence_leaves_the_opening_whole(text):
    done = _apply(text)
    opening = text.split("\n\n")[0]
    assert done.text == opening


@pytest.mark.parametrize("text", [
    # a later absence in scope form, but the opening sentence states, not denies
    "The plumbing bill prices copper risers in 22 mm and 28 mm. They total 455 m.\n\n"
    "I searched the bill for a 40 mm riser and found none.",
    "Irrigation is billed as drip line, valves and a controller cabinet. The drip "
    "line is 2,400 m.\n\nI searched the bill for a timer and it is not there.",
])
def test_stated_opening_stays_whole_beside_a_scoped_absence(text):
    done = _apply(text)
    assert done.text == text.split("\n\n")[0]


@pytest.mark.parametrize("text", [
    # negated opening, no scope stated anywhere
    "No 40 mm copper riser is billed in the plumbing bill. The bill prices risers "
    "in 22 mm and 28 mm only.\n\nThe riser diagram shows the same two sizes.",
    "The landscaping bill does not price an irrigation timer. Irrigation is billed "
    "as drip line and valves.\n\nThe planting plan names a controller cabinet.",
])
def test_negated_opening_without_any_scope_stays_whole(text):
    done = _apply(text)
    assert done.text == text.split("\n\n")[0]


@pytest.mark.parametrize("text", [
    # "No." numbers a bill or a sheet; it is not a negation
    "Bill No. 4 prices the gas pipework in 63 mm and 90 mm. It totals 1,820 m.\n\n"
    "I searched the bill for 110 mm pipe and found none.",
    "Sheet No. 12 shows the drainage run at 1:60. The run is 84 m long.\n\n"
    "I searched the drawings for a pump and there is no pump.",
])
def test_number_abbreviation_is_not_a_negation(text):
    done = _apply(text)
    assert done.text == text.split("\n\n")[0]


@pytest.mark.parametrize("kw", [{"computed": True}, {"lead": False}])
def test_turns_rule_5_does_not_run_on_stay_whole(kw):
    text = ("No 40 mm copper riser is billed in the plumbing bill. The bill prices 22 mm "
            "and 28 mm risers.\n\nI searched the bill and found no 40 mm line.")
    done = _apply(text, **kw)
    assert done.details is None
    assert "The bill prices 22 mm" in done.text
