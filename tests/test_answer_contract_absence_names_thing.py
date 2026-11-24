"""An absence answer opens on the missing thing, on any kind of turn.

Rule 1 of the answer contract (project-assistant prompt) gives the form of an
absence: "<What> is not in the project documents I searched." The model often
writes it the other way round -- "No <thing> is in the documents I searched"
-- and nothing at the exit held it to the contract's form. The exit now turns
a first line that opens "No <thing> is <presence predicate>" into "A <thing>
is not <predicate>":

* turned: a short noun phrase after "No" and a presence predicate (a place
  "in the / on the / in any ...", or stated, listed, billed, named ...);
* left as written: "No" opening no noun phrase ("No, ...", "No one",
  "No more than"), a predicate of size or degree ("longer than", "within
  tolerance", "in excess of"), an already-negated verb, a noun phrase that
  holds a verb, any line but the first, and an asking turn.

Every input here is written by the test; none is a project file or an exam
question.
"""
from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.answer_contract import _absence_names_thing


def _first_line(text: str) -> str:
    return next(ln.strip() for ln in text.split("\n") if ln.strip())


# ── turned: the thing leads, the verb is negated ────────────────────────────

@pytest.mark.parametrize("text, first", [
    ("No acoustic ceiling area is in the project documents I searched.",
     "An acoustic ceiling area is not in the project documents I searched."),
    ("No lift pit depth is stated in the structural notes.",
     "A lift pit depth is not stated in the structural notes."),
    ("No PVC conduit is billed in the electrical bill (Section 7).",
     "A PVC conduit is not billed in the electrical bill (Section 7)."),
    ("No handrail finishes are listed in the finishes register.",
     "Handrail finishes are not listed in the finishes register."),
    ("No surveyor was named in any appointment letter.",
     "A surveyor was not named in any appointment letter."),
])
def test_no_thing_lead_is_turned(text, first):
    assert _first_line(_absence_names_thing(text)) == first


def test_rest_of_the_line_and_answer_are_kept():
    text = ("No gutter length is billed in the roofing bill; the bill prices gutters as a lump sum [1].\n\n"
            "The roof plan shows 4 downpipes [2].")
    out = _absence_names_thing(text)
    assert out == ("A gutter length is not billed in the roofing bill; the bill prices gutters as a lump sum [1].\n\n"
                   "The roof plan shows 4 downpipes [2].")


def test_bold_lead_keeps_its_mark():
    out = _absence_names_thing("**No door hardware schedule is included** in the tender set.")
    assert out == "**A door hardware schedule is not included** in the tender set."


# ── left as written ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "No, the screed is 75 mm thick.",                       # "No" is an answer word
    "No one is named as the fire engineer.",                # not a thing
    "No more than 3 m is allowed between joints.",          # a limit
    "No pile is longer than 18 m.",                         # degree: universal claim
    "No anchor is within tolerance.",                       # degree, not a place
    "No slab is in excess of 300 mm thick.",                # "in" not before a place
    "No sample is not tested.",                             # verb already negated
    "No contractor who has tendered is listed in the register.",  # noun phrase holds a verb
    "No change is required to the pour sequence.",          # not a presence predicate
    "The kerb length is not in the bill.",                  # already in the contract's form
    "No rate on file for that - upload your priced rate schedule.",  # no "is"
])
def test_other_leads_stay(text):
    assert _absence_names_thing(text) == text


def test_only_the_first_line_is_read():
    text = "Cladding area: 1,240 m2 [1].\n\nNo soffit area is billed in the bill."
    assert _absence_names_thing(text) == text


def test_long_phrase_after_no_stays():
    text = ("No figure for the external works hard landscaping paving and kerbs and edging and drainage "
            "channels around the plant building is stated in the bill.")
    assert _absence_names_thing(text) == text


# ── at the exit ─────────────────────────────────────────────────────────────

def test_apply_turns_a_document_answer():
    out = answer_contract.apply("No fire stopping detail is shown in the drawings I searched.")
    assert _first_line(out.text) == "A fire stopping detail is not shown in the drawings I searched."


def test_apply_leaves_an_asking_turn():
    text = "No wall height is stated in the drawings I searched. What height should I use?"
    assert answer_contract.apply(text, asking=True).text == text
