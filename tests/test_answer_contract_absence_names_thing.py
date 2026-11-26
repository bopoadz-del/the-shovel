"""An absence answer opens on the missing thing, on any kind of turn.

Rule 1 of the answer contract (project-assistant prompt) gives the form of an
absence: "<What> is not in the project documents I searched." The model often
writes it the other way round -- "No <thing> is in the documents I searched"
-- and nothing at the exit held it to the contract's form. The exit now turns
a first line that opens "No <thing> is <presence predicate>" into "The <thing>
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
     "The acoustic ceiling area is not in the project documents I searched."),
    ("No lift pit depth is stated in the structural notes.",
     "The lift pit depth is not stated in the structural notes."),
    ("No PVC conduit is billed in the electrical bill (Section 7).",
     "The PVC conduit is not billed in the electrical bill (Section 7)."),
    ("No handrail finishes are listed in the finishes register.",
     "Handrail finishes are not listed in the finishes register."),
    ("No surveyor was named in any appointment letter.",
     "The surveyor was not named in any appointment letter."),
])
def test_no_thing_lead_is_turned(text, first):
    assert _first_line(_absence_names_thing(text)) == first


@pytest.mark.parametrize("text, turned", [
    ("No gutter length is billed in the roofing bill; the bill prices gutters as a lump sum [1].\n\n"
     "The roof plan shows 4 downpipes [2].",
     "The gutter length is not billed in the roofing bill; the bill prices gutters as a lump sum [1].\n\n"
     "The roof plan shows 4 downpipes [2]."),
    ("\n  No expansion joint is shown on the slab plan - the joints are by others.\nSee note 3 [1].",
     "\n  The expansion joint is not shown on the slab plan - the joints are by others.\nSee note 3 [1]."),
])
def test_rest_of_the_line_and_answer_are_kept(text, turned):
    assert _absence_names_thing(text) == turned


@pytest.mark.parametrize("text, turned", [
    ("**No door hardware schedule is included** in the tender set.",
     "**The door hardware schedule is not included** in the tender set."),
    ("**No paint systems are specified** for the plant room.",
     "**Paint systems are not specified** for the plant room."),
])
def test_bold_lead_keeps_its_mark(text, turned):
    assert _absence_names_thing(text) == turned


@pytest.mark.parametrize("text, first", [
    # uncountable nouns: never "An information", "A data"
    ("No information is provided in the project documents I searched.",
     "The information is not provided in the project documents I searched."),
    ("No evidence is recorded in the site diary.",
     "The evidence is not recorded in the site diary."),
    # an h-word and a figure: never "A hourly", "A 11 kV"
    ("No hourly rate is stated for the excavator.",
     "The hourly rate is not stated for the excavator."),
    ("No 11 kV cable is listed in the bill.",
     "The 11 kV cable is not listed in the bill."),
    ("No 8 mm bar is scheduled in the bending schedule.",
     "The 8 mm bar is not scheduled in the bending schedule."),
])
def test_singular_thing_takes_the_never_a_guessed_article(text, first):
    out = _first_line(_absence_names_thing(text))
    assert out == first
    assert not out.startswith(("A ", "An "))


@pytest.mark.parametrize("text", [
    "No drawing without a revision number is listed in the register.",
    "No pump with no duty point is scheduled in the plant list.",
])
def test_a_negated_noun_phrase_stays(text):
    # "No X without Y is listed" is not "X without Y is not listed": the
    # negative inside the noun phrase changes the claim, so it is left.
    assert _absence_names_thing(text) == text


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
    "",                                                     # nothing to read
    "   \n\n  ",                                            # blank lines only
])
def test_other_leads_stay(text):
    assert _absence_names_thing(text) == text


@pytest.mark.parametrize("text", [
    "Cladding area: 1,240 m2 [1].\n\nNo soffit area is billed in the bill.",
    "The bill has 3 sections [1].\nNo lift shaft lining is listed in section 2.",
])
def test_only_the_first_line_is_read(text):
    assert _absence_names_thing(text) == text


@pytest.mark.parametrize("text", [
    "No figure for the external works hard landscaping paving and kerbs and edging and drainage "
    "channels around the plant building is stated in the bill.",
    "No allowance for the temporary works access scaffold platforms ramps hoists and edge protection "
    "to every elevation of the block is included in the prelims.",
])
def test_long_phrase_after_no_stays(text):
    assert _absence_names_thing(text) == text


# ── at the exit ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, first", [
    ("No fire stopping detail is shown in the drawings I searched.",
     "The fire stopping detail is not shown in the drawings I searched."),
    ("No unit rate for topsoil is priced in the bill I searched.",
     "The unit rate for topsoil is not priced in the bill I searched."),
])
def test_apply_turns_a_document_answer(text, first):
    assert _first_line(answer_contract.apply(text).text) == first


@pytest.mark.parametrize("text", [
    "No wall height is stated in the drawings I searched. What height should I use?",
    "No concrete grade is specified in the notes I searched. Which grade applies?",
])
def test_apply_leaves_an_asking_turn(text):
    assert answer_contract.apply(text, asking=True).text == text


# ── both rules at the exit: an overridden absence goes, a true one is turned ─

@pytest.mark.parametrize("text, first", [
    ("No formwork rate is given.\n\nThe formwork rate is 35 per m2 [Bill.pdf, p. 2].",
     "The formwork rate is 35 per m2 [1]."),
    ("No pump duty is given.\n\nThe pump duty is 45 l/s [Notes.pdf, p. 4].",
     "The pump duty is 45 l/s [1]."),
])
def test_apply_drops_an_overridden_absence_before_any_turn(text, first):
    assert _first_line(answer_contract.apply(text).text) == first


@pytest.mark.parametrize("text, first", [
    ("No rate for formwork is stated in the bill.\n\nThe rebar rate is 2,400 per t [Bill.pdf, p. 2].",
     "The rate for formwork is not stated in the bill."),
    ("No pump duty is stated in the notes.\n\nThe fan duty is 3 m3/s [Notes.pdf, p. 4].",
     "The pump duty is not stated in the notes."),
])
def test_apply_turns_an_absence_nothing_overrides(text, first):
    assert _first_line(answer_contract.apply(text).text) == first


@pytest.mark.parametrize("text, first", [
    ("No formwork rate is given in the schedule.\n\nThe formwork rate in the schedule is 35 per m2 [Bill.pdf, p. 2].",
     "The formwork rate in the schedule is 35 per m2 [1]."),
    ("No pump duty is given in the notes.\n\nThe pump duty in the notes is 45 l/s [Notes.pdf, p. 4].",
     "The pump duty in the notes is 45 l/s [1]."),
])
def test_the_overridden_absence_is_read_before_it_is_turned(text, first):
    # Turned first, "The X is not given in the Y" reads as a negation between
    # two named things (not an absence) and would survive next to its own
    # cited figure. The drop must read the line as written.
    assert _first_line(answer_contract.apply(text).text) == first
