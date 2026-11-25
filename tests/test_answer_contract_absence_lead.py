"""An absence lead names the missing thing first: the answer exit moves a
negation that stands on the subject ("No X is in ...", "There is no X in
...") to the verb ("X is not in ..."), word for word otherwise, and leaves
every other lead alone. Each test builds its own input; every rule has at
least two inputs where it fires and two where it must not."""
from __future__ import annotations

import re
from collections import Counter

import pytest

from app.agents import answer_contract as ac


def _first_line(text: str) -> str:
    return next(ln for ln in text.split("\n") if ln.strip())


def _words(text: str) -> Counter:
    return Counter(w.lower() for w in re.findall(r"[\w']+", text))


# ---- fires: negation on the subject -------------------------------------

@pytest.mark.parametrize("lead,expected", [
    ("No billed rate for lift pit waterproofing is in the project documents I searched.",
     "Billed rate for lift pit waterproofing is not in the project documents I searched."),
    ("No acoustic ceiling tile quantity was stated in the tender drawings I searched.",
     "Acoustic ceiling tile quantity was not stated in the tender drawings I searched."),
    ("No rebar couplers are listed in the priced schedule I read.",
     "Rebar couplers are not listed in the priced schedule I read."),
])
def test_no_subject_lead_moves_negation_to_verb(lead, expected):
    assert ac._absence_lead(lead, computed=False) == expected


@pytest.mark.parametrize("lead,expected", [
    ("There is no PN16 fire main valve in the priced bill I searched.",
     "PN16 fire main valve is not in the priced bill I searched."),
    ("There are no curtain wall anchor loads in the structural notes I read.",
     "Curtain wall anchor loads are not in the structural notes I read."),
    ("There was no handover date in the programme I searched.",
     "Handover date was not in the programme I searched."),
])
def test_there_is_no_lead_moves_negation_to_verb(lead, expected):
    assert ac._absence_lead(lead, computed=False) == expected


@pytest.mark.parametrize("text,tail", [
    ("No billed rate for lift pit waterproofing is in the project documents I searched. "
     "The bill prices the pit walls only.\n\n- Section 3 lists the pit.",
     " The bill prices the pit walls only.\n\n- Section 3 lists the pit."),
    ("  There are no curtain wall anchor loads in the structural notes I read - "
     "the notes stop at level 4.  Upload the facade calc.",
     "  Upload the facade calc."),
])
def test_only_the_negation_moves_rest_is_byte_identical(text, tail):
    out = ac._absence_lead(text, computed=False)
    assert out != text
    # Content words: only "no"/"there" leave and "not" arrives.
    assert _words(out) - _words(text) == Counter({"not": 1})
    gone = _words(text) - _words(out)
    assert set(gone) <= {"no", "there"} and gone["no"] == 1
    # Everything after the first sentence, and the indent, is untouched.
    assert out.endswith(tail)
    assert out[: len(text) - len(text.lstrip())] == text[: len(text) - len(text.lstrip())]


def test_markdown_emphasis_before_the_lead_is_kept():
    assert ac._absence_lead("**No pile cap depth is in the drawings I searched.**", False) == \
        "**Pile cap depth is not in the drawings I searched.**"
    assert ac._absence_lead("_There is no slab camber in the notes I read._", False) == \
        "_Slab camber is not in the notes I read._"


# ---- must NOT fire ------------------------------------------------------

@pytest.mark.parametrize("lead", [
    "Lift pit waterproofing rate is not in the project documents I searched.",
    "Acoustic ceiling tile quantity is not stated in the tender drawings I searched.",
])
def test_lead_already_in_prescribed_form_is_unchanged(lead):
    assert ac._absence_lead(lead, computed=False) == lead


@pytest.mark.parametrize("lead", [
    "No, the slab is 250 mm thick [1].",
    "No. The handover date is 14 March [2].",
])
def test_no_as_an_answer_word_is_unchanged(lead):
    assert ac._absence_lead(lead, computed=False) == lead


@pytest.mark.parametrize("lead", [
    "No more than 3 bars are in each bundle.",
    "No other document is in the tender pack.",
    "No one is in the site office after 18:00.",
    "No such clause was stated in the contract I read.",
])
def test_no_that_does_not_open_the_missing_thing_is_unchanged(lead):
    assert ac._absence_lead(lead, computed=False) == lead


@pytest.mark.parametrize("lead", [
    "No rate is given in the documents for the lift pit.",
    "There is no mention in the specification about joint sealant.",
    "There is no line in the bill of quantities I read.",
])
def test_scope_that_names_the_topic_is_unchanged(lead):
    assert ac._absence_lead(lead, computed=False) == lead


@pytest.mark.parametrize("lead", [
    "No rate higher than the tender rate is in the bill I read.",
    "No rate [1] is in the bill I read.",
    "No permit is required in zone B.",
    "There is no slab camber.",
])
def test_noun_phrase_or_verb_out_of_shape_is_unchanged(lead):
    assert ac._absence_lead(lead, computed=False) == lead


@pytest.mark.parametrize("text", [
    "No pile cap depth is in the drawings I searched.",
    "There are no curtain wall anchor loads in the notes I read.",
])
def test_computed_turn_is_unchanged(text):
    assert ac._absence_lead(text, computed=True) == text


@pytest.mark.parametrize("text", [
    "The pile cap is 1.2 m deep [1]. No camber is in the notes I read.",
    "Handover is 14 March [2].\nThere is no slab camber in the notes I read.",
])
def test_absence_after_the_lead_is_unchanged(text):
    assert ac._absence_lead(text, computed=False) == text


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_empty_text_is_unchanged(text):
    assert ac._absence_lead(text, computed=False) == text


# ---- through the answer exit --------------------------------------------

@pytest.mark.parametrize("answer,lead", [
    ("No billed rate for lift pit waterproofing is in the project documents I searched. "
     "The bill prices the pit walls only.",
     "Billed rate for lift pit waterproofing is not in the project documents I searched."),
    ("There are no curtain wall anchor loads in the structural notes I read.\n\n"
     "The notes stop at level 4.",
     "Curtain wall anchor loads are not in the structural notes I read."),
])
def test_apply_restates_the_lead_on_a_document_turn(answer, lead):
    out = ac.apply(answer, sources=[]).text
    assert _first_line(out).startswith(lead)


@pytest.mark.parametrize("answer", [
    "No billed rate for lift pit waterproofing is in the project documents I searched.",
    "There are no curtain wall anchor loads in the structural notes I read.",
])
def test_apply_leaves_the_lead_on_an_asking_turn(answer):
    assert ac.apply(answer, sources=[], asking=True).text == answer


@pytest.mark.parametrize("answer", [
    "No billed rate for lift pit waterproofing is in the project documents I searched.",
    "There are no curtain wall anchor loads in the structural notes I read.",
])
def test_apply_leaves_the_lead_on_a_computed_turn(answer):
    # No figure line, so nothing else reorders it: only the computed guard
    # keeps the lead as written, and the same text on a document turn changes.
    assert ac.apply(answer, sources=[], computed=True).text == answer
    assert ac.apply(answer, sources=[], computed=False).text != answer
