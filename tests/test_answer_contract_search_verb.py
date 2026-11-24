"""Contract rule 3: the search verb said of the answer's own reading is
rewritten in place, in the reader's words.

Rule 3 already rewrote the search participle used as an adjective ("the
retrieved rows"). The verb use slipped past it: "not in what I retrieved",
"what I could retrieve", "No rate for it was retrieved". Two uses are the
search whatever their object: the answer itself as the agent (I / we), and
a passive whose subject is an absence (no / nothing / none). The verb
becomes "find" in the matching form; the sentence, its scope and its
figures stay. Any other use of the same verb (something on site retrieved)
is about the works and stays as written.

Every input here is written for this test; none is product output.
"""

from __future__ import annotations

import re

import pytest

from app.agents import answer_contract

_SEARCH_VERB = re.compile(r"\b(?:retriev\w*|fetch\w*)\b", re.IGNORECASE)


# ── It goes: the answer as the agent of the search ─────────────────────────

@pytest.mark.parametrize("text, kept", [
    ("The handrail finish is not in what I retrieved for this question.",
     "The handrail finish is not in what I found for this question."),
    ("A lintel schedule is not among what I could retrieve.",
     "A lintel schedule is not among what I could find."),
    ("I could not retrieve a fire rating for the riser doors.",
     "I could not find a fire rating for the riser doors."),
    ("We have fetched no row for skirting on level 4.",
     "We have found no row for skirting on level 4."),
])
def test_the_search_verb_with_the_answer_as_agent_is_rewritten(text, kept):
    out = answer_contract.apply(text).text
    assert out == kept
    assert not _SEARCH_VERB.search(out)


# ── It goes: a passive whose subject is what the search did not bring ───────

@pytest.mark.parametrize("text, kept", [
    ("The cladding total is 18,400 m². No project rate for cladding was retrieved to check it against.",
     "The cladding total is 18,400 m². No project rate for cladding was found to check it against."),
    ("Nothing on the sump pump duty was retrieved.",
     "Nothing on the sump pump duty was found."),
])
def test_a_passive_reporting_an_absence_is_rewritten_and_figures_stay(text, kept):
    out = answer_contract.apply(text).text
    assert out == kept
    assert not _SEARCH_VERB.search(out)


# ── It stays: the same verb about the works, or in the documents' words ─────

@pytest.mark.parametrize("text", [
    # A passive with a thing on site as its subject.
    "The damaged shutter panel was retrieved from the pit.\n",
    "Both survey pins were fetched before backfilling.\n",
    # Someone else on the project as the agent.
    "The contractor retrieved the tower crane counterweights on 12 May.\n",
    "Site staff fetch the cube moulds from the store each morning.\n",
    # A quoted clause is the document's own words.
    'The method statement says "I retrieved the sample at 10:00" in its log.\n',
    'Clause 4 reads "nothing shall be retrieved without a permit".\n',
])
def test_the_same_verb_that_is_not_the_search_stays(text):
    assert answer_contract.apply(text).text == text


@pytest.mark.parametrize("text", [
    "I could not retrieve a rate for the riser doors.",
    "What I retrieved names no supplier.",
])
def test_a_turn_asking_for_inputs_keeps_its_wording(text):
    # Rule 3 does not run on a turn that ends on its own question.
    assert answer_contract.apply(text, asking=True).text == text
