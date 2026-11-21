"""Contract rule 3: a search participle used as an adjective is rewritten in
place, never kept and never used to drop the sentence.

"the retrieved rows" tells the reader how the answer was made, not what the
documents say. The sentence around it often carries the refusal itself
("X is not in the retrieved text"), and rule 3 keeps a negative sentence as
written, so the word used to reach the reader. The rewrite keeps the scope
of the claim ("the rows I read") and the sentence's meaning.

Every input here is written for this test; none is product output.
"""

from __future__ import annotations

import re

import pytest

from app.agents import answer_contract

_SEARCH_WORD = re.compile(r"\b(?:retrieved|fetched|passages?|excerpts?|chunks?)\b", re.IGNORECASE)


# ── It goes: adjective uses, in refusals and in figure sentences ────────────

@pytest.mark.parametrize("text, kept", [
    # A refusal: the absence and its scope both stay.
    ("The curing period is not stated in the retrieved text.",
     "The curing period is not stated in the text I read."),
    ("Neither paving area is in the fetched rows, so no comparison can be made.",
     "Neither paving area is in the rows I read, so no comparison can be made."),
    # A noun phrase of more than one word, followed by a prepositional phrase.
    ("No row for handrails appears in the retrieved bill rows for this section.",
     "No row for handrails appears in the bill rows I read for this section."),
    # Opening the sentence: the capital moves to the noun.
    ("Retrieved rows are a partial view of the schedule, not the whole sheet.",
     "Rows I read are a partial view of the schedule, not the whole sheet."),
    # After its noun, pointing back at a document the sentence names.
    ("The only waterproofing note retrieved from it is a general clause on membranes.",
     "The only waterproofing note read from it is a general clause on membranes."),
    # A search-unit noun is machinery too: in the reader's words, documents.
    ("No lintel schedule is in the retrieved passages.",
     "No lintel schedule is in the documents I read."),
])
def test_a_search_participle_is_rewritten_and_the_sentence_stays(text, kept):
    out = answer_contract.apply(text).text
    assert out == kept
    assert not _SEARCH_WORD.search(out)


@pytest.mark.parametrize("text, figure", [
    ("Cover to footings is 75 mm in the retrieved material.", "75 mm"),
    ("The fetched rows give 1,240 m² of blockwork on level 2.", "1,240 m²"),
])
def test_a_figure_in_the_sentence_is_untouched(text, figure):
    out = answer_contract.apply(text).text
    assert figure in out
    assert not _SEARCH_WORD.search(out)


def test_a_bullet_keeps_its_bullet():
    text = "Doors:\n- retrieved rows list 14 doors on level 1\n- 9 doors on level 2"
    out = answer_contract.apply(text).text
    assert "- rows I read list 14 doors on level 1" in out
    assert "- 9 doors on level 2" in out


# ── It stays: the same words that are not the search ───────────────────────

@pytest.mark.parametrize("text", [
    # A verb after "be": something on site was retrieved.
    "The damaged formwork panel was retrieved and stored.\n",
    "Two anchors were fetched back from the yard.\n",
    # A physical object taken from a physical place.
    "Cores retrieved from the slab are tested at 28 days.\n",
    # The document's own words, quoted or cited.
    'The spec says "samples retrieved from site are labelled" for cube tests.\n',
])
def test_the_same_word_that_is_not_the_search_stays(text):
    assert answer_contract.apply(text).text == text


def test_a_cited_document_name_keeps_its_words():
    # Rule 4 moves the citation out of the text; the name it carries is the
    # document's own and reaches the citation entry unrewritten.
    out = answer_contract.apply("Cube results are in [retrieved samples register.xlsx] for level 2.\n")
    assert [c["document"] for c in out.citations] == ["retrieved samples register.xlsx"]
    assert "[1]" in out.text


def test_an_answer_with_no_search_word_is_unchanged():
    text = "The curing period is 7 days for slabs.\nColumns cure for 10 days.\n"
    assert answer_contract.apply(text).text == text


def test_a_machinery_only_answer_is_still_not_emptied():
    # Rule 3 drops it; the contract returns the original rather than nothing.
    text = "The search returned three passages from the retrieved documents."
    assert answer_contract.apply(text).text == text


def test_a_turn_asking_for_inputs_keeps_its_wording():
    text = "Which of the retrieved rows is the level 2 slab? Give the row number."
    assert answer_contract.apply(text, asking=True).text == text
