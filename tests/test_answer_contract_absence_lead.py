"""Rule 5, the opening paragraph: an answer that leads on an absence.

When the first sentence of the opening paragraph reports an absence in the
answer's own scope form ("X is not in the documents I searched"), that
sentence is the whole answer and leads; the sentences after it (what the
documents do hold) move word for word to the details. An opening paragraph
that states values is never cut, so an answer of several stated parts keeps
every part in view. Every input is invented here; each decision is checked
from both sides with at least two inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract

_DOC_ROW = {"doc_name": "Works register.pdf", "page_or_section": "p. 9"}


def _apply(text: str, **kw):
    kw.setdefault("lead", True)
    if "sources" not in kw and "evidence" not in kw:
        kw["sources"] = [dict(_DOC_ROW)]
    return answer_contract.apply(text, **kw)


def _words(text: str) -> list[str]:
    return text.replace("\n", " ").split()


# ── an absence lead stands alone; the rest is detail, word for word ─────────

@pytest.mark.parametrize("first,rest", [
    ("The colour of the reception blinds is not in the project documents I searched.",
     "The finishes schedule covers floors, walls and ceilings only, with no window "
     "treatment item. The only colour rule given is that doors match the facade."),
    ("No billed length of copper earth tape is in what I searched.",
     "The electrical bill prices earth rods and inspection pits as numbered items. "
     "Earthing conductors appear in the specification as a material clause only."),
])
def test_absence_lead_stands_alone_and_rest_moves(first, rest):
    done = _apply(first + " " + rest)
    assert done.text == first
    assert done.details == rest


@pytest.mark.parametrize("text", [
    "The colour of the reception blinds is not in the project documents I searched. "
    "The finishes schedule covers floors only.\n\n"
    "Ask for the interior design pack if blinds were specified there.",
    "No billed length of copper earth tape is in what I searched. "
    "Earth rods are billed as numbered items.\n\n"
    "The earthing layout drawing may carry a run length.",
])
def test_absence_lead_keeps_every_word_across_lead_and_details(text):
    done = _apply(text)
    assert done.details is not None
    assert sorted(_words(done.text + " " + done.details)) == sorted(_words(text))
    # The later paragraph still follows the moved sentences, in order.
    assert done.details.index("billed as numbered" if "earth" in text else "covers floors") \
        < done.details.index("\n\n")


@pytest.mark.parametrize("first,rest,marks", [
    ("A design wind speed for the canopy is not in the project documents I searched.",
     "The structural notes give dead and live loads [1], and the canopy drawing gives its span [2].",
     "[1][2]"),
    ("No unit rate for anti-graffiti coating is in the priced bill I searched.",
     "The coating is named in the painting specification [1].",
     "[1]"),
])
def test_markers_of_moved_sentences_ride_on_the_lead(first, rest, marks):
    done = _apply(first + " " + rest)
    assert done.text == first + " " + marks
    assert done.details == rest


# ── the other side: a paragraph that is not an absence lead stays whole ─────

@pytest.mark.parametrize("text", [
    # Several stated parts: cutting would hide the second part.
    "The main contractor is Harrowgate Civils. The supervising engineer is the borough works office.",
    "The slab is 250 mm thick. The topping is 75 mm of screed.",
])
def test_paragraph_stating_values_is_untouched(text):
    done = _apply(text)
    assert done.text == text and done.details is None


@pytest.mark.parametrize("text", [
    # A search scope with no negation is not an absence.
    "The ramp gradient is 1:12 in the documents I searched. The landing is 1.5 m long.",
    # A negation with no search scope is not the scope form.
    "The bill does not price the sump pumps. The pumps are named in the mechanical schedule.",
])
def test_scope_and_negation_are_both_needed(text):
    done = _apply(text)
    assert done.text == text and done.details is None


@pytest.mark.parametrize("text", [
    "The membrane warranty is not in the documents I searched, e.g. the roofing "
    "clause names only the product. The defects period is general.",
    "A lift pit depth is not in what I searched, i.e. no drawing gives it. "
    "The lift supplier sets it.",
])
def test_abbreviation_stop_does_not_end_the_lead(text):
    done = _apply(text)
    first = text.split(". ")[0] + ". " + text.split(". ")[1] + "."
    assert done.text == first
    assert done.details == text[len(first):].strip()


@pytest.mark.parametrize("text", [
    "The fence height is not in the documents I searched.\nThe site plan shows its line only.",
    "No gate width is in what I searched.\nThe access plan shows two gates.",
])
def test_paragraph_written_over_several_lines_stays_whole(text):
    done = _apply(text)
    assert done.text == text and done.details is None


@pytest.mark.parametrize("text", [
    "The fence height is not in the documents I searched.",
    "No gate width is in what I searched.",
])
def test_single_absence_sentence_is_untouched(text):
    done = _apply(text)
    assert done.text == text and done.details is None


@pytest.mark.parametrize("kw", [
    {"lead": False},
    {"asking": True},
    {"computed": True},
    {"sources": []},
])
@pytest.mark.parametrize("text", [
    "The colour of the reception blinds is not in the project documents I searched. "
    "The finishes schedule covers floors only.",
    "No billed length of copper earth tape is in what I searched. "
    "Earth rods are billed as numbered items.",
])
def test_rule_runs_only_on_a_finished_document_answer(kw, text):
    done = _apply(text, **kw)
    assert done.details is None
    assert "Earth rods" in done.text or "finishes schedule" in done.text
