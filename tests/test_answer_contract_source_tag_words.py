"""A ``[source: ...]`` tag that names its document in words, not by file
name, is a citation: it becomes the marker of the one Sources row its words
name, and when no single row backs it, it leaves the text. It never reaches
the reader as raw ``[source: ...]`` text, and no row is guessed for it.

Every input here is invented for the test; two inputs per decision."""

from __future__ import annotations

import re

import pytest

from app.agents import answer_contract

ROWS = [
    {"doc_name": "QX-77-Tender_Ledger of Rates (Priced).ledger.xlsx", "page_or_section": "Sheet 2",
     "snippet": "filling 75 per m3"},
    {"doc_name": "Gantry Frames Register.pdf", "page_or_section": "p. 9", "snippet": "frame G4 at 12 m"},
    {"doc_name": "Pump House Notes - Rev C.pdf", "page_or_section": "p. 3", "snippet": "duty 40 l/s"},
    {"doc_name": "Pump House Notes - Rev D.pdf", "page_or_section": "p. 3", "snippet": "duty 45 l/s"},
]


def _apply(text: str):
    return answer_contract.apply(text, sources=ROWS)


@pytest.mark.parametrize("text, doc, section", [
    ("Filling is 75 per m3 [source: Ledger of Rates, Part 4 - Earthmoving].",
     "QX-77-Tender_Ledger of Rates (Priced).ledger.xlsx", "Part 4 - Earthmoving"),
    ("Frame G4 stands at 12 m [source: Gantry Frames Register].", "Gantry Frames Register.pdf", "p. 9"),
])
def test_a_tag_naming_one_row_in_words_becomes_its_marker(text, doc, section):
    out = _apply(text)
    assert "source:" not in out.text.lower()
    assert re.search(r"\[1\]\.?$", out.text), out.text
    assert out.citations and out.citations[0]["document"] == doc
    assert out.citations[0]["section"] == section


@pytest.mark.parametrize("text, kept", [
    ("The kerb is 150 mm high [source: Kerb Detail Sheet].", "The kerb is 150 mm high."),
    ("The span is 18 m [source: 1].", "The span is 18 m."),
])
def test_a_tag_no_row_backs_leaves_the_text_and_adds_no_entry(text, kept):
    out = _apply(text)
    assert out.text == kept
    assert not out.citations


@pytest.mark.parametrize("text", [
    "Duty is 40 l/s [source: Pump House Notes].",
    "Duty is 45 l/s [source: Pump House Notes, p. 3].",
])
def test_a_tag_whose_words_fit_two_documents_is_not_guessed(text):
    out = _apply(text)
    assert "source:" not in out.text.lower()
    assert "[1]" not in out.text
    assert not out.citations


@pytest.mark.parametrize("text", [
    "The bill covers paving (Roads and Pavings) only.",
    "Use the larger bar [T] in the slab.",
])
def test_a_bracket_without_the_source_label_stays_prose(text):
    assert _apply(text).text == text


@pytest.mark.parametrize("text, doc", [
    ("Frame G4 stands at 12 m [source: Gantry Frames Register.pdf].", "Gantry Frames Register.pdf"),
    ("Duty is 45 l/s [source: Pump House Notes - Rev D.pdf, p. 3].", "Pump House Notes - Rev D.pdf"),
])
def test_a_tag_naming_a_file_still_reads_as_before(text, doc):
    out = _apply(text)
    assert "source:" not in out.text.lower()
    assert out.citations[0]["document"] == doc


def test_a_words_tag_and_a_file_tag_for_the_same_row_share_one_marker():
    out = _apply("Frame G4 is 12 m [source: Gantry Frames Register.pdf]. It is steel [source: Gantry Frames Register].")
    assert out.text.count("[1]") == 2 and "[2]" not in out.text
    assert len(out.citations) == 1
