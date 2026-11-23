"""Contract rule 4 reads a bracketed file citation by its balanced brackets.

A file name may carry its own brackets ("Ledger (Priced).xlsx", "Sheets
(2 of 5).pdf"). The citation is the whole bracketed group whose own text
names a file, so such a citation becomes a marker like any other, and a
bracket that is not a citation is left as written. Every input here is
written by the test; none is a project file or an exam question.
"""
from __future__ import annotations

import pytest

from app.agents import answer_contract


def _apply(text, sources=None):
    return answer_contract.apply(text, sources=sources or [])


# ── a citation whose file name carries brackets becomes a marker ────────────

@pytest.mark.parametrize("text, document", [
    ("The rate is 42 per m3 [source: Harbour Ledger (Priced).ledger.xlsx].",
     "Harbour Ledger (Priced).ledger.xlsx"),
    ("The wall is 300 mm thick [source: Quay Sheets (2 of 5).pdf].",
     "Quay Sheets (2 of 5).pdf"),
    ("The pipe is DN150 (source: Outfall Layout (QX-0042).pdf).",
     "Outfall Layout (QX-0042).pdf"),
    ("Cover is 40 mm [Pier Notes [rev C].docx].",
     "Pier Notes [rev C].docx"),
])
def test_a_bracketed_file_name_is_one_citation(text, document):
    out = _apply(text)
    assert "source:" not in out.text.lower()
    assert document not in out.text
    assert out.text.endswith("[1].")
    assert [c["document"] for c in out.citations] == [document]


@pytest.mark.parametrize("text, document, section", [
    ("Lap is 40 d [source: Rebar Manual (Ed. 3).pdf, p. 7].", "Rebar Manual (Ed. 3).pdf", "p. 7"),
    ("Slump is 100 mm (Mix Notes (site).docx, section 4.2).", "Mix Notes (site).docx", "section 4.2"),
])
def test_the_locator_after_a_bracketed_name_is_its_section(text, document, section):
    out = _apply(text)
    assert out.citations[0]["document"] == document
    assert out.citations[0]["section"] == section


@pytest.mark.parametrize("text", [
    "The rate is 42 [source: Ledger (Priced).xlsx] and the total is 9 [source: Ledger (Priced).xlsx].",
    "Depth is 2 m (Pit Plan (A).pdf) and width is 3 m (Pit Plan (A).pdf).",
])
def test_the_same_bracketed_name_twice_is_one_marker(text):
    out = _apply(text)
    assert out.text.count("[1]") == 2
    assert "[2]" not in out.text
    assert len(out.citations) == 1


# ── a bracket that does not cite a file stays as written ─────────────────────

@pytest.mark.parametrize("text", [
    "The drawings run to seven sheets (6 of 7 issued so far).",
    "Use grade C40 [see note (a) on the schedule].",
])
def test_a_bracket_naming_no_file_stays(text):
    out = _apply(text)
    assert out.text == text
    assert out.citations == []


@pytest.mark.parametrize("text, kept, document", [
    ("The quay (the figure in [source: Berth Ledger (Rev 2).xlsx] is net) is 220 m.",
     "(the figure in [1] is net)", "Berth Ledger (Rev 2).xlsx"),
    ("Fill depth [read with (Fill Plan (B).pdf) on site] is 1.2 m.",
     "[read with [1] on site]", "Fill Plan (B).pdf"),
])
def test_prose_around_a_nested_citation_stays(text, kept, document):
    out = _apply(text)
    assert kept in out.text
    assert [c["document"] for c in out.citations] == [document]


@pytest.mark.parametrize("text", [
    "The bay is 6 m [source: Bay Ledger (Priced.xlsx and more",
    "The bay is 6 m (source: Bay Ledger [Priced).xlsx] here",
])
def test_an_unbalanced_or_mismatched_bracket_is_not_a_citation(text):
    out = _apply(text)
    assert out.citations == []
    assert "Bay Ledger" in out.text


@pytest.mark.parametrize("filler", ["word " * 60, "long clause, " * 25])
def test_an_overlong_group_is_not_a_citation(filler):
    text = f"The slab is 250 mm ({filler}see Slab Notes.pdf)."
    out = _apply(text)
    assert out.citations == []
    assert "Slab Notes.pdf" in out.text


@pytest.mark.parametrize("text, document", [
    ("Bearing is 150 kPa (Soil Report.pdf, p. 3).", "Soil Report.pdf"),
    ("Bearing is 150 kPa [source: Soil Report.pdf].", "Soil Report.pdf"),
])
def test_a_plain_file_citation_still_becomes_a_marker(text, document):
    out = _apply(text)
    assert out.text == "Bearing is 150 kPa [1]."
    assert out.citations[0]["document"] == document
