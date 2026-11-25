"""A file name says BOQ whatever separator sits around the token.

``_BOQ_NAME_RE`` gates the BOQ parser, the finer chunker and the BOQ OCR
budget. Its ``boq`` token was bounded by ``\b``; underscore is a word
character, so an underscore-separated name never matched. Every name here is
invented; two inputs per decision.
"""

from __future__ import annotations

import pytest

from app.core.doc_index import _BOQ_NAME_RE, _chunker_for_document, _looks_like_boq


@pytest.mark.parametrize("name", [
    "Tender_BOQ_rev2.pdf",
    "BOQ_Section4.pdf",
    "lot3_boq.pdf",
    "Package-BOQs.pdf",
    "civil works BOQ.pdf",
    "XY-ABC-002-0000-STR-BOQ-CA-000004-A.pdf",
])
def test_boq_token_matches_across_separators(name: str) -> None:
    assert _BOQ_NAME_RE.search(name)
    assert _looks_like_boq(name, ".pdf")
    assert _chunker_for_document(name) == "finer"


@pytest.mark.parametrize("name", [
    "eboq_notes.pdf",
    "boqueron_site_photos.pdf",
    "Bog_survey_report.pdf",
    "specification_part8.pdf",
])
def test_boq_inside_another_word_does_not_match(name: str) -> None:
    assert not _BOQ_NAME_RE.search(name)
    assert not _looks_like_boq(name, ".pdf")
    assert _chunker_for_document(name) == "default"


def test_spreadsheet_gate_unchanged_by_name() -> None:
    assert _looks_like_boq("eboq_notes.xlsx", ".xlsx")
    assert _looks_like_boq("Tender_BOQ_rev2.csv", ".csv")
