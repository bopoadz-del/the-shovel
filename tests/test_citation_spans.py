"""The one citation-span rule (``app.lib.citation_spans``) shared by the
offline answer scorer and the conflict check. Every text here is invented for
this file; each decision is shown on two inputs or more."""

from __future__ import annotations

import pytest

from app.lib import citation_spans as cs


@pytest.mark.parametrize("text, gone, kept", [
    ("Cover is 40 mm [source: Deck Notes 12.pdf] here.", "Deck Notes 12", "40 mm"),
    ("Grade C35 (source: Bill (Priced) 7.xlsx) chunk=88 doc_id=zz-41 ok", "Bill (Priced) 7", "Grade C35"),
])
def test_blank_named_blanks_citations_and_keys_keeps_the_claim(text, gone, kept):
    out = cs.blank_named(text)
    assert len(out) == len(text)
    assert gone not in out
    assert kept in out
    assert "chunk=" not in out and "doc_id=" not in out


@pytest.mark.parametrize("text, listed, expected", [
    ("Curing lasts 7 days 2.", 3, "Curing lasts 7 days  ."),
    ("Curing lasts 7 days 5.", 3, "Curing lasts 7 days 5."),
    ("Cover is 50 mm 1. Next", 0, "Cover is 50 mm 1. Next"),
    ("Cover is 50 mm 1. Next", 1, "Cover is 50 mm  . Next"),
])
def test_blank_markers_only_blanks_what_the_panel_numbers(text, listed, expected):
    assert cs.blank_markers(text, listed) == expected


@pytest.mark.parametrize("text, expected", [
    ("Slab 200 mm [3][4] and [source: A 9.pdf].", "Slab 200 mm and"),
    ("Bars 16 mm [12] chunk=7", "Bars 16 mm"),
])
def test_blank_citations_leaves_only_the_claim(text, expected):
    out = cs.blank_citations(text)
    assert len(out) == len(text)
    assert " ".join(out.split()).rstrip(" .") == expected
