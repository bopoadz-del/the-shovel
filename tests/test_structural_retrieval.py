"""Tests for structural retrieval (item 2): two BOQ layouts."""
from __future__ import annotations

import pytest

from app.core.rag.structural_retrieval import render_rows, rows_matching

# Row-per-item sheet: one line item per row, a plain header at the top.
_ROW_PER_ITEM = [
    ["code", "description", "unit", "qty", "rate"],
    ["A1", "Bulk excavation in all classes", "m3", "1200", "8.50"],
    ["A2", "75 mm blinding to slab", "m2", "900", "3.20"],
    ["A3", "C40/50 shear walls", "m3", "450", "210.00"],
]

# Sectioned sheet: section header rows interspersed with line-item rows.
_SECTIONED = [
    ["Bill No.1 — General"],
    ["A1", "Bulk excavation in all classes", "m3", "1200", "8.50"],
    ["A2", "75 mm blinding to slab", "m2", "900", "3.20"],
    ["Bill No.4 — Concrete"],
    ["B1", "C40/50 shear walls", "m3", "450", "210.00"],
    ["B2", "Lift walls C50", "m3", "180", "240.00"],
]


@pytest.mark.parametrize("term,want", [
    ("excavation", ["Bulk excavation in all classes"]),
    ("blinding", ["75 mm blinding to slab"]),
])
def test_rows_matching_row_per_item_sheet(term, want):
    rows = rows_matching(_ROW_PER_ITEM, term)
    assert [r[1] for r in rows] == want


@pytest.mark.parametrize("term,want", [
    ("walls", ["C40/50 shear walls", "Lift walls C50"]),
    ("C50", ["Lift walls C50"]),
])
def test_rows_matching_sectioned_sheet_returns_items_not_section_headers(term, want):
    rows = rows_matching(_SECTIONED, term)
    # "Bill No.1/4 — ..." section header rows have no item term, so no header
    # is returned.
    assert [r[1] for r in rows] == want


@pytest.mark.parametrize("term,want", [
    ("EXCAVATION", 1),
    ("BULK EXCAVATION", 1),
])
def test_rows_matching_is_case_insensitive(term, want):
    assert len(rows_matching(_ROW_PER_ITEM, term)) == want


@pytest.mark.parametrize("term,want", [
    ("wall", []),      # "wall" (singular) is not a whole word of "walls"
    ("plaster", []),   # no match
])
def test_rows_matching_whole_word_and_no_match(term, want):
    assert rows_matching(_ROW_PER_ITEM, term) == want


@pytest.mark.parametrize("term", ["", "   "])
def test_rows_matching_empty_term(term):
    assert rows_matching(_ROW_PER_ITEM, term) == []


@pytest.mark.parametrize("term,want_first", [
    ("excavation", "A1\tBulk excavation"),
    ("blinding", "A2\t75 mm blinding"),
])
def test_render_rows_keeps_columns_aligned(term, want_first):
    text = render_rows(rows_matching(_ROW_PER_ITEM, term), _ROW_PER_ITEM[0])
    lines = text.splitlines()
    assert lines[0] == "code\tdescription\tunit\tqty\trate"
    assert lines[1].startswith(want_first)
