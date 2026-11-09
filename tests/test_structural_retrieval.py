"""Tests for structural retrieval (item 2): two BOQ layouts."""
from __future__ import annotations

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


def test_rows_matching_row_per_item_sheet():
    rows = rows_matching(_ROW_PER_ITEM, "excavation")
    assert len(rows) == 1
    assert rows[0][1] == "Bulk excavation in all classes"


def test_rows_matching_sectioned_sheet_returns_items_not_section_headers():
    rows = rows_matching(_SECTIONED, "walls")
    # "C40/50 shear walls" and "Lift walls C50" match; "Bill No.4 — Concrete"
    # does not, so no section header is returned.
    assert [r[1] for r in rows] == ["C40/50 shear walls", "Lift walls C50"]


def test_rows_matching_is_case_insensitive_and_whole_word():
    assert len(rows_matching(_ROW_PER_ITEM, "EXCAVATION")) == 1
    # "wall" (singular) is not a substring of "walls" as a whole word.
    assert rows_matching(_ROW_PER_ITEM, "wall") == []


def test_rows_matching_empty_term_or_no_match():
    assert rows_matching(_ROW_PER_ITEM, "") == []
    assert rows_matching(_ROW_PER_ITEM, "plaster") == []


def test_render_rows_keeps_columns_aligned():
    text = render_rows(rows_matching(_ROW_PER_ITEM, "excavation"),
                       _ROW_PER_ITEM[0])
    lines = text.splitlines()
    assert lines[0] == "code\tdescription\tunit\tqty\trate"
    assert lines[1].startswith("A1\tBulk excavation")
