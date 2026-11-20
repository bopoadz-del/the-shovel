"""A row chunk's band is sheet-scoped, and never silently missing.

A row chunk carries a locator chain: the sheet it lives on, then the band (the
bill/section heading that scopes the figure). Two properties are structural,
not corpus coincidences:

* a band may be inherited only from a heading row in the SAME sheet. A band
  stated on another sheet describes a different bill/section; inheriting it
  would attach a foreign locator to the figure.
* when the row's own sheet states no band, the chunk carries the explicit
  absence marker in the same field position. A band-scoped question cannot
  reach a row whose band field is simply missing, and an answer cannot say
  which band the never stated.

Every assertion below is built from the test's own synthetic workbook (no live
document, sheet or band name). Each decision is exercised by at least two
inputs. Reverting either rule fails a named assertion: dropping the marker
fails ``test_a_sheet_that_states_no_band_carries_the_marker``, and making
inheritance document-scoped instead of sheet-scoped fails both that test's
"never the other sheet's band" assertion and
``test_each_sheet_inherits_only_its_own_band``.
"""
from __future__ import annotations

import pytest
from openpyxl import Workbook

from app.core import doc_index

_HEADER = ["Item", "Description", "Qty", "Unit", "Rate", "Amount"]


def _write_workbook(path: str, sheets: list[tuple[str, list[list]]]) -> str:
    wb = Workbook()
    first = True
    for name, rows in sheets:
        if first:
            ws = wb.active
            ws.title = name
            first = False
        else:
            ws = wb.create_sheet(name)
        for row in rows:
            ws.append(list(row))
    wb.save(path)
    return path


# ── the pure decision ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sheet_name, band, band_sheet, expected",
    [
        # Stated on this sheet: inherited.
        ("Alpha", "SECTION 1 - EARTHWORKS", "Alpha", "SECTION 1 - EARTHWORKS"),
        # Stated on another sheet: refused, marker instead.
        ("Beta", "SECTION 1 - EARTHWORKS", "Alpha",
         doc_index._SPREADSHEET_NO_BAND),
        # The sheet states nothing: marker.
        ("Beta", "", "Beta", doc_index._SPREADSHEET_NO_BAND),
        # No band, and the (empty) state belongs to another sheet: marker.
        ("Alpha", "", "Beta", doc_index._SPREADSHEET_NO_BAND),
    ],
)
def test_the_band_is_inherited_only_from_the_same_sheet(
    sheet_name, band, band_sheet, expected
):
    assert (
        doc_index._spreadsheet_band_for(sheet_name, band, band_sheet) == expected
    )


# ── input 1: sheet A states a band, sheet B states none ──────────────────────


@pytest.mark.parametrize(
    "band, items, bannerless_items",
    [
        (
            "BAND ALPHA - FIRST SECTION",
            [
                ["A.1", "Excavation to formation", 1200, "m3", 25, 30000],
                ["A.2", "Blinding concrete", 300, "m2", 45, 13500],
            ],
            [
                ["B.1", "Ductile iron pipe", 240, "lm", 180, 43200],
                ["B.2", "Manhole cover", 12, "no", 1500, 18000],
            ],
        ),
        (
            "BAND GAMMA - SECOND TRADE",
            [
                ["G.1", "Shear wall concrete", 3533, "m3", 380, 1342540],
                ["G.2", "Formwork to wall", 900, "m2", 60, 54000],
            ],
            [
                ["D.1", "Wall tiling", 1200, "m2", 55, 66000],
                ["D.2", "Emulsion to walls", 540, "m2", 12, 6480],
            ],
        ),
    ],
)
def test_a_sheet_that_states_no_band_carries_the_marker(
    tmp_path, band, items, bannerless_items
):
    """Input 1 decides the marker AND the same-sheet rule together."""
    path = _write_workbook(
        str(tmp_path / "two_sheets.xlsx"),
        [
            ("Section A", [[band], _HEADER, *items]),
            ("Section B", [_HEADER, *bannerless_items]),
        ],
    )

    chunks = doc_index._spreadsheet_row_chunks(
        str(path), "two_sheets.xlsx", ".xlsx"
    )
    assert len(chunks) == len(items) + len(bannerless_items), chunks

    # Every row chunk's locator chain is complete: Sheet, then a band field in
    # the same position -- no chunk goes out with the band field missing.
    for chunk in chunks:
        assert "; Bill: " in chunk, chunk
        assert chunk.index("Sheet: ") < chunk.index("Bill: ") < chunk.index(
            "Item: "
        ), chunk

    # Rule 1 (positive): every row of the banded sheet inherits its band.
    for _, desc, *_ in items:
        chunk = next(c for c in chunks if desc in c)
        assert f"Bill: {band}" in chunk, chunk
        assert "Sheet: Section A" in chunk, chunk

    # Rules 1+2: every row of the bannerless sheet carries the marker, and the
    # other sheet's band is never used as a fallback.
    for _, desc, *_ in bannerless_items:
        chunk = next(c for c in chunks if desc in c)
        assert "Sheet: Section B" in chunk, chunk
        assert f"Bill: {doc_index._SPREADSHEET_NO_BAND}" in chunk, chunk
        assert band not in chunk, chunk


# ── input 2: each sheet states its own band ─────────────────────────────────


@pytest.mark.parametrize(
    "upper_band, lower_band, upper_item, lower_item",
    [
        (
            "BAND ONE - UPPER WORKS",
            "BAND TWO - LOWER WORKS",
            ["U.1", "Reinforced concrete wall", 3533, "m3", 380, 1342540],
            ["L.1", "Wall tiling", 1200, "m2", 55, 66000],
        ),
        (
            "PHASE NORTH - DRAINAGE",
            "PHASE SOUTH - ROADS",
            ["N.1", "Concrete culvert", 620, "m3", 410, 254200],
            ["S.1", "Kerb edging", 300, "lm", 22, 6600],
        ),
    ],
)
def test_each_sheet_inherits_only_its_own_band(
    tmp_path, upper_band, lower_band, upper_item, lower_item
):
    """Input 2 decides positive inheritance and the contamination guard."""
    path = _write_workbook(
        str(tmp_path / "two_banded_sheets.xlsx"),
        [
            ("Upper", [[upper_band], _HEADER, upper_item]),
            ("Lower", [[lower_band], _HEADER, lower_item]),
        ],
    )

    chunks = doc_index._spreadsheet_row_chunks(
        str(path), "two_banded_sheets.xlsx", ".xlsx"
    )
    assert len(chunks) == 2, chunks

    upper = next(c for c in chunks if upper_item[1] in c)
    lower = next(c for c in chunks if lower_item[1] in c)

    assert "Sheet: Upper" in upper, upper
    assert f"Bill: {upper_band}" in upper, upper
    assert lower_band not in upper, upper

    assert "Sheet: Lower" in lower, lower
    assert f"Bill: {lower_band}" in lower, lower
    assert upper_band not in lower, lower

    assert doc_index._SPREADSHEET_NO_BAND not in upper, upper
    assert doc_index._SPREADSHEET_NO_BAND not in lower, lower
