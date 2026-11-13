"""P-D1 (3): every sheet of a drawings volume yields its own title block.

Two general defects are covered:

* The title block was read from page 0 only. In a multi-sheet volume page 0
  is a cover or drawing list, so no sheet had an identity.
* A value was read from the label's own line only. CAD-generated PDF text
  puts the label and its value in separate runs, i.e. on separate lines.

Every input is built in the test; two layouts per decision.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.core import doc_index


@pytest.fixture(scope="module")
def container():
    return ConstructionContainer()


# ── extractor: value on the label's line vs on the next line ─────────────────

SAME_LINE = (
    "GENERAL NOTES\n1. ALL LEVELS IN METRES.\n"
    "DRAWING TITLE: PUMP STATION GENERAL ARRANGEMENT\n"
    "DRAWING NO: PS-4410-012\nREV: B3\nSCALE: 1:200\n"
)
STACKED = (
    "GENERAL NOTES\n1. ALL LEVELS IN METRES.\n"
    "DRAWING TITLE\nRETAINING WALL SECTIONS\n"
    "DRAWING NO.\nRW-0937-205\nREVISION\nD\nSCALE\n1:50\n"
)


@pytest.mark.parametrize(
    "text, number, title, rev, scale",
    [
        (SAME_LINE, "PS-4410-012", "PUMP STATION GENERAL ARRANGEMENT", "B3", "1:200"),
        (STACKED, "RW-0937-205", "RETAINING WALL SECTIONS", "D", "1:50"),
    ],
)
def test_number_title_revision_read_in_both_layouts(container, text, number, title, rev, scale):
    tb = container._extract_title_block({"raw_text": text})
    assert tb["drawing_number"] == number
    assert tb["drawing_title"] == title
    assert tb["revision"] == rev
    assert tb["scale"] == scale


@pytest.mark.parametrize(
    "text, drawn, checked",
    [
        # An empty DRAWN BY followed by a label: the label is never a name.
        ("DRAWN BY\nCHECKED BY\nR. Nakamura\n", None, "R. Nakamura"),
        ("DRAWN BY\nT. Okafor\nCHECKED BY\nR. Nakamura\n", "T. Okafor", "R. Nakamura"),
    ],
)
def test_next_line_is_never_a_label(container, text, drawn, checked):
    tb = container._extract_title_block({"raw_text": text})
    assert tb["drawn_by"] == drawn
    assert tb["checked_by"] == checked


@pytest.mark.parametrize(
    "text, rev",
    [
        ("REV\nGENERAL ARRANGEMENT PLAN\n", None),  # wrong shape: not a revision
        ("REV\nC2\n", "C2"),
    ],
)
def test_next_line_value_must_fit_the_field_shape(container, text, rev):
    assert container._extract_title_block({"raw_text": text})["revision"] == rev


def test_stacked_reading_prefers_the_last_pair(container):
    text = "DRAWING NO.\nAA-1000-001\nnotes\nDRAWING NO.\nBB-2000-002\n"
    assert container._extract_title_block({"raw_text": text})["drawing_number"] == "BB-2000-002"


# ── indexing: one title block per sheet ──────────────────────────────────────

_W, _H = 1190.0, 842.0  # A3 landscape, points


def _same_line_block(page, number: str, title: str, rev: str, scale: str) -> None:
    import fitz

    x = _W - 300
    page.insert_text(fitz.Point(x, _H - 90), f"DRAWING TITLE: {title}", fontsize=8)
    page.insert_text(fitz.Point(x, _H - 70), f"DRAWING NO: {number}", fontsize=8)
    page.insert_text(fitz.Point(x, _H - 50), f"REV: {rev}", fontsize=8)
    page.insert_text(fitz.Point(x, _H - 30), f"SCALE: {scale}", fontsize=8)


def _stacked_strip_block(page, number: str, title: str, rev: str) -> None:
    """Labels above values down the right edge, as CAD exports emit them."""
    import fitz

    x = _W - 150
    y = 200.0
    for label, value in (("DRAWING TITLE", title), ("DRAWING NO.", number), ("REVISION", rev)):
        page.insert_text(fitz.Point(x, y), label, fontsize=7)
        page.insert_text(fitz.Point(x, y + 14), value, fontsize=7)
        y += 40


def _volume(path, *, sheets) -> str:
    import fitz

    doc = fitz.open()
    cover = doc.new_page(width=_W, height=_H)
    cover.insert_text(fitz.Point(80, 120), "VOLUME OF DRAWINGS", fontsize=20)
    cover.insert_text(fitz.Point(80, 160), "LIST OF SHEETS FOLLOWS", fontsize=10)
    for draw in sheets:
        page = doc.new_page(width=_W, height=_H)
        page.insert_text(fitz.Point(80, 100), "GENERAL NOTES: SEE SPECIFICATION.", fontsize=9)
        draw(page)
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.mark.parametrize(
    "sheets, expected",
    [
        (
            [
                lambda p: _same_line_block(p, "GA-3310-101", "UPPER DECK PLAN", "A", "1:100"),
                lambda p: _stacked_strip_block(p, "GA-3310-102", "LOWER DECK PLAN", "C"),
            ],
            [("GA-3310-101", "A"), ("GA-3310-102", "C")],
        ),
        (
            [
                lambda p: _stacked_strip_block(p, "XS-7702-040", "CULVERT LONG SECTION", "2"),
                lambda p: _same_line_block(p, "XS-7702-041", "CULVERT HEADWALL", "B", "1:25"),
                lambda p: _stacked_strip_block(p, "XS-7702-042", "CULVERT APRON", "E"),
            ],
            [("XS-7702-040", "2"), ("XS-7702-041", "B"), ("XS-7702-042", "E")],
        ),
    ],
)
def test_every_sheet_after_the_cover_gets_a_title_block_chunk(tmp_path, sheets, expected):
    name = "VOL-DWG-volume.pdf"
    path = _volume(tmp_path / name, sheets=sheets)
    chunks = doc_index._drawing_chunks_for_document(path, name, ".pdf", "p1")
    blocks = [c for c in chunks if c.startswith("DRAWING TITLE BLOCK")]
    assert len(blocks) == len(expected), blocks
    for i, (block, (number, rev)) in enumerate(zip(blocks, expected)):
        assert f"page {i + 2}." in block, block  # the cover is page 1 and has none
        assert f"drawing number: {number}" in block, block
        assert f"revision: {rev}" in block, block


@pytest.mark.parametrize("callout_scale", ["1:20", "1:5"])
def test_title_block_region_wins_over_a_drawing_zone_callout(tmp_path, callout_scale):
    """A detail callout drawn AFTER the title block is the last SCALE on the
    page; the region crop still reads the sheet's own scale."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=_W, height=_H)
    _same_line_block(page, "DT-5521-300", "DRAINAGE DETAILS", "F", "1:100")
    page.insert_text(fitz.Point(120, 220), f"SCALE: {callout_scale}", fontsize=9)
    path = tmp_path / "DT-DWG-detail.pdf"
    doc.save(str(path))
    doc.close()

    chunks = doc_index._drawing_chunks_for_document(str(path), path.name, ".pdf", "p1")
    blocks = [c for c in chunks if c.startswith("DRAWING TITLE BLOCK")]
    assert len(blocks) == 1, chunks
    assert "scale: 1:100" in blocks[0], blocks[0]
    assert f"scale: {callout_scale}" not in blocks[0], blocks[0]


@pytest.mark.parametrize("numbers", [("TB-1000-001", "TB-1000-002"), ("QQ-88-7", "QQ-88-9")])
def test_a_sheet_table_is_attributed_to_its_own_sheet(tmp_path, monkeypatch, numbers):
    """Tables on sheet N carry sheet N's number, not sheet 1's."""
    table = {"header": ["MARK", "QTY"], "rows": [["P1", "4"]], "kind": "unclassified"}
    monkeypatch.setattr(
        ConstructionContainer, "_extract_tables_advanced", lambda self, page: [dict(table)],
    )
    import fitz

    doc = fitz.open()
    for n in numbers:
        page = doc.new_page(width=_W, height=_H)
        _same_line_block(page, n, "SHEET", "A", "1:50")
    path = tmp_path / "TB-DWG-set.pdf"
    doc.save(str(path))
    doc.close()

    chunks = doc_index._drawing_chunks_for_document(str(path), path.name, ".pdf", "p1")
    tables = [c for c in chunks if c.startswith("DRAWING TABLE")]
    assert len(tables) == len(numbers)
    for chunk, n in zip(tables, numbers):
        assert f"[{n}]" in chunk, chunk
