"""F6 — the drawing ingest path must actually extract, not return empties.

`_process_drawing_page` builds every drawing ingest result out of five
helpers. Three of them were hollow:

    "tables":      self._extract_tables_advanced(page)   -> []
    "annotations": self._extract_annotations(page)       -> []
    ...
    result["title_block"] = self._extract_title_block(...)  -> {}

So every drawing that went through the container came out with no tables,
no markups and no identity — and nothing said so. A drawing's numbers live
in its schedules, so this was the ingest path quietly dropping the most
retrievable content on the sheet.

These tests pin the real behaviour. Each one fails against the old bodies:
the table/annotation tests assert non-empty structured output where `[]`
was returned, and the title-block tests assert named fields where `{}` was.

The PDFs are built here rather than committed as fixtures so the assertions
state exactly what the sheet contains.
"""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is the drawing ingest reader")

from app.containers.construction import ConstructionContainer


@pytest.fixture
def container():
    return ConstructionContainer()


# ── tables ────────────────────────────────────────────────────────────────

def _ruled_table_pdf():
    """A page with a real ruled 3x3 grid — what a bar schedule looks like."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    xs, ys = [40, 150, 260, 360], [40, 80, 120, 160]
    for y in ys:
        page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y))
    for x in xs:
        page.draw_line(fitz.Point(x, ys[0]), fitz.Point(x, ys[-1]))
    cells = [
        ["BAR MARK", "DIAMETER", "QTY"],
        ["B1", "16", "48"],
        ["B2", "20", "24"],
    ]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text(fitz.Point(xs[c] + 5, ys[r] + 25), val, fontsize=9)
    return doc, page


def test_a_ruled_schedule_comes_back_with_its_rows(container):
    """The load-bearing one: a bar schedule must survive ingest."""
    doc, page = _ruled_table_pdf()
    try:
        tables = container._extract_tables_advanced(page)
    finally:
        doc.close()

    assert tables, "a ruled 3x3 grid produced no tables — ingest is still dropping schedules"
    table = tables[0]
    flat = " ".join(" ".join(r) for r in ([table["header"]] + table["rows"]))
    assert "B1" in flat and "48" in flat, f"row content lost: {table}"
    assert table["column_count"] == 3
    assert table["page"] == 1


def test_the_schedule_is_named_so_a_consumer_can_find_it(container):
    """Extraction without classification just moves the search problem."""
    doc, page = _ruled_table_pdf()
    try:
        tables = container._extract_tables_advanced(page)
    finally:
        doc.close()
    assert tables[0]["kind"] == "rebar_schedule", (
        f"a BAR MARK/DIAMETER/QTY table classified as {tables[0]['kind']!r}"
    )


def test_a_blank_page_reports_no_tables_without_raising(container):
    """Empty must stay a legal answer — it just has to be the true one."""
    doc = fitz.open()
    page = doc.new_page()
    try:
        assert container._extract_tables_advanced(page) == []
    finally:
        doc.close()


def test_a_reader_without_find_tables_warns_instead_of_faking_an_empty_page(container, caplog):
    """The distinction the old code could not make.

    An ancient PyMuPDF and a genuinely blank sheet both yield `[]`. Only one
    of them is a true statement about the drawing, so the other has to say so.
    """
    class _NoFinder:
        number = 0

    with caplog.at_level("WARNING"):
        assert container._extract_tables_advanced(_NoFinder()) == []
    assert any("find_tables" in r.message or "find_tables" in r.getMessage()
               for r in caplog.records), \
        "silently returned [] on a reader that cannot detect tables at all"


def test_a_detector_error_is_reported_not_swallowed(container, caplog):
    class _Angry:
        number = 3

        def find_tables(self):
            raise RuntimeError("mupdf exploded")

    with caplog.at_level("WARNING"):
        assert container._extract_tables_advanced(_Angry()) == []
    assert any("table detection" in r.getMessage() for r in caplog.records)


# ── annotations ───────────────────────────────────────────────────────────

def test_a_consultant_markup_survives_ingest(container):
    """Annotations are the review trail; dropping them loses the comments."""
    doc = fitz.open()
    page = doc.new_page()
    page.add_text_annot(fitz.Point(100, 100), "Confirm rebar cover to BS 8110")
    page.add_text_annot(fitz.Point(200, 200), "Revise slab thickness")
    try:
        annots = container._extract_annotations(page)
    finally:
        doc.close()

    assert len(annots) == 2, f"expected both markups, got {annots}"
    contents = {a["content"] for a in annots}
    assert "Confirm rebar cover to BS 8110" in contents
    for a in annots:
        assert a["page"] == 1
        assert a["rect"] and len(a["rect"]) == 4
        assert a["type"]


def test_a_clean_sheet_has_no_annotations(container):
    doc = fitz.open()
    page = doc.new_page()
    try:
        assert container._extract_annotations(page) == []
    finally:
        doc.close()


def test_an_unreadable_annotation_layer_is_reported(container, caplog):
    class _Angry:
        number = 1

        def annots(self):
            raise RuntimeError("broken xref")

    with caplog.at_level("WARNING"):
        assert container._extract_annotations(_Angry()) == []
    assert any("annotation layer" in r.getMessage() for r in caplog.records)


# ── title block ───────────────────────────────────────────────────────────

SHEET_TEXT = """
GENERAL NOTES
1. ALL DIMENSIONS IN MILLIMETRES UNLESS NOTED OTHERWISE.

PROJECT: THE CLIENT RESIDENTIAL TOWER
CLIENT: THE CLIENT PROPERTIES
DRAWING TITLE: TYPICAL FLOOR SLAB REINFORCEMENT
DRAWING NO: S-2101-004
REV: C2
SCALE: 1:50
DATE: 14/03/2026
DRAWN BY: A. Haddad
CHECKED BY: M. Rahman
APPROVED BY: K. Osman
SHEET: 4 OF 12
"""


def test_the_title_block_identifies_the_drawing(container):
    """Every field a submittal log needs to reference this sheet."""
    tb = container._extract_title_block({"raw_text": SHEET_TEXT})

    assert tb["drawing_number"] == "S-2101-004"
    assert tb["revision"] == "C2"
    assert tb["scale"] == "1:50"
    assert tb["date"] == "14/03/2026"
    # These three are the ones a label-vs-value bug silently blanks: the
    # regex matches `DRAWN` before `DRAWN BY` and captures "BY" as the name.
    assert tb["drawn_by"].startswith("A"), f"drawn_by={tb['drawn_by']!r}"
    assert tb["checked_by"].startswith("M"), f"checked_by={tb['checked_by']!r}"
    assert tb["approved_by"].startswith("K"), f"approved_by={tb['approved_by']!r}"
    assert "THE CLIENT PROPERTIES" in tb["client"].upper()
    assert "SLAB REINFORCEMENT" in tb["drawing_title"].upper()
    assert "THE CLIENT" in tb["project_name"].upper()
    assert tb["sheet_number"].replace(" ", "") == "4OF12"


def test_the_discipline_is_derived_from_the_number(container):
    """`S-2101-004` is structural. Nobody labels that field on a sheet."""
    tb = container._extract_title_block({"raw_text": SHEET_TEXT})
    assert tb["discipline"] == "structural"


def test_every_field_is_addressable_even_when_the_sheet_is_bare(container):
    """Callers index this dict directly — missing must be None, not absent."""
    tb = container._extract_title_block({"raw_text": "just some plan notes"})

    for field in ("drawing_number", "drawing_title", "revision", "scale",
                  "date", "drawn_by", "checked_by", "approved_by",
                  "project_name", "client", "sheet_number", "discipline"):
        assert field in tb, f"{field} missing entirely — caller would KeyError"
        assert tb[field] is None, f"{field} invented {tb[field]!r} from nothing"
    assert tb["fields_found"] == 0


def test_no_text_is_distinguishable_from_no_fields(container):
    """A scanned sheet yields no text; that is not the same as a sheet
    whose title block simply is not labelled. Both report 0 fields, and
    neither may invent one."""
    for payload in ({}, {"raw_text": ""}, {"raw_text": "   "}):
        tb = container._extract_title_block(payload)
        assert tb["fields_found"] == 0
        assert tb["drawing_number"] is None


def test_a_label_is_never_mistaken_for_a_value(container):
    """`DRAWN BY:` with nothing after it must stay None, not capture the
    next label — the failure mode that makes a title block look populated
    when it is empty."""
    tb = container._extract_title_block(
        {"raw_text": "DRAWN BY:\nCHECKED BY:\nDATE:\n"}
    )
    assert tb["drawn_by"] is None, f"captured {tb['drawn_by']!r} from an empty field"
    assert tb["fields_found"] == 0
