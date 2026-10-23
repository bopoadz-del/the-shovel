"""F6 — the construction happy path: a drawing goes in, structure comes out.

Document ingest -> structured extraction -> output, driven through the real
container entry point (`_process_drawing`) against a real PDF, with nothing
stubbed. This is the demo flow, so the `_extract_*` methods it calls are the
ones that had to be implemented for real rather than registered as roadmap.

Before that work every one of these assertions failed with an empty
container: `tables: []`, `annotations: []`, `title_block: {}`. The drawing
was "processed successfully" and carried none of its own content — which is
the precise shape of a demo that looks like it works.

The PDF is built here rather than committed so the assertions can state
exactly what is on the sheet.
"""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is the drawing reader")

from app.containers.construction import ConstructionContainer


TITLE_BLOCK = """PROJECT: THE CLIENT RESIDENTIAL TOWER
CLIENT: THE CLIENT PROPERTIES
DRAWING TITLE: TYPICAL FLOOR SLAB REINFORCEMENT
DRAWING NO: S-2101-004
REV: C2
SCALE: 1:50
DATE: 14/03/2026
DRAWN BY: A. Haddad
CHECKED BY: M. Rahman
SHEET: 4 OF 12"""


@pytest.fixture
def drawing(tmp_path):
    """A structural drawing sheet: title block, a ruled bar schedule, and a
    consultant markup — the three things a real issued drawing carries."""
    path = tmp_path / "S-2101-004_RevC2.pdf"
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)          # A4 landscape

    page.insert_text(fitz.Point(40, 60), "GENERAL NOTES", fontsize=10)
    page.insert_text(
        fitz.Point(40, 80),
        "1. ALL DIMENSIONS IN MILLIMETRES. CONCRETE GRADE C40/50.",
        fontsize=8,
    )

    # Ruled bar-bending schedule
    xs, ys = [40, 170, 300, 430], [140, 175, 210, 245]
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
        for c, value in enumerate(row):
            page.insert_text(fitz.Point(xs[c] + 6, ys[r] + 22), value, fontsize=8)

    # Title block, bottom right
    for i, line in enumerate(TITLE_BLOCK.splitlines()):
        page.insert_text(fitz.Point(520, 400 + i * 14), line, fontsize=8)

    page.add_text_annot(fitz.Point(300, 300), "Confirm cover to BS 8110 before pour")

    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def processed(drawing):
    """Run the real container path once; every test reads the same output."""
    import asyncio
    container = ConstructionContainer()
    return asyncio.run(container._process_drawing(drawing, {}))


def test_the_drawing_is_processed_at_all(processed):
    assert processed["status"] == "success", processed.get("error")
    assert processed["doc_type"] == "drawing"
    assert processed["total_pages"] == 1
    assert processed["sheets"], "no sheet data produced"


def test_the_bar_schedule_reaches_the_output(processed):
    """The load-bearing one. A drawing's quantities live in its schedules;
    this is what was being dropped."""
    assert processed["tables"], "the bar schedule did not survive ingest"

    flat = " ".join(
        " ".join(row)
        for table in processed["tables"]
        for row in ([table["header"]] + table["rows"])
    )
    assert "B1" in flat and "48" in flat, f"schedule content lost: {processed['tables']}"


def test_the_schedule_is_classified_so_it_can_be_found(processed):
    kinds = {t["kind"] for t in processed["tables"]}
    assert "rebar_schedule" in kinds, f"schedule classified as {kinds}"


def test_the_consultant_markup_reaches_the_output(processed):
    """The review trail. On an issued drawing this is the comment that has
    to reach the engineer."""
    assert processed["annotations"], "the markup did not survive ingest"
    contents = {a.get("content") for a in processed["annotations"]}
    assert "Confirm cover to BS 8110 before pour" in contents, contents


def test_the_drawing_identifies_itself(processed):
    """Title block. Without it an ingested sheet cannot be referenced in a
    submittal log, an RFI, or a citation."""
    tb = processed["title_block"]
    assert tb, "title block is empty — the sheet has no identity"
    assert tb["drawing_number"] == "S-2101-004"
    assert tb["revision"] == "C2"
    assert tb["scale"] == "1:50"
    assert tb["discipline"] == "structural"
    assert tb["fields_found"] >= 6, f"only {tb['fields_found']} fields read: {tb}"


def test_the_discipline_is_detected_from_the_sheet_content(processed):
    """Independent of the title block — driven off the drawing's own text."""
    assert "structural" in processed["detected_disciplines"], \
        processed["detected_disciplines"]


def test_the_specification_grade_on_the_sheet_is_extracted(processed):
    """General notes carry the grade; it must come through as a spec, not
    just as loose text."""
    values = {s.get("value") for s in processed["specifications"]}
    assert any(v and "C40" in v for v in values), \
        f"concrete grade not extracted from the general notes: {values}"


def test_a_file_that_is_not_a_pdf_reports_an_error_rather_than_empty_success(tmp_path):
    """The failure direction. A broken input must not come back as a
    successful ingest with empty content — that is indistinguishable from
    a drawing with nothing on it."""
    import asyncio
    bad = tmp_path / "not-a-drawing.pdf"
    bad.write_text("this is not a PDF", encoding="utf-8")

    result = asyncio.run(ConstructionContainer()._process_drawing(str(bad), {}))

    assert result["status"] == "error", f"a non-PDF reported {result['status']}"
    assert result.get("error"), "error status with no reason given"
