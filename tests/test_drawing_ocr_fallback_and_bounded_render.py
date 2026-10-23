"""Text-free drawings get an OCR attempt; rendering can never OOM the box.

Two live findings, one afternoon (2026-08-15):

F26 -- the OCR page renderer bounded NOTHING but dpi. An A1 CAD sheet at
150 dpi is a ~35-megapixel bitmap; a sync reindex of a 4-page guestroom
plot took the box from 585 MB to the 2 GB ceiling inside one minute and
dropped the whole service (Render memory metrics, 11:42Z). The renderer
now caps the PIXEL COUNT: small pages keep full dpi, large sheets scale
down proportionally with a 50 dpi floor.

Drawing-reader gap -- the same guestroom plots (text rendered as curves,
1,151 chars across 18 pages, none of it dimensions) returned 0 rooms with
no attempt. drawing_qto now falls back to OCR on pages whose text layer is
thin AND yielded no rooms, runs the same room extraction over the OCR
text, and marks OCR-sourced rooms for verification.
"""
from __future__ import annotations

import pytest

from app.blocks.drawing_qto import DrawingQTOBlock
from app.core.doc_index import _ocr_dpi_for_page

# -- F26: the pixel bound -------------------------------------------------

def test_a_letter_page_keeps_full_dpi():
    # 8.5x11in at 150dpi = 2.1MP -- under the 6MP cap
    assert _ocr_dpi_for_page(612, 792, 150, 6_000_000) == 150


def test_the_a1_cad_sheet_that_killed_the_box_is_bounded():
    """A1 = 594x841mm = 1684x2384pt. At 150dpi that is ~35MP; the box died.
    The bound must bring it under the cap."""
    dpi = _ocr_dpi_for_page(1684, 2384, 150, 6_000_000)
    assert dpi < 150
    w_px = 1684 / 72 * dpi
    h_px = 2384 / 72 * dpi
    assert w_px * h_px <= 6_000_000 * 1.05  # rounding headroom


def test_the_floor_keeps_ocr_legible():
    # absurdly huge sheet: dpi floors at 50, never 0
    assert _ocr_dpi_for_page(20000, 20000, 150, 6_000_000) == 50


# -- drawing-reader fallback machinery ------------------------------------

def _text_free_pdf(tmp_path):
    """A one-page PDF with no text layer (vector rectangle only)."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(50, 50, 300, 300))
    p = str(tmp_path / "plot.pdf")
    doc.save(p)
    return p


@pytest.mark.asyncio
async def test_a_text_free_pdf_gets_an_ocr_attempt_and_rooms(tmp_path, monkeypatch):
    """RED before the fix: 0 rooms, no attempt, silence."""
    import app.core.doc_index as di
    monkeypatch.setattr(di, "_ocr_pdf_page",
                        lambda page: "GUEST ROOM 4.20 X 6.10 BATHROOM 2.20 X 2.60")
    res = await DrawingQTOBlock().process({"file_path": _text_free_pdf(tmp_path)}, {})
    assert res["status"] == "success"
    assert res["rooms_count"] == 2, res.get("rooms")
    assert all(r["source"] == "ocr" for r in res["rooms"])
    assert res["ocr_fallback"]["pages_attempted"] == 1
    assert res["ocr_fallback"]["pages_yielded"] == 1
    assert "verify" in res["ocr_fallback"]["note"]
    assert res["net_room_area_m2"] == pytest.approx(4.20 * 6.10 + 2.20 * 2.60, abs=0.05)


@pytest.mark.asyncio
async def test_a_text_layer_plan_never_ocrs(tmp_path, monkeypatch):
    """Real text layers stay authoritative -- the fallback must not fire."""
    import fitz

    import app.core.doc_index as di

    def _boom(page):
        raise AssertionError("OCR must not run when the text layer has rooms")
    monkeypatch.setattr(di, "_ocr_pdf_page", _boom)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "BEDROOM 1 4.00 X 3.60  BATHROOM 2.35 X 1.60 "
                                "GENERAL NOTES REFER TO STRUCTURAL DRAWINGS")
    p = str(tmp_path / "plan.pdf")
    doc.save(p)
    res = await DrawingQTOBlock().process({"file_path": p}, {})
    assert res["rooms_count"] == 2
    assert all(r["source"] == "text_layer" for r in res["rooms"])
    assert res["ocr_fallback"]["pages_attempted"] == 0


@pytest.mark.asyncio
async def test_the_fallback_can_be_disabled(tmp_path, monkeypatch):
    import app.core.doc_index as di
    monkeypatch.setattr(di, "_ocr_pdf_page",
                        lambda page: "GUEST ROOM 4.20 X 6.10")
    res = await DrawingQTOBlock().process(
        {"file_path": _text_free_pdf(tmp_path)}, {"ocr_fallback": False})
    assert res["rooms_count"] == 0
    assert res["ocr_fallback"]["enabled"] is False
    assert res["ocr_fallback"]["pages_attempted"] == 0


@pytest.mark.asyncio
async def test_ocr_noise_does_not_become_rooms(tmp_path, monkeypatch):
    """OCR junk without the name+metres-pair structure yields nothing --
    the room pattern is the noise filter."""
    import app.core.doc_index as di
    monkeypatch.setattr(di, "_ocr_pdf_page",
                        lambda page: "|| .. 600x600 TILE  D07 900x2100  ~~ 154 2 -")
    res = await DrawingQTOBlock().process({"file_path": _text_free_pdf(tmp_path)}, {})
    assert res["rooms_count"] == 0
    assert res["ocr_fallback"]["pages_attempted"] == 1
    assert res["ocr_fallback"]["pages_yielded"] == 0


# -- F26b/F26c: the SECOND live OOM, same afternoon ------------------------
# The pixmap bound shipped and the box died again: pdfplumber and
# get_drawings() materialise per-object Python structures, and a dense A1
# CAD plot measured ~110,000 drawing objects / ~5 MB content stream PER
# PAGE (a normal floor plan: 4,631 objects / 257 KB). Both passes are now
# bounded by cheap pre-checks, and every skip is reported.

def test_table_extraction_refuses_oversize_sheets(tmp_path):
    """A1 sheets are drawings, not table BOQs -- pdfplumber never opens."""
    import fitz

    from app.core.doc_index import _pdf_tables_enabled
    doc = fitz.open()
    doc.new_page(width=2384, height=1684)  # the exact killer page size
    p = str(tmp_path / "a1.pdf")
    doc.save(p)
    assert _pdf_tables_enabled(p) is False


def test_table_extraction_still_runs_on_normal_pages(tmp_path):
    import fitz

    from app.core.doc_index import _pdf_tables_enabled
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # A4
    p = str(tmp_path / "a4.pdf")
    doc.save(p)
    assert _pdf_tables_enabled(p) is True


@pytest.mark.asyncio
async def test_dense_pages_skip_geometry_but_report_it(tmp_path, monkeypatch):
    """Content stream over the cap: the vector pass is skipped and COUNTED,
    rooms/OCR still run, and the turn succeeds fast instead of ballooning."""
    import app.blocks.drawing_qto as dq
    import app.core.doc_index as di
    monkeypatch.setattr(dq, "_GEOMETRY_MAX_CONTENT_BYTES", 10)  # everything is dense
    monkeypatch.setattr(di, "_ocr_pdf_page", lambda page: "SUITE 5.10 X 4.30")
    res = await DrawingQTOBlock().process({"file_path": _text_free_pdf(tmp_path)}, {})
    assert res["status"] == "success"
    assert res["pages_geometry_skipped"] == 1
    assert res["measurements_count"] == 0 and res["areas_count"] == 0
    assert res["rooms_count"] == 1  # OCR leg unaffected by the geometry skip
