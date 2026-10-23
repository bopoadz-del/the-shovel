"""The drawing reader finds title-block fields by STRUCTURE, never by name.

Every sheet here is generated with PyMuPDF at test time. The consultant,
drafter, checker and place names are invented and appear nowhere in the
source: the reader must find the drawing number, title, scale, revision,
date and drafter from the field labels (generic drawing vocabulary), where
the values sit relative to those labels, and where the title block sits on
the sheet. Title-block text is metadata, so it never comes back as a note
or a dimension. Text repeated on every sheet of a set is boilerplate.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.blocks import drawing_qto
from app.blocks.drawing_qto import DrawingQTOBlock

PAGE_W, PAGE_H = 842.0, 595.0  # A4 landscape, points

# Invented names. None of them may appear in app/ source.
CONSULTANT = "MORROW VARGA PARTNERS"
DRAFTER = "K. OSTLUND"
CHECKER = "R. HALVERSEN"
APPROVER = "T. QUILLON"
PLACE = "FENWICK REACH"
PROJECT = "FENWICK REACH DRAINAGE UPGRADE"

DRAWING_ZONE_NOTE = "ALL CONCRETE TO BE GRADE C40 UNLESS NOTED OTHERWISE"


def _text(page, x: float, y: float, s: str, size: float) -> None:
    page.insert_text((x, y), s, fontsize=size, fontname="helv")


def _cell(page, x: float, y: float, w: float, h: float,
          label: str, value: str | None, value_size: float = 7.0) -> None:
    """A boxed title-block cell: small label on top, value below it."""
    page.draw_rect(fitz.Rect(x, y, x + w, y + h), color=(0, 0, 0), width=0.5)
    _text(page, x + 3, y + 7, label, 5.0)
    if value:
        _text(page, x + 3, y + 7 + value_size + 3, value, value_size)


def _drawing_zone(page) -> None:
    page.draw_rect(fitz.Rect(40, 60, 600, 400), color=(0, 0, 0), width=1)
    _text(page, 60, 120, DRAWING_ZONE_NOTE, 8.0)
    _text(page, 60, 140, "BACKFILL WITH SELECTED GRANULAR MATERIAL IN LAYERS", 8.0)
    _text(page, 300, 300, "2400", 3.0)


def build_bottom_band_sheet(path: Path) -> Path:
    """Title block as a boxed band along the bottom of the sheet."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _drawing_zone(page)
    y1, y2, h = 512.0, 552.0, 38.0
    # Row 1 -- the consultant's name is the largest text in the block.
    _cell(page, 20, y1, 200, h, "PROJECT", PROJECT)
    _cell(page, 225, y1, 200, h, "CONSULTANT", CONSULTANT, value_size=12.0)
    _cell(page, 430, y1, 200, h, "TITLE", "CULVERT HEADWALL DETAILS", value_size=9.0)
    _cell(page, 635, y1, 190, h, "DRAWING NO", "MVP-FR-ST-0042", value_size=9.0)
    # Row 2
    _cell(page, 20, y2, 95, h, "DRAWN", DRAFTER)
    _cell(page, 120, y2, 100, h, "CHECKED", CHECKER)
    _cell(page, 225, y2, 100, h, "APPROVED", APPROVER)
    _cell(page, 330, y2, 95, h, "SCALE", "1:250")
    _cell(page, 430, y2, 95, h, "REV", "C")
    _cell(page, 530, y2, 100, h, "DATE", "14/03/2026")
    _cell(page, 635, y2, 190, h, "SHEET", "3 OF 7")
    doc.save(str(path))
    doc.close()
    return path


def build_right_column_sheet(path: Path) -> Path:
    """Title block as a column down the right-hand edge; nothing in the
    bottom band. Mixes inline ``LABEL: value`` with value-below-label, and
    the title wraps onto two lines."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _drawing_zone(page)
    x = 690.0
    page.draw_rect(fitz.Rect(x - 6, 20, PAGE_W - 10, 490), color=(0, 0, 0), width=0.5)
    _text(page, x, 50, "HOLLOWAY STROM ENGINEERS", 9.0)   # consultant, no label
    _text(page, x, 90, "PROJECT", 5.0)
    _text(page, x, 102, "ASHCOMBE RIDGE LINK", 7.0)
    _text(page, x, 140, "DRAWING TITLE", 5.0)
    _text(page, x, 152, "PUMP STATION", 8.0)
    _text(page, x, 163, "GENERAL ARRANGEMENT", 8.0)
    _text(page, x, 200, "DRAWN BY", 5.0)
    _text(page, x, 212, "L. BRANNIGAN", 7.0)
    _text(page, x, 250, "CHECKED BY", 5.0)
    _text(page, x, 262, "E. MARCHETTI", 7.0)
    _text(page, x, 300, "SCALE: 1:50", 7.0)
    _text(page, x, 340, "REV: B", 7.0)
    _text(page, x, 380, "DATE", 5.0)
    _text(page, x, 392, "02.11.2025", 7.0)
    _text(page, x, 430, "DRAWING NO.", 5.0)
    _text(page, x, 442, "HS-AR-0907", 8.0)
    _text(page, x, 475, "SHEET 1 OF 1", 6.0)
    doc.save(str(path))
    doc.close()
    return path


def build_keyplan_sheet(path: Path) -> Path:
    """A key plan: big place-name labels across the drawing area and a
    sparse title-block band (too few lines for either band on its own)."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _text(page, 150, 200, PLACE, 28.0)
    _text(page, 380, 330, "NORTH QUAY", 28.0)
    _text(page, 200, 420, "OSPREY WARD", 24.0)
    _text(page, 40, 540, "KEY PLAN", 10.0)
    _text(page, 600, 560, "MVP-FR-KP-0001", 8.0)
    doc.save(str(path))
    doc.close()
    return path


def build_two_sheet_set(path: Path) -> Path:
    """Two different drawings in one PDF. Both title blocks repeat the
    consultant's name and the project header at a large font and carry no
    TITLE label (only the drawing number is labelled): the repeated text is
    boilerplate, the per-sheet text is the title."""
    doc = fitz.open()
    for title, number in (("RETAINING WALL ELEVATIONS", "MVP-FR-ST-0101"),
                          ("RETAINING WALL SECTIONS", "MVP-FR-ST-0102")):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        _drawing_zone(page)
        _text(page, 20, 525, CONSULTANT, 14.0)
        _text(page, 20, 545, PROJECT, 12.0)
        _text(page, 330, 535, title, 9.0)
        _text(page, 330, 560, "1:100", 7.0)
        _text(page, 640, 527, "DRAWING NO", 5.0)
        _text(page, 640, 537, number, 8.0)
        _text(page, 640, 560, "14/03/2026", 7.0)
        _text(page, 330, 580, "FOR CONSTRUCTION", 6.0)
    doc.save(str(path))
    doc.close()
    return path


async def _read(path: Path) -> dict:
    return await DrawingQTOBlock().process({"file_path": str(path)}, {})


def _all_text_out(drawing: dict) -> str:
    return " | ".join((drawing.get("notes") or []) + (drawing.get("dimensions") or []))


# --- bottom-band layout ------------------------------------------------------

async def test_bottom_band_fields_found_by_label_and_position(tmp_path):
    r = await _read(build_bottom_band_sheet(tmp_path / "sheet_a.pdf"))
    d = r["drawing"]
    assert d["drawing_number"] == "MVP-FR-ST-0042", r["errors"]
    assert "drawing_number_fallback_to_filename" not in r["errors"]
    assert d["drawing_title"] == "CULVERT HEADWALL DETAILS"
    assert d["scale"] == "1:250"
    assert d["revision"] == "C"
    assert d["date"] == "14/03/2026"
    assert d["drafter"] == DRAFTER
    assert d["checked_by"] == CHECKER
    assert d["project_name"] == PROJECT
    assert d["sheet_number"] == "3 OF 7"


async def test_bottom_band_title_block_text_is_not_an_annotation(tmp_path):
    r = await _read(build_bottom_band_sheet(tmp_path / "sheet_a.pdf"))
    d = r["drawing"]
    out = _all_text_out(d).upper()
    for s in (CONSULTANT, DRAFTER, CHECKER, APPROVER, PROJECT,
              "CULVERT HEADWALL", "MVP-FR-ST-0042", "14/03/2026"):
        assert s not in out, f"title-block text {s!r} leaked into notes/dims: {out!r}"
    assert any(DRAWING_ZONE_NOTE in n for n in d["notes"]), d["notes"]
    assert any("2400" in x for x in d["dimensions"]), d["dimensions"]


# --- right-hand-column layout ------------------------------------------------

async def test_right_column_fields_found_by_label_and_position(tmp_path):
    r = await _read(build_right_column_sheet(tmp_path / "sheet_b.pdf"))
    d = r["drawing"]
    assert d["drawing_number"] == "HS-AR-0907", r["errors"]
    assert "drawing_number_fallback_to_filename" not in r["errors"]
    assert d["drawing_title"] == "PUMP STATION GENERAL ARRANGEMENT"
    assert d["scale"] == "1:50"
    assert d["revision"] == "B"
    assert d["date"] == "02.11.2025"
    assert d["drafter"] == "L. BRANNIGAN"
    assert d["checked_by"] == "E. MARCHETTI"
    assert d["project_name"] == "ASHCOMBE RIDGE LINK"


async def test_right_column_title_block_text_is_not_an_annotation(tmp_path):
    r = await _read(build_right_column_sheet(tmp_path / "sheet_b.pdf"))
    d = r["drawing"]
    out = _all_text_out(d).upper()
    for s in ("HOLLOWAY STROM", "BRANNIGAN", "MARCHETTI", "ASHCOMBE",
              "PUMP STATION", "HS-AR-0907", "DRAWN BY", "02.11.2025"):
        assert s not in out, f"title-block text {s!r} leaked into notes/dims: {out!r}"
    assert any(DRAWING_ZONE_NOTE in n for n in d["notes"]), d["notes"]


# --- position and repetition, no labels --------------------------------------

async def test_key_plan_place_names_lose_to_title_block_region_text(tmp_path):
    """Large place labels in the drawing area are sheet content: the title
    comes from the title-block region even when the reader had to widen to
    the whole page."""
    r = await _read(build_keyplan_sheet(tmp_path / "sheet_kp.pdf"))
    title = r["drawing"]["drawing_title"]
    assert title == "KEY PLAN", (title, r["errors"])


async def test_text_repeated_on_every_sheet_is_boilerplate_not_title(tmp_path):
    r = await _read(build_two_sheet_set(tmp_path / "set.pdf"))
    d = r["drawing"]
    assert d["drawing_title"] == "RETAINING WALL ELEVATIONS", (d["drawing_title"], r["errors"])


# --- no names, no name lists, no environment switches -------------------------

# The retired name-list variables, spelled from parts so that a repository
# grep for their names finds no reader, config or doc -- only absence checks.
_RETIRED_ENV = tuple("_".join(parts) for parts in (
    ("DRAWING", "PROJECT", "HEADER", "TERMS"),
    ("DRAWING", "KNOWN", "DRAFTERS"),
    ("DRAWING", "TITLEBLOCK", "BOILERPLATE"),
    ("DRAWING", "QTO", "EXCLUDED", "PLACE", "NAMES"),
))


def test_reader_source_carries_no_name_list_or_name_env_var():
    src = inspect.getsource(drawing_qto)
    for name in _RETIRED_ENV:
        assert name not in src, f"{name} is still read by the drawing reader"
    assert not hasattr(drawing_qto, "_EXCLUDED_PLACE_NAMES")
    assert not hasattr(drawing_qto, "_env_terms")
    # Every environment read in the module is a numeric threshold.
    for m in re.finditer(r"os\.(?:getenv|environ\.get)\(\s*\"([A-Z0-9_]+)\"", src):
        assert m.group(1).startswith("QTO_"), f"unexpected env read {m.group(1)}"
    for s in (CONSULTANT, DRAFTER, CHECKER, APPROVER, PLACE, "HOLLOWAY", "ASHCOMBE"):
        assert s not in src.upper()


async def test_retired_env_vars_have_no_effect(tmp_path, monkeypatch):
    sheet = build_bottom_band_sheet(tmp_path / "sheet_a.pdf")
    before = (await _read(sheet))["drawing"]
    for name in _RETIRED_ENV:
        monkeypatch.setenv(name, f"{CONSULTANT},{DRAFTER},CULVERT HEADWALL DETAILS")
    after = (await _read(sheet))["drawing"]
    assert before == after
