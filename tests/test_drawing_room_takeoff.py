"""Architectural plans carry room areas in their TEXT. Take them off.

Measured 2026-08-15 against real apartment floor plans (vector PDFs, 4,631
vector paths, no raster): `drawing_qto` returned 6,154 line segments and
`areas_count: 0`. It captured the room NAMES but computed no area, so nothing
downstream could produce a floor-tile, skirting, plaster or paint quantity --
the whole interior-finishes take-off was unreachable.

Yet the drawings state the dimensions outright:

    BEDROOM 1  4.00 X 3.60
    BATHROOM 2 2.35 X 1.60
    LIVING AND DINING 4.40 X 4.35

That is a take-off the block was throwing away. This is the fence for reading
it, and for NOT reading things that merely look like dimensions -- a tile
"600x600mm" or a door "900x2100mm" must never become a room.
"""
from __future__ import annotations

import pytest

from app.blocks.drawing_qto import DrawingQTOBlock


def rooms(text: str):
    return DrawingQTOBlock()._extract_rooms(text)


def test_a_room_label_yields_name_dimensions_and_area():
    r = rooms("BEDROOM 1 4.00 X 3.60")
    assert len(r) == 1, r
    room = r[0]
    assert "BEDROOM" in room["name"].upper()
    assert room["width_m"] == 4.00
    assert room["depth_m"] == 3.60
    assert room["area_m2"] == pytest.approx(14.40, abs=0.01)
    # perimeter is what a skirting or plaster quantity is measured from
    assert room["perimeter_m"] == pytest.approx(15.20, abs=0.01)


def test_every_room_on_a_real_plan_is_captured():
    """Text taken verbatim from one of the supplied apartment plans."""
    text = (
        "BEDROOM 1 4.00 X 3.60 BEDROOM 2 4.00 X 3.60 BATHROOM 2 2.35 X 1.60 "
        "BATHROOM 1 2.35 X 1.60 LIVING AND DINING 4.40 X 4.35 "
        "TOILET 1.10 X 2.30 Kitchen & Dining 3.15 X 3.85"
    )
    r = rooms(text)
    assert len(r) == 7, [x["name"] for x in r]
    total = sum(x["area_m2"] for x in r)
    assert total == pytest.approx(
        14.40 + 14.40 + 3.76 + 3.76 + 19.14 + 2.53 + 12.13, abs=0.05
    )


def test_lowercase_and_ampersand_names_survive():
    r = rooms("Kitchen & Dining 3.15 X 3.85")
    assert len(r) == 1
    assert "Kitchen" in r[0]["name"]
    assert r[0]["area_m2"] == pytest.approx(12.13, abs=0.01)


def test_a_lowercase_x_separator_is_accepted():
    assert rooms("TERRACE 4.35 x 1.80")[0]["area_m2"] == pytest.approx(7.83, abs=0.01)


# ── the false positives that would poison a take-off ─────────────────────

def test_tile_and_door_sizes_are_not_rooms():
    """Product sizes are integers in mm; room dimensions are metres to 2dp.
    Reading '600x600' as a 360,000 m2 room would swamp every real quantity."""
    assert rooms("Ceramic floor tiles 600x600mm to toilets") == []
    assert rooms("Solid core timber door 900x2100mm complete with frame") == []
    assert rooms("Mineral fibre acoustic ceiling tiles 600x600mm") == []


def test_implausible_dimensions_are_rejected():
    """A guard against grid references and levels parsed as a room."""
    assert rooms("GRID 0.05 X 0.05") == []
    assert rooms("SITE 450.00 X 380.00") == []


def test_text_with_no_rooms_yields_nothing():
    assert rooms("GENERAL NOTES. REFER TO STRUCTURAL DRAWINGS.") == []
    assert rooms("") == []


# ── what the take-off is FOR ──────────────────────────────────────────────

def test_areas_and_perimeters_support_a_finishes_takeoff():
    """Floor area drives tiling; perimeter drives skirting and wall finishes.
    Both must be present per room or the interior bill cannot be built."""
    for room in rooms("BEDROOM 1 4.00 X 3.60 BATHROOM 2 2.35 X 1.60"):
        assert room["area_m2"] > 0
        assert room["perimeter_m"] > 0
        assert set(room) >= {"name", "width_m", "depth_m", "area_m2", "perimeter_m"}
