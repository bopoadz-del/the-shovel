"""DWG take-off is removed; PDF and DXF take-off are unchanged.

Owner ruling: no DWG is converted or read. A DWG given to take-off
(``drawing_qto``) or BIM (``bim``) is refused with one message and a 415,
from the block and over HTTP (``POST /v1/execute``).

The PDF (``_extract_from_pdf``) and DXF paths must return exactly what they
returned before the DWG path was removed. The values pinned below were
recorded on these same synthetic drawings from the code as it stood before
the change, so a drift in either take-off fails here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.blocks.drawing_qto import DrawingQTOBlock
from app.core.cad_formats import DWG_NOT_SUPPORTED
from tests.conftest import requires_construction_kit
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


def test_the_message_is_the_owner_wording():
    assert DWG_NOT_SUPPORTED == (
        "DWG is not supported. Export DXF or PDF from your CAD software."
    )


# ── synthetic drawings ──────────────────────────────────────────────────────


def _build_plan_pdf(path: Path) -> None:
    """Two rectangles, two lines and two dimensioned room labels."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=1190, height=842)
    page.draw_rect(fitz.Rect(100, 100, 400, 300))
    page.draw_rect(fitz.Rect(450, 100, 600, 250))
    page.draw_line(fitz.Point(100, 400), fitz.Point(700, 400))
    page.draw_line(fitz.Point(100, 450), fitz.Point(100, 650))
    page.insert_text(fitz.Point(120, 200), "BEDROOM 1 4.00 X 3.60", fontsize=8)
    page.insert_text(fitz.Point(460, 200), "STORE 2.10 X 1.50", fontsize=8)
    doc.save(str(path))
    doc.close()


def _build_plan_dxf(path: Path) -> None:
    """Lines, two closed polylines, a circle, an arc and a hatch, in metres."""
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.units = 6
    msp = doc.modelspace()
    msp.add_line((0, 0), (12.5, 0), dxfattribs={"layer": "WALLS"})
    msp.add_line((0, 0), (0, 8), dxfattribs={"layer": "WALLS"})
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 6), (0, 6)], close=True,
                       dxfattribs={"layer": "ROOMS"})
    msp.add_lwpolyline([(20, 0), (24, 0), (24, 3), (20, 3)], close=True,
                       dxfattribs={"layer": "ROOMS"})
    msp.add_circle((30, 5), 2.0, dxfattribs={"layer": "COLUMNS"})
    msp.add_arc((40, 0), 3.0, 0, 90, dxfattribs={"layer": "WALLS"})
    hatch = msp.add_hatch(dxfattribs={"layer": "SLAB"})
    hatch.paths.add_polyline_path([(50, 0), (55, 0), (55, 5), (50, 5)], is_closed=True)
    doc.saveas(str(path))


def _run(path: Path) -> dict:
    return asyncio.run(DrawingQTOBlock().process({"file_path": str(path)}, {}))


# ── PDF take-off: same quantities as before ─────────────────────────────────


def test_pdf_takeoff_returns_the_recorded_quantities(tmp_path):
    p = tmp_path / "plan.pdf"
    _build_plan_pdf(p)
    r = _run(p)
    assert r["status"] == "success"
    assert r["measurements"] == [
        {"end": [700.0, 400.0], "length_pt": 600.0, "length_scaled": 600.0,
         "page": 1, "start": [100.0, 400.0], "type": "line"},
        {"end": [100.0, 650.0], "length_pt": 200.0, "length_scaled": 200.0,
         "page": 1, "start": [100.0, 450.0], "type": "line"},
    ]
    assert r["areas"] == [
        {"area_pt2": 60000.0, "area_scaled": 60000.0, "height_pt": 200.0,
         "page": 1, "type": "rect", "width_pt": 300.0},
        {"area_pt2": 22500.0, "area_scaled": 22500.0, "height_pt": 150.0,
         "page": 1, "type": "rect", "width_pt": 150.0},
    ]
    assert r["rooms"] == [
        {"area_m2": 14.4, "depth_m": 3.6, "name": "BEDROOM 1", "page": 1,
         "perimeter_m": 15.2, "source": "text_layer", "width_m": 4.0},
        {"area_m2": 3.15, "depth_m": 1.5, "name": "STORE", "page": 1,
         "perimeter_m": 7.2, "source": "text_layer", "width_m": 2.1},
    ]
    assert r["measurements_count"] == 2
    assert r["areas_count"] == 2


# ── DXF take-off: same quantities as before ─────────────────────────────────


def test_dxf_takeoff_returns_the_recorded_quantities(tmp_path):
    p = tmp_path / "plan.dxf"
    _build_plan_dxf(p)
    r = _run(p)
    assert r["status"] == "success"
    assert r["measurements"] == [
        {"end": [12.5, 0.0], "layer": "WALLS", "length_m": 12.5,
         "start": [0.0, 0.0], "type": "line"},
        {"end": [0.0, 8.0], "layer": "WALLS", "length_m": 8.0,
         "start": [0.0, 0.0], "type": "line"},
        {"bulge_segments": 0, "closed": True, "layer": "ROOMS", "length_m": 32.0,
         "type": "polyline", "vertex_count": 4},
        {"bulge_segments": 0, "closed": True, "layer": "ROOMS", "length_m": 14.0,
         "type": "polyline", "vertex_count": 4},
        {"circumference_m": 12.5664, "layer": "COLUMNS", "length_m": 12.5664,
         "radius_m": 2.0, "type": "circle"},
        {"arc_length_m": 4.7124, "layer": "WALLS", "length_m": 4.7124,
         "radius_m": 3.0, "type": "arc"},
    ]
    assert r["areas"] == [
        {"area_m2": 60.0, "layer": "ROOMS", "perimeter_m": 32.0,
         "type": "polyline_area", "vertex_count": 4},
        {"area_m2": 25.0, "hole_handling": "outer_minus_holes", "layer": "SLAB",
         "path_count": 1, "type": "hatch_area"},
        {"area_m2": 12.5664, "layer": "COLUMNS", "perimeter_m": 12.5664,
         "type": "circle"},
        {"area_m2": 12.0, "layer": "ROOMS", "perimeter_m": 14.0,
         "type": "polyline_area", "vertex_count": 4},
    ]
    assert [v["volume_m3"] for v in r["estimated_volumes"]] == [180.0, 75.0, 37.6992, 36.0]
    assert r["total_area_m2"] == 109.566
    assert r["total_length_m"] == 83.779
    assert r["entity_count"] == 7


# ── DWG: refused by the blocks ──────────────────────────────────────────────


def test_takeoff_refuses_a_dwg_without_reading_it(tmp_path):
    p = tmp_path / "Fenwick Reach level 02.dwg"
    p.write_bytes(b"AC1032 invented bytes")
    r = _run(p)
    assert r["status"] == "error"
    assert r["error"] == DWG_NOT_SUPPORTED
    assert r["client_error_status"] == 415


def test_takeoff_refuses_a_dwg_that_is_not_on_disk():
    r = asyncio.run(DrawingQTOBlock().process({"file_path": "/nowhere/plan.DWG"}, {}))
    assert r["error"] == DWG_NOT_SUPPORTED


def test_takeoff_accepts_dxf_and_pdf_only():
    assert DrawingQTOBlock.ui_schema["input"]["accept"] == [".dxf", ".pdf"]
    assert not hasattr(DrawingQTOBlock, "_try_convert_dwg")


# ── DWG: 415 over HTTP, from both entry points ──────────────────────────────


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@requires_construction_kit
@pytest.mark.parametrize("block, params", [
    ("drawing_qto", {}),
    ("bim", {"action": "parse_ifc"}),
])
def test_execute_answers_a_dwg_with_415_and_the_message(client, tmp_path, block, params):
    p = tmp_path / "morrow_varga_core.dwg"
    p.write_bytes(b"AC1032 invented bytes")
    r = client.post("/v1/execute", json={
        "block": block, "input": {"file_path": str(p)}, "params": params,
    }, headers=H)
    assert r.status_code == 415, r.text
    assert r.json()["detail"] == DWG_NOT_SUPPORTED


@requires_construction_kit
def test_execute_still_runs_a_dxf_takeoff(client, tmp_path):
    p = tmp_path / "plan.dxf"
    _build_plan_dxf(p)
    r = client.post("/v1/execute", json={
        "block": "drawing_qto", "input": {"file_path": str(p)}, "params": {},
    }, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["result"]["total_area_m2"] == 109.566
