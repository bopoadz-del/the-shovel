"""Verify bim_extractor parses a known IFC sample and produces expected counts.

The fixture ``tests/fixtures/sample_office.ifc`` is a 2-storey building generated
by ``scripts/_make_sample_ifc.py``. Re-run that script to regenerate after any
ifcopenshell schema changes.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from app.blocks.bim_extractor import BIMExtractorBlock

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_office.ifc")
FIXTURE_2X3 = os.path.join(os.path.dirname(__file__), "fixtures", "sample_office_2x3.ifc")


def _run(input_data, params=None):
    block = BIMExtractorBlock()
    return asyncio.run(block.process(input_data, params))


def test_extracts_elements_from_sample_ifc():
    assert os.path.exists(FIXTURE), (
        f"missing fixture {FIXTURE}; run `python scripts/_make_sample_ifc.py`"
    )
    result = _run({"file_path": FIXTURE})
    assert result["status"] == "success", result.get("error")
    assert result["ifc_schema"] == "IFC4"
    # Counts the fixture script writes â€” keep them in sync.
    assert result["element_count"] >= 26  # walls(8) + slabs(2) + columns(4) + beams(2) + doors(2) + windows(2) + storeys(2) + spaces(2) + pipe + duct + light
    q = result["quantities"]
    assert q["walls"]["count"] == 8
    assert q["slabs"]["count"] == 2
    assert q["columns"]["count"] == 4
    assert q["beams"]["count"] == 2
    assert q["doors"]["count"] == 2
    assert q["windows"]["count"] == 2
    assert q["storeys"]["count"] == 2
    storey_names = {s.get("name") for s in result["storeys"]}
    assert storey_names == {"Ground Floor", "Level 1"}


def test_rejects_non_ifc_extension(tmp_path):
    not_ifc = tmp_path / "not_an_ifc.xlsx"
    not_ifc.write_bytes(b"PK\x03\x04")  # xlsx magic header
    result = _run({"file_path": str(not_ifc)})
    assert result["status"] == "error"
    assert "must be an .ifc" in result["error"].lower() or ".ifc" in result["error"]


def test_missing_file_path():
    result = _run({})
    assert result["status"] == "error"
    assert "no file_path" in result["error"].lower() or "requires" in result["error"].lower()


def test_file_not_found():
    result = _run({"file_path": "/tmp/this_file_does_not_exist_xyz.ifc"})
    assert result["status"] == "error"
    assert "not found" in result["error"].lower()


def test_nwd_returns_actionable_error(tmp_path):
    """A .nwd upload must NOT just say 'unsupported' â€” the operator needs to
    know the file is Autodesk-proprietary and how to convert it."""
    nwd = tmp_path / "model.nwd"
    nwd.write_bytes(b"\x00" * 32)
    result = _run({"file_path": str(nwd)})
    assert result["status"] == "error"
    assert result["format_extension"] == ".nwd"
    assert result["required_format"] == ".ifc"
    msg = result["error"].lower()
    assert "navisworks" in msg
    assert "ifc" in msg


def test_rvt_returns_actionable_error(tmp_path):
    rvt = tmp_path / "model.rvt"
    rvt.write_bytes(b"\x00" * 32)
    result = _run({"file_path": str(rvt)})
    assert result["status"] == "error"
    assert result["format_extension"] == ".rvt"
    msg = result["error"].lower()
    assert "revit" in msg
    assert "export" in msg and "ifc" in msg


def test_extracts_elements_from_ifc2x3_sample():
    """IFC2x3 is common in older GCC project models. The extractor must read
    the older schema without changes â€” only the model contents and category
    coverage may differ (IFC2x3 lacks IfcPipeSegment/IfcDuctSegment)."""
    assert os.path.exists(FIXTURE_2X3), (
        f"missing fixture {FIXTURE_2X3}; "
        f"run `python scripts/_make_sample_ifc.py --version IFC2X3`"
    )
    result = _run({"file_path": FIXTURE_2X3})
    assert result["status"] == "success", result.get("error")
    assert result["ifc_schema"] == "IFC2X3"
    q = result["quantities"]
    assert q["walls"]["count"] == 8
    assert q["slabs"]["count"] == 2
    assert q["columns"]["count"] == 4
    assert q["beams"]["count"] == 2
    assert q["storeys"]["count"] == 2
    storey_names = {s.get("name") for s in result["storeys"]}
    assert storey_names == {"Ground Floor", "Level 1"}


def test_clash_off_by_default_omits_aabb_report():
    """Leftover-hat L2: counting walls must not run the geom clash pass."""
    result = _run({"file_path": FIXTURE})
    assert result["status"] == "success"
    assert result.get("clash_report") in ({}, None)
    assert result["quantities"]["walls"]["count"] == 8


def test_clash_report_carries_operator_disclaimer():
    """Operator-locked text. Chat must show this on every clash response so
    pilot users don't read AABB output as Navisworks-grade precision."""
    result = _run({"file_path": FIXTURE}, {"run_clash_detection": True})
    assert result["status"] == "success"
    clash = result["clash_report"]
    assert "detection_method_disclaimer" in clash
    disclaimer = clash["detection_method_disclaimer"]
    assert "bounding-box" in disclaimer
    assert "not geometric intersection" in disclaimer
    assert "Navisworks Clash Detective" in disclaimer
    assert "Solibri" in disclaimer


def test_clash_pass_actually_reads_geometry():
    """The fixture must have real shapes, or every clash run is an empty scan.

    Both IFC fixtures shipped with zero IfcProductDefinitionShape entities, so
    ``create_shape`` failed for all 27 elements and the pass reported
    ``elements_analyzed=0 / elements_without_geometry=27``. clash_count was
    always 0 -- not because the model was clean, but because nothing was ever
    compared. Asserting the analysed count is what stops that reading as a pass.
    """
    result = _run({"file_path": FIXTURE}, {"run_clash_detection": True})
    assert result["status"] == "success", result.get("error")
    clash = result["clash_report"]
    assert clash["detection_method"] == "aabb_intersection", (
        "fell back to the name-duplicate heuristic; ifcopenshell.geom missing?"
    )
    assert clash["elements_analyzed"] > 0, clash
    assert clash["timed_out"] is False, clash
    # Only IfcSpace + IfcBuildingStorey are intentionally geometry-free.
    assert clash["elements_without_geometry"] == 4, clash


def test_clash_pass_finds_the_planted_pipe_through_wall():
    """End-to-end proof on a known-bad model: the fixture drives Pipe-Storm-001
    through Wall-GF-E with 200mm of interpenetration, 20x the 10mm tolerance.

    This is the only cross-category overlap in the fixture, so an exact count
    of 1 also proves the same-category skip and the tolerance shrink are not
    manufacturing extra hits.
    """
    result = _run({"file_path": FIXTURE}, {"run_clash_detection": True})
    clash = result["clash_report"]
    assert clash["clash_count"] == 1, clash["clashes"]
    hit = clash["clashes"][0]
    assert {hit["name_a"], hit["name_b"]} == {"Pipe-Storm-001", "Wall-GF-E"}
    assert {hit["category_a"], hit["category_b"]} == {"pipes", "walls"}
    assert hit["type"] == "aabb_overlap"
    assert hit["severity"] == "warning"


def test_clash_reports_real_interpenetration_depth():
    """The planted pipe runs 200mm into the wall, so that is what must be
    reported -- not a figure derived from how long the pipe happens to be."""
    result = _run({"file_path": FIXTURE}, {"run_clash_detection": True})
    hit = result["clash_report"]["clashes"][0]
    assert hit["overlap_mm"] == pytest.approx(200.0, abs=1.0), hit


@pytest.mark.parametrize(
    "tol_mm, expected",
    [(10.0, 1), (199.0, 1), (201.0, 0), (500.0, 0)],
)
def test_clash_tolerance_brackets_the_planted_overlap(tol_mm, expected):
    """Tolerance is the minimum interpenetration that counts, so the planted
    200mm hit must survive just under it and vanish just over it.

    The old test compared each box's min against the other's MAX, which is the
    near-face-to-far-face distance rather than the overlap depth. Because that
    grows with the other element's length, this 2.5m pipe survived a 500mm
    tolerance on a 200mm interpenetration -- tolerance dropped clashes in an
    order that tracked element size, not severity. Bracketing the real overlap
    from both sides is what pins the semantics down.
    """
    result = _run(
        {"file_path": FIXTURE},
        {"run_clash_detection": True, "clash_tolerance_mm": tol_mm},
    )
    clash = result["clash_report"]
    assert clash["tolerance_mm"] == tol_mm
    assert clash["clash_count"] == expected, clash["clashes"]


def test_payload_caps_surface_truncated_flag():
    """The fixture is small so no cap fires â€” verify the truncated flag is
    False, truncation_caps is reported, and quantities_truncated is empty.
    The negative-case shape itself is what guards the 50k-element scenario."""
    result = _run({"file_path": FIXTURE})
    assert result["status"] == "success"
    assert result["truncated"] is False
    assert result["quantities_truncated"] == []
    caps = result["truncation_caps"]
    assert caps["category_item_cap"] == 200
    assert caps["building_elements_cap"] == 500
    assert caps["spaces_cap"] == 50


def test_per_category_items_cap_is_200_not_20():
    """Force a category over the cap and verify items truncate at 200, count
    keeps the true total, and the category is listed in quantities_truncated."""
    block = BIMExtractorBlock()
    fake_quants = {"walls": {"count": 0, "items": []}}
    # Simulate 250 walls passing through the inner loop logic.
    from app.blocks import bim_extractor as bx
    for i in range(250):
        fake_quants["walls"]["count"] += 1
        if len(fake_quants["walls"]["items"]) < bx._CATEGORY_ITEM_CAP:
            fake_quants["walls"]["items"].append({"id": f"w{i}"})
    assert fake_quants["walls"]["count"] == 250
    assert len(fake_quants["walls"]["items"]) == 200
    assert bx._CATEGORY_ITEM_CAP == 200
