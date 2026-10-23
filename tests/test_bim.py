"""bim block — action-dispatch + error-shape regression guards.

Locks in the contract that every bim block action returns a top-level
`status` field of either `"success"` or `"error"`, and that:

  - element_types filter is honoured (was silently dropped pre-1.1.0)
  - compare_versions surfaces parse errors (was masking them as
    old_count = 0)
  - Unknown actions return a valid_actions list, not a bare error

These tests instantiate the block directly; they don't exercise the
agent runtime, so they don't depend on construction-kit registration.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from app.blocks.bim import BIMBlock

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_office.ifc")


def _run(input_data, params=None):
    block = BIMBlock(hal_block=None, config={})
    return asyncio.run(block.process(input_data, params))


# ── Error shape: every failure path must surface status:error ───────────────


def test_unknown_action_returns_status_error_with_valid_actions():
    """Unknown action must return a structured error WITH the valid
    action list so callers can recover or surface help text."""
    r = _run({}, {"action": "nope"})
    assert r["status"] == "error"
    assert "Unknown action" in r["error"]
    assert isinstance(r.get("valid_actions"), list)
    assert "parse_ifc" in r["valid_actions"]
    assert "spatial_query" in r["valid_actions"]


def test_index_folder_missing_path_returns_status_error():
    r = _run({"folder_path": "/tmp/does_not_exist_xyz"}, {"action": "index_folder"})
    assert r["status"] == "error"
    assert "folder_path" in r["error"].lower()


def test_parse_ifc_missing_file_returns_status_error():
    r = _run({"file_path": "/tmp/missing.ifc"}, {"action": "parse_ifc"})
    assert r["status"] == "error"
    assert "file_path" in r["error"].lower()


def test_process_pdf_missing_file_returns_status_error():
    r = _run({"file_path": "/tmp/missing.pdf"}, {"action": "process_pdf"})
    assert r["status"] == "error"


@pytest.mark.parametrize("action", ["parse_ifc", "get_elements", "process_pdf"])
def test_a_dwg_is_refused_whatever_the_action(action):
    """DWG is not read (owner ruling): every action answers a DWG with the
    one export message and a 415, never a conversion attempt."""
    from app.core.cad_formats import DWG_NOT_SUPPORTED

    r = _run({"file_path": "/tmp/Fenwick Reach site plan.DWG"}, {"action": action})
    assert r["status"] == "error"
    assert r["error"] == DWG_NOT_SUPPORTED
    assert r["client_error_status"] == 415


def test_extract_dwg_metadata_is_no_longer_an_action():
    r = _run({}, {"action": "extract_dwg_metadata"})
    assert r["status"] == "error"
    assert "extract_dwg_metadata" not in r["valid_actions"]


def test_index_folder_lists_a_dwg_as_not_supported(tmp_path):
    from app.core.cad_formats import DWG_NOT_SUPPORTED

    (tmp_path / "level_02.dwg").write_bytes(b"AC1032")
    (tmp_path / "level_02.dxf").write_text("0\nEOF\n")
    r = _run({"folder_path": str(tmp_path), "project_id": "p1"}, {"action": "index_folder"})
    assert r["status"] == "success"
    assert r["indexed"] == 1
    assert r["not_supported"] == [{"name": "level_02.dwg", "error": DWG_NOT_SUPPORTED}]


def test_compare_versions_missing_paths_returns_status_error():
    r = _run({}, {"action": "compare_versions"})
    assert r["status"] == "error"
    assert "old_version" in r["error"] or "new_version" in r["error"]


def test_compare_versions_propagates_parse_error_instead_of_zeroing_count():
    """Pre-1.1.0 bug: a missing old_version was swallowed as old_count=0,
    leading to a misleading diff. Must now surface as status:error."""
    r = _run(
        {"old_version": "/tmp/nope_old.ifc", "new_version": "/tmp/nope_new.ifc"},
        {"action": "compare_versions"},
    )
    assert r["status"] == "error"
    assert "parse failed" in r["error"]


# ── Success path + element_types filter ──────────────────────────────────────


def _block_with_extractor():
    """BIMBlock with a real bim_extractor wired in (bypassing the
    construction kit registration which isn't loaded in unit tests)."""
    from app.blocks.bim_extractor import BIMExtractorBlock
    block = BIMBlock(hal_block=None, config={})
    block.wire("bim_extractor", BIMExtractorBlock())
    return block


def test_parse_ifc_success_shape_on_real_fixture():
    if not os.path.exists(FIXTURE):
        pytest.skip(f"missing IFC fixture {FIXTURE}")
    block = _block_with_extractor()
    r = asyncio.run(
        block.process({"file_path": FIXTURE}, {"action": "parse_ifc"})
    )
    assert r["status"] == "success", r.get("error")
    assert r["file"] == FIXTURE
    assert isinstance(r.get("elements"), list)
    assert r.get("count") == len(r["elements"])
    assert r.get("schema") in ("IFC4", "IFC2X3")


def test_element_types_filter_is_honoured_on_real_fixture():
    """Pre-1.1.0 bug: element_types was passed to bim_extractor but
    bim_extractor doesn't read that key, so the filter silently dropped
    and the caller got the FULL element list. After fix, the filter is
    applied in bim.py before the return."""
    if not os.path.exists(FIXTURE):
        pytest.skip(f"missing IFC fixture {FIXTURE}")
    block = _block_with_extractor()

    # First: unfiltered count
    full = asyncio.run(
        block.process(
            {"file_path": FIXTURE, "element_types": []},
            {"action": "parse_ifc"},
        )
    )
    assert full["status"] == "success", full.get("error")
    full_count = full["count"]
    assert full_count > 0

    # Then: ask only for IfcWall and assert the filter narrowed the list.
    walls = asyncio.run(
        block.process(
            {"file_path": FIXTURE, "element_types": ["IfcWall"]},
            {"action": "parse_ifc"},
        )
    )
    assert walls["status"] == "success", walls.get("error")
    assert walls["count"] < full_count, (
        "element_types filter did nothing — every returned element should "
        "have ifc_type='IfcWall'"
    )
    assert walls["element_types_filter"] == ["IfcWall"]
    for el in walls["elements"]:
        assert el.get("ifc_type") == "IfcWall", (
            f"filter let through a non-wall element: {el.get('ifc_type')}"
        )


def test_parse_ifc_without_extractor_returns_status_error():
    """When the bim_extractor dep isn't wired (kit not loaded), bim must
    surface a clean error rather than crash or silently no-op."""
    block = BIMBlock(hal_block=None, config={})
    r = asyncio.run(
        block.process({"file_path": FIXTURE}, {"action": "parse_ifc"})
    )
    assert r["status"] == "error"
    assert "bim_extractor block not available" in r["error"]


def test_version_field():
    assert BIMBlock.version == "1.1.0"
