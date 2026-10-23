"""extract_measurements must read the drawing, not just answer in the shape.

Audit bar 3, 2026-08-12. This is the one of three unreachable actions that
turned out to be real and was kept wired (the other two cannot run at all --
see tests/test_construction_actions_reachable.py). Wiring it was only half the
job: control-delete showed it had no coverage that could detect it breaking.

Replacing the whole body with

    return {"status": "success", "measurements": [], "specifications": [],
            "count": 0, "confidence": 0}

left tests/test_all_blocks.py and tests/test_blocks_simple.py green -- 24
passed. Both reference the action, but both call it with `{}` and no file, so
they only ever exercise the honest-error branch. The extraction itself was
untested, which is the usual reason an action is unreachable in the first
place: nothing ever ran it.

The drawing is a synthetic key plan generated per run
(``tests/_synthetic_fixtures.build_drawing_keyplan_pdf``): a two-sheet PDF
whose legend carries counted items and three material grades, so the expected
specification grades are known from the generator rather than copied from a
client document.
"""
from __future__ import annotations

import pytest

from tests._synthetic_fixtures import build_drawing_keyplan_pdf


@pytest.fixture(scope="module")
def drawing(tmp_path_factory):
    path = tmp_path_factory.mktemp("extract_measurements") / "drawing_qz_keyplan.pdf"
    spec = build_drawing_keyplan_pdf(path)
    return path, spec


def test_the_fixture_this_file_depends_on_is_present(drawing):
    """Without this, a broken generator would return the action to zero
    coverage SILENTLY. The drawing is generated, never committed, so it must
    exist and be a real multi-page PDF on every run, in every profile.
    """
    import fitz

    path, spec = drawing
    assert path.is_file(), f"{path} was not generated"
    with fitz.open(str(path)) as doc:
        assert doc.page_count == spec["pages"] >= 2


@pytest.fixture(scope="module")
def extracted(drawing):
    """Parse the PDF once -- it is a multi-page drawing."""
    import asyncio

    from app.containers.construction import ConstructionContainer

    return asyncio.run(
        ConstructionContainer().extract_measurements({"file_path": str(drawing[0])}, {})
    )


def test_measurements_come_off_the_drawing(extracted):
    """Non-empty, and varied enough that a constant cannot fake it."""
    assert extracted["status"] == "success", extracted
    measurements = extracted["measurements"]
    assert measurements, "no measurements extracted from a construction drawing"

    # A stub returning one repeated item would satisfy "non-empty". Distinct
    # source strings can only come from reading different parts of the file.
    distinct = {m.get("raw") for m in measurements}
    assert len(distinct) > 5, (
        f"{len(measurements)} measurements but only {len(distinct)} distinct source "
        "strings -- that is a repeated constant, not an extraction"
    )


def test_count_reconciles_with_the_items(extracted):
    """`count` must describe the list beside it.

    The two fields are returned independently, so a body that stopped
    extracting but kept a count -- or vice versa -- shows up here.
    """
    assert extracted["count"] == len(extracted["measurements"]), extracted["count"]


def test_specifications_are_the_grades_on_the_sheet(extracted, drawing):
    """The legend names three material grades; the specification pass must
    report exactly those, in sheet order."""
    grades = [s["value"] for s in extracted["specifications"]]
    assert grades == list(drawing[1]["grades"]), grades


def test_every_measurement_is_structurally_complete(extracted):
    """Half-populated rows are the failure mode a length check misses."""
    for m in extracted["measurements"]:
        assert {"type", "value", "unit", "item", "raw"} <= set(m), m
        assert isinstance(m["value"], (int, float)) and not isinstance(m["value"], bool), m
        assert m["unit"], m


def test_confidence_is_the_measured_one(extracted):
    """Guards the fix made alongside this file.

    The old code read `confidence["measurement_extraction"]`, a key
    `process_document` has never produced, defaulting to 0 -- so every
    successful extraction reported zero confidence. The real contract is a
    measured dict, and it is forwarded whole: the caveats explain why the
    score is not 1.0, so reducing it to a bare number would be a quieter
    version of the same defect.
    """
    confidence = extracted["confidence"]
    assert isinstance(confidence, dict), (
        f"confidence={confidence!r} -- expected the measured dict from "
        "process_document, not a scalar and not a placeholder"
    )
    assert confidence.get("measured") is True, confidence
    assert 0 < confidence["overall"] <= 1, confidence
    assert confidence.get("signals"), "a confidence with no signals is not measured"


@pytest.mark.asyncio
async def test_refuses_honestly_with_nothing_to_read():
    """No file and no PDF block must be an explicit refusal, never an empty
    success -- an empty success is what makes a gutted extractor invisible."""
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().extract_measurements({}, {})
    assert result["status"] == "error"
    assert "extract" in result["error"].lower()


@pytest.mark.asyncio
async def test_route_reaches_the_same_extraction(drawing):
    """The router path must produce the measurements too. The action was
    unreachable until 2026-08-12, so this direction has never been exercised."""
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().route(
        "extract_measurements", {"file_path": str(drawing[0])},
        {"action": "extract_measurements"},
    )
    assert result["status"] == "success", result
    assert result["count"] > 0, result
