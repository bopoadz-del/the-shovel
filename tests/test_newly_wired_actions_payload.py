"""Payload coverage for the two actions un-parked on 2026-08-12.

`track_progress` and `generate_construction_report` were unroutable because
they were unrunnable: four helpers they called had never been written, and the
photo/BIM chain under `track_progress` also read a key (`elements`) that
`BIMExtractorBlock` has never returned, so it compared every photograph
against an empty model.

The helpers are implemented and the chain is fixed. These tests exist so that
wiring them does not simply move the problem from unreachable to untested --
each assertion is on a value that can only come from the chain actually
running end to end.

The dangerous failure mode here is not a crash. It is a CONFIDENT ZERO: with
no image block, or no resolvable model, the old code returned
`{"status": "success", "progress_percentage": 0}` -- a manufactured finding
that nothing has been built, indistinguishable on screen from a real one.
Several tests below exist only to hold that line.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

IFC = Path("tests/fixtures/sample_office.ifc")
PHOTOS = sorted(glob.glob("tests/fixtures/photos/*.jpg"))[:2]


class StubImageBlock:
    """Test double for the vision block -- NOT shipped, and never imported by
    app code. Returns two labels present in the model and one that is not, so
    both the match path and the deviation path are exercised."""

    async def execute(self, input_data, params):
        return {"result": {"objects": [
            {"label": "wall"}, {"label": "door"}, {"label": "excavator"},
        ]}}


def _container(*, image=False, bim=False):
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    if bim:
        from app.blocks.bim_extractor import BIMExtractorBlock
        c.wire("bim_extractor", BIMExtractorBlock())
    if image:
        c.wire("image", StubImageBlock())
    return c


# ── track_progress: it must never manufacture a zero ─────────────────────────

@pytest.mark.asyncio
async def test_refuses_without_photographs():
    result = await _container().track_progress({"bim_file": str(IFC)}, {})
    assert result["status"] == "error"
    assert "photograph" in result["error"].lower()


@pytest.mark.asyncio
@pytest.mark.skipif(not PHOTOS, reason="no photo fixtures")
async def test_missing_image_block_is_a_refusal_not_zero_progress():
    """The finding this whole file is built around.

    No vision block means nothing looked at the photographs. Reporting 0%
    progress and 'high' delay risk from that is a fabricated finding, and it
    is the one that would land in front of a client.
    """
    result = await _container(bim=True).track_progress(
        {"photos": PHOTOS, "bim_file": str(IFC)}, {})

    assert result["status"] == "error", (
        f"expected a refusal, got {result.get('status')} with "
        f"progress_percentage={result.get('progress_percentage')!r} -- that is a "
        "progress claim manufactured out of a missing dependency"
    )
    assert "progress_percentage" not in result
    assert "image analysis" in result["error"].lower()


@pytest.mark.asyncio
@pytest.mark.skipif(not PHOTOS, reason="no photo fixtures")
async def test_missing_model_is_a_refusal_not_zero_progress():
    """Same rule on the other input: no model, no comparison, no number."""
    result = await _container(image=True).track_progress(
        {"photos": PHOTOS, "bim_file": ""}, {})

    assert result["status"] == "error"
    assert "progress_percentage" not in result
    assert "compared" in result["error"].lower()


@pytest.mark.asyncio
@pytest.mark.skipif(not PHOTOS or not IFC.is_file(), reason="fixtures missing")
async def test_identifies_model_elements_in_the_photographs():
    """The happy path, end to end: real IFC parse, real comparison.

    `sample_office.ifc` contains walls, so a detected "wall" must match them.
    Before the fixes this returned 0 matches for two separate reasons -- the
    model query read a key the extractor never returns, and the token
    comparison scored wall-vs-walls at 0.571 against a 0.6 threshold.
    """
    pytest.importorskip("ifcopenshell", reason="IFC parsing unavailable")

    result = await _container(image=True, bim=True).track_progress(
        {"photos": PHOTOS, "bim_file": str(IFC)}, {})

    assert result["status"] == "success", result
    assert result["elements_found"] > 0, (
        "no model element was identified from a photograph containing a label "
        "that matches elements in the model -- the comparison is not working"
    )
    # found + missing must account for the elements actually compared
    assert result["elements_found"] + result["elements_missing"] > 0
    assert 0 < result["progress_percentage"] <= 100, result["progress_percentage"]
    assert result["basis"], "a progress figure with no stated basis is a bare number"


@pytest.mark.asyncio
@pytest.mark.skipif(not PHOTOS or not IFC.is_file(), reason="fixtures missing")
async def test_objects_absent_from_the_model_are_reported_as_deviations():
    """`_find_deviations` catches seen-but-not-modelled.

    The stub reports an excavator, which no office model contains. It must
    surface as a deviation and must NOT be counted as a matched element.
    """
    pytest.importorskip("ifcopenshell", reason="IFC parsing unavailable")

    result = await _container(image=True, bim=True).track_progress(
        {"photos": PHOTOS, "bim_file": str(IFC)}, {})

    detected = {d["detected"] for d in result["deviations"]}
    assert "excavator" in detected, result["deviations"]
    for dev in result["deviations"]:
        assert dev["similarity"] < 0.6, dev


@pytest.mark.asyncio
async def test_delay_risk_says_not_assessed_rather_than_low():
    """A risk function with no data must refuse, not answer 'low'.

    'low' is the answer a reader acts on, so it is the expensive default.
    """
    risk = _container()._assess_delay_risk([{"elements_expected": 0}])
    assert risk["level"] == "not_assessed", risk
    assert "no shortfall" in risk["reason"].lower() or "no model" in risk["reason"].lower()


def test_element_similarity_is_symmetric_and_bounded():
    """Property check on the metric itself -- it is generic text similarity,
    so it must behave like one rather than like a tuned constant."""
    c = _container()
    wall = {"ifc_type": "IfcWall", "category": "walls", "name": "Wall-GF-N"}
    assert c._element_similarity(wall, {"label": "wall"}) > 0.6
    assert c._element_similarity(wall, {"label": "excavator"}) < 0.6
    assert c._element_similarity(wall, {"label": "wall"}) == \
        c._element_similarity({"label": "wall"}, wall)
    assert c._element_similarity(wall, {}) == 0.0


# ── generate_construction_report ─────────────────────────────────────────────

@pytest.fixture
def drawing(tmp_path):
    """A synthetic two-sheet key plan (tests/_synthetic_fixtures.py)."""
    from tests._synthetic_fixtures import build_drawing_keyplan_pdf

    path = tmp_path / "drawing_qz_keyplan.pdf"
    return path, build_drawing_keyplan_pdf(path)


@pytest.mark.asyncio
async def test_report_summarises_the_document_it_read(drawing):
    DRAWING, spec = drawing
    result = await _container().generate_construction_report(
        {"file_path": str(DRAWING)}, {})

    assert result["status"] == "success", result
    summary = result["summary"]
    assert summary["document"] == DRAWING.name
    assert summary["type"] == "drawing", summary
    assert summary["pages"] == spec["pages"], (
        "a report over a PDF that did not count its pages did not read it")
    assert summary["measurements_found"] > 0, summary


@pytest.mark.asyncio
async def test_report_survives_a_non_drawing_document():
    """The KeyError that kept this action unrouted.

    `process_document` dispatches by type and only the drawing branch returns
    doc_type / detected_disciplines / total_pages / measurements / tables.
    Indexing them raised KeyError: 'doc_type' on every specification, report
    and contract.
    """
    import tempfile

    from app.blocks.spec_analyzer import SpecAnalyzerBlock
    from scripts.synthetic_fixtures import build_spec_section_txt

    container = _container()
    container.wire("spec_analyzer", SpecAnalyzerBlock())
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "synthetic_spec_section_03_concrete.txt"
        build_spec_section_txt(spec)
        result = await container.generate_construction_report({"file_path": str(spec)}, {})

    assert result["status"] == "success", result
    assert result["summary"]["document"] == spec.name


@pytest.mark.asyncio
async def test_recommendations_are_derived_from_the_document(drawing):
    """Every recommendation must be conditional on something the parse found.

    Canned advice under a construction-analysis heading reads as a finding, so
    the measurement count in the text has to match the measurements actually
    extracted.
    """
    result = await _container().generate_construction_report(
        {"file_path": str(drawing[0])}, {})

    recs = result["recommendations"]
    assert recs and all(isinstance(r, str) and r for r in recs), recs
    found = result["summary"]["measurements_found"]
    assert any(str(found) in r for r in recs), (
        f"no recommendation references the {found} measurements that were "
        f"extracted, so none of them is derived from this document: {recs}"
    )
