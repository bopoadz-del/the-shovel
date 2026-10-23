"""Project-shaped schedule and BIM-format fixtures.

- A P6 baseline programme generated per run by
  ``tests/_synthetic_fixtures.build_baseline_xer`` (invented "Sample Works"
  programme: a 12-activity zero-float chain bracketed by two milestones plus
  four floated branches, 50 activities). Exercises the schedule family on a
  full P6-shaped export (PROJECT data date, CALENDAR, PROJWBS, RSRC, TASK,
  TASKPRED, TASKRSRC) instead of the minimal resource_loaded fixtures. The
  expected summary is computed by the generator from the same rows it writes.
- Navisworks / DWG: BIM models in the corpus are .nwd (no native IFC
  exists; "IFC" folders hold Issued-For-Construction drawings, not BIM).
  The honest paths those files hit are pinned here deterministically:
  .nwd -> convert-to-IFC guidance; .dwg -> "DWG is not supported" (DWG
  take-off is removed; export DXF or PDF).
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from tests._synthetic_fixtures import build_baseline_xer
from tests.conftest import requires_construction_kit


@pytest.fixture
def container():
    return ConstructionContainer()


@pytest.fixture
def baseline(tmp_path):
    path = tmp_path / "sample_works_baseline.xer"
    return str(path), build_baseline_xer(path)


@requires_construction_kit
class TestBaselineXer:
    @pytest.mark.asyncio
    async def test_parse_baseline_programme(self, container, baseline):
        xer, expected = baseline
        result = await container.parse_primavera_schedule({"file_path": xer}, {})
        assert result["status"] == "success"
        s = result["summary"]
        assert s["total_activities"] == expected["total_activities"] == 50
        assert s["critical_activities"] == expected["critical_activities"] == 14
        assert s["project_duration"] == expected["project_duration"] == 160
        assert s["data_date"] == expected["data_date"] == "2025-02-24"
        assert len(result["critical_path"]["activities"]) == 14

    @pytest.mark.asyncio
    async def test_programme_feeds_delay_analysis_shape(self, container, baseline):
        result = await container.parse_primavera_schedule({"file_path": baseline[0]}, {})
        # The parse deliverable carries the delay/risk sections downstream
        # actions consume — assert they exist and are list/dict shaped.
        assert isinstance(result["schedule_risks"], list)
        # A baseline programme has nothing to be delayed against — the key
        # must exist (downstream reads it) but None is the honest value here.
        assert "delay_analysis" in result
        assert isinstance(result["recovery_options"], list)


class TestNavisworksHonestRejection:
    @pytest.mark.asyncio
    async def test_nwd_gets_convert_to_ifc_guidance(self, tmp_path):
        from app.blocks.bim_extractor import BIMExtractorBlock

        fake = tmp_path / "federated_model.nwd"
        fake.write_bytes(b"not a real navisworks file")
        result = await BIMExtractorBlock().process({"file_path": str(fake)}, {})
        assert result["status"] == "error"
        assert ".nwd" in result["error"] or "Navisworks" in result["error"]
        assert "IFC" in result["error"]


class TestDwgNotSupported:
    @pytest.mark.asyncio
    async def test_dwg_is_refused_with_the_export_message(self, tmp_path):
        from app.blocks.drawing_qto import DrawingQTOBlock
        from app.core.cad_formats import DWG_NOT_SUPPORTED

        fake = tmp_path / "site_plan.dwg"
        fake.write_bytes(b"AC1032 not really a dwg")
        result = await DrawingQTOBlock().process({"file_path": str(fake)}, {})
        assert result["status"] == "error"
        assert result["error"] == DWG_NOT_SUPPORTED
