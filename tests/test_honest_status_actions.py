"""Honest-status contracts for actions that previously looked more done than
they are (2026-07-24 seam sweep).

- jetson_dispatch: no edge hardware exists — it must validate + stage the job
  and say `pending_hardware`, not pretend or bare-error.
- om_manual_generator: must never fabricate a TBC equipment schedule.
- daily_site_report: empty input sections must be named, not silently empty.
- tender_bid_analysis: unimplemented sub-analyses must be declared.
- drawing_qto: the geometry engine in use must be visible in the result.
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer


@pytest.fixture
def container():
    return ConstructionContainer()


class TestJetsonDispatchPendingHardware:
    @pytest.mark.asyncio
    async def test_valid_job_is_staged_pending_hardware(self, container):
        result = await container.jetson_dispatch(
            {"task": "safety_scan", "payload": {"camera": "gate-3"}}, {}
        )
        assert result["status"] == "pending_hardware"
        assert result["staged_job"]["task"] == "safety_scan"
        assert result["staged_job"]["payload_keys"] == ["camera"]
        assert "not yet installed" in result["reason"]

    @pytest.mark.asyncio
    async def test_missing_task_is_an_input_error(self, container):
        result = await container.jetson_dispatch({}, {})
        assert result["status"] == "error"
        assert "task" in result["error"]


class TestOmManualNoFabrication:
    @pytest.mark.asyncio
    async def test_no_equipment_refuses_instead_of_inventing(self, container):
        result = await container.om_manual_generator({}, {})
        assert result["status"] == "error"
        assert "equipment" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_supplied_equipment_still_generates(self, container):
        result = await container.om_manual_generator(
            {"equipment_list": [{
                "tag": "HVAC-01", "description": "AHU-1", "system_type": "HVAC",
                "manufacturer": "Carrier", "model": "39M", "location": "Roof",
                "warranty_years": 2,
            }]},
            {"project_name": "P1"},
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_the_summary_does_not_overstate_what_was_generated(self, container):
        """`maintenance_tasks_generated` used to be `len(daily) + len(monthly)`.

        With a hollow daily-task generator and a one-string monthly one, a
        whole plant came back as "1 maintenance task generated" — a false
        claim on a client deliverable. It now counts all five task lists,
        and the count has to agree with the section it describes.
        """
        result = await container.om_manual_generator(
            {"equipment_list": [{
                "tag": "HVAC-01", "description": "AHU-1", "system_type": "HVAC",
                "category": "hvac_equipment", "location": "Roof",
            }]},
            {"project_name": "P1"},
        )

        preventive = next(
            s["content"] for s in result["sections"] if s["section"].startswith("E.")
        )
        actual = sum(
            len(preventive.get(k) or [])
            for k in ("daily_tasks", "weekly_tasks", "monthly_tasks",
                      "quarterly_tasks", "annual_tasks")
        )
        assert result["summary"]["maintenance_tasks_generated"] == actual, (
            "the summary count disagrees with the section it summarises"
        )

    @pytest.mark.asyncio
    async def test_unpopulated_sections_are_named_rather_than_shipped_blank(self, container):
        """An empty Section E that says nothing reads as complete. The
        generators behind it are registered roadmap in docs/KNOWN_INCOMPLETE.md,
        so the manual has to say which parts are missing and what they need.
        """
        result = await container.om_manual_generator(
            {"equipment_list": [{
                "tag": "CH-01", "description": "Chiller 1", "system_type": "Chilled Water",
            }]},
            {"project_name": "P1"},
        )

        gaps = result["data_gaps"]
        assert gaps, "sections came back empty with no gap declared"
        joined = " ".join(gaps).lower()
        assert "section e" in joined, "the empty preventive-maintenance section is not declared"
        assert result["summary"]["sections_not_populated"] == len(gaps)
        # ...and it must say what would fill them, not just that they're empty.
        assert "manufacturer" in joined or "supplier" in joined


class TestDailySiteReportIncompleteSections:
    @pytest.mark.asyncio
    async def test_no_inputs_names_empty_sections(self, container):
        result = await container.daily_site_report({}, {"project_name": "P1"})
        assert result["status"] == "success"
        missing = set(result["sections_incomplete"])
        assert {"manpower", "work_completed", "equipment"} <= missing

    @pytest.mark.asyncio
    async def test_marker_key_always_present(self, container):
        result = await container.daily_site_report({}, {})
        assert "sections_incomplete" in result


class TestTenderBidAnalysisGaps:
    @pytest.mark.asyncio
    async def test_analysis_gaps_declared(self, container):
        bids = [
            {"contractor": "Alpha", "bid_amount": 1_000_000, "schedule_days": 300},
            {"contractor": "Beta", "bid_amount": 1_100_000, "schedule_days": 280},
        ]
        result = await container.tender_bid_analysis({"bids": bids}, {})
        assert result["status"] == "success"
        assert "qualification_gaps" in result["analysis_gaps"]
        assert "clarifications_required" in result["analysis_gaps"]


class TestDigitalTwinSyncPayloadsWellFormed:
    """Now that digital_twin_sync is routable (GENERATIVE_INTENTS), its
    prepared payloads are a real deliverable — assert they are well-formed
    and the honest 'prepared_not_pushed' contract holds."""

    @pytest.mark.asyncio
    async def test_prepared_payloads_are_well_formed(self, container):
        result = await container.digital_twin_sync(
            {"data": {"elements": [
                {"id": "W-101", "type": "IfcWall", "properties": {"level": "L1"}},
            ]}},
            {"platform": "azure", "mode": "update", "project_id": "p1"},
        )
        assert result["status"] == "success"
        assert result["sync_status"] == "prepared_not_pushed"
        assert "p1" in result["connection_strings"]
        assert result["authentication_required"]["type"] == "OAuth2"
        assert isinstance(result["operations"], list)
        assert isinstance(result["api_payloads"], list)


class TestDrawingQtoGeometryEngineVisible:
    @pytest.mark.asyncio
    async def test_dxf_result_reports_geometry_engine(self, tmp_path):
        ezdxf = pytest.importorskip("ezdxf")
        doc = ezdxf.new()
        msp = doc.modelspace()
        msp.add_lwpolyline(
            [(0, 0), (10, 0), (10, 10), (0, 10)], close=True
        )
        path = tmp_path / "plan.dxf"
        doc.saveas(path)

        from app.blocks.drawing_qto import DrawingQTOBlock
        result = await DrawingQTOBlock().process({"file_path": str(path)}, {})
        assert result["status"] == "success"
        try:
            import shapely  # noqa: F401
            expected = "shapely"
        except ImportError:
            expected = "fallback"
        assert result["geometry_engine"] == expected
