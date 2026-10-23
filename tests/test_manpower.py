"""Tests for the roadmap-named Manpower Planner shim.

The real histogram logic is tested in ``tests/test_pm_excel.py`` and
``tests/test_pm_computations.py``; this file only proves that
``app.blocks.manpower_planner.ManpowerPlannerBlock`` delegates to that working
implementation and returns the expected shape.
"""

import pytest

from app.blocks.manpower_planner import ManpowerPlannerBlock


@pytest.fixture
def block():
    return ManpowerPlannerBlock()


@pytest.mark.asyncio
async def test_manpower_planner_delegates_to_resource_histogram(block):
    activities = [
        {
            "id": "A",
            "duration": 5,
            "predecessors": [],
            "resources": [{"trade": "labor", "count": 4}],
        },
        {
            "id": "B",
            "duration": 5,
            "predecessors": [{"predecessor_id": "A"}],
            "resources": [{"trade": "labor", "count": 4}],
        },
    ]
    result = await block.process({"activities": activities, "period_unit": "week"})

    assert result["status"] == "success"
    assert result["period_unit"] == "week"
    assert result["periods"]
    assert result["by_trade_totals"]["labor"] > 0
    assert result["total_manhours"] > 0


@pytest.mark.asyncio
async def test_manpower_planner_returns_error_without_activities(block):
    result = await block.process({})
    assert result["status"] == "error"
    assert "No activities" in result["error"]


@pytest.mark.asyncio
async def test_manpower_planner_uses_cpm_results_when_provided(block):
    activities = [
        {
            "id": "A",
            "duration": 5,
            "predecessors": [],
            "resources": [{"trade": "labor", "count": 4}],
        },
    ]
    results = [
        {
            "id": "A",
            "name": "A",
            "duration": 5,
            "early_start_day": 0,
            "early_finish_day": 5,
            "late_start_day": 0,
            "late_finish_day": 5,
            "total_float": 0,
            "free_float": 0,
            "is_critical": True,
        }
    ]
    result = await block.process({"activities": activities, "results": results})
    assert result["status"] == "success"
    assert result["by_trade_totals"]["labor"] > 0


def test_manpower_planner_block_metadata():
    assert ManpowerPlannerBlock.name == "manpower_planner"
    assert "manpower" in ManpowerPlannerBlock.tags


@pytest.mark.asyncio
async def test_manpower_planner_accepts_boq_derived_activities(block):
    """schedule-from-boq emits duration_days + crew_size + string predecessors.

    /v1/schedule/manpower used to 422 those payloads because CPM Activity
    requires duration + resources=[{trade,count}] + Dependency objects.
    """
    activities = [
        {
            "id": "A",
            "name": "Bulk excavation",
            "duration_days": 10,
            "crew_size": 6,
            "predecessors": [],
        },
        {
            "id": "B",
            "name": "Raft concrete",
            "duration_days": 5,
            "crew_size": 8,
            "predecessors": ["A"],
        },
    ]
    result = await block.process({"activities": activities, "period_unit": "week"})
    assert result["status"] == "success", result
    assert result["by_trade_totals"]["General"] > 0
    assert result["total_manhours"] > 0
