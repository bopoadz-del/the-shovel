"""Tests for the roadmap-named CPM Engine shim.

The real CPM math is tested in ``tests/test_pm_computations.py``; this file
only proves that ``app.blocks.cpm_engine.CPMEngineBlock`` delegates to that
working implementation and returns the expected shape.
"""

import pytest

from app.blocks.cpm_engine import CPMEngineBlock


@pytest.fixture
def block():
    return CPMEngineBlock()


@pytest.mark.asyncio
async def test_cpm_engine_delegates_to_pm_computations(block):
    activities = [
        {"id": "A", "duration": 3, "predecessors": []},
        {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
        {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "A"}]},
        {"id": "D", "duration": 2, "predecessors": [{"predecessor_id": "B"}, {"predecessor_id": "C"}]},
    ]
    result = await block.process({"activities": activities})

    assert result["status"] == "success"
    assert result["project_duration"] == 10
    assert result["critical_path"] == ["A", "B", "D"]
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["C"]["total_float"] == 3
    assert by_id["C"]["is_critical"] is False
    assert by_id["A"]["is_critical"] is True


@pytest.mark.asyncio
async def test_cpm_engine_returns_error_without_activities(block):
    result = await block.process({})
    assert result["status"] == "error"
    assert "No activities" in result["error"]


@pytest.mark.asyncio
async def test_cpm_engine_supports_dependency_lag(block):
    activities = [
        {"id": "A", "duration": 3, "predecessors": []},
        {
            "id": "B",
            "duration": 2,
            "predecessors": [{"predecessor_id": "A", "type": "FS", "lag": 2}],
        },
    ]
    result = await block.process({"activities": activities})
    assert result["status"] == "success"
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["B"]["early_start_day"] == 5
    assert by_id["B"]["early_finish_day"] == 7


def test_cpm_engine_block_metadata():
    assert CPMEngineBlock.name == "cpm_engine"
    assert "cpm" in CPMEngineBlock.tags
