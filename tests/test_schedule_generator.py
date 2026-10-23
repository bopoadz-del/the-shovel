"""Tests for the roadmap-named Schedule Generator shim.

The real WBS generator is tested in ``tests/test_construction_generate_wbs.py``;
this file only proves that ``app.blocks.schedule_generator.ScheduleGeneratorBlock``
delegates to that working implementation and returns the expected shape.
"""

import pytest

from app.blocks.schedule_generator import ScheduleGeneratorBlock


@pytest.fixture
def block():
    return ScheduleGeneratorBlock()


@pytest.mark.asyncio
async def test_schedule_generator_delegates_to_generate_wbs(block):
    result = await block.process(
        {"brief": "Build a 50MW hyperscale data center."},
        params={"target_count": 60, "project_type": "data_center"},
    )

    assert result["status"] == "success"
    assert result["project_type"] == "data_center"
    assert result["actual_count"] >= 60
    assert len(result["activities"]) == result["actual_count"]
    assert isinstance(result.get("wbs_tree"), dict) and result["wbs_tree"]
    assert isinstance(result.get("summary"), dict)
    assert isinstance(result.get("assumptions"), list)


@pytest.mark.asyncio
async def test_schedule_generator_activity_shape(block):
    result = await block.process(
        {},
        params={"target_count": 30, "project_type": "building"},
    )
    activities = result["activities"]
    assert activities
    a = activities[0]
    assert isinstance(a.get("id"), str) and a["id"]
    assert isinstance(a.get("code"), str) and a["code"]
    assert "duration_days" in a
    assert "early_start_day" in a
    assert "critical" in a


@pytest.mark.asyncio
async def test_schedule_generator_clamps_low_target(block):
    result = await block.process({}, params={"target_count": 5})
    assert result["target_count"] == 20
    assert result["actual_count"] >= 20


def test_schedule_generator_block_metadata():
    assert ScheduleGeneratorBlock.name == "schedule_generator"
    assert "wbs" in ScheduleGeneratorBlock.tags
