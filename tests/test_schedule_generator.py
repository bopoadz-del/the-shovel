"""Tests for the roadmap-named Schedule Generator shim.

The real WBS generator is tested in ``tests/test_construction_generate_wbs.py``;
this file only proves that ``app.blocks.schedule_generator.ScheduleGeneratorBlock``
delegates to that working implementation and returns the expected shape.

PR#3 (ORDERS_28OCT_0345Z/0415Z): generate_wbs derives from real structured
input, so the delegation/shape tests are re-pointed to the BOQ scope (the
template-scaling clamp test is removed — the BOQ path keeps the measured row
count and never scales to a target).
"""

import pytest

from app.blocks.schedule_generator import ScheduleGeneratorBlock

BOQ_ASK = "Generate a high-level WBS for the demolition and site clearance bill of quantities scope."
# Stated rates so durations are kernel-derived (the shipped norm table is empty
# until sourced — ruling 1045Z); the shape test needs CPM fields, i.e. real
# durations.
BOQ_ITEMS = [
    {"item_key": "D110", "description": "General site clearance", "quantity": 10, "unit": "m2", "productivity_rate": "5 m2 per gang-day"},
    {"item_key": "D290.1", "description": "Removal of trees in existing sidewalks", "quantity": 20, "unit": "m", "productivity_rate": "10 m per gang-day"},
    {"item_key": "D549.2", "description": "Removal of existing chain link fence", "quantity": 30, "unit": "nr", "productivity_rate": "15 nr per gang-day"},
]


@pytest.fixture
def block():
    return ScheduleGeneratorBlock()


@pytest.mark.asyncio
async def test_schedule_generator_delegates_to_generate_wbs(block):
    result = await block.process(
        {"brief": BOQ_ASK, "boq_items": BOQ_ITEMS},
        params={"target_count": 60},
    )

    assert result["status"] == "success"
    assert result["scaffold"]["derived_from_boq"] is True
    # One activity per retrieved BOQ row — delegation produces input-derived rows.
    assert result["actual_count"] == len(BOQ_ITEMS)
    assert len(result["activities"]) == result["actual_count"]
    assert isinstance(result.get("wbs_tree"), dict) and result["wbs_tree"]
    assert isinstance(result.get("summary"), dict)
    assert isinstance(result.get("assumptions"), list)


@pytest.mark.asyncio
async def test_schedule_generator_activity_shape(block):
    result = await block.process(
        {"brief": BOQ_ASK, "boq_items": BOQ_ITEMS},
        params={"target_count": 30},
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
async def test_schedule_generator_prose_brief_derives_with_egress_else_no_scope(block, monkeypatch):
    """1030Z re-point: delegation structures a prose brief into a derived
    schedule when the egress is configured; a brief with no derivable scope is
    the no-scope error, never a template."""
    from app.core import llm_client

    async def fake(system, user, **k):
        if "pour" in (user or "").lower():
            return {"activities": [
                {"name": "Pour slab", "source_span": "pour the slab", "duration_days": 3, "predecessors": []},
            ]}
        return {"activities": []}
    monkeypatch.setattr(llm_client, "complete_json", fake)

    derived = await block.process({"brief": "Then pour the slab."}, params={"target_count": 5})
    assert derived["status"] == "success"
    assert any(a["name"] == "Pour slab" for a in derived["activities"])

    no_scope = await block.process({}, params={"target_count": 5})
    assert no_scope["status"] == "error"
    assert no_scope["activities"] == []


def test_schedule_generator_block_metadata():
    assert ScheduleGeneratorBlock.name == "schedule_generator"
    assert "wbs" in ScheduleGeneratorBlock.tags
