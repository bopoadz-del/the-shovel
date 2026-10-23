"""Tests for ConstructionContainer.generate_wbs — the deterministic WBS
generator that backs the heavy-reasoning agent's "create a 200 activity
schedule" path.

The method is template-based (no LLM), so the assertions are deterministic
on activity count, CPM well-formedness, and project-type detection. Keeps
the heavy-reasoning routing path testable without burning provider tokens.
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer


@pytest.fixture
def container():
    return ConstructionContainer()


@pytest.mark.asyncio
async def test_generate_wbs_default_data_center_target_200(container):
    """Canonical user case: data-center brief, target_count=200.

    Plan accepted ~225 activities (template scaling overshoots slightly to
    keep each zone's scaffold intact). Lock in the count >= target.
    """
    result = await container.generate_wbs(
        input_data={"brief": "Build a 50MW hyperscale data center."},
        params={"target_count": 200, "project_type": "data_center"},
    )

    assert result["status"] == "success"
    assert result["project_type"] == "data_center"
    assert result["target_count"] == 200
    # Template scaling rounds up to keep zones whole, never down.
    assert result["actual_count"] >= 200
    assert len(result["activities"]) == result["actual_count"]
    # WBS tree, summary, and assumptions are part of the contract.
    assert isinstance(result.get("wbs_tree"), dict) and result["wbs_tree"]
    assert isinstance(result.get("summary"), dict)
    assert isinstance(result.get("assumptions"), list) and result["assumptions"]


@pytest.mark.asyncio
async def test_generate_wbs_activity_shape(container):
    """Every activity carries the fields the agent / UI / CPM consumers need."""
    result = await container.generate_wbs(
        input_data={},
        params={"target_count": 60, "project_type": "building"},
    )
    activities = result["activities"]
    assert activities, "must return at least one activity"

    a = activities[0]
    # Identity + naming
    assert isinstance(a.get("id"), str) and a["id"]
    assert isinstance(a.get("code"), str) and a["code"]
    assert isinstance(a.get("name"), str) and a["name"]
    # Scheduling
    assert isinstance(a.get("duration_days"), (int, float)) and a["duration_days"] >= 0
    assert isinstance(a.get("predecessors"), list)
    # CPM fields attached (compute_cpm emits *_day suffixes).
    for k in (
        "early_start_day", "early_finish_day",
        "late_start_day", "late_finish_day",
        "total_float_days", "critical",
    ):
        assert k in a, f"activity missing CPM field {k!r}"
    # Resources may be empty per phase but the key exists.
    assert "resources" in a


@pytest.mark.asyncio
async def test_generate_wbs_clamps_low_and_high_targets(container):
    """target_count is clamped to [20, 1000] regardless of input."""
    low = await container.generate_wbs(input_data={}, params={"target_count": 5})
    high = await container.generate_wbs(input_data={}, params={"target_count": 5000})

    # Effective clamp is reflected in target_count field too.
    assert low["target_count"] == 20
    assert high["target_count"] == 1000
    # Actual count still scales to satisfy the clamped target.
    assert low["actual_count"] >= 20
    assert high["actual_count"] >= 1000


@pytest.mark.asyncio
async def test_generate_wbs_infers_project_type_from_brief(container):
    """When no project_type is passed, the brief is used to infer it."""
    result = await container.generate_wbs(
        input_data={"brief": "Construct a 200MW solar PV plant in Kenya."},
        params={"target_count": 50},
    )
    assert result["project_type"] == "solar_plant"

    result_wind = await container.generate_wbs(
        input_data={"brief": "Offshore wind farm 100MW with turbines and substation."},
        params={"target_count": 50},
    )
    assert result_wind["project_type"] == "wind_farm"


@pytest.mark.asyncio
async def test_generate_wbs_unknown_project_type_defaults_to_building(container):
    """An unknown project_type falls back to the GENERIC building template,
    not data_center (F39: generic briefs were getting hyperscale WBS items),
    and not an error."""
    result = await container.generate_wbs(
        input_data={},
        params={"target_count": 30, "project_type": "spacecraft"},
    )
    assert result["status"] == "success"
    assert result["project_type"] == "building"


@pytest.mark.asyncio
async def test_generate_wbs_cpm_well_formed(container):
    """CPM output: every activity's predecessors must reference existing
    activity codes (no dangling refs), and EF >= ES for every row."""
    result = await container.generate_wbs(
        input_data={},
        params={"target_count": 80, "project_type": "infrastructure"},
    )
    activities = result["activities"]
    code_set = {a["code"] for a in activities}

    for a in activities:
        for pred in a.get("predecessors", []):
            assert pred in code_set, f"dangling predecessor {pred!r} in {a['code']}"
        assert a["early_finish_day"] >= a["early_start_day"], f"EF<ES in {a['code']}"
        assert a["late_finish_day"] >= a["late_start_day"], f"LF<LS in {a['code']}"
        assert a["total_float_days"] >= 0, f"negative float in {a['code']}"


@pytest.mark.asyncio
async def test_generate_wbs_routes_via_route_dispatch(container):
    """The heavy-reasoning agent invokes generate_wbs via container.route().
    Pin that dispatch path so a registry typo can't silently break it."""
    result = await container.route(
        action="generate_wbs",
        input_data={"brief": "Office tower 30 floors."},
        params={"target_count": 40, "project_type": "building"},
    )
    assert result["status"] == "success"
    assert result["actual_count"] >= 40
