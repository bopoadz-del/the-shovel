"""Tests for ConstructionContainer.generate_wbs.

PR#3 (ORDERS_28OCT_0345Z/0415Z): generate_wbs no longer emits a per-type
template scaffold. The activity set is derived only from real structured
input — a project's measured BOQ demolition/site-clearance scope, or a
document's long-lead procurement feed. A prose brief with no structured scope
is refused as no-scope (status="error", activities=[]), which the export
routes translate to HTTP 422.

These exercise the generation mechanics (activity shape, CPM well-formedness,
dispatch) against the BOQ scope — real input-traceable rows that survive
Stage 2 — and pin the no-scope / label-only behaviour for the prose-brief path.
Re-pointed from the template path, with the template-scaling tests removed.
"""

from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer

# A demolition/site-clearance WBS-outline + BOQ ask elects the BOQ scope path;
# boq_items are the measured rows (one package + activity per row).
BOQ_ASK = "Generate a high-level WBS for the demolition and site clearance bill of quantities scope."
BOQ_ITEMS = [
    {"item_key": "D110", "description": "General site clearance", "quantity": 10, "unit": "m2"},
    {"item_key": "D290.1", "description": "Removal of trees in existing sidewalks", "quantity": 20, "unit": "m"},
    {"item_key": "D549.2", "description": "Removal of existing chain link fence", "quantity": 30, "unit": "nr"},
    {"item_key": "D599.5", "description": "Breaking out existing carriageway", "quantity": 40, "unit": "item"},
]


@pytest.fixture
def container():
    return ConstructionContainer()


async def _boq_wbs(container, **params):
    return await container.generate_wbs(
        input_data={"brief": BOQ_ASK, "boq_items": BOQ_ITEMS},
        params={"target_count": 60, **params},
    )


@pytest.mark.asyncio
async def test_generate_wbs_boq_scope_produces_a_cost_loadable_wbs(container):
    """Re-pointed from the template path: the BOQ scope yields one input-
    traceable activity per measured row — no template, no scaling to a target."""
    result = await _boq_wbs(container)
    assert result["status"] == "success"
    assert result["scaffold"]["derived_from_boq"] is True
    # One activity per retrieved BOQ row; nothing invented, nothing scaled.
    assert result["actual_count"] == len(BOQ_ITEMS)
    assert len(result["activities"]) == result["actual_count"]
    assert isinstance(result.get("wbs_tree"), dict) and result["wbs_tree"]
    assert isinstance(result.get("summary"), dict)
    assert isinstance(result.get("assumptions"), list) and result["assumptions"]


@pytest.mark.asyncio
async def test_generate_wbs_activity_shape(container):
    """Every activity carries the fields the agent / UI / CPM consumers need."""
    result = await _boq_wbs(container)
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
async def test_generate_wbs_prose_brief_is_refused_as_no_scope(container):
    """The subject Stage 2 (LLM extraction) will legitimately change: a prose
    brief with no structured scope yields no activities and an explicit
    no-scope error — never a template scaffold."""
    result = await container.generate_wbs(
        input_data={"brief": "Build a 50MW hyperscale data center."},
        params={"target_count": 200, "project_type": "data_center"},
    )
    assert result["status"] == "error"
    assert result["activities"] == []
    assert "no schedulable scope" in result["error"].lower()


@pytest.mark.asyncio
async def test_generate_wbs_infers_project_type_from_brief(container):
    """When no project_type is passed, the brief is used to infer it. The label
    is carried even on the no-scope path (project_type is a label only now)."""
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
async def test_generate_wbs_stated_project_type_passes_through_unremapped(container):
    """Re-written from 'unknown type defaults to building': there is no template
    to fall back to, so a stated project_type is an honest label that passes
    through unchanged — never silently remapped — and with no structured scope
    the call is no-scope, never a substituted template."""
    result = await container.generate_wbs(
        input_data={},
        params={"target_count": 30, "project_type": "spacecraft"},
    )
    assert result["project_type"] == "spacecraft"
    assert result["status"] == "error"
    assert result["activities"] == []


@pytest.mark.asyncio
async def test_generate_wbs_cpm_well_formed(container):
    """CPM output: every activity's predecessors must reference existing
    activity codes (no dangling refs), and EF >= ES for every row."""
    result = await _boq_wbs(container)
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
    Pin that dispatch path (over the BOQ scope so it succeeds) so a registry
    typo can't silently break it."""
    result = await container.route(
        action="generate_wbs",
        input_data={"brief": BOQ_ASK, "boq_items": BOQ_ITEMS},
        params={"target_count": 40},
    )
    assert result["status"] == "success"
    assert result["actual_count"] == len(BOQ_ITEMS)
