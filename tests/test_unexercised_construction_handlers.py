"""Dispatch the 14 previously-unsent construction actions plus extra calculators.

Chat never sent these leftover-matrix BLOCKED actions. The container handlers
must still be reachable through ``ConstructionContainer.route`` with honest
inputs (error for missing data is OK; NameError / Unknown action is not).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.conftest import requires_construction_kit

_IFC = Path(__file__).resolve().parent / "fixtures" / "sample_office.ifc"

_ACTIONS = [
    ("estimate_costs", {"quantities": {"Concrete Works": {"quantity": 24, "unit": "m3"}}}),
    ("spec_analyze", {"text": "C30 concrete per ACI 318. Cement to ASTM C150 Type I."}),
    ("process_document", {"text": "Project documents: sample_mini_boq.xlsx, drawing_qz_0100.pdf."}),
    ("procurement_list_generator", {"quantities": {"Rebar": {"quantity": 3.2, "unit": "t"}}}),
    ("rfi_management", {"text": "how many RFIs are open and which ones are overdue?"}),
    ("change_order_impact", {"text": "VO-12 adding 300m of storm drain", "cost_impact": 50000}),
    ("extract_quantities", {"measurements": [{"item": "slab", "quantity": 120, "unit": "m2"}]}),
    ("procurement_optimizer", {"quantities": {"Steel": {"quantity": 10, "unit": "t"}}}),
    ("process_specification_full", {"text": "CSI division 03 cast-in-place concrete C30 ACI 318."}),
    ("progress_tracker", {"text": "actual progress tracking against planned"}),
    ("forensic_delay_analysis", {"text": "6-week steel delivery delay EOT"}),
    ("variation_order_manager", {"text": "open VOs status and value"}),
    ("value_engineering", {"text": "basement options to cut cost without losing parking"}),
]

_EXTRA_CALCS = [
    ("wind_pressure", {"wind_speed_m_s": 50, "code": "aci"}),
    ("scaffold_load_capacity", {"platform_area_m2": 10, "duty": "medium"}),
    ("cost_per_area", {"total_cost": 1_000_000, "area": 5000}),
    ("critical_path_float", {"early_start": 5, "early_finish": 10, "late_start": 8, "late_finish": 13}),
    ("carbon_footprint_concrete", {"volume_m3": 100, "grade": "c30"}),
]


def _container():
    from app.containers.construction import ConstructionContainer
    return ConstructionContainer()


def _run(coro):
    return asyncio.run(coro)


@requires_construction_kit
@pytest.mark.parametrize("action,payload", _ACTIONS)
def test_previously_unsent_action_is_dispatchable(action, payload):
    result = _run(_container().route(action, payload, {"action": action, **payload}))
    assert isinstance(result, dict), result
    err = str(result.get("error") or "")
    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    nested_err = str(nested.get("error") or "")
    assert "Unknown action" not in err and "Unknown action" not in nested_err, result
    assert "NameError" not in err and "NameError" not in nested_err
    # Honest missing-input error is OK; the action must be wired and return a shape.
    status = result.get("status")
    if status is None and nested:
        status = nested.get("status")
    assert status in ("success", "error", "ok"), (
        f"{action}: expected status success|error|ok, got {result!r}"
    )


@requires_construction_kit
def test_bim_clash_detection_runs_on_sample_office_ifc():
    assert _IFC.is_file()
    result = _run(_container().route(
        "bim_clash_detection",
        {"ifc_file": str(_IFC)},
        {"action": "bim_clash_detection", "run_clash_detection": True},
    ))
    assert isinstance(result, dict), result
    assert result.get("status") == "success", result
    assert "No IFC file" not in (result.get("error") or "")
    # Handler returns action "clash_detection" with a clash_summary even at 0 hits.
    inner = result.get("result") if isinstance(result.get("result"), dict) else result
    assert (
        isinstance(inner.get("clash_summary"), dict)
        or isinstance(result.get("clash_summary"), dict)
        or "clashes" in inner
        or "clashes" in result
    ), result


@requires_construction_kit
@pytest.mark.parametrize("name,params", _EXTRA_CALCS)
def test_extra_named_calculator_via_construction_calc(name, params):
    result = _run(_container().route(
        "construction_calc",
        {},
        {"action": "construction_calc", "name": name, **params},
    ))
    assert isinstance(result, dict), result
    inner = result.get("result") if isinstance(result.get("result"), dict) else result
    err = str(inner.get("error") or result.get("error") or "")
    assert "Unknown calculation" not in err, result
    assert inner.get("status") == "success" or result.get("status") == "success", result
    assert name in str(result)
