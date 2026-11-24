"""Every registered calculation is exercised through the chat entry.

The chat loop reaches a calculator through ONE door: the ``construction_calc``
tool handler (``app.agents.base.tools.handle_construction_calc``) called with a
``ToolCall``. A calculation that is unit-tested only as a bare Python function
can still break at that door (parameter binding, unit conversion, the
missing-input ask, the result envelope).

Two parts:

1. ``test_every_registered_calculation_has_a_chat_entry_case`` walks
   ``formula_registry.all_specs()`` and fails for any registered calculation
   that has no case here AND no other test that drives it through the chat
   entry. A formula registered later therefore cannot slip through untested.

2. ``CASES``: per calculation
   - a typical case whose expected values are worked by hand in the comment
     (never copied from the product's output);
   - a missing-input case: one required input dropped -> status
     ``needs_input`` and the question names that input by its declared label;
   - a unit case where the formula takes a physical unit: the same input given
     as a string in another unit -> the same result plus a recorded conversion.

   ``missing=None`` is only allowed when the registry declares no required
   input; ``unit=None`` only when no input carries a physical unit. Both are
   checked against the registry, so neither can be used to skip a case.
"""
from __future__ import annotations

import asyncio
import pathlib
import re
from typing import Any, Dict, NamedTuple, Optional, Tuple

import pytest

from app.agents.base.tools import handle_construction_calc
from app.agents.core.tool_registry import ToolCall
from app.lib import formula_registry as fr

# Units that are not physical quantities: a value in them has no other unit to
# be converted from, so a unit case cannot exist for an input declared in them.
_NON_PHYSICAL_UNITS = {"", "-", "count", "ratio", "fraction", "%", "currency"}


class Case(NamedTuple):
    params: Dict[str, Any]
    expect: Dict[str, Any]
    missing: Optional[str]
    unit: Optional[Tuple[Dict[str, Any], Dict[str, Any]]]


def _dispatch(calculation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    call = ToolCall(
        agent=None,
        name="construction_calc",
        args={"calculation": calculation, "params": dict(params)},
        user_message=f"Calculate {calculation}",
        history=[],
    )
    out = asyncio.run(handle_construction_calc(call))
    assert isinstance(out, dict) and isinstance(out.get("result"), dict), out
    return out["result"]


def _value(result: Dict[str, Any], key: str) -> Any:
    body = result.get("result")
    assert isinstance(body, dict), result
    if key in body:
        return body[key]
    raise AssertionError(f"{key!r} not in result {sorted(body)}")


def _check(result: Dict[str, Any], expect: Dict[str, Any]) -> None:
    assert result.get("status") == "success", result
    for key, want in expect.items():
        got = _value(result, key)
        if isinstance(want, float):
            assert got == pytest.approx(want, abs=0.011), (key, got, want)
        else:
            assert got == want, (key, got, want)


_DEFL = {"ec_mpa": 200000, "i_mm4": 1e8}

CASES: Dict[str, Case] = {
    # Cantilever, UDL: d = wL^4/(8EI). w = 10 kN/m = 10 N/mm, L = 2000 mm,
    # E = 200000 MPa, I = 1e8 mm4 -> 10 * 2000^4 / (8 * 2e5 * 1e8)
    # = 1.6e14 / 1.6e14 = 1.00 mm.
    "beam_deflection_cantilever_udl": Case(
        {"w_kn_m": 10, "span_m": 2, **_DEFL}, {"value": 1.0}, "w_kn_m",
        ({"w_kn_m": 10, "span_m": "2000 mm", **_DEFL}, {"value": 1.0})),
    # SS, central point load: d = PL^3/(48EI). P = 10000 N, L = 4000 mm
    # -> 1e4 * 6.4e10 / (48 * 2e13) = 6.4e14 / 9.6e14 = 0.667 -> 0.67 mm.
    "beam_deflection_ss_point_load_midspan": Case(
        {"p_kn": 10, "span_m": 4, **_DEFL}, {"value": 0.67}, "p_kn",
        ({"p_kn": 10, "span_m": "4000 mm", **_DEFL}, {"value": 0.67})),
    # SS, UDL: d = 5wL^4/(384EI). w = 10 N/mm, L = 4000 mm
    # -> 5 * 10 * 2.56e14 / (384 * 2e13) = 1.28e16 / 7.68e15 = 1.667 -> 1.67 mm.
    "beam_deflection_ss_udl": Case(
        {"w_kn_m": 10, "span_m": 4, **_DEFL}, {"value": 1.67}, "w_kn_m",
        ({"w_kn_m": 10, "span_m": "4000 mm", **_DEFL}, {"value": 1.67})),
    # SS beam, UDL 10 kN/m over 6 m: M = wL^2/8 = 10*36/8 = 45 kN.m at L/2 = 3 m,
    # sagging.
    "beam_governing_moment": Case(
        {"span_m": 6, "loads": [{"kind": "udl", "w": 10}]},
        {"moment_kn_m": 45.0, "position_m": 3.0, "sense": "sagging"}, "span_m",
        ({"span_m": "6000 mm", "loads": [{"kind": "udl", "w": 10}]},
         {"moment_kn_m": 45.0, "position_m": 3.0})),
    # Fixed-fixed, UDL 12 kN/m over 6 m: support = wL^2/12 = 12*36/12 = 36;
    # midspan = wL^2/24 = 18 kN.m; max is the support moment, 36.
    "beam_moment_fixed_udl": Case(
        {"udl_w_kn_m": 12, "span_m": 6},
        {"support_moment_kn_m": 36.0, "midspan_moment_kn_m": 18.0,
         "max_moment_kn_m": 36.0}, "udl_w_kn_m",
        ({"udl_w_kn_m": 12, "span_m": "6000 mm"}, {"max_moment_kn_m": 36.0})),
    # SS, UDL 10 kN/m over 6 m: M = wL^2/8 = 360/8 = 45 kN.m.
    "beam_moment_simple": Case(
        {"udl_w_kn_m": 10, "span_m": 6}, {"max_moment_kn_m": 45.0}, "udl_w_kn_m",
        ({"udl_w_kn_m": 10, "span_m": "6000 mm"}, {"max_moment_kn_m": 45.0})),
    # SS 6 m, point load 12 kN at a = 2 m: R_right = P*a/L = 24/6 = 4 kN,
    # R_left = P - R_right = 8 kN; no end moments on a simple support.
    "beam_reactions": Case(
        {"span_m": 6, "loads": [{"kind": "point", "P": 12, "a": 2}]},
        {"reaction_left_kn": 8.0, "reaction_right_kn": 4.0,
         "moment_left_kn_m": 0.0, "moment_right_kn_m": 0.0}, "span_m",
        ({"span_m": "600 cm", "loads": [{"kind": "point", "P": 12, "a": 2}]},
         {"reaction_left_kn": 8.0, "reaction_right_kn": 4.0})),
    # Reference table, not arithmetic: the BEP convention the calculator
    # documents gives LOD 300 -> 25 mm hard-clash tolerance (LOD 200 50 mm,
    # 350 12 mm, 400 6 mm, 500 3 mm). LOD is a count: no unit case.
    "bim_clash_tolerance": Case(
        {"lod": 300}, {"lod": 300, "clash_tolerance_mm": 25.0}, None, None),
    # V = [100, 100], C = [80, 80], advance 10 %, retention 5 %, lag 15 d
    # -> ceil(15/30) = 1 month. Contract = 200; advance = 20; retention = 10.
    # Certified = [100-10-5, 100-10-5+10] = [85, 95]; cash in = [0, 85];
    # net = [-80, 5]; running from the advance: 20-80 = -60, -60+5 = -55
    # -> peak funding 60.
    "cash_flow": Case(
        {"monthly_value": [100, 100], "monthly_cost": [80, 80]},
        {"contract_value": 200.0, "advance_payment": 20.0,
         "retention_total": 10.0, "peak_funding_requirement": 60.0},
        "monthly_value",
        # A 6-week lag = 42 days -> ceil(42/30) = 2 months: cash in = [0, 0],
        # net = [-80, -80], running 20-80 = -60, -140 -> peak 140.
        ({"monthly_value": [100, 100], "monthly_cost": [80, 80],
          "payment_lag_days": "6 weeks"},
         {"peak_funding_requirement": 140.0})),
    # ACI tied, Ag = 400x400 = 160000 mm2, Ast = 2000, fc 30, fy 420:
    # P0 = 0.85*30*(160000-2000) + 420*2000 = 4029000 + 840000 = 4869000 N;
    # Pn,max = 0.80*P0 = 3895200; phi.Pn = 0.65*3895200 = 2531880 N = 2531.9 kN.
    "column_axial_capacity": Case(
        {"gross_area_mm2": 160000, "steel_area_mm2": 2000, "fc_mpa": 30,
         "fy_mpa": 420}, {"axial_capacity_kn": 2531.9}, "gross_area_mm2",
        ({"gross_area_mm2": "0.16 m2", "steel_area_mm2": 2000, "fc_mpa": 30,
          "fy_mpa": 420}, {"axial_capacity_kn": 2531.9})),
    # ACI 209 inverse: t = a*f/(1 - b*f), a = 4, b = 0.85, f = 0.6
    # -> 2.4/(1 - 0.51) = 2.4/0.49 = 4.898 -> 4.90 days. No required input;
    # its inputs are a fraction and two constants: no unit case.
    "concrete_curing_time": Case(
        {"target_strength_fraction": 0.6}, {"days_to_target": 4.9}, None, None),
    # f = P/A, A = pi/4 * 150^2 = 17671.5 mm2; P = 700 kN = 700000 N
    # -> 700000/17671.5 = 39.61 MPa.
    "concrete_cylinders": Case(
        {"failure_load_kn": 700, "cylinder_diameter_mm": 150},
        {"compressive_strength_mpa": 39.61, "cylinder_area_mm2": 17671.5},
        "failure_load_kn",
        ({"failure_load_kn": 700, "cylinder_diameter_mm": "0.15 m"},
         {"compressive_strength_mpa": 39.61})),
    # Documented slip-form mix: 1:2:2.6 at w/c 0.42, slump 150 +/- 30 mm,
    # max concrete temperature 32 C. It declares no input at all.
    "concrete_mix_slip_form": Case(
        {}, {"w_c_ratio": 0.42, "proportions": "1:2:2.6",
             "slump_mm": "150 +/- 30", "max_concrete_temp_c": 32}, None, None),
    # Core 65 C (<= 70 ok), surface 40 C: dT = 25 C > 20 -> delta fails -> risk.
    "concrete_thermal_cracking_check": Case(
        {"core_temp_c": 65, "surface_temp_c": 40},
        {"delta_t_c": 25, "core_ok": True, "delta_ok": False,
         "thermal_cracking_risk": True}, "surface_temp_c",
        # 338.15 K = 65 C; with 50 C at the surface dT = 15 -> no risk.
        ({"core_temp_c": "338.15 K", "surface_temp_c": 50},
         {"delta_t_c": 15.0, "thermal_cracking_risk": False})),
    # Per m2: shuttering 60/6 uses = 10; scaffolding 2/day * 10 days = 20;
    # material 30; labour 9 h * 10 = 90; crane 300/30 m2/h = 10; direct 130;
    # selling = 130 * 1.18 / (1 - 0.15) = 153.4/0.85 = 180.47; x 100 m2 = 18047.
    "cost_buildup_formwork": Case(
        {"area_m2": 100, "shuttering_supply_per_m2": 60,
         "scaffolding_per_m2_day": 2, "labour_rate_per_hr": 10,
         "crane_rate_per_hr": 300},
        {"material_per_m2": 30.0, "labour_per_m2": 90.0,
         "selling_price_per_m2": 180.47, "total_for_area": 18047.0},
        "labour_rate_per_hr",
        # 1,000,000 cm2 = 100 m2 -> same total.
        ({"area_m2": "1000000 cm2", "shuttering_supply_per_m2": 60,
          "scaffolding_per_m2_day": 2, "labour_rate_per_hr": 10,
          "crane_rate_per_hr": 300}, {"total_for_area": 18047.0})),
    # A(3) -> B(2), C(4) -> D(1): paths A-B-D = 6, A-C-D = 8 -> duration 8,
    # critical path A, C, D (B has 2 days of float). Durations are whole
    # working days in a table: no unit case.
    "cpm_network": Case(
        {"activities": [
            {"id": "A", "duration": 3},
            {"id": "B", "duration": 2, "predecessors": ["A"]},
            {"id": "C", "duration": 4, "predecessors": ["A"]},
            {"id": "D", "duration": 1, "predecessors": ["B", "C"]}]},
        {"project_duration": 8, "critical_path": ["A", "C", "D"]},
        "activities", None),
    # 2 cranes x 6 months; per crane per month 10000 + 2000 + 3000 = 15000;
    # hire 2*6*15000 = 180000; mobilisation 50 % of one month's dry hire per
    # crane = 2*5000 = 10000; demobilisation 75 % of that = 7500;
    # grand total 197500.
    "crane_cost_estimate": Case(
        {"num_cranes": 2, "crane_capacity_tons": 50, "duration_months": 6,
         "dry_hire_rate_per_month": 10000, "operator_rate_per_month": 2000,
         "riggers_rate_per_month": 3000},
        {"per_crane_monthly": 15000.0, "total_project_cost": 180000.0,
         "mobilization": 10000.0, "demobilization": 7500.0,
         "grand_total": 197500.0}, "num_cranes",
        ({"num_cranes": 2, "crane_capacity_tons": "50000 kg",
          "duration_months": 6, "dry_hire_rate_per_month": 10000,
          "operator_rate_per_month": 2000, "riggers_rate_per_month": 3000},
         {"crane_capacity_tons": 50.0, "grand_total": 197500.0})),
    # Net = 50 - 2 = 48 t; utilisation = 30/48 = 62.5 % (<= 85 % default) -> pass.
    "crane_lift_capacity": Case(
        {"chart_capacity_t": 50, "deductions_t": 2, "load_t": 30},
        {"net_capacity_t": 48.0, "utilization_percent": 62.5, "passed": True},
        "load_t",
        ({"chart_capacity_t": "50000 kg", "deductions_t": 2, "load_t": 30},
         {"net_capacity_t": 48.0, "utilization_percent": 62.5})),
    # 60/20 min = 3 lifts/h; per crane per day 10 t * 0.65 * 3 * 9.5 h * 2 shifts
    # = 370.5 t; per month * 26 = 9633 t; ceil(10000/9633) = 2 cranes.
    "crane_planning": Case(
        {"total_lift_demand_tons": 10000, "crane_capacity_tons": 10},
        {"lifts_per_hour_per_crane": 3.0, "daily_tonnage_per_crane": 370.5,
         "cranes_required": 2}, "total_lift_demand_tons",
        ({"total_lift_demand_tons": 10000, "crane_capacity_tons": "10000 kg"},
         {"daily_tonnage_per_crane": 370.5, "cranes_required": 2})),
    # Weights .45/.45/.07/.03. X: 36 + 31.5 + 6.3 + 1.5 = 75.3;
    # Y: 31.5 + 40.5 + 5.6 + 1.8 = 79.4 -> Y ranks first. Scores carry no unit.
    "evaluate_tender": Case(
        {"tenderers": [
            {"name": "X", "technical": 80, "commercial": 70, "hse": 90,
             "local_content": 50},
            {"name": "Y", "technical": 70, "commercial": 90, "hse": 80,
             "local_content": 60}]},
        {}, "tenderers", None),
    # F = m g (1 + H/d) = 80 * 9.81 * (1 + 1.8/1.2) = 784.8 * 2.5 = 1962 N
    # = 1.96 kN, within the 8 kN limit. No required input.
    "fall_arrest_force": Case(
        {"worker_mass_kg": 80, "free_fall_m": 1.8, "deceleration_distance_m": 1.2},
        {"max_arrest_force_kn": 1.96, "within_osha_limit": True}, None,
        # 120 cm = 1.2 m -> same force.
        ({"worker_mass_kg": 80, "free_fall_m": 1.8,
          "deceleration_distance_m": "120 cm"},
         {"max_arrest_force_kn": 1.96})),
    # Contribution per unit = 50 - 30 = 20; break-even = 12,000 / 20 = 600
    # units; revenue = 600 x 50 = 30,000. Money and counts carry no physical
    # unit, so there is no unit case.
    "break_even_point": Case(
        {"fixed_costs": 12_000, "price_per_unit": 50, "variable_cost_per_unit": 30},
        {"contribution_per_unit": 20.0, "break_even_units": 600.0,
         "break_even_revenue": 30_000.0}, "fixed_costs", None),
}


# ── registry guard ─────────────────────────────────────────────────────────
_CHAT_MARKERS = ("handle_construction_calc", '"construction_calc"', "'construction_calc'")
_THIS = pathlib.Path(__file__).resolve()


def _ids_driven_through_chat_elsewhere(ids):
    """Calculation ids that another test file names (quoted) next to the chat
    entry. Same rule the 19 Nov route audit used to find the untested ones."""
    found = set()
    for path in _THIS.parent.rglob("*.py"):
        if path.resolve() == _THIS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not any(m in text for m in _CHAT_MARKERS):
            continue
        for name in ids - found:
            if re.search(r"[\"']" + re.escape(name) + r"[\"']", text):
                found.add(name)
    return found


def test_every_registered_calculation_has_a_chat_entry_case():
    registered = {s.name for s in fr.all_specs()}
    assert registered, "the formula registry is empty"
    uncovered = registered - set(CASES)
    uncovered -= _ids_driven_through_chat_elsewhere(uncovered)
    assert not uncovered, (
        "registered calculations with no test through the chat entry "
        f"(add a Case to CASES): {sorted(uncovered)}")


def test_cases_name_registered_calculations():
    registered = {s.name for s in fr.all_specs()}
    assert set(CASES) <= registered, sorted(set(CASES) - registered)
    assert set(_UNIT_GAPS) <= {n for n, c in CASES.items() if c.unit}, _UNIT_GAPS


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_shape_matches_the_registry(name):
    params = fr.parameters(fr.get(name))
    case = CASES[name]
    required = {k for k, p in params.items() if p.required}
    if case.missing is None:
        assert not required, f"{name} declares required inputs {sorted(required)}"
    else:
        assert case.missing in required, (name, case.missing, sorted(required))
    physical = {k for k, p in params.items()
                if p.kind in ("number", "integer", "series")
                and (p.unit or "").strip() not in _NON_PHYSICAL_UNITS
                and "currency" not in (p.unit or "")}
    if case.unit is None:
        assert not physical, f"{name} takes physical units on {sorted(physical)}"
    else:
        given = {k for k, v in case.unit[0].items() if isinstance(v, str) and k in physical}
        assert given, f"{name}: the unit case gives no physical input as a unit string"


@pytest.mark.parametrize("name", sorted(CASES))
def test_typical_case_through_chat(name):
    case = CASES[name]
    result = _dispatch(name, case.params)
    _check(result, case.expect)
    if name == "evaluate_tender":
        body = result["result"]
        assert body["recommended"]["name"] == "Y", body
        totals = {t["name"]: t["weighted_total"] for t in body["ranked_tenderers"]}
        assert totals == {"Y": pytest.approx(79.4), "X": pytest.approx(75.3)}, totals


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c.missing))
def test_missing_input_asks_by_label(name):
    case = CASES[name]
    params = {k: v for k, v in case.params.items() if k != case.missing}
    result = _dispatch(name, params)
    assert result.get("status") == "needs_input", result
    assert case.missing in (result.get("missing") or []), result
    label = fr.parameters(fr.get(name))[case.missing].label
    assert label, f"{name}.{case.missing} declares no label"
    assert label.lower() in str(result.get("question") or "").lower(), result


# Unit cases that fail on main today. Each is a reported defect, not a
# skipped case: strict xfail turns RED the moment the gap is fixed, so the
# entry must then be removed. Never add an entry to make a new case pass.
_UNIT_GAPS = {
    "concrete_thermal_cracking_check": (
        "temperature in K or F is not converted to degC: the calculator asks "
        "again ('I do not recognise the unit') instead of converting"),
}


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=_UNIT_GAPS[n]))
    if n in _UNIT_GAPS else n
    for n in sorted(n for n, c in CASES.items() if c.unit)
])
def test_unit_case_converts(name):
    params, expect = CASES[name].unit
    result = _dispatch(name, params)
    _check(result, expect)
    converted = {c.get("parameter") for c in result.get("conversions") or []}
    strings = {k for k, v in params.items() if isinstance(v, str)}
    assert strings & converted, result
