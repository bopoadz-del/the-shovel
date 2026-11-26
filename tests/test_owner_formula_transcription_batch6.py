"""Transcription check, batch 6: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what the owner's stated form says.
For the strain-to-temperature conversion the stated form is the owner's
knowledge base (``concrete.equivalent_temperature_drop`` in
``app/knowledge/construction_kb.json``); for the crane check the knowledge
base states the rule (``equipment.crane_rating_check``: de-rate, then compare
with positive margin) and the docstring states the arithmetic. No KB entry
states the other eight, so their stated form is each formula's own docstring
(for the beam solver: the boundary conditions it names). Every expected value
below is a hand calculation from that stated form, written as a literal; none
is obtained by running the calculator. All inputs are invented figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.hats.design.formulas.beam_solver import (
    beam_governing_moment,
    beam_reactions,
)
from app.agents.hats.design.formulas.construction_formulas import (
    beam_deflection_cantilever_point_load,
    thermal_shrinkage_equivalence,
    unit_weight_concrete,
)
from app.agents.hats.qaqc.formulas.construction_formulas import fineness_modulus
from app.agents.hats.quantities.formulas.construction_formulas import (
    cost_buildup_concrete,
    cost_buildup_formwork,
    cost_buildup_rebar,
)
from app.agents.hats.safety.formulas.construction_formulas_safety import crane_lift_capacity
from app.lib.construction_formulas import run_calculation

TOL = 0.011
_KB = Path(__file__).resolve().parents[1] / "app" / "knowledge" / "construction_kb.json"


def _close(got, want, tol=TOL):
    assert got is not None
    assert abs(float(got) - float(want)) <= tol, (got, want)


def _kb_entry(entry_id):
    entries = json.loads(_KB.read_text(encoding="utf-8"))["entries"]
    return next(e for e in entries if e.get("id") == entry_id)


def _value(out):
    """A qaqc calculator may return its value bare or inside an envelope."""
    if isinstance(out, dict):
        for key in ("fineness_modulus", "value", "result"):
            if key in out:
                return out[key]
    return getattr(out, "fineness_modulus", out)


# --- beam_reactions: reactions from the boundary conditions -------------------
# Simply supported UDL: R = wL/2 each end. Point P at a: RL = P(L-a)/L, RR = Pa/L.
@pytest.mark.parametrize("span,loads,rl,rr", [
    (6, [{"kind": "udl", "w": 10}], 30.0, 30.0),
    (6, [{"kind": "point", "P": 60, "a": 2}], 40.0, 20.0),   # 60x4/6, 60x2/6
    (8, [{"kind": "point", "P": 24, "a": 6}], 6.0, 18.0),    # 24x2/8, 24x6/8
])
def test_beam_reactions_simply_supported(span, loads, rl, rr):
    out = beam_reactions(span, "simply_supported", loads)
    _close(out["reaction_left_kn"], rl)
    _close(out["reaction_right_kn"], rr)
    _close(out["moment_left_kn_m"], 0.0)
    _close(out["moment_right_kn_m"], 0.0)


# Fixed both ends UDL: R = wL/2, end moments -wL^2/12.
# Propped cantilever (fixed at x=0) UDL: R_fixed = 5wL/8, R_prop = 3wL/8, M_fixed = -wL^2/8.
# Cantilever (fixed at x=0) UDL: R = wL, M_fixed = -wL^2/2.
@pytest.mark.parametrize("support,span,w,rl,rr,ml,mr", [
    ("fixed_fixed", 5, 12, 30.0, 30.0, -25.0, -25.0),
    ("fixed_fixed", 6, 10, 30.0, 30.0, -30.0, -30.0),
    ("propped_cantilever", 4, 8, 20.0, 12.0, -16.0, 0.0),
    ("propped_cantilever", 8, 5, 25.0, 15.0, -40.0, 0.0),
    ("cantilever", 4, 5, 20.0, 0.0, -40.0, 0.0),
    ("cantilever", 3, 6, 18.0, 0.0, -27.0, 0.0),
])
def test_beam_reactions_indeterminate_and_cantilever(support, span, w, rl, rr, ml, mr):
    out = beam_reactions(span, support, [{"kind": "udl", "w": w}])
    _close(out["reaction_left_kn"], rl)
    _close(out["reaction_right_kn"], rr)
    _close(out["moment_left_kn_m"], ml)
    _close(out["moment_right_kn_m"], mr)


@pytest.mark.parametrize("span,support", [(0, "simply_supported"), (6, "overhang"), (6, "hinged_on_air")])
def test_beam_reactions_refuses_what_it_cannot_solve(span, support):
    assert "error" in beam_reactions(span, support, [{"kind": "udl", "w": 10}])


# --- beam_governing_moment ------------------------------------------------------
@pytest.mark.parametrize("span,support,loads,m,pos,sense", [
    (6, "simply_supported", [{"kind": "udl", "w": 10}], 45.0, 3.0, "sagging"),     # wL^2/8
    (6, "simply_supported", [{"kind": "point", "P": 60, "a": 2}], 80.0, 2.0, "sagging"),  # Pab/L
    (5, "fixed_fixed", [{"kind": "udl", "w": 12}], -25.0, None, "hogging"),       # wL^2/12 > wL^2/24
    (4, "cantilever", [{"kind": "udl", "w": 5}], -40.0, 0.0, "hogging"),          # wL^2/2 at the root
])
def test_beam_governing_moment_is_the_largest_magnitude(span, support, loads, m, pos, sense):
    out = beam_governing_moment(span, support, loads)
    _close(out["moment_kn_m"], m)
    assert out["sense"] == sense
    if pos is not None:
        _close(out["position_m"], pos)
    else:
        assert float(out["position_m"]) in (0.0, float(span))


@pytest.mark.parametrize("span", [0, -3])
def test_beam_governing_moment_refuses_a_span_not_above_zero(span):
    assert "error" in beam_governing_moment(span, "simply_supported", [{"kind": "udl", "w": 10}])


# --- beam_deflection_cantilever_point_load: delta = PL^3 / (3EI) ----------------
@pytest.mark.parametrize("p,l,e,i,d", [
    (30, 4, 200000, 1e8, 32.0),     # 30e3 x 4000^3 / (3 x 2e5 x 1e8)
    (10, 2, 30000, 5e8, 1.78),      # 1e4 x 2000^3 / (3 x 3e4 x 5e8) = 1.777
])
def test_cantilever_point_deflection_is_pl3_over_3ei(p, l, e, i, d):
    _close(beam_deflection_cantilever_point_load(p, l, e, i), d)


@pytest.mark.parametrize("p,l,e,i", [(-1, 4, 200000, 1e8), (30, 4, 0, 1e8), (30, 4, 200000, 0)])
def test_cantilever_point_deflection_refuses_bad_input(p, l, e, i):
    with pytest.raises(ValueError):
        beam_deflection_cantilever_point_load(p, l, e, i)


# --- unit_weight_concrete: RC 2500, plain 2400 (range 2330-2470) ---------------
@pytest.mark.parametrize("reinforced,uw,has_range", [(True, 2500, False), (False, 2400, True)])
def test_unit_weight_concrete_is_the_stated_value(reinforced, uw, has_range):
    out = unit_weight_concrete(reinforced)
    assert out["unit_weight_kg_m3"] == uw
    assert ("range_kg_m3" in out) is has_range
    if has_range:
        assert tuple(out["range_kg_m3"]) == (2330, 2470)


# --- fineness_modulus: FM = sum of cumulative % retained / 100 ------------------
@pytest.mark.parametrize("sieves,fm", [
    ([0, 5, 20, 45, 75, 95], 2.40),
    ([2, 10, 30, 60, 85, 97], 2.84),
])
def test_fineness_modulus_is_cumulative_sum_over_100(sieves, fm):
    _close(_value(fineness_modulus(sieves)), fm)


@pytest.mark.parametrize("sieves", [[0, 5, 101], [-1, 20, 40]])
def test_fineness_modulus_refuses_a_percentage_outside_0_100(sieves):
    out = fineness_modulus(sieves)
    assert isinstance(out, dict) and "error" in out


@pytest.mark.parametrize("sieves,fm", [([0, 40, 100], 1.40), ([10, 100, 100], 2.10)])
def test_fineness_modulus_accepts_exactly_100_percent(sieves, fm):
    _close(_value(fineness_modulus(sieves)), fm)


@pytest.mark.parametrize("sieves", [[0, 40, 100.5], [100.01]])
def test_fineness_modulus_refuses_just_above_100_percent(sieves):
    # Two layers refuse it: the declared 0-100 % range, and the body's own check.
    for calc in (fineness_modulus, fineness_modulus.__wrapped__):
        out = calc(sieves)
        assert isinstance(out, dict) and "error" in out


# --- crane_lift_capacity: net = chart - deductions; util = load / net ----------
@pytest.mark.parametrize("chart,ded,load,limit,net,util,passed", [
    (100, 5, 70, 0.75, 95.0, 73.68, True),    # 70/95
    (100, 5, 80, 0.75, 95.0, 84.21, False),   # 80/95 above 75 %
    (50, 2, 36, 0.75, 48.0, 75.00, True),     # exactly at the limit
    (50, 2, 30, 0.60, 48.0, 62.50, False),
])
def test_crane_net_capacity_and_utilisation(chart, ded, load, limit, net, util, passed):
    out = crane_lift_capacity(chart, ded, load, limit)
    _close(out["net_capacity_t"], net)
    _close(out["utilization_percent"], util)
    assert out["passed"] is passed
    assert "max_utilization_is_default" not in out


@pytest.mark.parametrize("chart,ded,load", [(100, 5, 96), (40, 10, 31)])
def test_crane_load_above_net_capacity_has_no_positive_margin(chart, ded, load):
    # equipment.crane_rating_check: reject any configuration without positive margin.
    out = crane_lift_capacity(chart, ded, load, 1.0)
    assert out["passed"] is False
    assert "exceeds the net capacity" in out["reason"]


@pytest.mark.parametrize("chart,ded,load", [(50, 2, 48), (100, 5, 95)])
def test_crane_load_equal_to_net_is_judged_on_utilisation_not_as_an_overload(chart, ded, load):
    # load == net: not above net capacity, so the overload branch must not fire;
    # 100 % utilisation then fails the 75 % limit on its own.
    out = crane_lift_capacity(chart, ded, load, 0.75)
    _close(out["utilization_percent"], 100.0)
    assert out["passed"] is False
    assert "reason" not in out


@pytest.mark.parametrize("load,passed", [(80, True), (81, False)])
def test_crane_default_limit_is_labelled_as_a_default(load, passed):
    out = crane_lift_capacity(100, 5, load)   # 80/95 = 84.2 %, 81/95 = 85.3 %
    assert out["passed"] is passed
    assert out["max_utilization_is_default"] is True


@pytest.mark.parametrize("chart,ded,load", [(0, 0, 10), (10, 10, 5), (10, -1, 5), (10, 1, -5)])
def test_crane_refuses_degenerate_input(chart, ded, load):
    assert "error" in crane_lift_capacity(chart, ded, load, 0.75)


# --- thermal_shrinkage_equivalence: dT = strain / alpha (owner KB) --------------
@pytest.mark.parametrize("field,want", [
    ("expression", "strain/thermal_coefficient"),
    ("variables", {"strain", "thermal_coefficient"}),
])
def test_owner_kb_states_strain_over_alpha(field, want):
    got = _kb_entry("concrete.equivalent_temperature_drop")[field]
    assert (set(got) if isinstance(want, set) else got) == want


@pytest.mark.parametrize("strain,alpha,dt", [
    (0.0002, 10e-6, 20.0),
    (0.00036, 7.2e-6, 50.0),
    (0.000108, 10.8e-6, 10.0),
])
def test_equivalent_temperature_drop_is_strain_over_alpha(strain, alpha, dt):
    _close(thermal_shrinkage_equivalence(strain, alpha)["equivalent_temp_drop_c"], dt)


# --- cost_buildup_rebar: per tonne, material wasted, then x(1+ind)/(1-markup) ---
@pytest.mark.parametrize("kw,mat,lab,eq,sell,total", [
    (dict(quantity_kg=2000, material_price_per_t=3000, labour_mhr_t=90, labour_rate_per_hr=20,
          crane_hr_t=2, crane_rate_per_hr=200, waste_pct=0.10, indirect_pct=0.18, markup_pct=0.15),
     3300, 1800, 400, 7635, 15270),       # 5500 x 1.18 / 0.85 = 7635.29
    (dict(quantity_kg=500, material_price_per_t=2500, labour_mhr_t=60, labour_rate_per_hr=15,
          crane_hr_t=1, crane_rate_per_hr=300, waste_pct=0.05, indirect_pct=0.18, markup_pct=0.15),
     2625, 900, 300, 5310, 2655),         # 3825 x 1.18 / 0.85 = 5310.0
])
def test_cost_buildup_rebar(kw, mat, lab, eq, sell, total):
    out = cost_buildup_rebar(**kw)
    assert (out["material_per_t"], out["labour_per_t"], out["equipment_per_t"]) == (mat, lab, eq)
    assert out["selling_price_per_t"] == sell
    assert out["total_project_value"] == total


@pytest.mark.parametrize("qty", [0, -100])
def test_cost_buildup_rebar_refuses_no_quantity(qty):
    with pytest.raises(ValueError):
        cost_buildup_rebar(qty, material_price_per_t=3000, labour_rate_per_hr=20, crane_rate_per_hr=200)


# --- cost_buildup_formwork: supply / 6 uses + scaffold x cycle + labour + crane/output
@pytest.mark.parametrize("kw,shut,scaf,mat,lab,sell,total", [
    (dict(area_m2=100, shuttering_supply_per_m2=120, scaffolding_per_m2_day=2, cycle_days=10,
          labour_mhr_m2=9, labour_rate_per_hr=20, crane_rate_per_hr=300, crane_output_m2_hr=30),
     20.0, 20.0, 40.0, 180.0, 319.29, 31929),     # direct 230 x 1.18 / 0.85
    (dict(area_m2=50, shuttering_supply_per_m2=60, scaffolding_per_m2_day=1.5, cycle_days=8,
          labour_mhr_m2=5, labour_rate_per_hr=25, crane_rate_per_hr=200, crane_output_m2_hr=40),
     10.0, 12.0, 22.0, 125.0, 211.01, 10551),     # direct 152 x 1.18 / 0.85 = 211.012
])
def test_cost_buildup_formwork(kw, shut, scaf, mat, lab, sell, total):
    out = cost_buildup_formwork(**kw)
    _close(out["shuttering_per_m2"], shut)
    _close(out["scaffolding_per_m2"], scaf)
    _close(out["material_per_m2"], mat)
    _close(out["labour_per_m2"], lab)
    _close(out["selling_price_per_m2"], sell)
    assert out["total_for_area"] == total


@pytest.mark.parametrize("area,output", [(-1, 30), (100, 0)])
def test_cost_buildup_formwork_refuses_degenerate_input(area, output):
    with pytest.raises(ValueError):
        cost_buildup_formwork(area, shuttering_supply_per_m2=120, scaffolding_per_m2_day=2,
                              labour_rate_per_hr=20, crane_rate_per_hr=300, crane_output_m2_hr=output)


# --- cost_buildup_concrete: aggregate 1839 kg/m3, water 0.16 m3; waste on material only
_CONCRETE_A = dict(quantity_m3=10, cement_kg_m3=400, cement_price_per_t=300, aggregate_price_per_t=50,
                   water_price_per_m3=5, microsilica_kg_m3=12, microsilica_price_per_kg=2,
                   plasticizer_lit_m3=5, plasticizer_price_per_lit=4, plant_cost_per_m3=10,
                   labour_cost_per_m3=30, erection_per_m3=5, power_per_m3=2, indirect_per_m3=8,
                   indirect_pct=0.18, markup_pct=0.15, waste_pct=0.03)
_CONCRETE_B = dict(quantity_m3=4, cement_kg_m3=350, cement_price_per_t=400, aggregate_price_per_t=40,
                   water_price_per_m3=10, microsilica_kg_m3=0, microsilica_price_per_kg=9,
                   plasticizer_lit_m3=4, plasticizer_price_per_lit=5, plant_cost_per_m3=0,
                   labour_cost_per_m3=20, erection_per_m3=0, power_per_m3=0, indirect_per_m3=0,
                   indirect_pct=0.10, markup_pct=0.10, waste_pct=0.05)


@pytest.mark.parametrize("kw,mat,direct,sell,total", [
    # 120 + 91.95 + 0.80 + 24 + 20 = 256.75; x1.03 = 264.4525; +55 = 319.45; x1.18/0.85 = 443.48
    (_CONCRETE_A, 256.75, 319.45, 443.48, 4435),
    # 140 + 73.56 + 1.60 + 0 + 20 = 235.16; x1.05 = 246.918; +20 = 266.92; x1.1/0.9 = 326.23
    (_CONCRETE_B, 235.16, 266.92, 326.23, 1305),
])
def test_cost_buildup_concrete(kw, mat, direct, sell, total):
    out = cost_buildup_concrete(**kw)
    _close(out["material_cost_per_m3"], mat)
    _close(out["direct_cost_per_m3"], direct)
    _close(out["selling_price_per_m3"], sell)
    assert out["total_project_value"] == total


@pytest.mark.parametrize("qty", [-1, -10])
def test_cost_buildup_concrete_refuses_a_negative_quantity(qty):
    with pytest.raises(ValueError):
        cost_buildup_concrete(**{**_CONCRETE_A, "quantity_m3": qty})


# --- the public route returns the same hand value --------------------------------
@pytest.mark.parametrize("name,params,key,want", [
    ("thermal_shrinkage_equivalence", {"shrinkage_strain": 0.0002, "alpha_c": 10e-6},
     "equivalent_temp_drop_c", 20.0),
    ("crane_lift_capacity", {"chart_capacity_t": 100, "deductions_t": 5, "load_t": 70,
                             "max_utilization": 0.75}, "utilization_percent", 73.68),
    ("beam_deflection_cantilever_point_load", {"p_kn": 30, "span_m": 4, "ec_mpa": 200000,
                                               "i_mm4": 1e8}, "value", 32.0),
    ("fineness_modulus", {"sieve_retained_percentages": [2, 10, 30, 60, 85, 97]}, "value", 2.84),
])
def test_public_route_returns_the_hand_value(name, params, key, want):
    env = run_calculation(name, params)
    assert env["status"] == "success", env
    res = env["result"]
    _close(res[key], want)


@pytest.mark.parametrize("sieves", [[], None])
def test_public_route_asks_for_sieve_figures_it_was_not_given(sieves):
    env = run_calculation("fineness_modulus", {"sieve_retained_percentages": sieves})
    assert env["status"] != "success"
    assert "result" not in env
