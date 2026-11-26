"""Transcription check, batch 8: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what its stated form says. The
stated form is the formula's own docstring, its registered description and
its named constants or table. Every expected value below is a hand
calculation from that stated form, written as a literal; none is obtained by
running the calculator. All inputs are invented.
"""
from __future__ import annotations

import pytest

from app.agents.hats.commercial.formulas.construction_formulas_additions import (
    calculate_interim_payment,
)
from app.agents.hats.commercial.formulas.construction_knowledge import calculate_payment
from app.agents.hats.design.formulas.construction_formulas import (
    composite_column_design,
    concrete_mix_slip_form,
)
from app.agents.hats.planning.formulas.construction_formulas import (
    electrical_installation_sequence,
)
from app.agents.hats.planning.formulas.construction_formulas_planning import cpm_network
from app.agents.hats.planning.formulas.project_control import cash_flow
from app.agents.hats.procurement.formulas.construction_formulas import crane_planning
from app.agents.hats.qaqc.formulas.construction_formulas_reference_tables import (
    bim_clash_tolerance,
)
from app.agents.hats.qaqc.formulas.design_review_rules import check_review_timeline

TOL = 0.011


def _close(got, want, tol=TOL):
    assert got is not None
    assert abs(float(got) - float(want)) <= tol, (got, want)


def _refused(out):
    return bool(out.get("error")) or out.get("status") == "out_of_range"


# --- bim_clash_tolerance: the LOD table 200/300/350/400/500 -> 50/25/12/6/3 mm -
@pytest.mark.parametrize("lod,tol_mm", [(200, 50.0), (300, 25.0), (350, 12.0), (400, 6.0), (500, 3.0)])
def test_clash_tolerance_reads_the_lod_table(lod, tol_mm):
    out = bim_clash_tolerance(lod)
    assert out["lod"] == lod
    assert out["clash_tolerance_mm"] == tol_mm


def test_clash_tolerance_lod_100_is_undefined_not_zero():
    out = bim_clash_tolerance(100)
    assert out["clash_tolerance_mm"] is None
    assert "undefined" in out["note"]


@pytest.mark.parametrize("lod", [250, 600, 50])
def test_clash_tolerance_refuses_a_level_outside_the_table(lod):
    assert _refused(bim_clash_tolerance(lod))


# --- calculate_interim_payment: net = gross - gross x r% ------------------------
@pytest.mark.parametrize("gross,pct,ret,net", [
    (1_250_000, 5, 62_500.0, 1_187_500.0),
    (840_000, 10, 84_000.0, 756_000.0),
    (500_000, 0, 0.0, 500_000.0),
    (500_000, 100, 500_000.0, 0.0),
])
def test_interim_payment_takes_retention_off_gross(gross, pct, ret, net):
    out = calculate_interim_payment(gross, pct)
    _close(out["retention_amount"], ret)
    _close(out["net_payment"], net)
    _close(out["gross_valuation"], gross)


@pytest.mark.parametrize("gross,pct", [(-1, 5), (100_000, 100.5), (100_000, -0.5)])
def test_interim_payment_refuses_out_of_range(gross, pct):
    assert "error" in calculate_interim_payment(gross, pct)


@pytest.mark.parametrize("gross", [100_000, 0])
def test_interim_payment_has_no_default_retention(gross):
    out = calculate_interim_payment(gross, None)
    assert out["missing"] == ["retention_percent"]


# --- calculate_payment: retention on THIS period's certified; prior only adds ---
@pytest.mark.parametrize("claimed,cert,r,prev,cv,held,net,cum,pct,disputed", [
    (1_000_000, 900_000, 0.05, 2_000_000, 10_000_000, 45_000, 855_000, 2_900_000, 29.0, 100_000),
    (400_000, 400_000, 0.10, 0, 0, 40_000, 360_000, 400_000, None, 0),
])
def test_payment_due_is_certified_less_retention(claimed, cert, r, prev, cv, held, net, cum, pct, disputed):
    out = calculate_payment(claimed, cert, r, prev, cv)
    _close(out["retention_held"], held)
    _close(out["net_payment_due"], net)
    _close(out["cumulative_certified"], cum)
    assert out["percent_complete"] == pct
    _close(out["disputed_amount"], disputed)
    _close(out["retention_rate_pct"], r * 100)


@pytest.mark.parametrize("args", [
    (100, 100, None),          # no default rate
    (100, 100, 5),             # a percent passed where a fraction is stated
    (-1, 100, 0.05),
    (100, 100, 0.05, -5),
])
def test_payment_refuses_bad_input(args):
    assert "error" in calculate_payment(*args)


# --- cash_flow: advance at month 0, pro-rata recovery, retention released at end,
#     payment lag = ceil(days / 30) months --------------------------------------
@pytest.mark.parametrize("lag_days,peak", [
    (15, 175.0),   # lag 1: cum 40-150=-110, -110-65=-175, -175+120=-55
    (30, 175.0),   # 30/30 = exactly 1 month
    (31, 260.0),   # lag 2: cum -110, -260, -225
    (0, 25.0),     # lag 0: cum 40-65=-25, -5, 50
])
def test_cash_flow_peak_funding(lag_days, peak):
    out = cash_flow([100, 200, 100], [150, 150, 50], 0.10, 0.05, lag_days)
    _close(out["contract_value"], 400)
    _close(out["advance_payment"], 40)
    _close(out["retention_total"], 20)
    _close(out["check_advance_recovery"], -40)
    _close(out["check_retention"], -20)
    _close(out["check_valuation"], 360)   # 85 + 170 + (85 + 20 released)
    _close(out["peak_funding_requirement"], peak)


def test_cash_flow_second_series():
    # V 300, 300; C 100, 400; adv 20%, ret 10%, lag 1 month.
    # certified 210, 210+60 = 270; cash in 0, 210; net -100, -190;
    # cum 120-100 = 20, 20-190 = -170 -> peak 170.
    out = cash_flow([300, 300], [100, 400], 0.20, 0.10, 20)
    _close(out["advance_payment"], 120)
    _close(out["check_valuation"], 480)
    _close(out["peak_funding_requirement"], 170)


@pytest.mark.parametrize("value,cost", [([], []), ([100, 200], [50])])
def test_cash_flow_refuses_empty_or_mismatched_series(value, cost):
    assert "error" in cash_flow(value, cost)


# --- check_review_timeline: minimum 7 days, maximum 14 (advisory) --------------
@pytest.mark.parametrize("dist,workshop,days,ok,word", [
    ("2026-03-01", "2026-03-07", 6, False, "minimum 7"),
    ("2026-03-01", "2026-03-08", 7, True, "compliant"),
    ("2026-02-25", "2026-03-04", 7, True, "compliant"),     # across a 28-day February
    ("2026-03-01", "2026-03-15", 14, True, "compliant"),
    ("2026-03-01", "2026-03-16", 15, True, "maximum is 14"),
])
def test_review_timeline_window(dist, workshop, days, ok, word):
    out = check_review_timeline(dist, workshop)
    assert out["distribution_period_days"] == days
    assert out["compliant"] is ok
    assert word in out["message"]


@pytest.mark.parametrize("dist,workshop", [("01/03/2026", "2026-03-10"), ("2026-03-01", "next week")])
def test_review_timeline_refuses_unparsed_dates(dist, workshop):
    assert "error" in check_review_timeline(dist, workshop)


# --- composite_column_design: Npl = Aa fy + 0.85 Ac fck/1.5 + As fsk/1.15 -----
# D 600: gross = pi x 300^2 = 282,743.3 mm2, fcd = 40/1.5 = 26.667
#   A: Ac = 282,743.3 - 10,000 - 4,000 = 268,743.3
#      3,550.0 + 0.85 x 268,743.3 x 26.667 / 1000 (6,091.5) + 4,000 x 434.78 / 1000 (1,739.1) = 11,381
#   B: Ac = 278,743.3 -> 6,318.2 + 1,739.1 = 8,057
@pytest.mark.parametrize("load,rec,a_safe,b_safe", [
    (9_000, "A", True, False),
    (8_000, "A or B", True, True),
    (12_000, "NEITHER", False, False),
])
def test_composite_column_squash_resistance(load, rec, a_safe, b_safe):
    out = composite_column_design(load, 600, 40, True, 0.0, 10_000, 355, 4_000, 500)
    assert out["option_a"]["capacity_kn"] == 11_381
    assert out["option_b"]["capacity_kn"] == 8_057
    assert (out["option_a"]["is_safe"], out["option_b"]["is_safe"]) == (a_safe, b_safe)
    assert out["recommended"] == rec


def test_composite_column_without_steel_section():
    # D 400, C30: gross 125,663.7; Ac 123,663.7 x 0.85 x 20 = 2,102.3 kN; + 2,000 x 434.78 = 869.6
    out = composite_column_design(2_500, 400, 30, False, 3.0, 9_000, 355, 2_000, 500)
    assert out["option_a"]["capacity_kn"] == out["option_b"]["capacity_kn"] == 2_972
    assert out["recommended"] == "A or B"


@pytest.mark.parametrize("kw", [
    dict(column_diameter_mm=0),
    dict(concrete_grade_n_mm2=0),
    dict(steel_area_mm2=300_000),          # more than the 282,743 mm2 gross
])
def test_composite_column_refuses_degenerate_input(kw):
    base = dict(axial_load_kn=1_000, column_diameter_mm=600, concrete_grade_n_mm2=40)
    base.update(kw)
    with pytest.raises(ValueError):
        composite_column_design(**base)


# --- concrete_mix_slip_form: 1:2:2.6, W/C 0.42 through the SG mix ---------------
# C = 1 / (0.42 + 1/3.15 + 2/2.67 + 2.6/2.54) = 1/2.51014 t = 398 kg
@pytest.mark.parametrize("extra", [{}, {"slump": 200, "w_c": 0.6}])
def test_slip_form_mix_constants_are_locked(extra):
    out = concrete_mix_slip_form(**extra)
    assert (out["cement_kg_m3"], out["water_litres_m3"]) == (398, 167)
    assert (out["fine_aggregate_kg_m3"], out["coarse_aggregate_kg_m3"]) == (796, 1035)
    assert out["slump_mm"] == "150 +/- 30"
    assert (out["retarder_20c_lit_m3"], out["retarder_10c_lit_m3"]) == (3.8, 2.9)
    assert out["max_concrete_temp_c"] == 32


# --- cpm_network: forward / backward pass, TF = LS - ES ------------------------
def test_cpm_network_with_one_float_path():
    # A3; B2 after A; C4 after A; D1 after B,C -> A 0-3, B 3-5, C 3-7, D 7-8.
    # B: LF 7, LS 5, TF 2, FF 7-5 = 2.  Critical A-C-D, duration 8.
    out = cpm_network([
        {"id": "A", "duration": 3, "predecessors": []},
        {"id": "B", "duration": 2, "predecessors": ["A"]},
        {"id": "C", "duration": 4, "predecessors": "A"},
        {"id": "D", "duration": 1, "predecessors": "B, C"},
    ])
    rows = {r["id"]: r for r in out["activities"]}
    assert out["project_duration"] == 8
    assert list(out["critical_path"]) == ["A", "C", "D"]
    assert (rows["B"]["early_start"], rows["B"]["late_start"]) == (3, 5)
    assert (rows["B"]["total_float"], rows["B"]["free_float"]) == (2, 2)
    assert [rows[k]["total_float"] for k in "ACD"] == [0, 0, 0]


def test_cpm_network_total_float_differs_from_free_float():
    # A2; B5; C2 after A; D2 after B,C -> A 0-2, B 0-5, C 2-4, D 5-7.
    # C: LS 3, TF 1, FF 5-4 = 1.  A: LF 3, TF 1, FF 2-2 = 0.  Critical B-D, 7.
    out = cpm_network([
        {"id": "A", "duration": 2},
        {"id": "B", "duration": 5},
        {"id": "C", "duration": 2, "predecessors": ["A"]},
        {"id": "D", "duration": 2, "predecessors": ["B", "C"]},
    ])
    rows = {r["id"]: r for r in out["activities"]}
    assert out["project_duration"] == 7
    assert list(out["critical_path"]) == ["B", "D"]
    assert (rows["A"]["total_float"], rows["A"]["free_float"]) == (1, 0)
    assert (rows["C"]["total_float"], rows["C"]["free_float"]) == (1, 1)


@pytest.mark.parametrize("acts", [
    [],
    [{"id": "A", "duration": 2.5}],
    [{"id": "A", "duration": -1}],
    [{"duration": 3}],
    [{"id": "A", "duration": 1, "predecessors": ["B"]}, {"id": "B", "duration": 1, "predecessors": ["A"]}],
])
def test_cpm_network_refuses_bad_network(acts):
    assert "error" in cpm_network(acts)


# --- crane_planning: ceil(demand / (cap x util x 60/cycle x h x shifts x days)) -
# 10 t, 65%, 20 min, 9.5 h x 2, 26 d: 6.5 t x 3 x 19 = 370.5 t/day, 9,633 t/month
@pytest.mark.parametrize("demand,cranes", [(20_000, 3), (9_633, 1), (9_634, 2), (0, 0)])
def test_crane_count_per_month(demand, cranes):
    out = crane_planning(demand, 10, 65, 2, 9.5, 20, 26, 50_000)
    _close(out["lifts_per_hour_per_crane"], 3.0)
    _close(out["daily_tonnage_per_crane"], 370.5)
    assert out["cranes_required"] == cranes
    assert out["total_monthly_cost"] == cranes * 50_000.0


def test_crane_count_second_case_without_a_rate():
    # 8 t, 50%, 15 min, 10 h x 1, 25 d: 4 t x 4 x 10 = 160 t/day, 4,000 t/month
    out = crane_planning(5_000, 8, 50, 1, 10, 15, 25)
    _close(out["daily_tonnage_per_crane"], 160.0)
    assert out["cranes_required"] == 2
    assert out["total_monthly_cost"] is None
    assert any("dry hire rate" in n for n in out["notes"])


@pytest.mark.parametrize("kw", [
    dict(total_lift_demand_tons=-1),
    dict(crane_capacity_tons=0),
    dict(cycle_time_minutes=0),
    dict(hours_per_shift=0),
])
def test_crane_planning_refuses_degenerate_input(kw):
    base = dict(total_lift_demand_tons=1_000, crane_capacity_tons=10)
    base.update(kw)
    with pytest.raises(ValueError):
        crane_planning(**base)


# --- electrical_installation_sequence: days = ceil(area x floors / rate) per stage
@pytest.mark.parametrize("area,floors,rates,stages,total", [
    (1_200, 5, {"first_fix": 250, "second_fix": 400, "testing": 1_000},
     {"first_fix": 24, "second_fix": 15, "testing": 6}, 45),
    (850, 3, {"containment": 300, "cabling": 500, "zero_rate": 0},
     {"containment": 9, "cabling": 6}, 15),            # 2,550/300 = 8.5 -> 9; 5.1 -> 6
])
def test_electrical_stage_days(area, floors, rates, stages, total):
    out = electrical_installation_sequence(area, floors, stage_output_rates=rates)
    assert out["total_area_m2"] == area * floors
    assert out["stages"] == stages
    assert out["total_days"] == total


@pytest.mark.parametrize("rates", [{}, None])
def test_electrical_has_no_built_in_rates(rates):
    out = electrical_installation_sequence(1_000, 2, stage_output_rates=rates)
    assert out["needs_input"] == ["stage_output_rates"]


@pytest.mark.parametrize("area,floors", [(0, 2), (1_000, 0)])
def test_electrical_refuses_degenerate_area(area, floors):
    with pytest.raises(ValueError):
        electrical_installation_sequence(area, floors, stage_output_rates={"x": 100})
