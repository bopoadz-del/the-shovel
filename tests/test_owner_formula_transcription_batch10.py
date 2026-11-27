"""Transcription check, batch 10: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them.
They check that each calculator's code does what its stated form says. The
stated form is the formula's own docstring, its registered description, and
its named constants, table or procedure record. Every expected value below is
a hand calculation from that stated form, written as a literal; none is
obtained by running the calculator. All inputs are invented.

One DIFFERENCE between the stated form and the code is recorded here and NOT
fixed (the owner rules on it):

* material_consumption -- the docstring says to pass ``waste_factor`` as the
  multiplier ("e.g. 1.05"). The declared input range is 0 to 1, so on the
  public calculation route the multiplier 1.05 is refused. A value below 1 is
  read as a fraction (0.05 -> x1.05), and a waste factor of 0 gives a material
  quantity of 0. A direct call still reads 0.05 as the multiplier itself.

The tests below pin the code as it is, so a change is seen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.base.formulas.construction_formulas_general import tolerance_check
from app.agents.base.formulas.construction_formulas_planning import pe_unit_convert
from app.agents.hats.contracts.formulas.construction_knowledge import score_risk
from app.agents.hats.contracts.formulas.document_rules import next_ncr_status
from app.agents.hats.design.formulas.construction_formulas import (
    precast_beam_erection_check,
)
from app.agents.hats.planning.formulas.construction_formulas import (
    plumbing_flow_programme,
    supervision_ratio,
)
from app.agents.hats.planning.formulas.productivity_benchmarks import (
    productivity_benchmark,
)
from app.agents.hats.qaqc.formulas.design_review_rules import validate_design_status
from app.agents.hats.quantities.formulas.construction_formulas import (
    mobilization_cost_estimate,
)
from app.agents.hats.quantities.formulas.construction_formulas_planning import (
    material_consumption,
)
from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    resource_line_cost,
)
from app.lib.construction_formulas import run_calculation

TOL = 0.011
_PROCEDURES = Path(__file__).resolve().parents[1] / "app" / "data" / "procedures" / "procedures_db.json"


def _close(got, want, tol=TOL):
    assert got is not None
    assert abs(float(got) - float(want)) <= tol, (got, want)


def _fields(out):
    """A qaqc calculator may return a dataclass, its fields, or an envelope."""
    if isinstance(out, dict):
        return out.get("result", out) if "status" in out and "result" in out else out
    return out


def _get(out, key):
    out = _fields(out)
    return out[key] if isinstance(out, dict) else getattr(out, key)


def _procedure(kind):
    return json.loads(_PROCEDURES.read_text(encoding="utf-8"))["procedures"][kind]


# --- material_consumption: Q / output per unit (x waste factor) --------------
@pytest.mark.parametrize("q,out_u,kw,base,want", [
    (90, 4, {}, 22.5, 22.5),                               # no waste given
    (120, 8, {"waste_percent": 5}, 15.0, 15.75),           # 1 + 5/100 = 1.05
    (300, 12, {"waste_percent": 10}, 25.0, 27.5),          # 1 + 10/100 = 1.10
    (120, 8, {"waste_factor": 1.05}, 15.0, 15.75),         # direct call: the multiplier
])
def test_material_consumption_direct(q, out_u, kw, base, want):
    out = material_consumption(q, out_u, **kw)
    _close(out["base_without_waste"], base)
    _close(out["material_required"], want)


@pytest.mark.parametrize("kw", [
    {"quantity_of_work": 100},
    {"output_per_unit": 5},
])
def test_material_consumption_refuses_a_missing_input(kw):
    assert "error" in material_consumption(**kw)


@pytest.mark.parametrize("out_u", [0, -2])
def test_material_consumption_refuses_a_non_positive_output(out_u):
    assert "error" in material_consumption(50, out_u)


# DIFFERENCE (recorded, not fixed): on the public route waste_factor is
# declared 0..1. The docstring's multiplier 1.05 is refused; a value below 1 is
# read as a fraction; 0 gives zero material.
@pytest.mark.parametrize("factor", [1.05, 1.1])
def test_material_consumption_difference_public_route_refuses_the_multiplier(factor):
    out = run_calculation("material_consumption",
                          {"quantity_of_work": 120, "output_per_unit": 8, "waste_factor": factor})
    assert out["status"] == "error"
    assert out["rejected"][0]["parameter"] == "waste_factor"


@pytest.mark.parametrize("factor,want", [(0.05, 15.75), (0.1, 16.5)])
def test_material_consumption_difference_public_route_reads_a_fraction(factor, want):
    out = run_calculation("material_consumption",
                          {"quantity_of_work": 120, "output_per_unit": 8, "waste_factor": factor})
    assert out["status"] == "success"
    _close(out["result"]["material_required"], want)
    # the same value through a direct call is the multiplier itself
    _close(material_consumption(120, 8, waste_factor=factor)["material_required"], 15.0 * factor)


@pytest.mark.parametrize("q,out_u", [(120, 8), (60, 3)])
def test_material_consumption_difference_zero_waste_factor_gives_zero(q, out_u):
    out = run_calculation("material_consumption",
                          {"quantity_of_work": q, "output_per_unit": out_u, "waste_factor": 0})
    assert out["status"] == "success"
    assert out["result"]["material_required"] == 0.0
    assert out["result"]["base_without_waste"] == q / out_u


# --- mobilization_cost_estimate: the project's own rates ---------------------
_MOB_FULL = {
    "office_rate_per_person_month": 100, "camp_rate_per_person_month": 300,
    "transport_rate_per_person_month": 50, "safety_medical_rate_per_person_month": 20,
    "it_rate_per_person_month": 10, "office_setup_cost": 20000, "camp_setup_cost": 80000,
    "safety_setup_cost": 5000, "warehouse_equipment_cost": 15000, "demobilization_cost": 10000,
}


_MOB_OFFICES_ONLY = {
    "include_camp": False, "include_transport": False, "include_safety_medical": False,
    "office_rate_per_person_month": 150, "it_rate_per_person_month": 25,
    "office_setup_cost": 10000, "warehouse_equipment_cost": 4000, "demobilization_cost": 3000,
}


@pytest.mark.parametrize("n,months,kw,monthly,recurring,fixed,total,offices", [
    # (100+300+50+20) x 50 x 1.2 = 28,200/month ; x12 = 338,400
    # fixed (20,000+80,000+10,000+5,000 + IT 50x10x12=6,000 + 15,000) x 1.2 = 163,200
    # offices line (20,000 + 100x50x12) x 1.2 = 96,000
    (50, 12, {**_MOB_FULL, "remote_area_factor": 1.2}, 28200, 338400, 163200, 501600, 96000),
    # 150 x 20 = 3,000/month ; x6 = 18,000 ; fixed 10,000+3,000+20x25x6=3,000+4,000 = 20,000
    # offices line 10,000 + 150x20x6 = 28,000
    (20, 6, _MOB_OFFICES_ONLY, 3000, 18000, 20000, 38000, 28000),
])
def test_mobilization_totals(n, months, kw, monthly, recurring, fixed, total, offices):
    out = mobilization_cost_estimate(n, months, **kw)
    assert out["recurring_monthly"] == monthly
    assert out["total_recurring"] == recurring
    assert out["total_fixed"] == fixed
    assert out["grand_total"] == total
    assert out["breakdown"]["offices"] == offices
    assert sum(out["breakdown"].values()) == out["grand_total"]


@pytest.mark.parametrize("drop", ["camp_rate_per_person_month", "demobilization_cost"])
def test_mobilization_refuses_a_missing_project_rate(drop):
    rates = {k: v for k, v in _MOB_FULL.items() if k != drop}
    out = mobilization_cost_estimate(10, 3, **rates)
    assert out["error"] == "project rates not given"
    assert drop in out["needs_input"]


def test_mobilization_excluded_item_needs_no_rate():
    rates = {k: v for k, v in _MOB_FULL.items() if not k.startswith("camp")}
    out = mobilization_cost_estimate(10, 3, include_camp=False, **rates)
    assert "error" not in out


# --- next_ncr_status: the procedure's status sequence ------------------------
@pytest.mark.parametrize("pos,status", [(0, "RAISED"), (4, "APPROVED"), (7, "CLOSED")])
def test_ncr_sequence_is_the_procedure_record(pos, status):
    from app.core.construction_knowledge import NCR_WORKFLOW_SEQUENCE
    record = _procedure("non_conformance")["statuses"]
    assert NCR_WORKFLOW_SEQUENCE == record
    assert NCR_WORKFLOW_SEQUENCE[pos] == status


@pytest.mark.parametrize("given,want", [
    ("RAISED", "ACKNOWLEDGED"),
    ("disposition proposed", "DISPOSITION_REVIEWED"),
    ("approved", "IMPLEMENTING"),
    ("Verifying", "CLOSED"),
])
def test_ncr_next_status(given, want):
    out = next_ncr_status(given)
    assert out["next_status"] == want
    assert out["workflow_complete"] is False


def test_ncr_closed_is_the_end():
    out = next_ncr_status("closed")
    assert out["next_status"] is None
    assert out["workflow_complete"] is True


@pytest.mark.parametrize("given", ["open", ""])
def test_ncr_refuses_a_status_outside_the_workflow(given):
    assert "error" in next_ncr_status(given)


# --- pe_unit_convert: through the SI base ------------------------------------
@pytest.mark.parametrize("value,src,dst,want,tol", [
    (10, "ft", "m", 3.048, 1e-9),                 # 0.3048 m/ft
    (100, "psi", "kPa", 689.4757, 1e-3),          # 6.894757 kPa/psi
    (100, "degF", "degC", 37.7778, 1e-3),         # (100-32) x 5/9
    (2, "m3", "l", 2000.0, 1e-9),
])
def test_unit_convert(value, src, dst, want, tol):
    out = pe_unit_convert(value, src, dst)
    _close(out["value_out"], want, tol)


@pytest.mark.parametrize("crew,hours,want", [(4, 8, 2.5), (2, 10, 4.0)])
def test_unit_convert_labour_effort_to_days(crew, hours, want):
    # 80 man-hours / (crew x hours per day)
    out = pe_unit_convert(80, "man-hours", "days", crew_size=crew, hours_per_day=hours)
    _close(out["value_out"], want)


def test_unit_convert_asks_for_missing_hours_per_day():
    out = pe_unit_convert(80, "man-hours", "days", crew_size=4)
    assert "hours_per_day" in str(out.get("needs_input", "")) + str(out.get("missing", ""))


@pytest.mark.parametrize("src,dst", [("m", "kg"), ("", "m")])
def test_unit_convert_refuses_an_unsupported_or_missing_unit(src, dst):
    assert "error" in pe_unit_convert(5, src, dst)


# --- plumbing_flow_programme: declared stage outputs -------------------------
# Stage outputs 40 / 60 / 80 / 50 m2 per day; fixed 7, 21, 7 and 3 days.
@pytest.mark.parametrize("area,floors,first,second,final,testing,total", [
    (1200, 5, 150, 100, 75, 120, 483),    # 6,000 m2: 7+21+150+100+75+7+120+3
    (30, 1, 1, 1, 1, 1, 42),              # each area stage at least 1 day
    (2000, 1, 50, 34, 25, 40, 187),       # ceil(2000/60)=34
])
def test_plumbing_programme(area, floors, first, second, final, testing, total):
    out = plumbing_flow_programme(area, floors)
    st = out["stages"]
    assert st["1st_fix_marking_piping"]["days"] == first
    assert st["2nd_fix_sanitary_fittings"]["days"] == second
    assert st["final_fix_pumps_tanks"]["days"] == final
    assert st["testing_commissioning"]["days"] == testing
    assert st["material_submittal"]["days"] == 7
    assert st["procurement"]["days"] == 21
    assert out["total_days"] == total
    assert st["handover"]["cumulative_days"] == total


# --- precast_beam_erection_check: (weight + rigging) x dynamic / allowable ----
@pytest.mark.parametrize("w,cap,kw,eff,util,safe", [
    (12, 20, {}, 15.62, 78.1, True),                                   # (12+0.5)x1.25 = 15.625
    (15.5, 20, {}, 20.0, 100.0, True),                                 # exactly at the allowable
    (16, 20, {}, 20.62, 103.1, False),                                 # (16+0.5)x1.25 = 20.625
    (8, 10, {"rigging_weight_t": 1.0, "dynamic_factor": 1.1}, 9.9, 99.0, True),
])
def test_precast_erection_demand(w, cap, kw, eff, util, safe):
    out = precast_beam_erection_check(w, 12, cap, 18, **kw)
    _close(out["effective_lift_weight_t"], eff)
    _close(out["utilization_pct"], util, 0.051)
    assert out["is_safe"] is safe


@pytest.mark.parametrize("radius,length", [(6, 8), (30, 24)])
def test_precast_radius_and_length_do_not_enter_the_sum(radius, length):
    # the allowable pick weight is the one stated for the radius; the radius
    # and member length are labels in this calculation
    out = precast_beam_erection_check(12, length, 20, radius)
    assert out["utilization_pct"] == 78.1


# --- productivity_benchmark: the owner's planner table -----------------------
@pytest.mark.parametrize("item,rate,unit", [
    ("Blockwork (walls)", 12.0, "m2"),
    ("internal wall plaster", 20.0, "m2"),
    ("Ceramic tiles - floors", 40.0, "m2"),
])
def test_benchmark_reads_the_table(item, rate, unit):
    out = productivity_benchmark(item)
    assert out["rate_per_crew_day"] == rate
    assert out["unit"] == unit
    assert "unverified" in out["provenance"]


@pytest.mark.parametrize("item", ["glazing", ""])
def test_benchmark_has_no_rate_for_an_unlisted_item(item):
    out = productivity_benchmark(item)
    assert out["rate_per_crew_day"] is None
    assert "project's own rate" in out["flag"]


# --- resource_line_cost: Q / output x day rate, plant share, material --------
@pytest.mark.parametrize("q,out_d,rate,mat,plant,days,lab,pl,total", [
    (500, 20, 400, 30, 0.1, 25.0, 10000.0, 1000.0, 26000.0),
    (240, 16, 350, 12.5, 0.25, 15.0, 5250.0, 1312.5, 9562.5),
])
def test_resource_line_with_material(q, out_d, rate, mat, plant, days, lab, pl, total):
    out = resource_line_cost(q, out_d, rate, mat, plant)
    _close(out["crew_days"], days)
    _close(out["labour_cost"], lab)
    _close(out["plant_cost"], pl)
    _close(out["total_cost"], total)
    assert out["total_is_partial"] is False


@pytest.mark.parametrize("q,out_d,rate,total", [
    (240, 16, 350, 5250.0),     # 240/16 x 350
    (100, 5, 200, 4000.0),      # 100/5 x 200
])
def test_resource_line_without_material_is_partial(q, out_d, rate, total):
    out = resource_line_cost(q, out_d, rate)
    assert out["material_cost"] == "NO SOURCED RATE"
    assert out["total_is_partial"] is True
    _close(out["total_cost"], total)


@pytest.mark.parametrize("out_d,rate", [(0, 400), (20, 0)])
def test_resource_line_refuses_a_missing_output_or_rate(out_d, rate):
    assert resource_line_cost(100, out_d, rate)["status"] == "error"


# --- score_risk: P x I, bands from the risk procedure ------------------------
@pytest.mark.parametrize("band,low,high,low_pi,high_pi", [
    ("green", 1, 4, (1, 1), (2, 2)),
    ("amber", 5, 9, (1, 5), (3, 3)),
    ("red", 10, 25, (2, 5), (5, 5)),
])
def test_risk_bands_are_the_procedure_record(band, low, high, low_pi, high_pi):
    th = _procedure("risk_management")["scoring"]["thresholds"]
    assert th[band] == [low, high]
    # the code gives this band at both ends of the recorded range
    for (p, i), score in ((low_pi, low), (high_pi, high)):
        out = score_risk(p, i)
        assert out["score"] == score
        assert out["band"] == band.upper()


@pytest.mark.parametrize("p,i,score,band,action", [
    (2, 2, 4, "GREEN", False),     # top of green
    (1, 5, 5, "AMBER", True),      # bottom of amber
    (3, 3, 9, "AMBER", True),      # top of amber
    (2, 5, 10, "RED", True),       # bottom of red
    (5, 5, 25, "RED", True),
])
def test_risk_score_and_band(p, i, score, band, action):
    out = score_risk(p, i)
    assert out["score"] == score
    assert out["band"] == band
    assert out["requires_action"] is action


@pytest.mark.parametrize("p,i", [(0, 3), (3, 6)])
def test_risk_refuses_a_rating_outside_1_to_5(p, i):
    assert "error" in score_risk(p, i)


# --- supervision_ratio: ceil(quantity / project ratio) -----------------------
@pytest.mark.parametrize("kw,want", [
    # ceil(12000/2500)=5 ; 30000/10000=3 ; total 8 ; HSE ceil(8/4)=2 ; DC ceil(8/6)=2 ; 12
    ({"concrete_m3": 12000, "area_m2": 30000, "concrete_m3_per_supervisor": 2500,
      "area_m2_per_supervisor": 10000, "supervisors_per_hse_officer": 4,
      "staff_per_document_controller": 6},
     {"civil_supervisors": 5, "general_supervisors": 3, "total_supervisors": 8,
      "hse_officers": 2, "document_controllers": 2, "total_supervision_staff": 12}),
    # ceil(900/400)=3 ; HSE ceil(3/7)=1 ; DC ceil(3/2)=2 ; 6
    ({"structural_steel_t": 900, "steel_t_per_supervisor": 400,
      "supervisors_per_hse_officer": 7, "staff_per_document_controller": 2},
     {"structural_supervisors": 3, "civil_supervisors": 0, "total_supervisors": 3,
      "hse_officers": 1, "document_controllers": 2, "total_supervision_staff": 6}),
])
def test_supervision_staff(kw, want):
    out = supervision_ratio(**kw)
    for key, value in want.items():
        assert out[key] == value, key


@pytest.mark.parametrize("kw,missing", [
    ({"structural_steel_t": 900, "steel_t_per_supervisor": 400}, "supervisors_per_hse_officer"),
    ({"concrete_m3": 500, "supervisors_per_hse_officer": 4, "staff_per_document_controller": 6},
     "concrete_m3_per_supervisor"),
])
def test_supervision_refuses_a_missing_ratio(kw, missing):
    out = supervision_ratio(**kw)
    assert missing in out["needs_input"]


@pytest.mark.parametrize("kw", [{}, {"concrete_m3": 0, "area_m2": 0}])
def test_supervision_nothing_given_is_zero(kw):
    out = supervision_ratio(**kw)
    assert out["total_supervision_staff"] == 0


# --- tolerance_check: specified - minus <= measured <= specified + plus ------
@pytest.mark.parametrize("m,s,plus,minus,within,dev,margin", [
    (1003, 1000, 5, None, True, 3.0, 2.0),        # minus defaults to plus
    (994, 1000, 5, 3, False, -6.0, -3.0),         # below 997
    (1005, 1000, 5, 3, True, 5.0, 0.0),           # exactly on the upper limit
    (997, 1000, 5, 3, True, -3.0, 0.0),           # exactly on the lower limit
    (996.9, 1000, 5, 3, False, -3.1, -0.1),       # just under the lower limit
])
def test_tolerance_check(m, s, plus, minus, within, dev, margin):
    out = tolerance_check(m, s, plus, minus)
    assert out["within_tolerance"] is within
    _close(out["deviation"], dev, 1e-6)
    _close(out["margin"], margin, 1e-6)


# --- validate_design_status: the design review procedure ---------------------
@pytest.mark.parametrize("status", ["for comment", "Buy-Off", "acceptance"])
def test_design_status_valid(status):
    assert _get(validate_design_status(status), "valid") is True


@pytest.mark.parametrize("status,word", [
    ("Approved", "forbidden"),
    ("sign-off", "forbidden"),
    ("released", "not a valid"),
])
def test_design_status_not_valid(status, word):
    out = validate_design_status(status)
    assert _get(out, "valid") is False
    assert word in _get(out, "message")


def test_design_status_review_statuses_are_in_the_procedure_record():
    from app.core.construction_knowledge import VALID_DESIGN_STATUSES
    assert set(_procedure("design_review")["review_statuses"]) <= VALID_DESIGN_STATUSES
