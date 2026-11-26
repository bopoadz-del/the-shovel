"""Transcription check, batch 9: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what its stated form says. The
stated form is the formula's own docstring, its registered description and
its named constants or table. Every expected value below is a hand
calculation from that stated form, written as a literal; none is obtained by
running the calculator. All inputs are invented.

Two DIFFERENCES between the stated form and the code are recorded here and
NOT fixed (the owner rules on them):

* formwork_striking_time -- the registered description says the striking
  time comes "from element type, temperature and test strength". The code
  takes no element type and no temperature: the answer is the same at any
  temperature, and the BS 8110 note always says "wall formwork".
* enforce_critical_rules -- the registered description says it "lists every
  rule it breaks". The rule catalogue it reads holds seven critical rules;
  the code scans the text for one of them (the word APPROVED on a design
  document). A text that calls an instruction a variation order is passed
  as compliant.

The tests below pin the code as it is, so a change to either is seen.
"""
from __future__ import annotations

import pytest

from app.agents.hats.commercial.formulas.construction_formulas_reference_tables import (
    leed_points_estimate,
)
from app.agents.hats.contracts.formulas.document_rules import (
    enforce_critical_rules,
    generate_doc_number,
)
from app.agents.hats.planning.formulas.project_control import (
    floor_cycle_progress,
    manhours_scurve,
)
from app.agents.hats.procurement.formulas.construction_knowledge import evaluate_tender
from app.agents.hats.qaqc.formulas.construction_formulas import formwork_striking_time
from app.agents.hats.qaqc.formulas.construction_formulas_reference_tables import (
    laser_scan_accuracy,
)
from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    interior_finishes_takeoff,
)
from app.agents.hats.safety.formulas.construction_formulas_additions import (
    guardrail_top_rail_height,
)

TOL = 0.011


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


# --- generate_doc_number: the register templates -----------------------------
# RFI/IR/JR/PR/WP 4 digits; VO/RFM/DD 3 digits; NCR/PDN carry the year when
# given; any other type is TYPE-#### ; the type is upper-cased.
@pytest.mark.parametrize("kind,seq,year,want", [
    ("RFI", 42, None, "RFI-0042"),
    ("ir", 7, None, "IR-0007"),
    ("VO", 15, None, "VO-015"),
    ("dd", 230, None, "DD-230"),
    ("NCR", 1, 2024, "NCR-2024-001"),
    ("ncr", 38, None, "NCR-038"),
    ("PDN", 12, 2031, "PDN-2031-012"),
    ("RFI", 9, 2031, "RFI-0009"),          # the year is not part of an RFI number
    ("SUB", 5, None, "SUB-0005"),          # an unlisted type: TYPE-####
    ("xyz", 123, None, "XYZ-0123"),
])
def test_doc_number_reads_the_template(kind, seq, year, want):
    out = generate_doc_number(kind, seq, year)
    assert out["document_number"] == want
    assert out["doc_type"] == kind.upper()


@pytest.mark.parametrize("kind", ["", "   ", None])
def test_doc_number_refuses_a_missing_type(kind):
    assert "error" in generate_doc_number(kind, 3)


# --- enforce_critical_rules: APPROVED on a design document --------------------
@pytest.mark.parametrize("body", [
    "The drawing package is approved for construction.",
    "Consultant approval received on the architect's layout.",
])
def test_critical_rules_flags_approved_on_a_design_document(body):
    out = enforce_critical_rules(body)
    assert out["compliant"] is False
    assert out["violation_count"] == 1
    assert out["violations"][0]["rule_id"] == "no_approved_on_design"


@pytest.mark.parametrize("body", [
    "The pump was approved by the client.",            # no design context word
    "The drawing package is accepted for comment.",    # design context, no APPROVED
])
def test_critical_rules_passes_text_without_the_breach(body):
    out = enforce_critical_rules(body)
    assert out["compliant"] is True
    assert out["violation_count"] == 0
    assert out["violations"] == []


# DIFFERENCE (recorded, not fixed): only one of the seven catalogued rules is
# scanned. These texts break the RFM rule and the stop-work rule as written in
# the catalogue, and are passed as compliant.
@pytest.mark.parametrize("body", [
    "Treat RFM-003 as a variation order and proceed.",
    "Resume work today after the STOP WORK notice; no sign-off needed.",
])
def test_critical_rules_difference_other_rules_are_not_scanned(body):
    from app.core.construction_knowledge import CRITICAL_RULES
    assert len(CRITICAL_RULES) == 7
    out = enforce_critical_rules(body)
    assert out["compliant"] is True


@pytest.mark.parametrize("body", ["", "   "])
def test_critical_rules_refuses_empty_text(body):
    assert "error" in enforce_critical_rules(body)


# --- evaluate_tender: weighted sum, default 0.45/0.45/0.07/0.03 ---------------
@pytest.mark.parametrize("rows,order,totals", [
    # A: 80x.45 + 70x.45 + 90x.07 + 50x.03 = 36 + 31.5 + 6.3 + 1.5 = 75.3
    # B: 70x.45 + 90x.45 + 60x.07 + 100x.03 = 31.5 + 40.5 + 4.2 + 3 = 79.2
    ([{"name": "A", "technical": 80, "commercial": 70, "hse": 90, "local_content": 50},
      {"name": "B", "technical": 70, "commercial": 90, "hse": 60, "local_content": 100}],
     ["B", "A"], {"A": 75.3, "B": 79.2}),
    # C: 60x.45 + 60x.45 + 100x.07 + 0 = 27 + 27 + 7 = 61.0
    # D: 50x.45 + 80x.45 + 80x.07 + 100x.03 = 22.5 + 36 + 5.6 + 3 = 67.1
    ([{"name": "C", "technical": 60, "commercial": 60, "hse": 100, "local_content": 0},
      {"name": "D", "technical": 50, "commercial": 80, "hse": 80, "local_content": 100}],
     ["D", "C"], {"C": 61.0, "D": 67.1}),
])
def test_tender_default_weights_rank_by_weighted_total(rows, order, totals):
    out = evaluate_tender(rows)
    ranked = out["ranked_tenderers"]
    assert [r["name"] for r in ranked] == order
    for r in ranked:
        _close(r["weighted_total"], totals[r["name"]])
    assert out["recommended"]["name"] == order[0]
    assert [r["rank"] for r in ranked] == [1, 2]


@pytest.mark.parametrize("rows,weights,order,totals,price_scores,applied", [
    # Weights 60/40 sum to 100: read as 0.6/0.4.
    # price score = lowest / amount x 100: A 800/1000x100 = 80, B 100.
    # A: 80x.6 + 80x.4 = 80.0 ; B: 60x.6 + 100x.4 = 76.0
    ([{"name": "A", "technical": 80, "price": 1000},
      {"name": "B", "technical": 60, "price": 800}],
     {"technical": 60, "price": 40}, ["A", "B"], {"A": 80.0, "B": 76.0},
     {"A": 80.0, "B": 100.0}, {"technical": 0.6, "price": 0.4}),
    # 50/50: E price 900/1200x100 = 75, F 100.
    # E: 90x.5 + 75x.5 = 82.5 ; F: 70x.5 + 100x.5 = 85.0
    ([{"name": "E", "technical": 90, "price": 1200},
      {"name": "F", "technical": 70, "price": 900}],
     {"technical": 50, "price": 50}, ["F", "E"], {"E": 82.5, "F": 85.0},
     {"E": 75.0, "F": 100.0}, {"technical": 0.5, "price": 0.5}),
])
def test_tender_percent_weights_and_price_scored_lowest_over_amount(
        rows, weights, order, totals, price_scores, applied):
    out = evaluate_tender(rows, weights)
    ranked = out["ranked_tenderers"]
    assert [r["name"] for r in ranked] == order
    for r in ranked:
        _close(r["weighted_total"], totals[r["name"]])
        _close(r["price_score"], price_scores[r["name"]])
    assert out["weights_applied"] == applied
    assert out.get("weights_read_as_percent") is True


def test_tender_suffixed_score_wins_over_bare_value():
    # technical_score 50 is read, the bare 99 is not: 50x0.5 + 40x0.5 = 45.0
    # the other tenderer: 70x0.5 + 30x0.5 = 50.0
    out = evaluate_tender(
        [{"name": "P", "technical_score": 50, "technical": 99, "hse": 40},
         {"name": "Q", "technical": 70, "hse": 30}],
        {"technical": 0.5, "hse": 0.5},
    )
    totals = {r["name"]: r["weighted_total"] for r in out["ranked_tenderers"]}
    _close(totals["P"], 45.0)
    _close(totals["Q"], 50.0)


@pytest.mark.parametrize("tenderers", [[], None])
def test_tender_refuses_no_tenderers(tenderers):
    assert "error" in evaluate_tender(tenderers)


def test_tender_refuses_tenderers_without_any_criterion_score():
    out = evaluate_tender([{"name": "A", "colour": "red"}, {"name": "B"}])
    assert "error" in out


# --- floor_cycle_progress: SUM(c/t x w) / SUM(w) ------------------------------
@pytest.mark.parametrize("comp,tot,wt,elements,weighted", [
    ([3, 10], [4, 20], [2, 1], [0.75, 0.5], 0.667),        # (1.5 + 0.5) / 3
    ([1, 9, 0], [2, 10, 5], [1, 1, 2], [0.5, 0.9, 0.0], 0.35),  # 1.4 / 4
])
def test_floor_cycle_weighted_progress(comp, tot, wt, elements, weighted):
    out = floor_cycle_progress(comp, tot, wt)
    assert out["element_progress"] == elements
    _close(out["weighted_progress"], weighted, 0.0011)


def test_floor_cycle_weights_summing_to_zero_are_undefined_not_zero():
    out = floor_cycle_progress([1, 2], [2, 4], [0, 0])
    assert out["weighted_progress"] is None
    assert "undefined" in out["note"]


@pytest.mark.parametrize("comp,tot,wt", [
    ([1, 2], [2, 4], [1]),        # unequal lengths
    ([], [], []),                 # empty
    ([1, 2], [2, 0], [1, 1]),     # a zero total
])
def test_floor_cycle_refuses_degenerate_input(comp, tot, wt):
    assert "error" in floor_cycle_progress(comp, tot, wt)


# --- formwork_striking_time: both minima met -> proposed, else max() ----------
@pytest.mark.parametrize("strength,design,ciria,bs_h,prop_h,hours,based_on", [
    (3.0, 0.5, 3.0, 12, 8, 8, "test_data"),       # exactly at the CIRIA minimum
    (2.99, 0.5, 3.0, 12, 8, 12, "insufficient"),  # just below it
    (5.0, 5.01, 3.0, 12, 8, 12, "insufficient"),  # below the design strength
    (2.0, 1.0, 3.0, 10, 16, 16, "insufficient"),  # proposed already longer
    (6.0, 4.0, 3.0, 24, 18, 18, "test_data"),
])
def test_formwork_striking_rule(strength, design, ciria, bs_h, prop_h, hours, based_on):
    out = formwork_striking_time(
        concrete_strength_7h=strength, design_required_strength=design,
        ciria_surface_strength=ciria, bs8110_minimum_hours=bs_h, proposed_hours=prop_h,
    )
    _close(_get(out, "recommended_hours"), hours)
    assert _get(out, "based_on") == based_on


@pytest.mark.parametrize("strength,ciria,says", [
    (3.0, 3.0, "meets the CIRIA"),     # equal to the minimum meets it
    (2.99, 3.0, "is below the CIRIA"),
    (4.5, 4.5, "meets the CIRIA"),
])
def test_formwork_note_states_the_ciria_comparison(strength, ciria, says):
    out = formwork_striking_time(concrete_strength_7h=strength, design_required_strength=0.5,
                                 ciria_surface_strength=ciria, bs8110_minimum_hours=12,
                                 proposed_hours=8)
    assert any(says in n for n in _get(out, "notes"))


def test_formwork_difference_no_element_type_or_temperature_input():
    # DIFFERENCE (recorded, not fixed): the description names element type and
    # temperature; the declared inputs carry neither.
    from app.lib.formula_registry import parameters
    spec = formwork_striking_time.__formula__
    assert "temperature" in spec.description and "element type" in spec.description
    names = set(parameters(spec))
    assert not any("temp" in n for n in names)
    assert not any("element" in n for n in names)


# --- guardrail_top_rail_height: 42 in +/- 3 in, mid-rail halfway --------------
@pytest.mark.parametrize("part,inches,mm,m", [
    ("top_rail", 42, 1066.8, 1.07),     # 42 x 25.4
    ("mid_rail", 21, 533.4, 0.53),      # halfway: 21 x 25.4
])
def test_guardrail_heights_convert_from_the_stated_inches(part, inches, mm, m):
    out = guardrail_top_rail_height()
    assert out[f"{part}_height_in"] == inches
    _close(out[f"{part}_height_mm"], mm, 0.41)
    _close(out[f"{part}_height_m"], m, 0.001)
    assert out["tolerance_in"] == 3
    _close(out["tolerance_mm"], 76.2, 0.21)                     # 3 x 25.4


# --- interior_finishes_takeoff -----------------------------------------------
@pytest.mark.parametrize("kw,want", [
    # 2 rooms x 1 door 0.9 x 2.1 = 3.78 m2; 30 x 3 = 90; paint 90 - 3.78 - 4
    # = 82.22; blockwork x 0.5 = 41.11; skirting 30 - 2 x 0.9 = 28.2
    (dict(floor_area_m2=50, perimeter_m=30, room_count=2, floor_to_ceiling_m=3,
          door_width_m=0.9, door_height_m=2.1, doors_per_room=1,
          shared_wall_fraction=0.5, window_deduction_m2=4),
     dict(floor_screed_m2=50, ceiling_finish_m2=50, gross_wall_face_m2=90,
          door_deduction_m2=3.78, wall_paint_m2=82.22, blockwork_m2=41.11, skirting_m=28.2)),
    # 3 rooms x 2 doors 1.0 x 2.0 = 12 m2; 40 x 2.7 = 108; paint 96;
    # blockwork x 0.25 = 24; skirting 40 - 6 = 34
    (dict(floor_area_m2=72.5, perimeter_m=40, room_count=3, floor_to_ceiling_m=2.7,
          door_width_m=1.0, door_height_m=2.0, doors_per_room=2,
          shared_wall_fraction=0.25),
     dict(floor_screed_m2=72.5, ceiling_finish_m2=72.5, gross_wall_face_m2=108,
          door_deduction_m2=12, wall_paint_m2=96, blockwork_m2=24, skirting_m=34)),
])
def test_interior_finishes_takeoff(kw, want):
    out = interior_finishes_takeoff(**kw)
    for key, value in want.items():
        _close(out[key], value)


@pytest.mark.parametrize("kw", [
    # 4 x 1 = 4 m2 of wall; deductions 4.5 x 1 + 5 > 4; skirting 4 - 4.5 < 0
    dict(floor_area_m2=1, perimeter_m=4, room_count=1, floor_to_ceiling_m=1,
         door_width_m=4.5, door_height_m=1, doors_per_room=1, window_deduction_m2=5),
    # 6 x 1 = 6 m2; 2 rooms x 3.5 x 1 = 7 > 6; skirting 6 - 7 < 0
    dict(floor_area_m2=2, perimeter_m=6, room_count=2, floor_to_ceiling_m=1,
         door_width_m=3.5, door_height_m=1, doors_per_room=1),
])
def test_interior_finishes_deductions_never_go_below_zero(kw):
    out = interior_finishes_takeoff(**kw)
    assert out["wall_paint_m2"] == 0.0
    assert out["blockwork_m2"] == 0.0
    assert out["skirting_m"] == 0.0


@pytest.mark.parametrize("height", [0, -2.5])
def test_interior_finishes_refuses_a_missing_height(height):
    out = interior_finishes_takeoff(floor_area_m2=10, perimeter_m=12, room_count=1,
                                    floor_to_ceiling_m=height)
    assert "error" in out


# --- laser_scan_accuracy: accuracy at 10 m x range / 10 (linear) --------------
@pytest.mark.parametrize("rng,acc10,want", [
    (25, 2.0, 5.0),
    (4, 3.5, 1.4),
    (10, 1.2, 1.2),
])
def test_laser_scan_accuracy_scales_linearly(rng, acc10, want):
    out = laser_scan_accuracy(rng, acc10)
    _close(_get(out, "estimated_accuracy_mm"), want)
    _close(_get(out, "range_m"), rng)


# --- leed_points_estimate: 40 / 50 / 60 / 80 thresholds ----------------------
@pytest.mark.parametrize("points,level", [
    (39.9, "Not certified"), (40, "Certified"), (49, "Certified"),
    (50, "Silver"), (59.5, "Silver"), (60, "Gold"), (79, "Gold"),
    (80, "Platinum"), (110, "Platinum"), (0, "Not certified"),
])
def test_leed_level_reads_the_thresholds(points, level):
    out = leed_points_estimate(points)
    assert out["certification_level"] == level
    _close(out["points"], points)


# --- manhours_scurve: both series over the PLANNED total; index = plan / actual
@pytest.mark.parametrize("planned,actual,pp,ap,index", [
    # planned 300/1000 = 30.0 %; actual 370/1000 = 37.0 %; 30/37 = 0.811
    ([100, 200, 300, 400], [120, 250], 30.0, 37.0, 0.811),
    # periods compared = min(len): 3. planned 150/200 = 75 %; actual 120/200
    # = 60 %; 75/60 = 1.25
    ([50, 50, 50, 50], [40, 40, 40], 75.0, 60.0, 1.25),
])
def test_manhours_scurve(planned, actual, pp, ap, index):
    out = manhours_scurve(planned, actual)
    _close(out["cum_planned_pct"], pp, 0.051)
    _close(out["cum_actual_pct"], ap, 0.051)
    _close(out["productivity_index"], index, 0.0011)


@pytest.mark.parametrize("planned,actual,pp", [
    ([10, 20], [0, 0], 100.0),
    ([5, 5, 5], [0], 33.3),             # one period compared: 5 / 15
])
def test_manhours_scurve_no_actual_hours_gives_no_index(planned, actual, pp):
    out = manhours_scurve(planned, actual)
    assert out["productivity_index"] is None
    assert out["cum_planned_pct"] == pp
    assert out["cum_actual_pct"] == 0.0


@pytest.mark.parametrize("planned", [[0, 0], []])
def test_manhours_scurve_refuses_a_zero_plan(planned):
    assert "error" in manhours_scurve(planned, [1])


# --- the public route returns the same hand values ---------------------------
@pytest.mark.parametrize("name,params,key,want", [
    ("floor_cycle_progress", {"completed_units": [1, 9, 0], "total_units": [2, 10, 5],
                              "weights": [1, 1, 2]}, "weighted_progress", 0.35),
    ("laser_scan_accuracy", {"range_m": 4, "accuracy_at_10m_mm": 3.5},
     "estimated_accuracy_mm", 1.4),
    ("interior_finishes_takeoff", {"floor_area_m2": 72.5, "perimeter_m": 40, "room_count": 3,
                                   "floor_to_ceiling_m": 2.7, "door_width_m": 1.0,
                                   "door_height_m": 2.0, "doors_per_room": 2,
                                   "shared_wall_fraction": 0.25}, "wall_paint_m2", 96.0),
    ("manhours_scurve", {"planned_mh": [100, 200, 300, 400], "actual_mh": [120, 250]},
     "productivity_index", 0.811),
])
def test_public_route_returns_the_hand_value(name, params, key, want):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation(name, params)
    assert env["status"] == "success"
    _close(env["result"][key], want, 0.0011)


@pytest.mark.parametrize("points,level", [(59.5, "Silver"), (80, "Platinum")])
def test_public_route_leed_level(points, level):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation("leed_points_estimate", {"points": points})
    assert env["status"] == "success"
    assert env["result"]["certification_level"] == level


@pytest.mark.parametrize("name,params", [
    ("leed_points_estimate", {"points": 111}),
    ("laser_scan_accuracy", {"range_m": 2500}),
])
def test_public_route_refuses_outside_the_declared_range(name, params):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation(name, params)
    assert env["status"] == "error"
