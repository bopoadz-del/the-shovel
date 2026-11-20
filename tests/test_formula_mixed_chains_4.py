"""Formula mixed scenarios, part 4: CI chains, no model (exam 6).

From the formula test plan, part 3 (mixed scenarios). Each chain calls the
calculators in order through ``run_calculation`` (the entry point
``construction_calc`` uses). The named output of one step is typed into the
named input of the next. Every intermediate value in the plan's "Hand" line is
asserted, not only the final value.

Every expected value is a HAND calculation from the published definition,
written beside its row; none was copied from the product's output. Each
chain runs on at least two inputs: the plan's case and an invented case
worked by hand the same way. Where the chain makes a decision (who wins,
ahead or behind, does the contractor need financing), one input goes each
way. Plan tolerance: 0.5 % on any number; counts rounded up are exact.

Chains here: M06, M07 with the M32 correction sets (CI part), M18, M27,
M28 and M29. All inputs are invented numbers; no project, person or
document is named.

Conventions, read from each docstring before writing the test (plan rule for
a flagged convention):
- M06 ``cash_flow``: the horizon is the months given; the advance is paid at
  month 0; retention is released with the last certificate; a certificate is
  paid ceil(lag / 30) months later. The plan's month 7 receipt is outside
  that horizon, so the closing-cash line is not asserted; the plan's peak
  funding (months 3-4) is inside it and is.
- M27 ``interior_finishes_takeoff``: ``perimeter_m`` is the summed wall
  length of all rooms (the plan's convention). Durations and crews are
  rounded up by the chain, as the plan states.
- M29 ``manhours_scurve``: the productivity index is planned % / actual %
  (the docstring's definition); the plan accepts either definition once
  stated.
"""
from __future__ import annotations

import math

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 5e-3  # plan tolerance for mixed scenarios: +-0.5 %


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, env)
    return env["result"]


# ----------------------------- M06 cash flow -> peak funding (6-month horizon)
# certified_m = V_m (1 - adv - ret), plus all retention on the last month;
# cash_in_m = certified_(m - lag); cum_m = advance + sum(cash_in - cost).

@pytest.mark.parametrize(
    "value, cost_ratio, adv, ret, lag_days, advance, retention, valuation, "
    "peak, needs_funding", [
        # plan (M): V 1,2,3,3,2,1 = 12; cost 0.85V; lag ceil(30/30) = 1.
        # advance 1.2; retention 0.6; cert 0.85,1.7,2.55,2.55,1.7,0.85+0.6 -> 10.8
        # in 0,0.85,1.7,2.55,2.55,1.7; net -0.85,-0.85,-0.85,0,+0.85,+0.85
        # cum 0.35,-0.50,-1.35,-1.35,-0.50,+0.35 -> peak 1.35
        ([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], 0.85, 0.10, 0.05, 30,
         1.2, 0.6, 10.8, 1.35, True),
        # invented: V 2,4,4 = 10; cost 0.8V = 1.6,3.2,3.2; lag ceil(60/30) = 2.
        # advance 2.0; retention 1.0; cert 1.4,2.8,2.8+1.0 -> 8.0
        # in 0,0,1.4; net -1.6,-3.2,-1.8; cum 0.4,-2.8,-4.6 -> peak 4.6
        ([2.0, 4.0, 4.0], 0.8, 0.20, 0.10, 60,
         2.0, 1.0, 8.0, 4.6, True),
        # invented: V 1,1 = 2; cost 0.5V; adv 30 %, ret 5 %; lag ceil(15/30) = 1.
        # advance 0.6; retention 0.1; cert 0.65, 0.65+0.1 -> 1.4
        # in 0,0.65; net -0.5,+0.15; cum 0.10, 0.25 -> never negative, peak 0
        ([1.0, 1.0], 0.5, 0.30, 0.05, 15,
         0.6, 0.1, 1.4, 0.0, False),
    ])
def test_m06_cash_flow_peak_funding(value, cost_ratio, adv, ret, lag_days, advance,
                                    retention, valuation, peak, needs_funding):
    cost = [round(v * cost_ratio, 6) for v in value]
    cf = _ok("cash_flow", {"monthly_value": value, "monthly_cost": cost,
                           "advance_pct": adv, "retention_pct": ret,
                           "payment_lag_days": lag_days})
    assert cf["contract_value"] == pytest.approx(sum(value), rel=_REL)
    assert cf["advance_payment"] == pytest.approx(advance, rel=_REL)
    assert cf["retention_total"] == pytest.approx(retention, rel=_REL)
    # the advance is recovered in full and all retention is deducted
    assert cf["check_advance_recovery"] == pytest.approx(-advance, rel=_REL)
    assert cf["check_retention"] == pytest.approx(-retention, rel=_REL)
    assert cf["check_valuation"] == pytest.approx(valuation, rel=_REL)
    assert cf["peak_funding_requirement"] == pytest.approx(peak, rel=_REL, abs=1e-9)
    assert (cf["peak_funding_requirement"] > 0) is needs_funding


# ---------------- M07 progress -> interim payment -> advance recovery (+ M32)
# actual % = done / total; gross to date = % x contract; this period =
# gross - previously certified; retention = rate x this period; advance
# recovery = advance % x this period (labelled step, no formula); net =
# this period - retention - recovery.

@pytest.mark.parametrize(
    "ret_rate, previous, planned_pct, actual_pct, remaining, this_period, "
    "retention, net_before_advance, recovery, net, cumulative", [
        # M07 plan: 900/2,500 = 36.0 %; 0.36 x 8.0 M = 2,880,000; - 1,600,000 =
        # 1,280,000; 5 % = 64,000; 10 % = 128,000; net 1,280,000-64,000-128,000
        (0.05, 1_600_000, 40.0, 36.0, 1_600, 1_280_000, 64_000, 1_216_000,
         128_000, 1_088_000, 2_880_000),
        # M32 turn 2: retention 10 %: 128,000; net 1,280,000-128,000-128,000
        (0.10, 1_600_000, 40.0, 36.0, 1_600, 1_280_000, 128_000, 1_152_000,
         128_000, 1_024_000, 2_880_000),
        # M32 turn 3: previously 1,500,000: this period 1,380,000; 10 % = 138,000;
        # recovery 138,000; net 1,380,000-138,000-138,000 = 1,104,000
        (0.10, 1_500_000, 40.0, 36.0, 1_600, 1_380_000, 138_000, 1_242_000,
         138_000, 1_104_000, 2_880_000),
    ])
def test_m07_m32_interim_payment_with_advance_recovery(ret_rate, previous, planned_pct,
                                                       actual_pct, remaining, this_period,
                                                       retention, net_before_advance,
                                                       recovery, net, cumulative):
    contract, total_qty, advance_pct = 8_000_000, 2_500, 0.10
    prog = _ok("progress_quantity", {"total_qty": total_qty, "planned_qty": 1_000,
                                     "actual_qty": 900})
    assert prog["planned_percent"] == pytest.approx(planned_pct, rel=_REL)
    assert prog["actual_percent"] == pytest.approx(actual_pct, rel=_REL)
    assert prog["remaining_qty"] == pytest.approx(remaining, rel=_REL)
    gross_to_date = prog["actual_percent"] / 100 * contract
    period = gross_to_date - previous
    assert period == pytest.approx(this_period, rel=_REL)
    pay = _ok("calculate_payment", {"claimed_amount": period, "certified_amount": period,
                                    "retention_rate": ret_rate,
                                    "cumulative_previous_certified": previous,
                                    "contract_value": contract})
    assert pay["retention_held"] == pytest.approx(retention, rel=_REL)
    assert pay["net_payment_due"] == pytest.approx(net_before_advance, rel=_REL)
    assert pay["cumulative_certified"] == pytest.approx(cumulative, rel=_REL)
    assert pay["percent_complete"] == pytest.approx(actual_pct, rel=_REL)
    advance_recovery = advance_pct * period
    assert advance_recovery == pytest.approx(recovery, rel=_REL)
    assert pay["net_payment_due"] - advance_recovery == pytest.approx(net, rel=_REL)


# ------------------------- M18 tender evaluation -> award -> first payment
# price score = lowest price / price x 100 (the request's rule, typed in);
# total = w_price x price score + w_tech x technical; pay x % of the winner's
# price, less retention.

@pytest.mark.parametrize(
    "tenders, w_price, price_scores, totals, order, pay_pct, ret_pct, gross, "
    "retention, net, lowest_wins", [
        # plan: lowest 4.50; A 4.5/4.8 = 93.75, B 100, C 4.5/5.1 = 88.24
        # A 0.7*93.75+0.3*80 = 89.63; B 70+21 = 91.00; C 61.76+27 = 88.76 -> B, A, C
        # 20 % of 4.5 M = 900,000; 5 % = 45,000; net 855,000
        ({"A": (4_800_000, 80), "B": (4_500_000, 70), "C": (5_100_000, 90)}, 0.7,
         {"A": 93.75, "B": 100.0, "C": 88.24}, {"A": 89.63, "B": 91.0, "C": 88.76},
         ["B", "A", "C"], 0.20, 5, 900_000, 45_000, 855_000, True),
        # invented: lowest 2.0; P 100, Q 2.0/2.5 = 80; weights 50/50:
        # P 50+30 = 80.0; Q 40+47.5 = 87.5 -> Q, P (not the lowest price)
        # 10 % of 2.5 M = 250,000; 10 % = 25,000; net 225,000
        ({"P": (2_000_000, 60), "Q": (2_500_000, 95)}, 0.5,
         {"P": 100.0, "Q": 80.0}, {"P": 80.0, "Q": 87.5},
         ["Q", "P"], 0.10, 10, 250_000, 25_000, 225_000, False),
    ])
def test_m18_tender_award_then_first_payment(tenders, w_price, price_scores, totals, order,
                                             pay_pct, ret_pct, gross, retention, net,
                                             lowest_wins):
    lowest = min(price for price, _ in tenders.values())
    rows = []
    for name, (price, tech) in tenders.items():
        score = lowest / price * 100
        assert score == pytest.approx(price_scores[name], rel=_REL)
        rows.append({"name": name, "commercial": score, "technical": tech})
    ev = _ok("evaluate_tender", {"tenderers": rows,
                                 "weights": {"commercial": w_price,
                                             "technical": round(1 - w_price, 6)}})
    ranked = ev["ranked_tenderers"]
    assert [r["name"] for r in ranked] == order
    for r in ranked:
        assert r["weighted_total"] == pytest.approx(totals[r["name"]], rel=_REL)
    winner = ev["recommended"]["name"]
    assert (tenders[winner][0] == lowest) is lowest_wins
    pay = _ok("calculate_interim_payment", {"gross_valuation": pay_pct * tenders[winner][0],
                                            "retention_percent": ret_pct})
    assert pay["gross_valuation"] == pytest.approx(gross, rel=_REL)
    assert pay["retention_amount"] == pytest.approx(retention, rel=_REL)
    assert pay["net_payment"] == pytest.approx(net, rel=_REL)


# ------- M27 finishes take-off -> painting man-hours -> duration and crew
# paint = perimeter x height - doors; coats x paint; man-hours = area / rate;
# daily output = rate x painters x hours; days = area / daily output (UP);
# painters for D days = man-hours / (D x hours) (UP).

@pytest.mark.parametrize(
    "floor, perimeter, rooms, height, door_w, door_h, coats, rate, painters, hours, "
    "target_days, paint, skirting, coat_area, manhours, mh_per_day, days, crew", [
        # plan: 44*3 - 4*0.9*2.1 = 124.44; skirting 44-3.6 = 40.4; x2 = 248.88;
        # /3 = 82.96 mh; 4*8 = 32 mh/day; 248.88/96 = 2.59 -> 3 d;
        # 2 days: 82.96/16 = 5.19 -> 6 painters
        (120, 44, 4, 3.0, 0.9, 2.1, 2, 3, 4, 8, 2,
         124.44, 40.4, 248.88, 82.96, 32, 3, 6),
        # invented: 30*2.7 - 2*0.8*2.0 = 77.8; skirting 30-1.6 = 28.4; x3 = 233.4;
        # /4 = 58.35 mh; 2*8 = 16 mh/day; 233.4/64 = 3.65 -> 4 d;
        # 1 day: 58.35/8 = 7.29 -> 8 painters
        (60, 30, 2, 2.7, 0.8, 2.0, 3, 4, 2, 8, 1,
         77.8, 28.4, 233.4, 58.35, 16, 4, 8),
    ])
def test_m27_finishes_then_painting_duration_and_crew(floor, perimeter, rooms, height,
                                                      door_w, door_h, coats, rate,
                                                      painters, hours, target_days, paint,
                                                      skirting, coat_area, manhours,
                                                      mh_per_day, days, crew):
    take = _ok("interior_finishes_takeoff", {
        "floor_area_m2": floor, "perimeter_m": perimeter, "room_count": rooms,
        "floor_to_ceiling_m": height, "door_width_m": door_w, "door_height_m": door_h,
        "doors_per_room": 1})
    assert take["wall_paint_m2"] == pytest.approx(paint, rel=_REL)
    assert take["floor_tiling_m2"] == pytest.approx(floor, rel=_REL)
    assert take["skirting_m"] == pytest.approx(skirting, rel=_REL)
    area = take["wall_paint_m2"] * coats
    assert area == pytest.approx(coat_area, rel=_REL)
    pmd = _ok("productivity_manpower_duration", {
        "quantity": area, "productivity": str(rate), "manpower": painters,
        "working_hours": hours})
    assert pmd["manhours_required"] == pytest.approx(manhours, rel=_REL)
    assert pmd["manhours"] == pytest.approx(mh_per_day, rel=_REL)
    dur = _ok("productivity_manpower_duration", {
        "quantity": area, "daily_production": rate * pmd["manhours"]})
    assert math.ceil(dur["duration"]) == days
    need = _ok("productivity_manpower_duration", {
        "remaining_manhours": pmd["manhours_required"],
        "available_hours": target_days * hours})
    assert math.ceil(need["required_manpower"]) == crew


# ------------------------------ M28 floor-cycle progress -> progress KPI
# weighted = sum(done/total x w) / sum(w); variance = actual - plan; SPI = actual / plan

@pytest.mark.parametrize(
    "done, floors, weights, plan_pct, weighted, variance, spi, ahead", [
        # plan: 0.3*0.8 + 0.3*0.7 + 0.4*0.6 = 0.690; 69-75 = -6.0; 69/75 = 0.920
        ([8, 7, 6], 10, [30, 30, 40], 75, 0.690, -6.0, 0.920, False),
        # invented: 0.2*0.6 + 0.3*0.5 + 0.5*0.4 = 0.470; 47-40 = +7.0; 47/40 = 1.175
        ([12, 10, 8], 20, [20, 30, 50], 40, 0.470, 7.0, 1.175, True),
    ])
def test_m28_floor_cycle_then_progress_kpi(done, floors, weights, plan_pct, weighted,
                                           variance, spi, ahead):
    fc = _ok("floor_cycle_progress", {"completed_units": done,
                                      "total_units": [floors] * len(done),
                                      "weights": weights})
    assert fc["weighted_progress"] == pytest.approx(weighted, rel=_REL)
    kpi = _ok("progress_kpi", {"planned_cum": plan_pct,
                               "actual_cum": fc["weighted_progress"] * 100,
                               "planned_total": 100})
    assert kpi["actual_pct"] == pytest.approx(weighted * 100, rel=_REL)
    assert kpi["variance_pct"] == pytest.approx(variance, rel=_REL)
    assert kpi["spi_progress"] == pytest.approx(spi, rel=_REL)
    assert (kpi["spi_progress"] >= 1) is ahead


# --------------------- M29 man-hour S-curve -> recovery manpower
# cum % against the planned total; index = planned % / actual %;
# remaining = planned total x (1 - actual %); people = remaining / (days x hours)

@pytest.mark.parametrize(
    "planned, actual, cum_planned, cum_actual, index, days, hours, people", [
        # plan, month 4: 6,000/6,000 = 100.0; 5,200/6,000 = 86.67; 100/86.67 = 1.154;
        # remaining 800 mh; 800 / (10 x 8) = 10.0
        ([1000, 1500, 2000, 1500], [900, 1400, 1800, 1100], 100.0, 86.67, 1.154, 10, 8, 10.0),
        # plan, month 3 point: 4,500/6,000 = 75.0; 4,100/6,000 = 68.33;
        # 75/68.33 = 1.098; remaining 1,900; 1,900 / 80 = 23.75
        ([1000, 1500, 2000, 1500], [900, 1400, 1800], 75.0, 68.33, 1.098, 10, 8, 23.75),
        # invented: total 1,000; 750/1,000 = 75.0 vs 100.0; 100/75 = 1.333;
        # remaining 250; 250 / (5 x 10) = 5.0
        ([400, 600], [300, 450], 100.0, 75.0, 1.333, 5, 10, 5.0),
    ])
def test_m29_manhour_scurve_then_recovery_manpower(planned, actual, cum_planned, cum_actual,
                                                   index, days, hours, people):
    sc = _ok("manhours_scurve", {"planned_mh": planned, "actual_mh": actual})
    assert sc["cum_planned_pct"] == pytest.approx(cum_planned, rel=_REL)
    assert sc["cum_actual_pct"] == pytest.approx(cum_actual, rel=_REL)
    assert sc["productivity_index"] == pytest.approx(index, rel=_REL)
    remaining = sum(planned) * (1 - sc["cum_actual_pct"] / 100)
    need = _ok("productivity_manpower_duration", {"remaining_manhours": remaining,
                                                  "available_hours": days * hours})
    assert need["required_manpower"] == pytest.approx(people, rel=_REL)
