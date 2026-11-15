"""Formula mixed scenarios, part 2: the critical-path chains, no model.

From the formula test plan, part 3 (mixed scenarios): M05 (take-off ->
crews -> network -> float -> histogram -> look-ahead), M13 (curing time ->
network), M17 (network delay -> damages -> risk score) and M30 (compression
-> new critical path). The plan's inventory says three CPM implementations
exist; the same network is run through all three and must give the same
duration and critical path:

  1. ``app.lib.pm_computations.compute_cpm``
  2. ``app.lib.pm_excel._cpm`` (the cost-loaded schedule workbook)
  3. ``ConstructionContainer._calculate_cpm`` (the schedule analysis path)

Every expected value is a HAND calculation written beside its row; none was
copied from the product's output. Each decision runs on two inputs: the
plan's case and a second invented network worked by hand the same way.
Durations are whole days; the M05 request counts calendar days, so the
calendar used for dates is seven working days a week.
All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from app.lib.construction_formulas import run_calculation
from app.lib.pm_computations import (
    compress_schedule,
    compute_cpm,
    resource_histogram,
    select_look_ahead,
)
from app.lib.pm_excel import _cpm as excel_cpm
from app.schemas.cpm import (
    Activity,
    CPMInput,
    Dependency,
    ResourceAssignment,
    WorkCalendar,
)

_REL = 5e-3  # plan tolerance for mixed scenarios: +-0.5 %
_CALENDAR_DAYS = WorkCalendar(work_weekdays=[0, 1, 2, 3, 4, 5, 6])


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, env)
    return env["result"]


def _net(spec: dict, crews: dict | None = None) -> list[Activity]:
    """spec: id -> (duration, [predecessor ids]) ; crews: id -> persons."""
    return [
        Activity(id=i, name=i, duration=d,
                 predecessors=[Dependency(predecessor_id=p) for p in preds],
                 resources=([ResourceAssignment(trade="crew", count=crews[i])] if crews else []))
        for i, (d, preds) in spec.items()
    ]


def _as_dicts(spec: dict) -> list[dict]:
    return [{"id": i, "duration": d, "predecessors": list(preds)} for i, (d, preds) in spec.items()]


# M05 network (plan): raft A 10; columns B 8 and walls C 6 after A; slab D 9 after B and C; test E 3 after D.
# Hand: A 0-10, B 10-18, C 10-16, D 18-27, E 27-30 | duration 30 | critical A-B-D-E |
#       C: ES 10 EF 16 LS 12 LF 18, float 2.
M05 = {"A": (10, []), "B": (8, ["A"]), "C": (6, ["A"]), "D": (9, ["B", "C"]), "E": (3, ["D"])}
M05_HAND = {"duration": 30, "critical": ["A", "B", "D", "E"],
            "times": {"A": (0, 10, 0, 10), "B": (10, 18, 10, 18), "C": (10, 16, 12, 18),
                      "D": (18, 27, 18, 27), "E": (27, 30, 27, 30)}}
# Invented twin: P 5; Q 4 and R 7 after P; S 2 after Q and R.
# Hand: P 0-5, Q 5-9, R 5-12, S 12-14 | duration 14 | critical P-R-S | Q: LS 8 LF 12, float 3.
TWIN = {"P": (5, []), "Q": (4, ["P"]), "R": (7, ["P"]), "S": (2, ["Q", "R"])}
TWIN_HAND = {"duration": 14, "critical": ["P", "R", "S"],
             "times": {"P": (0, 5, 0, 5), "Q": (5, 9, 8, 12), "R": (5, 12, 5, 12), "S": (12, 14, 12, 14)}}

NETWORKS = [pytest.param(M05, M05_HAND, id="M05"), pytest.param(TWIN, TWIN_HAND, id="twin")]


# ----------------------------------------------- M05 step 1: raft -> crews
# volume = L x W x t; crews = Q / (D x R), rounded up; crew-days = Q / R;
# duration with the rounded crews = crew-days / crews.

@pytest.mark.parametrize("dims, rate, days, hours, volume, crews, crew_days, duration", [
    ((40, 20, 0.8), 32, 10, 10, 640.0, 2, 20.0, 10.0),  # 640; 640/(10x32) = 2.000 -> 2; 640/32 = 20; 20/2 = 10
    ((25, 12, 1.0), 25, 6, 8, 300.0, 2, 12.0, 6.0),     # 300; 300/(6x25) = 2.000 -> 2; 300/25 = 12; 12/2 = 6
])
def test_m05_raft_volume_then_crews_and_duration(dims, rate, days, hours, volume, crews, crew_days, duration):
    length, width, thick = dims
    vol = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick,
                                  "waste_factor": 0})
    assert vol["net_volume_m3"] == pytest.approx(volume, rel=_REL)
    crew = _ok("crew_sizing", {"quantity": vol["net_volume_m3"], "duration_days": days,
                               "rate_per_crew_day": rate, "hours_per_day": hours})
    assert crew["crews_required_rounded_up"] == crews
    assert crew["crew_days"] == pytest.approx(crew_days, rel=_REL)
    assert crew["crew_days"] / crew["crews_required_rounded_up"] == pytest.approx(duration, rel=_REL)


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect (wrong number): crew_sizing man_hours = crew-days x hours ignores the "
    "persons per crew; the request gives a crew of 6, so man-hours = 2 x 10 x 6 x 10 = 1,200, "
    "the formula returns 20 x 10 = 200"))
@pytest.mark.parametrize("q, days, rate, hours, size, man_hours", [
    (640, 10, 32, 10, 6, 1_200.0),  # crews 2 x 10 d x 6 persons x 10 h = 1,200
    (300, 6, 25, 8, 5, 480.0),      # crews 2 x 6 d x 5 persons x 8 h = 480
])
def test_m05_man_hours_count_every_person_in_the_crew(q, days, rate, hours, size, man_hours):
    crew = _ok("crew_sizing", {"quantity": q, "duration_days": days, "rate_per_crew_day": rate,
                               "hours_per_day": hours, "crew_size": size})
    assert crew["man_hours"] == pytest.approx(man_hours, rel=_REL)


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect (input meaning): crew_sizing labels crew_size 'crew size' (persons) but "
    "computes duration_for_crews = Q / (R x crew_size), i.e. reads it as a NUMBER of crews; "
    "with a crew of 6 persons and 2 crews the raft takes 10 d, the formula returns 3.33"))
@pytest.mark.parametrize("q, days, rate, size, duration", [
    (640, 10, 32, 6, 10.0),  # 2 crews (rounded) -> 640 / (32 x 2) = 10 d
    (300, 6, 25, 5, 6.0),    # 2 crews -> 300 / (25 x 2) = 6 d
])
def test_m05_duration_for_crews_does_not_read_persons_as_crews(q, days, rate, size, duration):
    crew = _ok("crew_sizing", {"quantity": q, "duration_days": days, "rate_per_crew_day": rate,
                               "hours_per_day": 10, "crew_size": size})
    assert crew["duration_for_crews"] == pytest.approx(duration, rel=_REL)


# --------------------------------------- M05 step 2: one network, three engines

@pytest.mark.parametrize("spec, hand", NETWORKS)
def test_engine_1_compute_cpm(spec, hand):
    out = compute_cpm(CPMInput(activities=_net(spec)))
    assert out.project_duration == hand["duration"]
    assert out.critical_path == hand["critical"]
    got = {r.id: (r.early_start_day, r.early_finish_day, r.late_start_day, r.late_finish_day)
           for r in out.results}
    assert got == hand["times"]


@pytest.mark.parametrize("spec, hand", NETWORKS)
def test_engine_2_cost_loaded_workbook_cpm(spec, hand):
    times, duration, _order = excel_cpm(_as_dicts(spec))
    assert duration == hand["duration"]
    assert [i for i in spec if times[i]["critical"]] == hand["critical"]
    assert {i: (t["es"], t["ef"], t["ls"], t["lf"]) for i, t in times.items()} == hand["times"]


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect (not an engine): ConstructionContainer._calculate_cpm does no forward or "
    "backward pass; it reads total_float and start/finish dates from the input. Given only "
    "durations and predecessors it marks every activity critical (all floats default to 999) "
    "and returns duration 0, so the same network does not give the same answer"))
@pytest.mark.parametrize("spec, hand", NETWORKS)
def test_engine_3_container_schedule_cpm(spec, hand):
    from app.containers.construction import ConstructionContainer

    out = ConstructionContainer._calculate_cpm(None, {"activities": _as_dicts(spec)})
    assert out["project_duration_days"] == hand["duration"]
    assert out["critical_path"] == hand["critical"]


# ------------------------------- M05 step 3: float of the non-critical activity
# TF = LS - ES; the inputs are the network's own times, not typed by hand.

@pytest.mark.parametrize("spec, act, tf", [
    (M05, "C", 2.0),   # C: LS 12 - ES 10 = 2 (= LF 18 - EF 16)
    (TWIN, "Q", 3.0),  # Q: LS 8 - ES 5 = 3 (= LF 12 - EF 9)
])
def test_m05_float_of_the_non_critical_branch(spec, act, tf):
    res = {r.id: r for r in compute_cpm(CPMInput(activities=_net(spec))).results}[act]
    out = _ok("critical_path_float", {"early_start": res.early_start_day, "early_finish": res.early_finish_day,
                                      "late_start": res.late_start_day, "late_finish": res.late_finish_day})
    assert out["total_float"] == pytest.approx(tf, abs=1e-9)
    assert out["consistency_check_lf_minus_ef"] == pytest.approx(tf, abs=1e-9)
    assert out["is_critical"] is False
    assert res.total_float == tf


# -------------------------------------------------- M05 step 4: manpower
# Persons: M05 A 12, B 10, C 6, D 12, E 4 ; twin P 4, Q 3, R 5, S 6.
# Week buckets (5 days) sum every crew whose span overlaps the bucket (the
# docstring's definition). Man-days = sum persons x duration.
M05_CREWS = {"A": 12, "B": 10, "C": 6, "D": 12, "E": 4}
TWIN_CREWS = {"P": 4, "Q": 3, "R": 5, "S": 6}


@pytest.mark.parametrize("spec, crews, weeks, man_days", [
    # W1 A 12 | W2 A 12 | W3 B+C 16 | W4 (15-20) B+C+D 28 | W5 D 12 | W6 (25-30) D+E 16
    # man-days 12x10 + 10x8 + 6x6 + 12x9 + 4x3 = 120+80+36+108+12 = 356
    (M05, M05_CREWS, [12, 12, 16, 28, 12, 16], 356),
    # W1 P 4 | W2 (5-10) Q+R 8 | W3 (10-15) R+S 11 ; man-days 20+12+35+12 = 79
    (TWIN, TWIN_CREWS, [4, 8, 11], 79),
])
def test_m05_week_histogram_and_man_days(spec, crews, weeks, man_days):
    acts = _net(spec, crews)
    hist = resource_histogram(compute_cpm(CPMInput(activities=acts)).results, acts, period_unit="week")
    assert [p.total for p in hist.periods] == weeks
    assert hist.total_manhours / 8 == pytest.approx(man_days, rel=_REL)  # engine uses 8 h per day


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect (wrong number): the histogram peak sums every crew that touches a week, "
    "even crews that never work on the same day; the plan's peak is the largest number of "
    "persons on site on one day (M05: 16 on days 11-16; the week sum gives 28)"))
@pytest.mark.parametrize("spec, crews, daily_peak", [
    (M05, M05_CREWS, 16),  # days 11-16: B 10 + C 6 = 16; no other day is higher
    (TWIN, TWIN_CREWS, 8),  # days 6-9: Q 3 + R 5 = 8; S (6) never overlaps R
])
def test_m05_peak_is_the_most_persons_on_one_day(spec, crews, daily_peak):
    acts = _net(spec, crews)
    hist = resource_histogram(compute_cpm(CPMInput(activities=acts)).results, acts, period_unit="week")
    assert hist.peak_total == daily_peak


# ------------------------------------------------ M05 step 5: look-ahead
# Dates from the network on a calendar-day calendar; window as_of .. as_of + n - 1.

@pytest.mark.parametrize("spec, start, as_of, days, selected", [
    # 1 Dec start: A 1-11 Dec, B 11-19, C 11-17, D 19-28, E 28-31. Window 8-21 Dec: A B C D; E out.
    (M05, date(2026, 12, 1), date(2026, 12, 8), 14, ["A", "B", "C", "D"]),
    # 1 Mar start: P 1-6 Mar, Q 6-10, R 6-13, S 13-15. Window 8-10 Mar: Q R; P ended, S later.
    (TWIN, date(2027, 3, 1), date(2027, 3, 8), 3, ["Q", "R"]),
])
def test_m05_look_ahead_from_network_dates(spec, start, as_of, days, selected):
    out = compute_cpm(CPMInput(activities=_net(spec), project_start=start, calendar=_CALENDAR_DAYS))
    rows = [{"id": r.id, "early_start": r.early_start, "early_finish": r.early_finish,
             "total_float": r.total_float} for r in out.results]
    la = select_look_ahead(rows, as_of=as_of, window_days=days)
    assert sorted(a["id"] for a in la["activities"]) == selected
    assert la["count"] == len(selected)


# ------------------------------------------- M13 curing time -> network
# t = a f / (1 - b f), rounded up; A pour 1; B cure t after A; C prep 3 after A;
# D strike 1 after B; E next pour 1 after D and C.

@pytest.mark.parametrize("frac, days_exact, cure, duration, c_ls, c_lf, c_tf", [
    (0.7, 6.914, 7, 10, 6, 9, 5),  # 4x0.7/(1-0.595) = 6.914 -> 7; A 0-1 B 1-8 D 8-9 E 9-10; C 1-4, LF 9
    (0.5, 3.478, 4, 7, 3, 6, 2),   # 4x0.5/(1-0.425) = 3.478 -> 4; A 0-1 B 1-5 D 5-6 E 6-7; C 1-4, LF 6
])
def test_m13_curing_time_drives_the_network(frac, days_exact, cure, duration, c_ls, c_lf, c_tf):
    cur = _ok("concrete_curing_time", {"target_strength_fraction": frac, "gain_a": 4, "gain_b": 0.85})
    assert cur["days_to_target"] == pytest.approx(days_exact, rel=_REL)
    spec = {"A": (1, []), "B": (math.ceil(cur["days_to_target"]), ["A"]), "C": (3, ["A"]),
            "D": (1, ["B"]), "E": (1, ["D", "C"])}
    assert spec["B"][0] == cure
    out = compute_cpm(CPMInput(activities=_net(spec)))
    assert out.project_duration == duration
    assert out.critical_path == ["A", "B", "D", "E"]
    c = {r.id: r for r in out.results}["C"]
    assert (c.late_start_day, c.late_finish_day, c.total_float) == (c_ls, c_lf, c_tf)
    times, excel_duration, _ = excel_cpm(_as_dicts(spec))
    assert excel_duration == duration and times["C"]["tf"] == c_tf


# ------------------------------- M17 network delay -> damages -> risk score
# delay = duration - contract days; daily = rate% x contract; total = daily x delay.

@pytest.mark.parametrize("spec, critical, contract_days, amount, rate, delay, daily, total", [
    # A 0-20, B 20-55, C 20-45, D 55-95, E 95-128 -> 128; delay 8; 0.05% x 20M = 10,000/d; 80,000
    ({"A": (20, []), "B": (35, ["A"]), "C": (25, ["A"]), "D": (40, ["B", "C"]), "E": (33, ["D"])},
     ["A", "B", "D", "E"], 120, 20_000_000, 0.05, 8, 10_000.0, 80_000.0),
    # A 0-10, B 10-25, C 10-22, D 25-45 -> 45; delay 5; 0.1% x 5M = 5,000/d; 25,000
    ({"A": (10, []), "B": (15, ["A"]), "C": (12, ["A"]), "D": (20, ["B", "C"])},
     ["A", "B", "D"], 40, 5_000_000, 0.1, 5, 5_000.0, 25_000.0),
])
def test_m17_network_delay_priced_as_damages(spec, critical, contract_days, amount, rate, delay, daily, total):
    out = compute_cpm(CPMInput(activities=_net(spec)))
    assert out.critical_path == critical
    late = out.project_duration - contract_days
    assert late == delay
    dmg = _ok("delay_damages_daily", {"rate_percent": rate, "contract_amount": amount, "days": late})
    assert dmg["daily_amount"] == pytest.approx(daily, rel=_REL)
    assert dmg["total_amount"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("p, i, score, band", [
    (4, 3, 12, "RED"),   # 4 x 3 = 12 (> 9)
    (2, 2, 4, "GREEN"),  # 2 x 2 = 4 (<= 4)
])
def test_m17_risk_score_of_the_delay(p, i, score, band):
    from app.agents.hats.contracts.formulas.construction_knowledge import score_risk

    out = score_risk(p, i)
    assert (out["score"], out["band"]) == (score, band)


# ------------------------------------------ M30 compression -> new path
# The least-cost choice is worked by hand (the engine takes the cuts as input).

@pytest.mark.parametrize("spec, cuts, duration, saved, critical, branch, branch_tf", [
    # M05, cut B 1 (3,000) + A 2 (2 x 4,000) = 11,000: A 8 B 7 C 6 D 9 E 3 -> 27; C path 26, float 1
    (M05, {"B": 1, "A": 2}, 27, 3, ["A", "B", "D", "E"], "C", 1),
    # twin, cut R 2: P 5 Q 4 R 5 S 2 -> P-R-S 12, P-Q-S 11; Q float 1
    (TWIN, {"R": 2}, 12, 2, ["P", "R", "S"], "Q", 1),
])
def test_m30_compression_on_the_critical_path(spec, cuts, duration, saved, critical, branch, branch_tf):
    revised, delta = compress_schedule(CPMInput(activities=_net(spec)), cuts)
    assert revised.project_duration == duration
    assert delta == saved
    assert revised.critical_path == critical
    assert {r.id: r.total_float for r in revised.results}[branch] == branch_tf


@pytest.mark.parametrize("spec, cuts, duration", [
    (M05, {"C": 2}, 30),   # C is not critical (float 2): cutting it saves nothing
    (TWIN, {"Q": 3}, 14),  # Q has float 3: saves nothing
])
def test_m30_twin_cutting_a_non_critical_activity_saves_nothing(spec, cuts, duration):
    revised, delta = compress_schedule(CPMInput(activities=_net(spec)), cuts)
    assert revised.project_duration == duration
    assert delta == 0
