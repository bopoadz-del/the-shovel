"""Formula mixed scenarios, part 5: CI chains, no model (exam 6).

From the formula test plan, part 3 (mixed scenarios). Each chain calls the
product's calculators in order; the named output of one step is typed into
the named input of the next. Every intermediate value in the plan's "Hand"
line is asserted, not only the final value.

Every expected value is a HAND calculation from the published definition,
written beside its row; none was copied from the product's output. Each
chain runs on the plan's case and an invented case worked by hand the same
way. Where the chain makes a decision (one crane or more, a fractional crew
rounded up or not, a norm found or refused), one input goes each way. Plan
tolerance: 0.5 % on any number; counts rounded up are exact.

Chains here: M19, M25, M26 and M31. All inputs are invented numbers; no project,
person or document is named.

Conventions, read from each docstring before writing the test (plan rule for
a flagged convention):
- M19 ``crane_planning``: cranes = ceil(demand / (capacity x util x
  60 / cycle x hours x shifts x working_days)). The plan's route (crane-hours
  needed / crane-hours available per crane) is the same ratio, so the plan's
  count holds. ``crane_cost_estimate`` adds mobilisation (50 % of one month's
  dry hire per crane) and demobilisation (75 % of mobilisation); the plan
  asks only for the month's running cost, which is ``total_project_cost``.
- M25 ``boq_schedule``: duration_days = ceil(man-hours / (crew x hours)), the
  plan's convention. ``schedule_bridge.bridge_activity`` gives man-days =
  crew x rounded days and cost = man-days x day_rate, again the plan's.
- M26 ``price_line_items``: exact (category + unit) median, else the
  category median, else the asset median, each over rates of the same unit
  class (docstring). The card has no reinforcement or formwork row, so those
  lines take the asset median of the mass / area rows. The oracle is the
  card's rows, read here as data (plan rule), and the medians are worked from
  them by hand. ``reconcile_unit`` flags a stated unit that contradicts the
  work type. SUSPECTED DEFECT (strict xfail below): ``kg`` and ``t`` share
  one unit class with no scale, so a 9,000 kg line is priced at the per-tonne
  rate per kg, 1,000 times over.
- M31 ``productivity_benchmark`` holds finishing trades only; slab concrete
  has no row, so the product must return no rate (never invent one). The
  cited norm then comes from the published-ranges table the plan names,
  read here as data: "structural concrete place + finish (gang of 6)",
  typical 35 m3 per day. The row is labelled per gang, so one gang is one
  crew (convention stated; the live answer must state it too).
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pytest

from app.lib.boq_pricing import categorize
from app.lib.boq_pricing import price_line_items
from app.lib.boq_schedule import MissingProductivity, activities_from_boq
from app.lib.boq_units import reconcile_unit
from app.lib.construction_formulas import run_calculation
from app.lib.pm_excel import _cpm
from app.lib.schedule_bridge import bridge_activity

_REL = 5e-3  # plan tolerance for mixed scenarios: +-0.5 %
_CARD = (Path(__file__).resolve().parents[1]
         / "app" / "lib" / "data" / "boq_rate_card.json")
_NORMS = (Path(__file__).resolve().parents[1]
          / "docs" / "knowledge" / "productivity_norms_planning_defaults.md")


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, env)
    return env["result"]


# ------------------------------------- M19 crane programme -> crane hire cost
# lifts/h = 60/cycle; t/day/crane = cap*util*lifts/h*h*shifts;
# cranes = ceil(demand / (t/day * days)); cost = (dry+op+rig) * n * months.

@pytest.mark.parametrize(
    "demand, months, daily_t, cranes, monthly_cost, running, mob, demob, grand", [
        # plan: 25 t * 0.6 * 3 lifts/h * 8 h * 2 = 720 t/day; *20 = 14,400 t
        # per crane-month; 6,000 / 14,400 = 0.417 -> 1 crane (the plan's
        # 133.3 crane-h / 320 h = 0.417, same ratio). 1 * 55,000 dry hire.
        # running 76,000 * 1 * 1 = 76,000; mob 27,500; demob 20,625;
        # grand 124,125.
        (6_000, 1, 720.0, 1, 55_000, 76_000, 27_500, 20_625, 124_125),
        # invented: 30,000 / 14,400 = 2.083 -> 3 cranes; dry 3 * 55,000 =
        # 165,000; running 76,000 * 3 * 2 = 456,000; mob 3 * 27,500 = 82,500;
        # demob 61,875; grand 600,375.
        (30_000, 2, 720.0, 3, 165_000, 456_000, 82_500, 61_875, 600_375),
        # boundary: exactly one crane-month of capacity -> 1, not 2.
        (14_400, 1, 720.0, 1, 55_000, 76_000, 27_500, 20_625, 124_125),
        # one tonne over -> 14,401 / 14,400 = 1.00007 -> 2 cranes.
        (14_401, 1, 720.0, 2, 110_000, 152_000, 55_000, 41_250, 248_250),
    ],
)
def test_m19_crane_programme_to_hire_cost(
        demand, months, daily_t, cranes, monthly_cost, running, mob, demob,
        grand):
    plan = _ok("crane_planning", {
        "total_lift_demand_tons": demand, "crane_capacity_tons": 25,
        "utilization_pct": 60, "shifts_per_day": 2, "hours_per_shift": 8,
        "cycle_time_minutes": 20, "working_days": 20,
        "dry_hire_rate_per_month": 55_000,
    })
    assert plan["lifts_per_hour_per_crane"] == pytest.approx(3.0)
    assert plan["daily_tonnage_per_crane"] == pytest.approx(daily_t, rel=_REL)
    assert plan["cranes_required"] == cranes  # count rounded up: exact
    assert plan["total_monthly_cost"] == pytest.approx(monthly_cost, rel=_REL)

    cost = _ok("crane_cost_estimate", {
        "num_cranes": plan["cranes_required"],  # step 1 -> step 2
        "crane_capacity_tons": 25, "duration_months": months,
        "dry_hire_rate_per_month": 55_000,
        "operator_rate_per_month": 9_000, "riggers_rate_per_month": 12_000,
    })
    assert cost["per_crane_monthly"] == pytest.approx(76_000, rel=_REL)
    assert cost["total_project_cost"] == pytest.approx(running, rel=_REL)
    assert cost["mobilization"] == pytest.approx(mob, rel=_REL)
    assert cost["demobilization"] == pytest.approx(demob, rel=_REL)
    assert cost["grand_total"] == pytest.approx(grand, rel=_REL)


@pytest.mark.parametrize("given, missing", [
    # operator included but its rate not given -> named, never defaulted.
    ({"riggers_rate_per_month": 12_000}, ["operator_rate_per_month"]),
    ({"operator_rate_per_month": 9_000}, ["riggers_rate_per_month"]),
])
def test_m19_missing_crew_rate_is_named_not_defaulted(given, missing):
    out = run_calculation("crane_cost_estimate", {
        "num_cranes": 1, "crane_capacity_tons": 25, "duration_months": 1,
        "dry_hire_rate_per_month": 55_000, **given,
    })
    assert out.get("status") != "success", out
    assert out.get("needs_input") is True, out
    assert out.get("missing") == missing, out
    assert "total_project_cost" not in (out.get("result") or {})


# --------------------------- M25 BOQ -> man-hours -> durations -> CPM -> cost
# mh = q * norm; days = ceil(mh / (crew * 8)); man-days = crew * days;
# cost = man-days * day rate; FS chain, so the programme is the sum of days.

_M25_CASES = [
    pytest.param(
        # plan: 1,200 * 0.30 = 360 mh; 18 * 28 = 504; 216 * 2.5 = 540.
        # days 360/48 = 7.5 -> 8; 504/64 = 7.875 -> 8; 540/72 = 7.5 -> 8.
        # programme 24 d, all critical, float 0. man-days 48, 64, 72;
        # cost 48*120 = 5,760; 64*140 = 8,960; 72*130 = 9,360;
        # cumulative 5,760, 14,720, 24,080.
        [("Excavation in any material", 1_200, "m3", 0.30, 6, 120),
         ("High yield steel reinforcement in slabs", 18, "t", 28.0, 8, 140),
         ("Concrete C30 in slabs", 216, "m3", 2.5, 9, 130)],
        [360.0, 504.0, 540.0], [8, 8, 8], [48, 64, 72],
        [5_760, 8_960, 9_360], [5_760, 14_720, 24_080], 24,
        id="plan"),
    pytest.param(
        # invented: 500 * 0.4 = 200 mh, crew 5 -> 200/40 = 5.0 -> 5 (exact,
        # not rounded up); 100 * 3.0 = 300 mh, crew 6 -> 300/48 = 6.25 -> 7.
        # man-days 25, 42; cost 25*100 = 2,500; 42*150 = 6,300;
        # cumulative 2,500, 8,800; programme 12 d.
        [("Excavation in any material", 500, "m3", 0.4, 5, 100),
         ("Concrete C30 in slabs", 100, "m3", 3.0, 6, 150)],
        [200.0, 300.0], [5, 7], [25, 42],
        [2_500, 6_300], [2_500, 8_800], 12,
        id="invented"),
]


@pytest.mark.parametrize(
    "lines, mh, days, man_days, cost, cum, programme", _M25_CASES)
def test_m25_boq_to_cost_loaded_programme(
        lines, mh, days, man_days, cost, cum, programme):
    items = [{"description": d, "quantity": q, "unit": u}
             for d, q, u, _n, _c, _r in lines]
    norms = {categorize(d): n for d, _q, _u, n, _c, _r in lines}
    crews = {categorize(d): c for d, _q, _u, _n, c, _r in lines}
    acts = activities_from_boq(items, manhours_per_unit=norms,
                               crew_size=crews, hours_per_day=8.0)
    # the request's order is the build order: one FS chain, first has none.
    assert [a["total_manhours"] for a in acts] == pytest.approx(mh, rel=_REL)
    assert [a["duration_days"] for a in acts] == days  # rounded up: exact
    assert acts[0]["predecessors"] == []
    for prev, nxt in zip(acts, acts[1:]):
        assert nxt["predecessors"] == [prev["id"]]

    rates = [r for _d, _q, _u, _n, _c, r in lines]
    loaded = [bridge_activity(a, day_rate=r) for a, r in zip(acts, rates)]
    assert [a["manpower"] * a["duration"] for a in loaded] == man_days
    assert [a["cost"] for a in loaded] == pytest.approx(cost, rel=_REL)
    running = 0.0
    cumulative = []
    for a in loaded:  # cumulative by activity: labelled step, no formula
        running += a["cost"]
        cumulative.append(running)
    assert cumulative == pytest.approx(cum, rel=_REL)

    cpm, project, order = _cpm(loaded)
    assert project == programme
    assert order == [a["id"] for a in loaded]
    assert all(cpm[a["id"]]["tf"] == 0 and cpm[a["id"]]["critical"]
               for a in loaded)


@pytest.mark.parametrize("given, refused", [
    # a category with no norm is refused by name; nothing is defaulted.
    ({"Earthworks/Excavation": 0.30}, "Concrete"),
    ({"Concrete": 2.5}, "Earthworks/Excavation"),
])
def test_m25_missing_norm_is_refused_by_name(given, refused):
    items = [{"description": "Excavation in any material",
              "quantity": 1_200, "unit": "m3"},
             {"description": "Concrete C30 in slabs",
              "quantity": 216, "unit": "m3"}]
    with pytest.raises(MissingProductivity) as err:
        activities_from_boq(items, manhours_per_unit=given)
    assert refused in str(err.value)


# ------------------------------- M26 price a BOQ -> unit check -> payment
# line = qty * card median (exact, else same-class fallback); total = sum;
# gross = pct * total; retention = ret * gross; net = gross - retention.

_M26_LINES = [
    {"description": "Concrete C30 in slabs", "quantity": 120, "unit": "m3"},
    {"description": "High yield steel reinforcement", "quantity": 9,
     "unit": "t"},
    {"description": "Formwork to slab soffits", "quantity": 450, "unit": "m2"},
]


def _card_rows(asset: str, ccy: str) -> list:
    return json.loads(_CARD.read_text(encoding="utf-8"))["cards"][asset][ccy]


def _m26_hand_lines() -> list:
    """Hand rate per line from the card rows (data): concrete exact m3 row;
    reinforcement and formwork have no row of their own, so the asset median
    of the rows measured in t (mass) and in m2 (area)."""
    rows = _card_rows("Buildings/Towers", "AED")
    concrete = [r["median"] for r in rows
                if r["cat"] == "Concrete" and r["unit"] == "m3"]
    assert len(concrete) == 1
    assert not [r for r in rows if r["cat"] in ("Reinforcement", "Formwork")]
    mass = statistics.median(r["median"] for r in rows
                             if r["unit"] in ("t", "kg", "mt", "ton", "tonne"))
    area = statistics.median(r["median"] for r in rows if r["unit"] == "m2")
    return [120 * concrete[0], 9 * mass, 450 * area]


@pytest.mark.parametrize("pct, ret, gross_share, net_share", [
    # plan: 60 % complete, retention 10 % -> gross 0.6 T, net 0.54 T.
    (0.60, 10, 0.60, 0.54),
    # invented: 25 % complete, retention 5 % -> gross 0.25 T, net 0.2375 T.
    (0.25, 5, 0.25, 0.2375),
])
def test_m26_price_boq_to_interim_payment(pct, ret, gross_share, net_share):
    hand = _m26_hand_lines()
    total = sum(hand)
    priced, summary = price_line_items(_M26_LINES, "Buildings/Towers", "AED")
    amounts = [x["qty"] * x["rate"] for x in priced]
    assert amounts == pytest.approx(hand, rel=_REL)
    # rate basis shown per line: exact for concrete, fallback for the rest.
    assert priced[0]["rate_source"].startswith("exact")
    assert all(x["rate_source"] != "NO RATE" for x in priced)
    assert summary["no_rate"] == 0
    assert summary["grand_total"] == pytest.approx(total, rel=_REL)

    pay = _ok("calculate_interim_payment", {
        "gross_valuation": pct * summary["grand_total"],  # step 2 -> step 3
        "retention_percent": ret,
    })
    assert pay["gross_valuation"] == pytest.approx(gross_share * total, rel=_REL)
    assert pay["retention_amount"] == pytest.approx(
        (gross_share - net_share) * total, rel=_REL)
    assert pay["net_payment"] == pytest.approx(net_share * total, rel=_REL)


@pytest.mark.parametrize("unit, suspect, expected", [
    ("t", False, None),   # stated unit agrees with the work type
    ("kg", True, "t"),    # '9,000 kg' contradicts a tonnage item: flagged
])
def test_m26_unit_contradiction_is_flagged(unit, suspect, expected):
    kept, source, flagged, want = reconcile_unit(
        unit, "High yield steel reinforcement")
    assert kept == unit  # the stated unit is kept, never overwritten
    assert source == "parsed"
    assert flagged is suspect
    assert want == expected


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: kg and t share one unit class with no scale, so a "
    "per-tonne median prices a per-kg line 1,000 times over"))
def test_m26_same_steel_in_kg_costs_the_same_as_in_t():
    # 9 t = 9,000 kg: the two lines are the same steel and must cost the same.
    priced, _ = price_line_items([
        {"description": "High yield steel reinforcement", "quantity": 9,
         "unit": "t"},
        {"description": "High yield steel reinforcement", "quantity": 9_000,
         "unit": "kg"},
    ], "Buildings/Towers", "AED")
    in_t, in_kg = (x["qty"] * x["rate"] for x in priced)
    assert in_kg == pytest.approx(in_t, rel=_REL)


# --------------------------------------------- M31 cited norm -> crews
# crews = Q / (D * R) rounded up; crew-days = Q / R; man-hours = crew-days * h.

def _cited_typical(row_label: str) -> float:
    """Typical output of one row of the published-ranges table (data)."""
    for line in _NORMS.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].lower().startswith(row_label):
            return float(cells[3])
    raise AssertionError(f"row {row_label!r} not in {_NORMS.name}")


def test_m31_slab_concrete_has_no_benchmark_so_none_is_invented():
    out = _ok("productivity_benchmark", {"work_item": "slab concrete"})
    assert out["rate_per_crew_day"] is None
    assert "no benchmark entry" in out["flag"]
    assert out["tier"] == "GENERAL"


def test_m31_cited_norm_to_crews_plan_case():
    rate = _cited_typical("structural concrete place")
    assert rate == 35.0  # the table value read at write time (plan rule)
    out = _ok("crew_sizing", {"quantity": 640, "duration_days": 10,
                              "rate_per_crew_day": rate, "hours_per_day": 8})
    # 640 / (10 * 35) = 1.829 -> 2 crews; crew-days 640/35 = 18.29;
    # man-hours 18.29 * 8 = 146.3; identity 2 * 10 * 35 = 700 >= 640.
    assert out["crews_required"] == pytest.approx(1.829, rel=_REL)
    assert out["crews_required_rounded_up"] == 2
    assert out["crew_days"] == pytest.approx(18.29, rel=_REL)
    assert out["man_hours"] == pytest.approx(146.3, rel=_REL)
    assert out["identity_check"] == pytest.approx(700, rel=_REL)


@pytest.mark.parametrize("quantity, crews, crews_up, crew_days, man_hours", [
    # invented: blockwork benchmark 12 m2 per crew-day, 25 days, 10 h.
    # 1,500 / (25 * 12) = 5.0 -> 5 exactly; crew-days 125; man-hours 1,250.
    (1_500, 5.0, 5, 125.0, 1_250.0),
    # 1,510 / 300 = 5.033 -> 6; crew-days 125.83; man-hours 1,258.3.
    (1_510, 5.033, 6, 125.83, 1_258.3),
])
def test_m31_benchmark_rate_to_crews_invented(
        quantity, crews, crews_up, crew_days, man_hours):
    bench = _ok("productivity_benchmark", {"work_item": "Blockwork (walls)"})
    assert bench["rate_per_crew_day"] == 12.0
    assert bench["unit"] == "m2"
    assert re.search(r"unverified", bench["provenance"])  # source shown
    out = _ok("crew_sizing", {
        "quantity": quantity, "duration_days": 25,
        "rate_per_crew_day": bench["rate_per_crew_day"],  # step 1 -> step 2
        "hours_per_day": 10,
    })
    assert out["crews_required"] == pytest.approx(crews, rel=_REL)
    assert out["crews_required_rounded_up"] == crews_up
    assert out["crew_days"] == pytest.approx(crew_days, rel=_REL)
    assert out["man_hours"] == pytest.approx(man_hours, rel=_REL)
