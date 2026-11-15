"""Course program step 2: roads and earthworks formulas as KB calculators.

Every entry is a ``type=formula`` knowledge entry evaluated by
``app.blocks._knowledge.evaluate``. Worked examples come from the
owner-supplied course material (roads and earthworks sections 1-2); each
test states its hand calculation. Where the source rounds a printed figure,
the test asserts the exact value and the comment names the printed one.

The two fleet studies in the course reach their machine counts by a
different route (outputs summed over several haul distances). Their printed
answers are deliberately NOT used here; the cycle formula is tested on
invented cases worked by hand.
"""
from __future__ import annotations

import pytest

from app.blocks import _knowledge as kb

NEW_IDS = [
    "earthworks.loader_buckets_per_truck",
    "earthworks.wet_density",
    "earthworks.moisture_content",
    "earthworks.dry_density",
    "earthworks.sand_cone_hole_volume",
    "ground_improvement.dynamic_compaction_vibration_radius",
    "equipment.truck_cycle_time",
    "equipment.cycle_production",
    "equipment.machines_required",
    "earthworks.fill_layers",
    "earthworks.compaction_water",
]


def ev(rule_id, **values):
    return kb.evaluate(rule_id, **values)


# ---------------------------------------------------------------------------
# Envelope: every new entry is a general formula with a course source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule_id", NEW_IDS)
def test_entry_carries_course_source_and_declared_units(rule_id):
    e = kb.get_rule(rule_id)
    assert e is not None and e["type"] == "formula"
    ref = e["provenance"]["course_reference"]
    assert ref.startswith("owner-supplied course material, ")
    assert e["provenance"]["project"] is None
    assert e["applicability"]["project_specific"] is False
    assert "not a code or standard requirement" in e["statement"]
    # every symbol in the expression is a declared variable with a unit
    import sympy

    symbols = {str(s) for s in sympy.sympify(e["expression"]).free_symbols}
    assert symbols == set(e["variables"]), rule_id
    assert all(v.get("unit") for v in e["variables"].values())
    assert e["thresholds"]["unit"]


@pytest.mark.parametrize(
    "rule_id, partial",
    [
        ("equipment.machines_required", {"required_rate": 100}),
        ("earthworks.dry_density", {"wet_density": 2.0}),
        ("earthworks.sand_cone_hole_volume", {"sand_before": 8000, "sand_after": 2000, "sand_in_cone": 1500}),
        ("equipment.truck_cycle_time", {"fixed_time": 12, "haul_distance": 5, "loaded_speed": 45, "return_distance": 5}),
    ],
)
def test_missing_required_input_raises_not_defaulted(rule_id, partial):
    with pytest.raises(ValueError):
        ev(rule_id, **partial)


# ---------------------------------------------------------------------------
# Unrounded figure shown beside a rounded-up count (evaluator mechanism)
# ---------------------------------------------------------------------------


def test_rounded_up_count_shows_unrounded_figure():
    # Course W4: 18 / 3.1 = 5.806..., printed "5.8 say 6".
    out = ev("earthworks.loader_buckets_per_truck", truck_capacity=18, bucket_capacity=3.1)
    assert out["result"] == 6
    assert out["unrounded"] == pytest.approx(18 / 3.1)
    # Invented: 10 / 3 = 3.333 -> 4 (a part bucket is still a pass).
    out = ev("earthworks.loader_buckets_per_truck", truck_capacity=10, bucket_capacity=3)
    assert out["result"] == 4
    assert out["unrounded"] == pytest.approx(10 / 3)


def test_exact_whole_count_is_not_rounded_up():
    # Invented: 12 / 4 = 3 exactly -> 3, unrounded 3.
    out = ev("earthworks.loader_buckets_per_truck", truck_capacity=12, bucket_capacity=4)
    assert out["result"] == 3 and out["unrounded"] == 3
    # Invented: 720 / 360 = 2 exactly -> 2 machines.
    out = ev("equipment.machines_required", required_rate=720, production_per_machine=360)
    assert out["result"] == 2 and out["unrounded"] == 2


@pytest.mark.parametrize(
    "expression, values, result, unrounded",
    [
        # one ceiling: its argument is the unrounded figure (7/2 = 3.5 -> 4)
        ("ceiling(a / b)", {"a": 7, "b": 2}, 4, 3.5),
        # one ceiling inside other arithmetic: 2 x ceiling(9/4) = 2 x 3 = 6, unrounded 2.25
        ("2 * ceiling(a / b)", {"a": 9, "b": 4}, 6, 2.25),
        # two ceilings: no single figure to show -> no key
        ("ceiling(a / b) + ceiling(b / a)", {"a": 7, "b": 2}, 5, None),
        ("ceiling(a) * ceiling(b)", {"a": 1.5, "b": 2.5}, 6, None),
    ],
)
def test_unrounded_reads_the_expression_structure(monkeypatch, expression, values, result, unrounded):
    synthetic = {"id": "synthetic.count", "type": "formula", "expression": expression,
                 "thresholds": {"unit": "items"}, "provenance": {}, "credibility_tier": 3}
    monkeypatch.setattr(kb, "get_rule", lambda rule_id: synthetic)
    out = kb.evaluate("synthetic.count", **values)
    assert out["result"] == result
    if unrounded is None:
        assert "unrounded" not in out
    else:
        assert out["unrounded"] == pytest.approx(unrounded)


def test_formula_without_rounding_has_no_unrounded_key():
    # No ceiling() in the expression -> nothing to show.
    assert "unrounded" not in ev("earthworks.dry_density", wet_density=2.0, moisture_percent=10)
    assert "unrounded" not in ev("ground_improvement.dynamic_compaction_vibration_radius", M=10, H=10)


# ---------------------------------------------------------------------------
# Proctor (compaction) worksheet: five points reproduce to the last digit
# ---------------------------------------------------------------------------

# mould 4562 g, mould volume 2114 cm3 (unit inferred; the sheet prints none)
PROCTOR = [
    # mould+sample, cont+wet, cont+dry, cont, wet(printed), w%(printed), dry(printed)
    (8764, 117.0, 111.9, 16.2, 1.988, 5.3, 1.887),
    (8969, 129.2, 121.2, 16.0, 2.085, 7.6, 1.937),
    (9130, 129.0, 119.3, 15.9, 2.161, 9.4, 1.976),
    (9245, 132.7, 120.8, 16.0, 2.215, 11.4, 1.989),
    (9228, 137.2, 122.7, 16.0, 2.207, 13.6, 1.943),
]


@pytest.mark.parametrize("ms, cw, cd, c, wet_p, w_p, dry_p", PROCTOR)
def test_proctor_worksheet_point(ms, cw, cd, c, wet_p, w_p, dry_p):
    # Point 1 by hand: (8764 - 4562) / 2114 = 4202 / 2114 = 1.98770;
    # w = 5.1 / 95.7 x 100 = 5.3292 %; dry = 1.98770 / 1.053292 = 1.88713.
    wet = ev("earthworks.wet_density", gross_mass=ms, container_mass=4562, volume=2114)["result"]
    w = ev("earthworks.moisture_content", wet_with_container=cw, dry_with_container=cd, container_mass=c)["result"]
    dry = ev("earthworks.dry_density", wet_density=wet, moisture_percent=w)["result"]
    assert round(wet, 3) == wet_p
    assert round(w, 1) == w_p
    assert round(dry, 3) == dry_p


def test_proctor_point_values_exact():
    # Point 4 by hand: 4683 / 2114 = 2.215232; w = 11.9 / 104.8 x 100 =
    # 11.354962 %; dry = 2.215232 / 1.11354962 = 1.989343.
    wet = ev("earthworks.wet_density", gross_mass=9245, container_mass=4562, volume=2114)["result"]
    w = ev("earthworks.moisture_content", wet_with_container=132.7, dry_with_container=120.8, container_mass=16.0)["result"]
    assert wet == pytest.approx(4683 / 2114)
    assert w == pytest.approx(11.9 / 104.8 * 100)
    assert ev("earthworks.dry_density", wet_density=wet, moisture_percent=w)["result"] == pytest.approx(1.989343, abs=1e-6)


# ---------------------------------------------------------------------------
# Field density (sand replacement): two positions reproduce, chain to %
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before, after, wet_soil, cw, cd, c, v_hand, w_hand, dry_p, pct_p",
    [
        # pos 1: (8000-2514-1564)/1.512 = 3922/1.512 = 2593.915 cm3;
        # wet 5880/2593.915 = 2.26684; w = 10/128.9 = 7.7580 %;
        # dry 2.26684/1.077580 = 2.10364; 2.10364/2.131 = 98.716 % (printed 98.7)
        (8000, 2514, 5880, 241.2, 231.2, 102.3, 3922 / 1.512, 10 / 128.9 * 100, 2.104, 98.7),
        # pos 2: 4158/1.512 = 2750.0; wet 6309/2750 = 2.29418; w = 11.4/138.6 =
        # 8.2251 %; dry 2.11982 (printed 2.12); 99.476 % (printed 99.5)
        (8000, 2278, 6309, 250.8, 239.4, 100.8, 4158 / 1.512, 11.4 / 138.6 * 100, 2.120, 99.5),
    ],
)
def test_field_density_chain(before, after, wet_soil, cw, cd, c, v_hand, w_hand, dry_p, pct_p):
    V = ev("earthworks.sand_cone_hole_volume", sand_before=before, sand_after=after,
           sand_in_cone=1564, sand_density=1.512)["result"]
    assert V == pytest.approx(v_hand)
    # soil from the hole is weighed net: container_mass = 0, stated
    wet = ev("earthworks.wet_density", gross_mass=wet_soil, container_mass=0, volume=V)["result"]
    w = ev("earthworks.moisture_content", wet_with_container=cw, dry_with_container=cd, container_mass=c)["result"]
    assert w == pytest.approx(w_hand)
    dry = ev("earthworks.dry_density", wet_density=wet, moisture_percent=w)["result"]
    assert round(dry, 3) == dry_p
    pct = ev("compaction.field_density", field_dry_density=dry, lab_mdd=2.131)["result"]
    assert round(pct, 1) == pct_p


# ---------------------------------------------------------------------------
# Dynamic compaction vibration radius
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M, H, hand",
    [
        (25, 20, 130 * 500 ** 0.5),  # course: 130 x 22.3607 = 2906.9 m, printed 2.9 km
        (10, 10, 1300.0),            # invented: 130 x sqrt(100) = 1300 m
    ],
)
def test_vibration_radius(M, H, hand):
    out = ev("ground_improvement.dynamic_compaction_vibration_radius", M=M, H=H)
    assert out["result"] == pytest.approx(hand)
    assert out["unit"] == "meter"


# ---------------------------------------------------------------------------
# Truck cycle -> hourly production -> machines (invented cases, by hand)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixed, d_out, v_l, d_back, v_e, hand",
    [
        (12, 5, 45, 5, 50, 12 + 60 * 5 / 45 + 60 * 5 / 50),  # 12 + 6.6667 + 6 = 24.6667 min
        (10, 3, 25, 3, 30, 10 + 7.2 + 6.0),                 # 23.2 min
    ],
)
def test_truck_cycle_time(fixed, d_out, v_l, d_back, v_e, hand):
    out = ev("equipment.truck_cycle_time", fixed_time=fixed, haul_distance=d_out,
             loaded_speed=v_l, return_distance=d_back, empty_speed=v_e)
    assert out["result"] == pytest.approx(hand)
    assert out["unit"] == "minute"


def test_fleet_chain_cycle_to_machine_count():
    # Invented A: cycle 24.6667 min, 18 m3 body -> 60/24.6667 x 18 = 43.784 m3/hr;
    # required 300 m3/hr -> 300/43.784 = 6.852 -> 7 trucks.
    cyc = ev("equipment.truck_cycle_time", fixed_time=12, haul_distance=5, loaded_speed=45,
             return_distance=5, empty_speed=50)["result"]
    prod = ev("equipment.cycle_production", cycle_time=cyc, unit_capacity=18)["result"]
    assert prod == pytest.approx(60 / (74 / 3) * 18)
    n = ev("equipment.machines_required", required_rate=300, production_per_machine=prod)
    assert n["result"] == 7 and n["unrounded"] == pytest.approx(300 / prod)
    # Invented B: cycle 2.5 min, 15 m3 -> 24 x 15 = 360 m3/hr; 1000/360 = 2.778 -> 3.
    prod = ev("equipment.cycle_production", cycle_time=2.5, unit_capacity=15)["result"]
    assert prod == pytest.approx(360)
    n = ev("equipment.machines_required", required_rate=1000, production_per_machine=prod)
    assert n["result"] == 3 and n["unrounded"] == pytest.approx(1000 / 360)


# ---------------------------------------------------------------------------
# Fill layers, compaction water and tanker count (invented, by hand)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vol, area, t, layers, unrounded",
    [
        (60000, 40000, 0.3, 5, 5.0),         # 60000 / 12000 = 5 exactly
        (70000, 40000, 0.3, 6, 70000 / 12000),  # 5.833 -> 6
    ],
)
def test_fill_layers(vol, area, t, layers, unrounded):
    out = ev("earthworks.fill_layers", fill_volume=vol, area=area, layer_thickness=t)
    assert out["result"] == layers
    assert out["unrounded"] == pytest.approx(unrounded)


@pytest.mark.parametrize(
    "q, rho, omc, nmc, hand",
    [
        (2000, 2.0, 12, 4, 320.0),      # 2000 x 2.0 x 8 / 100 = 320 t/day
        (1500, 1.9, 10.5, 3, 213.75),   # 1500 x 1.9 x 7.5 / 100 = 213.75 t/day
    ],
)
def test_compaction_water(q, rho, omc, nmc, hand):
    out = ev("earthworks.compaction_water", daily_quantity=q, density=rho,
             optimum_moisture=omc, natural_moisture=nmc)
    assert out["result"] == pytest.approx(hand)
    assert out["unit"] == "tonne"


def test_tanker_count_chain():
    # 320 t/day over 10 h = 32 t/hr; tanker 12 t, cycle 30 min -> 24 t/hr each;
    # 32 / 24 = 1.333 -> 2 tankers.
    water = ev("earthworks.compaction_water", daily_quantity=2000, density=2.0,
               optimum_moisture=12, natural_moisture=4)["result"]
    per_tanker = ev("equipment.cycle_production", cycle_time=30, unit_capacity=12)["result"]
    assert per_tanker == pytest.approx(24)
    n = ev("equipment.machines_required", required_rate=water / 10, production_per_machine=per_tanker)
    assert n["result"] == 2 and n["unrounded"] == pytest.approx(32 / 24)
    # 213.75 t/day over 8 h = 26.72 t/hr; one 30 t tanker at 60 min -> 30 t/hr: 0.89 -> 1.
    water = ev("earthworks.compaction_water", daily_quantity=1500, density=1.9,
               optimum_moisture=10.5, natural_moisture=3)["result"]
    per_tanker = ev("equipment.cycle_production", cycle_time=60, unit_capacity=30)["result"]
    n = ev("equipment.machines_required", required_rate=water / 8, production_per_machine=per_tanker)
    assert n["result"] == 1 and n["unrounded"] == pytest.approx(213.75 / 8 / 30)
