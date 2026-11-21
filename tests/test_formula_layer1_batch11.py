"""Formula Layer 1, batch 11 (exam 5): unit weight, embodied carbon, cost build-ups, striking.

Same rules as batches 1-10 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation, written in the comment beside
its parametrize row:
  * concrete unit weight         the calculator's stated table: reinforced
                                 2500 kg/m3, plain 2400 kg/m3 (2330-2470)
  * embodied carbon of concrete  total = volume x factor; factor from the
                                 calculator's stated table by grade (c20 260,
                                 c25 290, c30 320, c35 350, c40 390 kgCO2e/m3)
                                 or the EPD figure when one is given
  * resource line cost           crew days = Q / daily output;
                                 labour = crew days x day rate;
                                 plant = labour x plant fraction;
                                 material = Q x material rate;
                                 total = labour + plant + material
  * formwork striking time       strength ok = strength >= CIRIA surface
                                 minimum AND >= design required strength;
                                 ok -> proposed hours; not ok ->
                                 max(proposed, BS 8110 minimum hours)
  * rebar cost build-up (per t)  material = price x (1 + waste);
                                 labour = mhr/t x rate; plant = crane h/t x
                                 rate; sell = direct x (1 + indirect) /
                                 (1 - markup); total = sell x tonnes
  * formwork cost build-up       shuttering = supply / 6 uses (stated);
    (per m2)                     scaffold = rate/m2/day x cycle days;
                                 labour = mhr/m2 x rate; plant = crane rate /
                                 crane output; sell = direct x (1 + indirect)
                                 / (1 - markup); total = sell x area
None was copied from the product's output. A case where the product
disagrees with the hand value would be marked ``xfail(strict=True)`` with a
one-line reason and reported as a suspected defect; the expected value is
never moved to match. Every value test runs on two inputs worked by hand
the same way.

No required-input M case for F68 and F71: every input has a stated default,
so nothing can be missing. Their M case is the OMITTED input instead: the
stated default is used and must be reported in ``defaults_used``, and a given
input must not be (``test_f68_m_*``, ``test_f71_m_*``). No U case for
F68 (a yes/no input) or F70 (counts and currency in the project's own unit).

Unit factors (exact by definition): 1 ft = 0.3048 m, so 1 ft2 =
0.09290304 m2 and 1 ft3 = 0.028316846592 m3; 1 t = 1000 kg.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 1e-3   # plan tolerance: 0.1 %


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", env
    return env["result"]


def _asks(name: str, params: dict, missing: str, withheld: tuple[str, ...]) -> dict:
    """The call must ask for ``missing`` and withhold every key in ``withheld``."""
    env = run_calculation(name, params)
    assert env.get("status") in ("needs_input", "error"), env
    text = " ".join(str(env.get(k) or "") for k in ("error", "question", "message", "note"))
    text += " " + " ".join(str(m) for m in (env.get("missing") or []))
    assert missing.lower() in text.lower(), (missing, env)
    result = env.get("result") or {}
    for key in withheld:
        assert result.get(key) is None, (key, env)
    return env


def _refuses(name: str, params: dict, withheld: tuple[str, ...]) -> dict:
    """A refusal: not success, no number in the withheld keys, no exception."""
    env = run_calculation(name, params)
    assert env.get("status") != "success", env
    result = env.get("result") or {}
    for key in withheld:
        assert result.get(key) is None, (key, env)
    return env


# ---------------------------------------------------------------- F68 unit weight
F68 = "unit_weight_concrete"


@pytest.mark.parametrize("reinforced, weight", [
    (True, 2500),    # stated table: reinforced concrete 2500 kg/m3
    (False, 2400),   # stated table: plain concrete 2400 kg/m3
])
def test_f68_t(reinforced, weight):
    r = _ok(F68, {"reinforced": reinforced})
    assert r["unit_weight_kg_m3"] == pytest.approx(weight, rel=_REL)


@pytest.mark.parametrize("reinforced", [False, True])
def test_f68_b_plain_inside_stated_range_reinforced_above(reinforced):
    # stated plain range 2330-2470 kg/m3: 2400 sits inside it, 2500 above it
    r = _ok(F68, {"reinforced": reinforced})
    inside = 2330 <= r["unit_weight_kg_m3"] <= 2470
    assert inside is (not reinforced)


# F68 declares one input, ``reinforced``, and it is optional (it has a
# declared default), so there is no required input to withhold. The M case
# here is the omitted input: the declared default must be used AND disclosed
# in ``defaults_used``; a given input must not be listed as defaulted.
@pytest.mark.parametrize("params, defaulted, weight", [
    ({}, {"reinforced"}, 2500),             # omitted -> declared default True -> reinforced 2500
    ({"reinforced": False}, set(), 2400),   # given -> nothing defaulted -> plain 2400
])
def test_f68_m_omitted_input_defaults_disclosed(params, defaulted, weight):
    r = _ok(F68, params)
    listed = {row["parameter"] for row in r.get("defaults_used") or []}
    assert listed == defaulted, r
    assert r["unit_weight_kg_m3"] == pytest.approx(weight, rel=_REL)


# ---------------------------------------------------------------- F69 embodied carbon
F69 = "carbon_footprint_concrete"
_KG = "total_kgco2e"


@pytest.mark.parametrize("vol, grade, kg, t", [
    (100, "c40", 39000.0, 39.0),   # 100 x 390 = 39 000 kg = 39.0 t
    (25, "c20", 6500.0, 6.5),      # 25 x 260 = 6 500 kg = 6.5 t
])
def test_f69_t(vol, grade, kg, t):
    r = _ok(F69, {"volume_m3": vol, "grade": grade})
    assert r[_KG] == pytest.approx(kg, rel=_REL)
    assert r["total_tco2e"] == pytest.approx(t, rel=_REL)


@pytest.mark.parametrize("vol, grade, epd, kg", [
    (50, "c30", 210, 10500.0),   # EPD governs: 50 x 210 = 10 500 (table c30 would give 16 000)
    (12, "c45", 400, 4800.0),    # EPD for a grade not in the table: 12 x 400 = 4 800
])
def test_f69_t_epd_governs(vol, grade, epd, kg):
    r = _ok(F69, {"volume_m3": vol, "grade": grade, "embodied_kgco2e_m3": epd})
    assert r[_KG] == pytest.approx(kg, rel=_REL)


@pytest.mark.parametrize("vol, grade, kg", [
    # 10 m3 = 10 / 0.028316846592 = 353.1466672 ft3; 10 x 320 = 3 200
    ("353.1466672 ft3", "c30", 3200.0),
    # 4 m3 = 141.2586669 ft3; 4 x 350 = 1 400
    ("141.2586669 ft3", "c35", 1400.0),
])
def test_f69_u(vol, grade, kg):
    r = _ok(F69, {"volume_m3": vol, "grade": grade})
    assert r[_KG] == pytest.approx(kg, rel=_REL)


@pytest.mark.parametrize("grade", ["c30", "c40"])
def test_f69_m_no_volume(grade):
    _asks(F69, {"grade": grade}, "volume", (_KG, "total_tco2e"))


@pytest.mark.parametrize("grade", ["c25", "c40"])
def test_f69_b_zero_volume(grade):
    r = _ok(F69, {"volume_m3": 0, "grade": grade})   # 0 x factor = 0
    assert r[_KG] == 0.0


@pytest.mark.parametrize("params", [
    {"volume_m3": -1, "grade": "c30"},   # negative volume
    {"volume_m3": 10, "grade": "c45"},   # grade not in the table, no EPD figure
    {"volume_m3": 10, "grade": "c15"},   # grade not in the table, no EPD figure
])
def test_f69_b_refused(params):
    _refuses(F69, params, (_KG, "total_tco2e"))


# ---------------------------------------------------------------- F70 resource line
F70 = "resource_line_cost"


@pytest.mark.parametrize("q, out, rate, plant, mat, days, lab, pl, m, tot", [
    # 100/20 = 5 d; 5 x 500 = 2 500; 0.1 x 2 500 = 250; 100 x 3 = 300; 3 050
    (100, 20, 500, 0.10, 3, 5.0, 2500.0, 250.0, 300.0, 3050.0),
    # 450/30 = 15 d; 15 x 800 = 12 000; 0.25 x 12 000 = 3 000; 450 x 12 = 5 400; 20 400
    (450, 30, 800, 0.25, 12, 15.0, 12000.0, 3000.0, 5400.0, 20400.0),
])
def test_f70_t(q, out, rate, plant, mat, days, lab, pl, m, tot):
    r = _ok(F70, {"quantity": q, "daily_output": out, "day_rate": rate,
                  "plant_fraction_of_labour": plant, "material_rate_per_unit": mat})
    assert r["crew_days"] == pytest.approx(days, rel=_REL)
    assert r["labour_cost"] == pytest.approx(lab, rel=_REL)
    assert r["plant_cost"] == pytest.approx(pl, rel=_REL)
    assert r["material_cost"] == pytest.approx(m, rel=_REL)
    assert r["total_cost"] == pytest.approx(tot, rel=_REL)
    assert r["total_is_partial"] is False


@pytest.mark.parametrize("q, out, rate, plant, tot", [
    (100, 20, 500, 0.10, 2750.0),   # no material rate: 2 500 + 250, partial
    (60, 12, 400, 0.0, 2000.0),     # no material rate: 60/12 = 5 d x 400 = 2 000, partial
])
def test_f70_m_no_material_rate_is_partial(q, out, rate, plant, tot):
    r = _ok(F70, {"quantity": q, "daily_output": out, "day_rate": rate,
                  "plant_fraction_of_labour": plant})
    assert r["material_cost"] == "NO SOURCED RATE"
    assert r["total_cost"] == pytest.approx(tot, rel=_REL)
    assert r["total_is_partial"] is True


@pytest.mark.parametrize("params, missing", [
    ({"quantity": 100, "day_rate": 500}, "daily_output"),
    ({"quantity": 100, "daily_output": 20}, "day_rate"),
])
def test_f70_m_required(params, missing):
    _asks(F70, params, missing, ("labour_cost", "total_cost"))


@pytest.mark.parametrize("params", [
    {"quantity": 100, "daily_output": 0, "day_rate": 500},   # zero output
    {"quantity": 100, "daily_output": 20, "day_rate": 0},    # zero day rate
])
def test_f70_b_zero_rate_refused(params):
    _refuses(F70, params, ("labour_cost", "total_cost"))


@pytest.mark.parametrize("q, out, rate, tot", [
    (100, 20, 500, 2500.0),   # material rate 0: 100 x 0 = 0; total = labour 2 500
    (40, 8, 300, 1500.0),     # material rate 0: 40/8 = 5 d x 300 = 1 500
])
def test_f70_b_zero_material_rate_is_complete(q, out, rate, tot):
    r = _ok(F70, {"quantity": q, "daily_output": out, "day_rate": rate,
                  "material_rate_per_unit": 0})
    assert r["material_cost"] == 0.0
    assert r["total_cost"] == pytest.approx(tot, rel=_REL)
    assert r["total_is_partial"] is False


# ---------------------------------------------------------------- F71 striking time
F71 = "formwork_striking_time"
_H = "recommended_hours"


@pytest.mark.parametrize("fc, req, ciria, bs, prop, hours, basis", [
    (5.0, 0.332, 3.0, 12, 8, 8.0, "test_data"),       # 5 >= 3 and >= 0.332 -> proposed 8
    (2.0, 0.332, 3.0, 12, 8, 12.0, "insufficient"),   # 2 < 3 -> max(8, 12) = 12
    (5.0, 6.0, 3.0, 12, 8, 12.0, "insufficient"),     # 5 < 6 design -> max(8, 12) = 12
    (2.0, 0.332, 3.0, 12, 16, 16.0, "insufficient"),  # 2 < 3 -> max(16, 12) = 16
    (4.0, 1.0, 3.5, 18, 10, 10.0, "test_data"),       # 4 >= 3.5 and >= 1 -> proposed 10
])
def test_f71_t(fc, req, ciria, bs, prop, hours, basis):
    r = _ok(F71, {"concrete_strength_7h": fc, "design_required_strength": req,
                  "ciria_surface_strength": ciria, "bs8110_minimum_hours": bs,
                  "proposed_hours": prop})
    assert r[_H] == pytest.approx(hours, rel=_REL)
    assert r["based_on"] == basis


@pytest.mark.parametrize("fc, hours", [
    (3.0, 8.0),    # on the CIRIA limit: 3.0 >= 3.0 -> proposed 8
    (2.99, 12.0),  # just below: max(8, 12) = 12
])
def test_f71_b_ciria_limit(fc, hours):
    r = _ok(F71, {"concrete_strength_7h": fc, "design_required_strength": 0.332,
                  "ciria_surface_strength": 3.0, "bs8110_minimum_hours": 12,
                  "proposed_hours": 8})
    assert r[_H] == pytest.approx(hours, rel=_REL)


# F71 declares five inputs, all optional (each has a declared default:
# strength 7.0, design required 0.332, CIRIA 3.0, minimum 12 h, proposed 8 h),
# so there is no required input to withhold. The M case is the omitted input:
# every omitted one must be listed in ``defaults_used`` and the hours worked
# from its declared default; a given input must not be listed.
_F71_ALL = {"concrete_strength_7h", "design_required_strength",
            "ciria_surface_strength", "bs8110_minimum_hours", "proposed_hours"}


@pytest.mark.parametrize("params, hours", [
    # strength 2.0 given; CIRIA defaults to 3.0: 2 < 3 -> max(8, 12) = 12
    ({"concrete_strength_7h": 2.0}, 12.0),
    # only CIRIA omitted (3.0): 5 >= 3 and 5 >= 0.332 -> proposed 16
    ({"concrete_strength_7h": 5.0, "design_required_strength": 0.332,
      "bs8110_minimum_hours": 12, "proposed_hours": 16}, 16.0),
    # nothing given: 7.0 >= 3.0 and >= 0.332 -> proposed default 8
    ({}, 8.0),
])
def test_f71_m_omitted_inputs_defaults_disclosed(params, hours):
    r = _ok(F71, params)
    listed = {row["parameter"] for row in r.get("defaults_used") or []}
    assert listed == _F71_ALL - set(params), r
    assert r[_H] == pytest.approx(hours, rel=_REL)


# ---------------------------------------------------------------- F72 rebar build-up
F72 = "cost_buildup_rebar"
_SELL = "selling_price_per_t"
_BASE72 = {"material_price_per_t": 2000, "labour_rate_per_hr": 10, "crane_rate_per_hr": 100}


@pytest.mark.parametrize("params, mat, lab, eq, sell, total", [
    # 1 t; 2000 x 1.10 = 2 200; 90 x 10 = 900; 2 x 100 = 200; direct 3 300;
    # 3 300 x 1.18 / 0.85 = 4 581.2 -> 4 581; x 1 t = 4 581
    ({"quantity_kg": 1000, **_BASE72}, 2200, 900, 200, 4581, 4581),
    # 2.5 t; 3000 x 1.05 = 3 150; 80 x 12 = 960; 1.5 x 150 = 225; direct 4 335;
    # 4 335 x 1.2 / 0.9 = 5 780; x 2.5 t = 14 450
    ({"quantity_kg": 2500, "material_price_per_t": 3000, "labour_mhr_t": 80,
      "labour_rate_per_hr": 12, "crane_hr_t": 1.5, "crane_rate_per_hr": 150,
      "waste_pct": 0.05, "indirect_pct": 0.20, "markup_pct": 0.10},
     3150, 960, 225, 5780, 14450),
])
def test_f72_t(params, mat, lab, eq, sell, total):
    r = _ok(F72, params)
    assert r["material_per_t"] == pytest.approx(mat, rel=_REL)
    assert r["labour_per_t"] == pytest.approx(lab, rel=_REL)
    assert r["equipment_per_t"] == pytest.approx(eq, rel=_REL)
    assert r[_SELL] == pytest.approx(sell, rel=_REL)
    assert r["total_project_value"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("qty, total", [
    ("2.5 t", 11452.9),   # 2.5 t = 2 500 kg; 4 581.18 x 2.5 = 11 452.9
    ("4 t", 18324.7),     # 4 t = 4 000 kg; 4 581.18 x 4 = 18 324.7
])
def test_f72_u(qty, total):
    r = _ok(F72, {"quantity_kg": qty, **_BASE72})
    assert r["total_project_value"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("drop", ["crane_rate_per_hr", "labour_rate_per_hr"])
def test_f72_m_required(drop):
    params = {"quantity_kg": 1000, **_BASE72}
    del params[drop]
    _asks(F72, params, drop, (_SELL, "total_project_value"))


@pytest.mark.parametrize("qty", [0, -5])
def test_f72_b_nonpositive_quantity_refused(qty):
    _refuses(F72, {"quantity_kg": qty, **_BASE72}, (_SELL, "total_project_value"))


@pytest.mark.parametrize("indirect, sell", [
    (0.18, 3894),   # markup 0: 3 300 x 1.18 / 1 = 3 894
    (0.0, 3300),    # markup 0, indirect 0: sell = direct 3 300
])
def test_f72_b_zero_markup(indirect, sell):
    r = _ok(F72, {"quantity_kg": 1000, **_BASE72, "indirect_pct": indirect, "markup_pct": 0})
    assert r[_SELL] == pytest.approx(sell, rel=_REL)


# ---------------------------------------------------------------- F73 formwork build-up
F73 = "cost_buildup_formwork"
_SELLM2 = "selling_price_per_m2"
_BASE73 = {"shuttering_supply_per_m2": 50, "scaffolding_per_m2_day": 1,
           "labour_rate_per_hr": 10, "crane_rate_per_hr": 200}


@pytest.mark.parametrize("params, shut, scaf, lab, sell, total", [
    # 50/6 = 8.333; 1 x 10 d = 10; 9 x 10 = 90; 200/30 = 6.667; direct 115;
    # 115 x 1.18 / 0.85 = 159.647; x 100 m2 = 15 965
    ({"area_m2": 100, **_BASE73}, 8.333, 10.0, 90.0, 159.647, 15965),
    # 120/6 = 20; 0.5 x 14 = 7; 6 x 15 = 90; 300/25 = 12; direct 129;
    # 129 x 1.1 / 0.8 = 177.375; x 250 m2 = 44 343.75
    ({"area_m2": 250, "shuttering_supply_per_m2": 120, "scaffolding_per_m2_day": 0.5,
      "cycle_days": 14, "labour_mhr_m2": 6, "labour_rate_per_hr": 15,
      "crane_rate_per_hr": 300, "crane_output_m2_hr": 25,
      "indirect_pct": 0.10, "markup_pct": 0.20},
     20.0, 7.0, 90.0, 177.375, 44343.75),
])
def test_f73_t(params, shut, scaf, lab, sell, total):
    r = _ok(F73, params)
    assert r["shuttering_per_m2"] == pytest.approx(shut, rel=_REL)
    assert r["scaffolding_per_m2"] == pytest.approx(scaf, rel=_REL)
    assert r["labour_per_m2"] == pytest.approx(lab, rel=_REL)
    assert r[_SELLM2] == pytest.approx(sell, rel=_REL)
    assert r["total_for_area"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("area, total", [
    ("1076.391042 ft2", 15965),   # 100 m2 = 100 / 0.09290304 ft2; 159.647 x 100
    ("2152.782084 ft2", 31929),   # 200 m2; 159.647 x 200 = 31 929.4
])
def test_f73_u(area, total):
    r = _ok(F73, {"area_m2": area, **_BASE73})
    assert r["total_for_area"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("drop", ["crane_rate_per_hr", "shuttering_supply_per_m2"])
def test_f73_m_required(drop):
    params = {"area_m2": 100, **_BASE73}
    del params[drop]
    _asks(F73, params, drop, (_SELLM2, "total_for_area"))


@pytest.mark.parametrize("extra", [{}, {"cycle_days": 20}])
def test_f73_b_zero_area(extra):
    # area 0: the rate per m2 still stands; the total is 0 x rate = 0
    r = _ok(F73, {"area_m2": 0, **_BASE73, **extra})
    assert r[_SELLM2] > 0
    assert r["total_for_area"] == 0


@pytest.mark.parametrize("params", [
    {"area_m2": 100, **_BASE73, "crane_output_m2_hr": 0},   # plant rate / 0
    {"area_m2": -1, **_BASE73},                              # negative area
])
def test_f73_b_refused(params):
    _refuses(F73, params, (_SELLM2, "total_for_area"))
