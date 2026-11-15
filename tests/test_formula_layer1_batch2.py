"""Formula Layer 1, batch 2: ten more formulas, four case kinds each.

Same rules as batch 1 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it (status
     ``needs_input`` or ``error`` naming the input) and must NOT return the
     figure it would have needed that input for;
  B  boundary -- zero, a value on the limit, rounding direction.

Every expected value is a HAND calculation from the published definition,
written in the comment beside its parametrize row. None was copied from the
product's output. A case where the product disagrees with the hand value is
marked ``xfail(strict=True)`` with a one-line reason and is a suspected
formula defect for the advisor; the expected value is never moved to match.
Every value test runs on two inputs: the plan's case and a second invented
case worked by hand the same way.

Where a formula fills a missing input from a signature default, two things
are tested separately: that the default is FLAGGED (named in
``defaults_used`` -- an assumption must be named, never silent), and, as a
strict expected failure, that the plan's "ask, do not guess" rule is not yet
met for an input that has no code-given value.

Unit factors used in the hand values (exact by definition unless noted):
  1 ft = 0.3048 m; 1 in = 25.4 mm; 1 lb = 0.45359237 kg;
  1 kip = 4.4482216 kN; 1 psi = 6.894757 kPa; 1 mph = 0.44704 m/s;
  1 ft2 = 0.09290304 m2; 1 pcf = 0.01601846 t/m3; 1 km/h = 1/3.6 m/s.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 1e-3  # plan tolerance: 0.1 % unless a case states otherwise


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


def _flagged_default(result: dict, param: str) -> None:
    """A default the calculator used must be named in ``defaults_used``."""
    used = {str(d.get("parameter")) for d in (result.get("defaults_used") or [])}
    assert param in used, (param, result)


# ------------------------------------------------------- F10 beam deflection
# Simply supported, UDL: delta = 5 w L^4 / (384 E I); w in kN/m = N/mm, L in mm.

@pytest.mark.parametrize("w, span, ec, i, delta", [
    (10, 5, 30_000, 8e8, 3.391),     # 5x10x5000^4 = 3.125e16; 384x30000x8e8 = 9.216e15; 3.391 mm
    (20, 6, 25_000, 1.2e9, 11.25),   # 5x20x6000^4 = 1.296e17; 384x25000x1.2e9 = 1.152e16; 11.25 mm
])
def test_f10_t_deflection_typical(w, span, ec, i, delta):
    r = _ok("beam_deflection_ss_udl", {"w_kn_m": w, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, rel=5e-3)  # product rounds to 0.01 mm


@pytest.mark.parametrize("w, span, ec, i, delta", [
    (10, "15 ft", 30_000, 8e8, 2.371),    # 15 ft = 4.572 m; 5x10x4572^4 / 9.216e15 = 2.3706 mm
    (20, "20 ft", 25_000, 1.2e9, 11.987),  # 20 ft = 6.096 m; 5x20x6096^4 / 1.152e16 = 11.9875 mm
])
def test_f10_u_deflection_span_in_feet(w, span, ec, i, delta):
    r = _ok("beam_deflection_ss_udl", {"w_kn_m": w, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, rel=5e-3)


@pytest.mark.parametrize("params, missing", [
    ({"w_kn_m": 10, "span_m": 5, "ec_mpa": 30_000}, "i_mm4"),  # no I
    ({"w_kn_m": 10, "span_m": 5, "i_mm4": 8e8}, "ec_mpa"),     # no E
])
def test_f10_m_deflection_missing_stiffness_asks(params, missing):
    _asks("beam_deflection_ss_udl", params, missing, ("value",))


@pytest.mark.parametrize("w, span", [
    (0, 5),   # w = 0: 5 x 0 x L^4 / (384EI) = 0
    (10, 0),  # L = 0: 5 x w x 0 / (384EI) = 0
])
def test_f10_b_deflection_zero_load_or_span_is_zero(w, span):
    r = _ok("beam_deflection_ss_udl", {"w_kn_m": w, "span_m": span, "ec_mpa": 30_000, "i_mm4": 8e8})
    assert r["value"] == 0


# ---------------------------------------------------------- F11 wind pressure
# EN 1991-1-4: qb = 0.5 rho v^2.  ASCE 7-16 26.10 (SI), all factors 1: q = 0.613 V^2.

@pytest.mark.parametrize("v, rho, q", [
    (40, 1.225, 980.0),  # 0.5 x 1.225 x 1600 = 980 Pa (plan L2-04)
    (30, 1.25, 562.5),   # 0.5 x 1.25 x 900 = 562.5 Pa
])
def test_f11_t_wind_basic_pressure(v, rho, q):
    r = _ok("wind_pressure", {"wind_speed_m_s": v, "code": "eurocode", "air_density_kg_m3": rho})
    assert r["velocity_pressure_pa"] == pytest.approx(q, rel=_REL)
    assert r["velocity_pressure_kn_m2"] == pytest.approx(q / 1000, rel=_REL)


@pytest.mark.parametrize("v, q", [
    ("144 km/h", 980.8),  # 144 / 3.6 = 40 m/s; 0.613 x 1600 = 980.8 Pa
    ("90 mph", 992.29),   # 90 x 0.44704 = 40.2336 m/s; 0.613 x 1618.743 = 992.29 Pa
])
def test_f11_u_wind_speed_in_other_units(v, q):
    r = _ok("wind_pressure", {"wind_speed_m_s": v})
    assert r["velocity_pressure_pa"] == pytest.approx(q, rel=_REL)


@pytest.mark.parametrize("params", [
    {"code": "eurocode", "air_density_kg_m3": 1.225},
    {"kz": 0.85, "kzt": 1.0},
])
def test_f11_m_wind_missing_speed_asks(params):
    _asks("wind_pressure", params, "wind_speed_m_s", ("velocity_pressure_pa",))


@pytest.mark.parametrize("v, q", [
    (0, 0.0),        # V = 0: 0.613 x 0 = 0
    (120, 8827.2),   # on the input limit 120 m/s: 0.613 x 14,400 = 8,827.2 Pa
])
def test_f11_b_wind_zero_and_on_limit(v, q):
    r = _ok("wind_pressure", {"wind_speed_m_s": v})
    assert r["velocity_pressure_pa"] == pytest.approx(q, rel=_REL, abs=0.05)


@pytest.mark.parametrize("v", [121, 250])
def test_f11_b2_wind_above_limit_refused(v):
    _refuses("wind_pressure", {"wind_speed_m_s": v}, ("velocity_pressure_pa",))


# ---------------------------------------------------------- F12 seismic shear
# ASCE 7-16 12.8.1.1: Cs = SDS x Ie / R; V = Cs x W.

@pytest.mark.parametrize("w, sds, r_, ie, cs, v", [
    (20_000, 0.6, 8, 1.0, 0.075, 1_500),    # 0.6 x 1.0 / 8 = 0.075; 0.075 x 20,000 = 1,500 kN (plan L2-05)
    (12_000, 0.9, 6, 1.25, 0.1875, 2_250),  # 0.9 x 1.25 / 6 = 0.1875; 0.1875 x 12,000 = 2,250 kN
])
def test_f12_t_seismic_typical(w, sds, r_, ie, cs, v):
    r = _ok("seismic_base_shear", {"seismic_weight_kn": w, "sds": sds, "r": r_, "ie": ie})
    assert r["seismic_response_coefficient_cs"] == pytest.approx(cs, rel=_REL)
    assert r["base_shear_kn"] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("w, sds, r_, v", [
    ("4500 kip", 0.6, 8, 1501.27),  # 4,500 x 4.4482216 = 20,016.997 kN; x 0.075 = 1,501.27 kN
    ("1000 kip", 1.0, 5, 889.64),   # 1,000 x 4.4482216 = 4,448.22 kN; Cs = 1.0/5 = 0.2; 889.64 kN
])
def test_f12_u_seismic_weight_in_kips(w, sds, r_, v):
    r = _ok("seismic_base_shear", {"seismic_weight_kn": w, "sds": sds, "r": r_, "ie": 1.0})
    assert r["base_shear_kn"] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("params", [
    {"sds": 0.6, "r": 8, "ie": 1.0},
    {"sds": 0.9, "r": 6, "ie": 1.25},
])
def test_f12_m_seismic_missing_weight_asks(params):
    _asks("seismic_base_shear", params, "seismic_weight_kn", ("base_shear_kn",))


@pytest.mark.parametrize("w", [20_000, 12_000])
def test_f12_m2_seismic_default_sds_is_flagged(w):
    # Without SDS the calculator fills a default; it must at least be named.
    r = _ok("seismic_base_shear", {"seismic_weight_kn": w, "r": 8, "ie": 1.0})
    _flagged_default(r, "sds")


@pytest.mark.xfail(strict=True, reason="suspected defect: SDS (a site value with no code default) "
                   "is filled from a signature default of 1.0 instead of asked (plan rule: ask, do not guess)")
@pytest.mark.parametrize("w", [20_000, 12_000])
def test_f12_m3_seismic_missing_sds_asks(w):
    # A default SDS of 1.0 gives Cs = 1.0/8 = 0.125 and V = 2,500 kN for W = 20,000:
    # 67 % above the plan's 1,500 kN at SDS 0.6. Must ask for SDS instead.
    _asks("seismic_base_shear", {"seismic_weight_kn": w, "r": 8, "ie": 1.0}, "sds", ("base_shear_kn",))


@pytest.mark.parametrize("r_, ie, cs", [
    (8, 1.5, 0.1125),  # both on their input limits: 0.6 x 1.5 / 8 = 0.1125
    (1, 1.0, 0.6),     # R on its lower limit: 0.6 x 1.0 / 1 = 0.6
])
def test_f12_b_seismic_factors_on_limits(r_, ie, cs):
    r = _ok("seismic_base_shear", {"seismic_weight_kn": 10_000, "sds": 0.6, "r": r_, "ie": ie})
    assert r["seismic_response_coefficient_cs"] == pytest.approx(cs, rel=_REL)
    assert r["base_shear_kn"] == pytest.approx(cs * 10_000, rel=_REL)


@pytest.mark.parametrize("r_", [0.5, 9])
def test_f12_b2_seismic_r_outside_range_refused(r_):
    _refuses("seismic_base_shear", {"seismic_weight_kn": 10_000, "sds": 0.6, "r": r_, "ie": 1.0},
             ("base_shear_kn",))


# ------------------------------------------------------ F13 concrete cylinder
# f = P / A; A = pi/4 x d^2.

@pytest.mark.parametrize("p, d, f", [
    (450, 150, 25.46),  # A = 17,671.46 mm2; 450,000 / 17,671.46 = 25.465 MPa (plan L2-07)
    (300, 100, 38.20),  # A = 7,853.98 mm2; 300,000 / 7,853.98 = 38.197 MPa
])
def test_f13_t_cylinder_typical(p, d, f):
    r = _ok("concrete_cylinders", {"failure_load_kn": p, "cylinder_diameter_mm": d})
    assert r["compressive_strength_mpa"] == pytest.approx(f, rel=_REL)


@pytest.mark.parametrize("p, d, f", [
    ("100 kip", "6 in", 24.39),  # 444.822 kN; 152.4 mm; A = 18,241.5 mm2; 444,822 / 18,241.5 = 24.385
    ("50 kip", "4 in", 27.43),   # 222.411 kN; 101.6 mm; A = 8,107.3 mm2; 222,411 / 8,107.3 = 27.433
])
def test_f13_u_cylinder_imperial(p, d, f):
    r = _ok("concrete_cylinders", {"failure_load_kn": p, "cylinder_diameter_mm": d})
    assert r["compressive_strength_mpa"] == pytest.approx(f, rel=_REL)


@pytest.mark.parametrize("d", [150, 100])
def test_f13_m_cylinder_missing_load_asks(d):
    _asks("concrete_cylinders", {"cylinder_diameter_mm": d}, "failure_load_kn",
          ("compressive_strength_mpa",))


@pytest.mark.parametrize("d", [150, 100])
def test_f13_b_cylinder_zero_load_is_zero(d):
    # P = 0: 0 / A = 0 MPa
    r = _ok("concrete_cylinders", {"failure_load_kn": 0, "cylinder_diameter_mm": d})
    assert r["compressive_strength_mpa"] == 0


@pytest.mark.parametrize("d", [50, 400])
def test_f13_b2_cylinder_non_standard_diameter_refused(d):
    # Outside the standard specimen range 100-300 mm: refuse, no strength.
    _refuses("concrete_cylinders", {"failure_load_kn": 450, "cylinder_diameter_mm": d},
             ("compressive_strength_mpa",))


# --------------------------------------------------------- F14 compaction
# compaction % = field dry density / maximum dry density x 100; pass when >= required.

@pytest.mark.parametrize("field, mdd, pct, passed", [
    (1.82, 1.95, 93.33, False),  # 1.82 / 1.95 = 0.93333 -> 93.33 %; below 95 -> FAIL (plan L2-08)
    (1.90, 1.95, 97.44, True),   # 1.90 / 1.95 = 0.97436 -> 97.44 %; above 95 -> PASS
])
def test_f14_t_compaction_typical(field, mdd, pct, passed):
    r = _ok("compaction_control", {"field_dry_density": field, "max_dry_density": mdd,
                                   "required_compaction_percent": 95})
    assert r["compaction_percent"] == pytest.approx(pct, rel=_REL)
    assert r["passed"] is passed


@pytest.mark.parametrize("field, mdd, pct", [
    ("1820 kg/m3", 1.95, 93.33),  # 1,820 kg/m3 = 1.82 t/m3; 1.82 / 1.95 = 93.33 %
    ("113.6 pcf", 1.95, 93.32),   # 113.6 x 0.01601846 = 1.81970 t/m3; / 1.95 = 93.32 %
])
def test_f14_u_compaction_mixed_density_units(field, mdd, pct):
    r = _ok("compaction_control", {"field_dry_density": field, "max_dry_density": mdd,
                                   "required_compaction_percent": 95})
    assert r["compaction_percent"] == pytest.approx(pct, rel=_REL)
    assert r["passed"] is False


@pytest.mark.parametrize("field", [1.82, 1.90])
def test_f14_m_compaction_missing_mdd_asks(field):
    _asks("compaction_control", {"field_dry_density": field, "required_compaction_percent": 95},
          "max_dry_density", ("compaction_percent", "passed"))


@pytest.mark.parametrize("field, mdd", [
    (1.8525, 1.95),  # 1.8525 / 1.95 = 0.95 exactly: ON the 95 % limit -> PASS (>=)
    (1.71, 1.80),    # 1.71 / 1.80 = 0.95 exactly -> PASS
])
def test_f14_b_compaction_on_the_limit_passes(field, mdd):
    r = _ok("compaction_control", {"field_dry_density": field, "max_dry_density": mdd,
                                   "required_compaction_percent": 95})
    assert r["compaction_percent"] == pytest.approx(95.0, abs=0.005)
    assert r["passed"] is True


# ------------------------------------------------------------ F15 rebar weight
# unit mass = pi/4 x d^2 x 7,850 kg/m3; mass = unit mass x length x bars.

@pytest.mark.parametrize("d, length, qty, kg", [
    (16, 100, 1, 157.83),  # pi/4 x 0.016^2 x 7850 = 1.578336 kg/m; x 100 = 157.83 kg (plan L2-10)
    (25, 12, 10, 462.40),  # pi/4 x 0.025^2 x 7850 = 3.853360 kg/m; x 12 x 10 = 462.40 kg
])
def test_f15_t_rebar_weight_typical(d, length, qty, kg):
    r = _ok("rebar_weight", {"bar_diameter_mm": d, "total_length_m": length, "quantity": qty})
    assert r["total_mass_kg"] == pytest.approx(kg, rel=_REL)


@pytest.mark.parametrize("d, length, kg", [
    (16, "328.084 ft", 157.83),  # 328.084 x 0.3048 = 100.000 m; x 1.578336 = 157.83 kg
    (12, "500 ft", 135.30),      # 152.4 m; pi/4 x 0.012^2 x 7850 = 0.887814 kg/m; x 152.4 = 135.30 kg
])
def test_f15_u_rebar_length_in_feet(d, length, kg):
    r = _ok("rebar_weight", {"bar_diameter_mm": d, "total_length_m": length})
    assert r["total_mass_kg"] == pytest.approx(kg, rel=_REL)


@pytest.mark.parametrize("d", [16, 25])
def test_f15_m_rebar_missing_length_asks(d):
    _asks("rebar_weight", {"bar_diameter_mm": d}, "total_length_m", ("total_mass_kg",))


@pytest.mark.parametrize("d, length", [(16, 100), (25, 12)])
def test_f15_b_rebar_zero_bars_never_one_bar(d, length):
    # quantity = 0: 0 kg, or a refusal (a zero quantity refused is acceptable).
    # Never the one-bar figure (1.578336 x 100 = 157.83 kg for the first row).
    env = run_calculation("rebar_weight", {"bar_diameter_mm": d, "total_length_m": length, "quantity": 0})
    if env.get("status") == "success":
        assert env["result"]["total_mass_kg"] == 0, env
    else:
        assert (env.get("result") or {}).get("total_mass_kg") is None, env


# ------------------------------------------------------------ F16 cost per area
# rate = total cost / area; money to 2 decimals.

@pytest.mark.parametrize("total, area, rate", [
    (12_600_000, 3_500, 3_600.00),  # 12,600,000 / 3,500 = 3,600 (plan L2-11)
    (4_500_000, 1_800, 2_500.00),   # 4,500,000 / 1,800 = 2,500
])
def test_f16_t_cost_per_area_typical(total, area, rate):
    r = _ok("cost_per_area", {"total_cost": total, "area": area})
    assert r["cost_per_area"] == pytest.approx(rate, abs=0.01)


@pytest.mark.parametrize("total, area, per_m2, per_ft2", [
    ("12.6 million", "10000 ft2", 13_562.53, 1_260.00),  # 929.0304 m2; 12.6e6 / 929.0304 = 13,562.53 /m2 (= 1,260 /ft2)
    ("4.5 million", "20000 ft2", 2_421.88, 225.00),      # 1,858.0608 m2; 4.5e6 / 1,858.0608 = 2,421.88 /m2 (= 225 /ft2)
])
def test_f16_u_cost_per_area_area_in_ft2(total, area, per_m2, per_ft2):
    # The answer must be per m2 (the formula's unit), or per ft2 and labelled so.
    r = _ok("cost_per_area", {"total_cost": total, "area": area})
    unit = str(r.get("area_unit") or "").lower().replace(" ", "")
    if unit in ("ft2", "sf", "sqft", "ft²"):
        assert r["cost_per_area"] == pytest.approx(per_ft2, abs=0.01)
    else:
        assert unit in ("m2", "m²"), r
        assert r["cost_per_area"] == pytest.approx(per_m2, abs=0.01)


@pytest.mark.parametrize("total", [12_600_000, 4_500_000])
def test_f16_m_cost_per_area_missing_area_asks(total):
    _asks("cost_per_area", {"total_cost": total}, "area", ("cost_per_area",))


@pytest.mark.parametrize("total", [12_600_000, 4_500_000])
def test_f16_b_cost_per_area_zero_area_refused(total):
    _refuses("cost_per_area", {"total_cost": total, "area": 0}, ("cost_per_area",))


@pytest.mark.parametrize("total, area, rate", [
    (1_000_000, 3, 333_333.33),  # 333,333.333... -> 333,333.33 (minor unit, round half up)
    (100, 7, 14.29),             # 14.2857... -> 14.29
])
def test_f16_b2_cost_per_area_money_rounding(total, area, rate):
    r = _ok("cost_per_area", {"total_cost": total, "area": area})
    assert r["cost_per_area"] == rate


# ----------------------------------------------------------- F17 delay damages
# daily = contract amount x rate %; total = daily x days.

@pytest.mark.parametrize("amount, pct, days, daily, total", [
    (8_000_000, 0.05, 20, 4_000, 80_000),  # 8,000,000 x 0.0005 = 4,000/day; x 20 = 80,000 (plan L2-14)
    (5_000_000, 0.1, 10, 5_000, 50_000),   # 5,000,000 x 0.001 = 5,000/day; x 10 = 50,000
])
def test_f17_t_delay_damages_typical(amount, pct, days, daily, total):
    r = _ok("delay_damages_daily", {"contract_amount": amount, "rate_percent": pct, "days": days})
    assert r["daily_amount"] == pytest.approx(daily, abs=0.01)
    assert r["total_amount"] == pytest.approx(total, abs=0.01)


@pytest.mark.parametrize("amount, pct, daily, total", [
    ("8 million", "0.05 percent", 4_000, 80_000),    # 8,000,000 x 0.0005 x 20
    ("2.5 million", "0.1 percent", 2_500, 50_000),   # 2,500,000 x 0.001 = 2,500; x 20 = 50,000
])
def test_f17_u_delay_damages_written_figures(amount, pct, daily, total):
    r = _ok("delay_damages_daily", {"contract_amount": amount, "rate_percent": pct, "days": 20})
    assert r["daily_amount"] == pytest.approx(daily, abs=0.01)
    assert r["total_amount"] == pytest.approx(total, abs=0.01)


@pytest.mark.parametrize("amount, pct, days, total", [
    (8_000_000, 0.05, "3 weeks", 84_000),  # 3 weeks = 21 days; 4,000 x 21 = 84,000
    (2_500_000, 0.1, "2 weeks", 35_000),   # 2 weeks = 14 days; 2,500 x 14 = 35,000
])
def test_f17_u2_delay_damages_period_in_weeks(amount, pct, days, total):
    r = _ok("delay_damages_daily", {"contract_amount": amount, "rate_percent": pct, "days": days})
    assert r["total_amount"] == pytest.approx(total, abs=0.01)


@pytest.mark.parametrize("pct", [0.05, 0.1])
def test_f17_m_delay_damages_missing_contract_amount_asks(pct):
    _asks("delay_damages_daily", {"rate_percent": pct, "days": 20}, "contract_amount",
          ("daily_amount", "total_amount"))


@pytest.mark.parametrize("amount, pct, daily", [
    (1_234_567, 0.05, 617.28),  # 617.2835 -> 617.28 (minor unit)
    (999_999, 0.03, 300.00),    # 299.9997 -> 300.00
])
def test_f17_b_delay_damages_money_rounding(amount, pct, daily):
    r = _ok("delay_damages_daily", {"contract_amount": amount, "rate_percent": pct})
    assert r["daily_amount"] == daily


@pytest.mark.parametrize("amount", [8_000_000, 5_000_000])
def test_f17_b2_delay_damages_zero_rate_is_zero(amount):
    # rate 0 % against a stated contract amount: 0 per day, 0 total.
    r = _ok("delay_damages_daily", {"contract_amount": amount, "rate_percent": 0, "days": 20})
    assert r["daily_amount"] == 0
    assert r.get("total_amount", 0) == 0


# ------------------------------------------------------- F18 fall arrest force
# energy balance: F = m g (1 + H/d) = m g (H + d) / d; g = 9.81.

@pytest.mark.parametrize("m, h, d, kn", [
    (100, 2, 0.6, 4.251),  # 981 x (1 + 2/0.6) = 981 x 4.3333 = 4,251 N (plan L2-16)
    (80, 1.5, 1.0, 1.962),  # 784.8 x 2.5 = 1,962 N
])
def test_f18_t_fall_arrest_typical(m, h, d, kn):
    r = _ok("fall_arrest_force", {"worker_mass_kg": m, "free_fall_m": h, "deceleration_distance_m": d})
    assert r["max_arrest_force_kn"] == pytest.approx(kn, rel=5e-3)  # product rounds to 0.01 kN


@pytest.mark.parametrize("m, h, d, kn", [
    ("200 lb", "6 ft", "3.5 ft", 2.416),  # 90.7185 kg x 9.81 = 889.95 N; 1 + 6/3.5 = 2.71429; 2,415.6 N
    ("250 lb", "4 ft", "2 ft", 3.337),    # 113.398 kg x 9.81 = 1,112.44 N; x (1 + 2) = 3,337.3 N
])
def test_f18_u_fall_arrest_imperial(m, h, d, kn):
    r = _ok("fall_arrest_force", {"worker_mass_kg": m, "free_fall_m": h, "deceleration_distance_m": d})
    assert r["max_arrest_force_kn"] == pytest.approx(kn, rel=5e-3)


@pytest.mark.parametrize("m", [100, 80])
def test_f18_m_fall_arrest_default_free_fall_is_flagged(m):
    r = _ok("fall_arrest_force", {"worker_mass_kg": m, "deceleration_distance_m": 0.6})
    _flagged_default(r, "free_fall_m")


@pytest.mark.xfail(strict=True, reason="suspected defect: free_fall_m (a site value) is filled from a "
                   "signature default of 1.8 m instead of asked (plan rule: ask, do not guess)")
@pytest.mark.parametrize("m", [100, 80])
def test_f18_m2_fall_arrest_missing_free_fall_asks(m):
    # Default 1.8 m with d = 0.6 gives 981 x 4 = 3,924 N; the plan's 2 m gives 4,251 N.
    _asks("fall_arrest_force", {"worker_mass_kg": m, "deceleration_distance_m": 0.6},
          "free_fall_m", ("max_arrest_force_kn",))


@pytest.mark.xfail(strict=True, reason="suspected defect: worker_mass_kg is filled from a signature "
                   "default of 100 kg instead of asked (plan rule: ask, do not guess)")
@pytest.mark.parametrize("h", [2, 1.5])
def test_f18_m3_fall_arrest_missing_mass_asks(h):
    _asks("fall_arrest_force", {"free_fall_m": h, "deceleration_distance_m": 0.6},
          "worker_mass_kg", ("max_arrest_force_kn",))


@pytest.mark.parametrize("d, within", [
    (0.25, False),  # 981 x (1 + 2/0.25) = 981 x 9 = 8,829 N > 8 kN limit
    (0.3, True),    # 981 x (1 + 2/0.3) = 981 x 7.6667 = 7,521 N <= 8 kN
])
def test_f18_b_fall_arrest_either_side_of_osha_limit(d, within):
    r = _ok("fall_arrest_force", {"worker_mass_kg": 100, "free_fall_m": 2, "deceleration_distance_m": d})
    assert r["within_osha_limit"] is within


@pytest.mark.parametrize("m, kn", [
    (100, 0.981),  # H = 0: F = m g = 981 N
    (80, 0.785),   # 784.8 N
])
def test_f18_b2_fall_arrest_zero_free_fall_is_body_weight(m, kn):
    r = _ok("fall_arrest_force", {"worker_mass_kg": m, "free_fall_m": 0, "deceleration_distance_m": 0.5})
    assert r["max_arrest_force_kn"] == pytest.approx(kn, abs=0.006)


@pytest.mark.parametrize("m", [100, 80])
def test_f18_b3_fall_arrest_zero_deceleration_refused(m):
    _refuses("fall_arrest_force", {"worker_mass_kg": m, "free_fall_m": 2, "deceleration_distance_m": 0},
             ("max_arrest_force_kn",))


# --------------------------------------------------- F19 modulus of rupture
# ACI 318-19 19.2.3.1: fr = 0.62 sqrt(f'c) MPa.
# metric-technical: fr = 2.4 sqrt(f'c in kg/cm2) kg/cm2, / 10 -> MPa (f'c kg/cm2 = 10 x MPa).

@pytest.mark.parametrize("fck, fr", [
    (30, 3.396),  # 0.62 x 5.47723 = 3.3959 MPa
    (40, 3.921),  # 0.62 x 6.32456 = 3.9212 MPa
])
def test_f19_t_rupture_aci(fck, fr):
    r = _ok("modulus_of_rupture", {"fck_n_mm2": fck, "code": "ACI 318"})
    assert r["modulus_of_rupture_n_mm2"] == pytest.approx(fr, rel=_REL)


@pytest.mark.parametrize("fck, fr", [
    (30, 4.157),  # 2.4 x sqrt(300) / 10 = 2.4 x 17.3205 / 10 = 4.1569
    (40, 4.800),  # 2.4 x sqrt(400) / 10 = 2.4 x 20 / 10 = 4.8
])
def test_f19_t2_rupture_metric_technical(fck, fr):
    r = _ok("modulus_of_rupture", {"fck_n_mm2": fck})
    assert r["modulus_of_rupture_n_mm2"] == pytest.approx(fr, rel=_REL)


@pytest.mark.parametrize("fck, fr", [
    ("4351 psi", 3.396),  # 4,351 x 0.006894757 = 29.999 MPa; 0.62 x 5.47714 = 3.3958
    ("5000 psi", 3.640),  # 34.4738 MPa; 0.62 x 5.87144 = 3.6403
])
def test_f19_u_rupture_strength_in_psi(fck, fr):
    r = _ok("modulus_of_rupture", {"fck_n_mm2": fck, "code": "ACI 318"})
    assert r["modulus_of_rupture_n_mm2"] == pytest.approx(fr, rel=_REL)


@pytest.mark.parametrize("code", ["ACI 318", "metric_technical"])
def test_f19_m_rupture_missing_strength_asks(code):
    _asks("modulus_of_rupture", {"code": code}, "fck_n_mm2", ("modulus_of_rupture_n_mm2",))


@pytest.mark.parametrize("code, fr", [
    ("ACI 318", 1.386),        # on the lower input limit 5 MPa: 0.62 x 2.23607 = 1.3864
    ("metric_technical", 1.697),  # 2.4 x sqrt(50) / 10 = 2.4 x 7.07107 / 10 = 1.6971
])
def test_f19_b_rupture_on_lower_limit(code, fr):
    r = _ok("modulus_of_rupture", {"fck_n_mm2": 5, "code": code})
    assert r["modulus_of_rupture_n_mm2"] == pytest.approx(fr, rel=_REL)


@pytest.mark.parametrize("fck", [0, 4])
def test_f19_b2_rupture_below_range_refused(fck):
    _refuses("modulus_of_rupture", {"fck_n_mm2": fck, "code": "ACI 318"}, ("modulus_of_rupture_n_mm2",))
