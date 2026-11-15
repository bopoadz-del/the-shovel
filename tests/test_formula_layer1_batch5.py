"""Formula Layer 1, batch 5 (exam 5): six more formulas, four case kinds each.

Same rules as batches 1-4 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units (or a different written form) that must convert to
     the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on the limit, a value past it.

Every expected value is a HAND calculation from the published definition,
written in the comment beside its parametrize row:
  * ACI 318-19 Eq. 19.2.2.1b (SI)   Ec = 4700 sqrt(f'c)            MPa
  * metric-technical form           Ec = 15000 sqrt(f'c kg/cm2)    kg/cm2
  * ACI 318-19 Eq. 19.2.3.1         fr = 0.62 sqrt(f'c)            MPa
  * ASCE 7-16 26.10 (SI)            qz = 0.613 Kz Kzt Kd V^2       Pa
  * EN 1991-1-4 4.5                 qb = 0.5 rho vb^2, rho = 1.25  Pa
  * arithmetic                      cost / area
  * ACI 318-19 Table 7.3.1.1        h = L / {20, 24, 28, 10} x (0.4 + fy/700)
  * EN 1992-1-1 7.4.2 + 4.4.1       h = L / {20, 26, 30, 8} + cover + bar/2
  * statics                         reactions of SS, cantilever, propped
                                    cantilever and fixed-fixed beams
None was copied from the product's output. A case where the product
disagrees with the hand value is marked ``xfail(strict=True)`` with a
one-line reason and is a suspected defect for the advisor; the expected
value is never moved to match. Every value test runs on two inputs worked
by hand the same way.

Unit factors (exact by definition): 1 psi = 6.894757293 kPa;
1 ft = 0.3048 m; 1 mph = 0.44704 m/s; 1 km/h = 1/3.6 m/s;
1 kgf/cm2 = 0.0980665 MPa.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 1e-3   # plan tolerance: 0.1 % unless a case states otherwise


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


# ----------------------------------------- F32 concrete modulus of elasticity

F32 = "modulus_of_elasticity_concrete"


@pytest.mark.parametrize("fck, ec", [
    (25.0, 23500.0),   # 4700 x sqrt(25) = 4700 x 5 = 23,500 MPa
    (36.0, 28200.0),   # 4700 x sqrt(36) = 4700 x 6 = 28,200 MPa
])
def test_f32_t_aci_si_form(fck, ec):
    r = _ok(F32, {"fck_n_mm2": fck, "code": "aci"})
    assert r["unit"] == "MPa"
    assert r["value"] == pytest.approx(ec, abs=0.6)


@pytest.mark.parametrize("fck, ec_kg", [
    # 25 MPa / 0.0980665 = 254.929 kg/cm2; sqrt = 15.96650; x15000 = 239,498
    (25.0, 239498.0),
    # 40 MPa / 0.0980665 = 407.886 kg/cm2; sqrt = 20.19619; x15000 = 302,943
    (40.0, 302943.0),
])
def test_f32_t_metric_technical_form_carries_its_unit(fck, ec_kg):
    r = _ok(F32, {"fck_n_mm2": fck, "code": "metric_technical"})
    assert r["unit"] == "kg/cm2"
    assert r["value"] == pytest.approx(ec_kg, rel=_REL)
    # its MPa twin is the same number converted, not a second formula
    assert r["value_mpa"] == pytest.approx(ec_kg * 0.0980665, rel=_REL)


@pytest.mark.parametrize("fck, ec", [
    # 4000 psi = 27.579 MPa; sqrt = 5.25158; x4700 = 24,682 MPa
    ("4000 psi", 24682.0),
    # 5000 psi = 34.474 MPa; sqrt = 5.87144; x4700 = 27,596 MPa
    ("5000 psi", 27596.0),
])
def test_f32_u_strength_in_psi(fck, ec):
    r = _ok(F32, {"fck_n_mm2": fck, "code": "aci"})
    assert r["value"] == pytest.approx(ec, rel=_REL)


@pytest.mark.parametrize("code", ["aci", "metric_technical"])
def test_f32_m_no_strength_asks(code):
    _asks(F32, {"code": code}, "fck", ("value", "value_mpa"))


@pytest.mark.parametrize("fck", [0, -5])
def test_f32_b_zero_or_negative_strength_refused(fck):
    _refuses(F32, {"fck_n_mm2": fck, "code": "aci"}, ("value",))


# -------------------------------------------------- F33 modulus of rupture

F33 = "modulus_of_rupture"


@pytest.mark.parametrize("fck, fr", [
    (25.0, 3.100),   # 0.62 x sqrt(25) = 0.62 x 5 = 3.100 MPa
    (36.0, 3.720),   # 0.62 x sqrt(36) = 0.62 x 6 = 3.720 MPa
])
def test_f33_t_aci(fck, fr):
    r = _ok(F33, {"fck_n_mm2": fck, "code": "aci"})
    assert r["unit"] == "MPa"
    assert r["value"] == pytest.approx(fr, abs=0.0006)


@pytest.mark.parametrize("fck, fr", [
    ("4 ksi", 3.256),     # 27.579 MPa; sqrt 5.25158 x 0.62 = 3.2560
    ("3000 psi", 2.820),  # 20.684 MPa; sqrt 4.54795 x 0.62 = 2.8197
])
def test_f33_u_strength_in_us_units(fck, fr):
    r = _ok(F33, {"fck_n_mm2": fck, "code": "aci"})
    assert r["value"] == pytest.approx(fr, abs=0.0006)


@pytest.mark.parametrize("code", ["aci", "metric_technical"])
def test_f33_m_no_strength_asks(code):
    _asks(F33, {"code": code}, "fck", ("value", "modulus_of_rupture_n_mm2"))


@pytest.mark.parametrize("fck", [0, -1])
def test_f33_b_zero_or_negative_strength_refused(fck):
    _refuses(F33, {"fck_n_mm2": fck, "code": "aci"}, ("value",))


# ----------------------------------------------------- F34 wind pressure

F34 = "wind_pressure"


@pytest.mark.parametrize("params, q", [
    ({"wind_speed_m_s": 30}, 551.7),   # 0.613 x 900 = 551.70 Pa
    ({"wind_speed_m_s": 40}, 980.8),   # 0.613 x 1600 = 980.80 Pa
    # 0.613 x 0.85 x 0.85 x 1600 = 0.613 x 0.7225 x 1600 = 708.63 Pa
    ({"wind_speed_m_s": 40, "kz": 0.85, "kd": 0.85}, 708.6),
])
def test_f34_t_asce(params, q):
    r = _ok(F34, params)
    assert r["velocity_pressure_pa"] == pytest.approx(q, abs=0.06)
    assert r["velocity_pressure_kn_m2"] == pytest.approx(q / 1000.0, abs=6e-5)


@pytest.mark.parametrize("v, q", [
    (30, 562.5),   # 0.5 x 1.25 x 900 = 562.5 Pa
    (26, 422.5),   # 0.5 x 1.25 x 676 = 422.5 Pa
])
def test_f34_t_eurocode_basic_pressure(v, q):
    r = _ok(F34, {"wind_speed_m_s": v, "code": "eurocode"})
    assert r["velocity_pressure_pa"] == pytest.approx(q, abs=0.06)


@pytest.mark.parametrize("v", [30, 40])
def test_f34_t_directionality_default_is_named(v):
    # Kd not supplied: the factor used must be declared, never silent.
    _flagged_default(_ok(F34, {"wind_speed_m_s": v}), "kd")


@pytest.mark.parametrize("v, q", [
    ("108 km/h", 551.7),   # 108 / 3.6 = 30 m/s; 0.613 x 900 = 551.7
    ("100 mph", 1225.0),   # 44.704 m/s; ^2 = 1998.448; x0.613 = 1225.05
])
def test_f34_u_speed_in_other_units(v, q):
    r = _ok(F34, {"wind_speed_m_s": v})
    assert r["velocity_pressure_pa"] == pytest.approx(q, abs=0.06)


@pytest.mark.parametrize("params", [{}, {"code": "eurocode"}])
def test_f34_m_no_speed_asks(params):
    _asks(F34, params, "wind_speed", ("velocity_pressure_pa", "velocity_pressure_kn_m2"))


@pytest.mark.parametrize("code", ["aci", "eurocode"])
def test_f34_b_zero_speed_is_zero_pressure(code):
    assert _ok(F34, {"wind_speed_m_s": 0, "code": code})["velocity_pressure_pa"] == 0


@pytest.mark.parametrize("v", [-10, -0.5])
def test_f34_b_negative_speed_refused(v):
    _refuses(F34, {"wind_speed_m_s": v}, ("velocity_pressure_pa",))


# ----------------------------------------------------- F35 cost per area

F35 = "cost_per_area"


@pytest.mark.parametrize("total, area, rate", [
    (120000, 400, 300.00),     # 120,000 / 400 = 300.00
    (98765, 321, 307.68),      # 98,765 / 321 = 307.679 -> 307.68
])
def test_f35_t(total, area, rate):
    r = _ok(F35, {"total_cost": total, "area": area})
    assert r["cost_per_area"] == pytest.approx(rate, abs=0.006)


@pytest.mark.parametrize("total, area, rate", [
    # 400 ft2 x 0.09290304 = 37.161216 m2; 120,000 / 37.161216 = 3229.17 /m2
    (120000, "400 ft2", 3229.17),
    # 1076.391 sq ft x 0.09290304 = 100.000 m2; 50,000 / 100 = 500.00 /m2
    (50000, "1076.391 sq ft", 500.00),
])
def test_f35_u_area_in_square_feet_is_rated_per_m2(total, area, rate):
    r = _ok(F35, {"total_cost": total, "area": area})
    # the rate and its unit must agree: converted to m2, so per m2
    assert r["area_unit"] == "m2"
    assert r["cost_per_area"] == pytest.approx(rate, abs=0.006)


@pytest.mark.parametrize("params, missing", [
    ({"total_cost": 120000}, "area"),
    ({"area": 100}, "total_cost"),
])
def test_f35_m_missing_input_asks(params, missing):
    _asks(F35, params, missing, ("cost_per_area",))


@pytest.mark.parametrize("area", [0, -10])
def test_f35_b_zero_or_negative_area_refused(area):
    _refuses(F35, {"total_cost": 50000, "area": area}, ("cost_per_area",))


@pytest.mark.parametrize("area", [100, 250])
def test_f35_b_zero_cost_is_zero_rate(area):
    assert _ok(F35, {"total_cost": 0, "area": area})["cost_per_area"] == 0


# ------------------------------------------- F36 minimum one-way slab thickness

F36 = "slab_thickness_min"


@pytest.mark.parametrize("span, support, h", [
    (6000, "simply_supported", 300.0),       # 6000 / 20 = 300
    (4800, "one_end_continuous", 200.0),     # 4800 / 24 = 200
    (7000, "both_ends_continuous", 250.0),   # 7000 / 28 = 250
    (1500, "cantilever", 150.0),             # 1500 / 10 = 150
])
def test_f36_t_aci_table(span, support, h):
    r = _ok(F36, {"span_mm": span, "support_condition": support})
    assert r["min_thickness_mm"] == pytest.approx(h, abs=0.06)


@pytest.mark.parametrize("fy, h", [
    (280, 160.0),   # 4000/20 = 200 x (0.4 + 280/700 = 0.8) = 160.0
    (520, 228.6),   # 200 x (0.4 + 520/700 = 1.142857) = 228.57
])
def test_f36_t_aci_other_steel_grade(fy, h):
    r = _ok(F36, {"span_mm": 4000, "fy_mpa": fy})
    assert r["min_thickness_mm"] == pytest.approx(h, abs=0.06)


@pytest.mark.parametrize("params, h", [
    # d = 4000/20 = 200; h = 200 + 25 + 0/2 = 225
    ({"span_mm": 4000, "code": "eurocode"}, 225.0),
    # d = 1600/8 = 200; h = 200 + 25 + 12/2 = 231
    ({"span_mm": 1600, "code": "eurocode", "support_condition": "cantilever",
      "bar_diameter_mm": 12}, 231.0),
])
def test_f36_t_eurocode_depth_plus_cover(params, h):
    r = _ok(F36, params)
    assert r["min_thickness_mm"] == pytest.approx(h, abs=0.06)


@pytest.mark.parametrize("span, h", [
    ("4 m", 200.0),          # 4000 mm / 20 = 200
    ("13.12336 ft", 200.0),  # 13.12336 x 304.8 = 4000.0 mm / 20 = 200
])
def test_f36_u_span_in_m_or_ft(span, h):
    r = _ok(F36, {"span_mm": span})
    assert r["min_thickness_mm"] == pytest.approx(h, abs=0.06)


@pytest.mark.parametrize("params", [{}, {"support_condition": "cantilever"}])
def test_f36_m_no_span_asks(params):
    _asks(F36, params, "span", ("min_thickness_mm",))


@pytest.mark.parametrize("span", [0, -4000])
def test_f36_b_zero_or_negative_span_refused(span):
    _refuses(F36, {"span_mm": span}, ("min_thickness_mm",))


# ---------------------------------------------------- F37 beam reactions

F37 = "beam_reactions"


def _udl(w: float) -> list[dict]:
    return [{"kind": "udl", "w": w}]


@pytest.mark.parametrize("span, loads, rl, rr", [
    (6, _udl(10), 30.0, 30.0),                             # wL/2 = 30 each
    (6, [{"kind": "point", "P": 60, "a": 2}], 40.0, 20.0),  # Pb/L = 60x4/6; Pa/L
])
def test_f37_t_simply_supported(span, loads, rl, rr):
    r = _ok(F37, {"span_m": span, "loads": loads})
    assert r["reaction_left_kn"] == pytest.approx(rl, abs=0.002)
    assert r["reaction_right_kn"] == pytest.approx(rr, abs=0.002)


@pytest.mark.parametrize("support, span, w, rl, rr, ml", [
    # cantilever fixed at left: R = wL = 40; M = wL^2/2 = 80
    ("cantilever", 4, 10, 40.0, 0.0, 80.0),
    # fixed-fixed: R = wL/2 = 30 each; end M = wL^2/12 = 12 x 25 / 12 = 25
    ("fixed_fixed", 5, 12, 30.0, 30.0, 25.0),
    # propped cantilever, fixed left: R = 5wL/8 = 20, prop 3wL/8 = 12,
    # fixed-end M = wL^2/8 = 8 x 16 / 8 = 16
    ("propped_cantilever", 4, 8, 20.0, 12.0, 16.0),
])
def test_f37_t_other_supports(support, span, w, rl, rr, ml):
    r = _ok(F37, {"span_m": span, "support": support, "loads": _udl(w)})
    assert r["reaction_left_kn"] == pytest.approx(rl, abs=0.002)
    assert r["reaction_right_kn"] == pytest.approx(rr, abs=0.002)
    assert abs(r["moment_left_kn_m"]) == pytest.approx(ml, abs=0.002)


@pytest.mark.parametrize("span, w, r_each", [
    ("20 ft", 10, 30.48),   # 6.096 m x 10 / 2 = 30.48
    ("15 ft", 8, 18.288),   # 4.572 m x 8 / 2 = 18.288
])
def test_f37_u_span_in_feet(span, w, r_each):
    r = _ok(F37, {"span_m": span, "loads": _udl(w)})
    assert r["reaction_left_kn"] == pytest.approx(r_each, abs=0.002)
    assert r["reaction_right_kn"] == pytest.approx(r_each, abs=0.002)


@pytest.mark.parametrize("params", [{"loads": _udl(10)}, {"support": "cantilever", "loads": _udl(5)}])
def test_f37_m_no_span_asks(params):
    _asks(F37, params, "span", ("reaction_left_kn", "reaction_right_kn"))


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: with no load given the calculator returns success with "
    "0 kN reactions instead of asking for the load"))
@pytest.mark.parametrize("params", [{"span_m": 6}, {"span_m": 4, "support": "cantilever"}])
def test_f37_m_no_load_asks(params):
    _asks(F37, params, "load", ("reaction_left_kn", "reaction_right_kn"))


@pytest.mark.parametrize("span", [0, -3])
def test_f37_b_zero_or_negative_span_refused(span):
    _refuses(F37, {"span_m": span, "loads": _udl(10)}, ("reaction_left_kn",))


@pytest.mark.parametrize("span, loads, total", [
    # equilibrium: the two reactions carry the whole load
    (7, _udl(9) + [{"kind": "point", "P": 25, "a": 3}], 9 * 7 + 25),   # 88
    (5, [{"kind": "point", "P": 12, "a": 1}, {"kind": "point", "P": 18, "a": 4}], 30),
])
def test_f37_b_reactions_sum_to_the_load(span, loads, total):
    r = _ok(F37, {"span_m": span, "loads": loads})
    assert r["reaction_left_kn"] + r["reaction_right_kn"] == pytest.approx(total, abs=0.004)
