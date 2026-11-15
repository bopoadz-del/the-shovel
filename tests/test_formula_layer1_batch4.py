"""Formula Layer 1, batch 4 (exam 5): six more formulas, four case kinds each.

Same rules as batches 1-3 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units (or a different written form) that must convert to
     the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on the limit, a value just past it.

Every expected value is a HAND calculation from the published definition
(elastic beam theory for the three deflections, ASTM C136 / C33 for the
fineness modulus, q = P / A and net q = q - gamma D for bearing, the mass
balance for cut and fill), written in the comment beside its parametrize
row. None was copied from the product's output. A case where the product
disagrees with the hand value is marked ``xfail(strict=True)`` with a
one-line reason and is a suspected defect for the advisor; the expected
value is never moved to match. Every value test runs on two inputs worked
by hand the same way.

Deflections: w in kN/m = N/mm, P in kN x 1e3 = N, L in mm, E in MPa =
N/mm2, I in mm4; the product rounds to 0.01 mm, so the tolerance is
0.006 mm absolute.

Unit factors (exact by definition): 1 ft = 0.3048 m; 1 yd3 = 0.764554858 m3.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 1e-3   # plan tolerance: 0.1 % unless a case states otherwise
_MM = 0.006   # deflection: product rounds to 0.01 mm


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


# ------------------------------------ F26 simply supported, central point load
# delta = P L^3 / (48 E I)

F26 = "beam_deflection_ss_point_load_midspan"


@pytest.mark.parametrize("p, span, ec, i, delta", [
    (50, 6, 30_000, 8e8, 9.375),     # 5e4 x 6000^3 = 1.08e16; 48 x 3e4 x 8e8 = 1.152e15; 9.375 mm
    (20, 4, 200_000, 1e8, 1.333),    # 2e4 x 4000^3 = 1.28e15; 48 x 2e5 x 1e8 = 9.6e14; 1.333 mm
])
def test_f26_t_ss_point_deflection_typical(p, span, ec, i, delta):
    r = _ok(F26, {"p_kn": p, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, abs=_MM)


@pytest.mark.parametrize("p, span, ec, i, delta", [
    (50, "20 ft", 30_000, 8e8, 9.832),   # 6096 mm; 5e4 x 6096^3 = 1.13267e16 / 1.152e15 = 9.832 mm
    (20, "10 ft", 200_000, 1e8, 0.590),  # 3048 mm; 2e4 x 3048^3 = 5.6634e14 / 9.6e14 = 0.590 mm
])
def test_f26_u_ss_point_deflection_span_in_feet(p, span, ec, i, delta):
    r = _ok(F26, {"p_kn": p, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, abs=_MM)


@pytest.mark.parametrize("params, missing", [
    ({"p_kn": 50, "span_m": 6, "ec_mpa": 30_000}, "i_mm4"),  # no I
    ({"p_kn": 50, "span_m": 6, "i_mm4": 8e8}, "ec_mpa"),     # no E
])
def test_f26_m_ss_point_deflection_missing_stiffness_asks(params, missing):
    _asks(F26, params, missing, ("value",))


@pytest.mark.parametrize("p, span", [
    (0, 6),   # P = 0: 0 x L^3 / (48EI) = 0
    (50, 0),  # L = 0: P x 0 / (48EI) = 0
])
def test_f26_b_ss_point_deflection_zero_load_or_span_is_zero(p, span):
    r = _ok(F26, {"p_kn": p, "span_m": span, "ec_mpa": 30_000, "i_mm4": 8e8})
    assert r["value"] == 0


# ------------------------------------------ F27 cantilever, point load at tip
# delta = P L^3 / (3 E I)

F27 = "beam_deflection_cantilever_point_load"


@pytest.mark.parametrize("p, span, ec, i, delta", [
    (30, 4, 200_000, 1e8, 32.0),    # 3e4 x 4000^3 = 1.92e15; 3 x 2e5 x 1e8 = 6e13; 32.0 mm
    (10, 2, 30_000, 5e8, 1.778),    # 1e4 x 2000^3 = 8e13; 3 x 3e4 x 5e8 = 4.5e13; 1.778 mm
])
def test_f27_t_cantilever_point_deflection_typical(p, span, ec, i, delta):
    r = _ok(F27, {"p_kn": p, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, abs=_MM)


@pytest.mark.parametrize("p, span, ec, i, delta", [
    (10, "5 ft", 30_000, 5e8, 0.787),     # 1524 mm; 1e4 x 1524^3 = 3.5396e13 / 4.5e13 = 0.787 mm
    (30, "10 ft", 200_000, 1e8, 14.158),  # 3048 mm; 3e4 x 3048^3 = 8.4951e14 / 6e13 = 14.158 mm
])
def test_f27_u_cantilever_point_deflection_span_in_feet(p, span, ec, i, delta):
    r = _ok(F27, {"p_kn": p, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, abs=_MM)


@pytest.mark.parametrize("params, missing", [
    ({"span_m": 4, "ec_mpa": 200_000, "i_mm4": 1e8}, "p_kn"),  # no load
    ({"p_kn": 30, "ec_mpa": 200_000, "i_mm4": 1e8}, "span_m"),  # no span
])
def test_f27_m_cantilever_point_deflection_missing_input_asks(params, missing):
    _asks(F27, params, missing, ("value",))


@pytest.mark.parametrize("p, span, ec, i, ratio", [
    # Same P, L, E, I: point-at-tip PL^3/3EI against a total load P spread as
    # w = P/L, wL^4/8EI = PL^3/8EI. Ratio 8/3 = 2.667, by hand, independent
    # of the numbers -- the two calculators must not be the same formula.
    (30, 4, 200_000, 1e8, 8 / 3),   # 32.0 mm against 12.0 mm
    (12, 3, 30_000, 4e8, 8 / 3),    # 1.2e4 x 2.7e10 / 3.6e13 = 9.0 mm against 3.375 mm
])
def test_f27_b_point_at_tip_is_eight_thirds_of_same_total_spread(p, span, ec, i, ratio):
    point = _ok(F27, {"p_kn": p, "span_m": span, "ec_mpa": ec, "i_mm4": i})["value"]
    spread = _ok("beam_deflection_cantilever_udl",
                 {"w_kn_m": p / span, "span_m": span, "ec_mpa": ec, "i_mm4": i})["value"]
    assert point / spread == pytest.approx(ratio, rel=5e-3)


# ------------------------------------------------- F28 cantilever, uniform load
# delta = w L^4 / (8 E I)

F28 = "beam_deflection_cantilever_udl"


@pytest.mark.parametrize("w, span, ec, i, delta", [
    (10, 3, 30_000, 8e8, 4.219),   # 10 x 3000^4 = 8.1e14; 8 x 3e4 x 8e8 = 1.92e14; 4.219 mm
    (5, 2, 200_000, 1e8, 0.5),     # 5 x 2000^4 = 8e13; 8 x 2e5 x 1e8 = 1.6e14; 0.5 mm
])
def test_f28_t_cantilever_udl_deflection_typical(w, span, ec, i, delta):
    r = _ok(F28, {"w_kn_m": w, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, abs=_MM)


@pytest.mark.parametrize("w, span, ec, i, delta", [
    (10, "10 ft", 30_000, 8e8, 4.495),  # 3048 mm; 10 x 3048^4 = 8.6310e14 / 1.92e14 = 4.495 mm
    (20, "8 ft", 200_000, 1e8, 4.419),  # 2438.4 mm; 20 x 2438.4^4 = 7.0705e14 / 1.6e14 = 4.419 mm
])
def test_f28_u_cantilever_udl_deflection_span_in_feet(w, span, ec, i, delta):
    r = _ok(F28, {"w_kn_m": w, "span_m": span, "ec_mpa": ec, "i_mm4": i})
    assert r["value"] == pytest.approx(delta, abs=_MM)


@pytest.mark.parametrize("params, missing", [
    ({"span_m": 3, "ec_mpa": 30_000, "i_mm4": 8e8}, "w_kn_m"),  # no load
    ({"w_kn_m": 10, "span_m": 3, "i_mm4": 8e8}, "ec_mpa"),      # no E
])
def test_f28_m_cantilever_udl_deflection_missing_input_asks(params, missing):
    _asks(F28, params, missing, ("value",))


@pytest.mark.parametrize("w, span", [
    (-1, 3),   # negative load: refused, not a negative deflection
    (10, -2),  # negative span: refused
])
def test_f28_b_cantilever_udl_negative_input_refused(w, span):
    _refuses(F28, {"w_kn_m": w, "span_m": span, "ec_mpa": 30_000, "i_mm4": 8e8}, ("value",))


# ---------------------------------------------------- F29 fineness modulus
# FM = (sum of cumulative % retained on the standard sieves) / 100.

F29 = "fineness_modulus"


@pytest.mark.parametrize("series, fm", [
    ([2, 15, 35, 60, 85, 97], 2.94),   # 2+15+35+60+85+97 = 294; /100 = 2.94
    ([0, 10, 30, 55, 80, 95], 2.70),   # 0+10+30+55+80+95 = 270; /100 = 2.70
])
def test_f29_t_fineness_modulus_typical(series, fm):
    assert _ok(F29, {"sieve_retained_percentages": series})["value"] == pytest.approx(fm, rel=_REL)


@pytest.mark.parametrize("series, fm", [
    (["2%", "15%", "35%", "60%", "85%", "97%"], 2.94),  # same as T row 1, each item carries %
    ("0, 10, 30, 55, 80, 95", 2.70),                    # same as T row 2, one comma-separated line
])
def test_f29_u_fineness_modulus_written_forms(series, fm):
    assert _ok(F29, {"sieve_retained_percentages": series})["value"] == pytest.approx(fm, rel=_REL)


@pytest.mark.xfail(
    strict=True,
    reason="suspected parser defect: one comma line whose items carry % is refused "
           "('do not recognise the unit'), while the same items as a list are accepted",
)
@pytest.mark.parametrize("series, fm", [
    ("2%, 15%, 35%, 60%, 85%, 97%", 2.94),  # 294 / 100
    ("0%,10%,30%,55%,80%,95%", 2.70),       # 270 / 100
])
def test_f29_u_fineness_modulus_percent_signs_in_one_line(series, fm):
    assert _ok(F29, {"sieve_retained_percentages": series})["value"] == pytest.approx(fm, rel=_REL)


@pytest.mark.parametrize("params", [
    {},                                   # nothing given
    {"sieve_retained_percentages": []},   # an empty series is not a zero modulus
])
def test_f29_m_fineness_modulus_without_series_asks(params):
    _asks(F29, params, "cumulative percentages retained", ("value",))


@pytest.mark.parametrize("series", [
    [2, 15, 105],   # a cumulative percentage cannot exceed 100
    [-1, 15, 35],   # nor be negative
])
def test_f29_b_fineness_modulus_out_of_range_refused(series):
    _refuses(F29, {"sieve_retained_percentages": series}, ("value",))


# ---------------------------------------------- F30 foundation bearing pressure
# q = P / (B x L); net q = q - gamma x D; FOS = q_allow / net q; ok when FOS >= 1.

F30 = "foundation_bearing_pressure"


@pytest.mark.parametrize("b, l_, p, qa, q, fos, ok", [
    (2, 2, 1000, 300, 250.0, 1.2, True),     # 1000 / 4 = 250; 300 / 250 = 1.20
    (3, 2, 1800, 200, 300.0, 0.667, False),  # 1800 / 6 = 300; 200 / 300 = 0.667
])
def test_f30_t_bearing_pressure_typical(b, l_, p, qa, q, fos, ok):
    r = _ok(F30, {"foundation_width_m": b, "foundation_length_m": l_,
                  "column_load_kn": p, "soil_bearing_capacity_kn_m2": qa})
    assert r["bearing_pressure_kn_m2"] == pytest.approx(q, rel=_REL)
    assert r["fos_against_bearing"] == pytest.approx(fos, abs=0.006)  # product rounds to 0.01
    assert r["bearing_ok"] is ok
    _flagged_default(r, "foundation_depth_m")


@pytest.mark.parametrize("b, l_, p, qa, q, fos", [
    ("10 ft", "10 ft", 1000, 300, 107.639, 2.787),  # 3.048^2 = 9.290304 m2; 1000 / 9.2903 = 107.639
    ("5 ft", 2, 600, 250, 196.850, 1.270),          # 1.524 x 2 = 3.048 m2; 600 / 3.048 = 196.850
])
def test_f30_u_bearing_pressure_plan_in_feet(b, l_, p, qa, q, fos):
    r = _ok(F30, {"foundation_width_m": b, "foundation_length_m": l_,
                  "column_load_kn": p, "soil_bearing_capacity_kn_m2": qa})
    assert r["bearing_pressure_kn_m2"] == pytest.approx(q, rel=_REL)
    assert r["fos_against_bearing"] == pytest.approx(fos, abs=0.006)


@pytest.mark.parametrize("params, missing", [
    ({"foundation_width_m": 2, "foundation_length_m": 2, "soil_bearing_capacity_kn_m2": 300},
     "column_load_kn"),
    ({"foundation_width_m": 2, "foundation_length_m": 2, "column_load_kn": 1000},
     "soil_bearing_capacity_kn_m2"),
])
def test_f30_m_bearing_missing_input_asks(params, missing):
    _asks(F30, params, missing, ("fos_against_bearing", "bearing_ok"))


@pytest.mark.parametrize("p, ok", [
    (800, True),    # 800 / 4 = 200 = q_allow: FOS = 1.00, on the limit, adequate
    (804, False),   # 804 / 4 = 201 > 200: FOS = 0.995, just past it, not adequate
])
def test_f30_b_bearing_on_the_limit(p, ok):
    r = _ok(F30, {"foundation_width_m": 2, "foundation_length_m": 2,
                  "column_load_kn": p, "soil_bearing_capacity_kn_m2": 200})
    assert r["bearing_ok"] is ok


@pytest.mark.parametrize("b, l_", [
    (0, 2),  # zero width: q = P / 0 has no value
    (2, 0),  # zero length
])
def test_f30_b_bearing_zero_plan_dimension_refused(b, l_):
    _refuses(F30, {"foundation_width_m": b, "foundation_length_m": l_,
                   "column_load_kn": 1000, "soil_bearing_capacity_kn_m2": 300},
             ("bearing_pressure_kn_m2", "fos_against_bearing", "bearing_ok"))


@pytest.mark.xfail(
    strict=True,
    reason="suspected undeclared default: with a depth given, net q subtracts a soil "
           "unit weight the caller never supplied, and defaults_used does not name it",
)
@pytest.mark.parametrize("depth", [0.5, 2.0])
def test_f30_m_bearing_net_pressure_names_the_soil_weight_it_assumed(depth):
    r = _ok(F30, {"foundation_width_m": 3, "foundation_length_m": 2, "column_load_kn": 1800,
                  "soil_bearing_capacity_kn_m2": 250, "foundation_depth_m": depth})
    # net q = q - gamma D needs gamma; when gamma is not an input, the value
    # the calculator assumed must be declared beside the answer.
    named = " ".join(
        f"{d.get('parameter')} {d.get('label')}" for d in (r.get("defaults_used") or [])
    ).lower()
    assert "unit weight" in named or "gamma" in named or "density" in named, r


# ------------------------------------------------------ F31 cut and fill balance
# balance = cut - fill (bank m3); haul loose = |balance| x (1 + bulking).

F31 = "cut_fill_balance"


@pytest.mark.parametrize("cut, fill, bulk, bal, status, loose", [
    (5000, 3000, 0.25, 2000.0, "export_surplus", 2500.0),   # 2000 x 1.25 = 2500
    (2000, 3500, 0.20, -1500.0, "import_deficit", 1800.0),  # |-1500| x 1.20 = 1800
])
def test_f31_t_cut_fill_balance_typical(cut, fill, bulk, bal, status, loose):
    r = _ok(F31, {"cut_volume_m3": cut, "fill_volume_m3": fill, "bulking_factor": bulk})
    assert r["balance_bank_m3"] == pytest.approx(bal, rel=_REL)
    assert r["status"] == status
    assert r["haul_loose_m3"] == pytest.approx(loose, rel=_REL)


@pytest.mark.parametrize("cut, fill, bulk, bal, loose", [
    ("1000 yd3", 500, 0.25, 264.555, 330.694),    # 764.554858 - 500 = 264.555; x 1.25 = 330.694
    (2000, "1000 yd3", 0.30, 1235.445, 1606.079),  # 2000 - 764.554858 = 1235.445; x 1.30 = 1606.079
])
def test_f31_u_cut_fill_balance_in_cubic_yards(cut, fill, bulk, bal, loose):
    r = _ok(F31, {"cut_volume_m3": cut, "fill_volume_m3": fill, "bulking_factor": bulk})
    assert r["balance_bank_m3"] == pytest.approx(bal, rel=_REL)
    assert r["haul_loose_m3"] == pytest.approx(loose, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"cut_volume_m3": 5000}, "fill_volume_m3"),
    ({"fill_volume_m3": 3000}, "cut_volume_m3"),
])
def test_f31_m_cut_fill_missing_volume_asks(params, missing):
    _asks(F31, params, missing, ("balance_bank_m3", "haul_loose_m3"))


@pytest.mark.parametrize("cut, fill", [
    (5000, 3000),  # bulking omitted: the default used must be named
    (100, 400),
])
def test_f31_m_cut_fill_default_bulking_is_named(cut, fill):
    _flagged_default(_ok(F31, {"cut_volume_m3": cut, "fill_volume_m3": fill}), "bulking_factor")


@pytest.mark.parametrize("vol", [0, 2500])
def test_f31_b_cut_equals_fill_is_balanced(vol):
    # cut = fill: balance 0, nothing to haul either way
    r = _ok(F31, {"cut_volume_m3": vol, "fill_volume_m3": vol, "bulking_factor": 0.25})
    assert r["balance_bank_m3"] == 0
    assert r["status"] == "balanced"
    assert r["haul_loose_m3"] == 0


@pytest.mark.parametrize("cut, fill", [
    (-100, 300),  # a negative volume is refused
    (300, -100),
])
def test_f31_b_cut_fill_negative_volume_refused(cut, fill):
    _refuses(F31, {"cut_volume_m3": cut, "fill_volume_m3": fill, "bulking_factor": 0.25},
             ("balance_bank_m3", "haul_loose_m3"))
