"""Formula Layer 1, batch 3 (exam 5): six more formulas, four case kinds each.

Same rules as batches 1 and 2 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on the limit, rounding direction.

Every expected value is a HAND calculation from the published definition
(statics for the beams, AISC 360-16 D2 / EN 1993-1-1 6.2.3 for tension,
geometry for backfill, arithmetic for a line extension), written in the
comment beside its parametrize row. None was copied from the product's
output. A case where the product disagrees with the hand value is marked
``xfail(strict=True)`` with a one-line reason and is a suspected formula
defect for the advisor; the expected value is never moved to match.
Every value test runs on two inputs worked by hand the same way.

Unit factors (exact by definition): 1 ft = 0.3048 m; 1 in2 = 645.16 mm2;
1 yd3 = 0.764554858 m3.

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


# ------------------------------------------------ F20 beam, one point load
# Simply supported. Central: M = P L / 4. At a from the left: M = P a b / L.

@pytest.mark.parametrize("p, span, a, moment", [
    (40, 6, None, 60.0),    # 40 x 6 / 4 = 60 kN.m
    (100, 8, None, 200.0),  # 100 x 8 / 4 = 200 kN.m
    (60, 10, 4, 144.0),     # b = 6; 60 x 4 x 6 / 10 = 144 kN.m
    (30, 5, 1, 24.0),       # b = 4; 30 x 1 x 4 / 5 = 24 kN.m
])
def test_f20_t_point_load_moment_typical(p, span, a, moment):
    params = {"point_load_kn": p, "span_m": span}
    if a is not None:
        params["distance_from_left_m"] = a
    r = _ok("beam_moment_point_load", params)
    assert r["max_moment_kn_m"] == pytest.approx(moment, rel=_REL)


@pytest.mark.parametrize("p, span, moment", [
    (40, "20 ft", 60.96),  # 6.096 m; 40 x 6.096 / 4 = 60.96
    (10, "10 ft", 7.62),   # 3.048 m; 10 x 3.048 / 4 = 7.62
])
def test_f20_u_point_load_span_in_feet(p, span, moment):
    r = _ok("beam_moment_point_load", {"point_load_kn": p, "span_m": span})
    assert r["max_moment_kn_m"] == pytest.approx(moment, rel=_REL)


@pytest.mark.parametrize("span", [6, 9])
def test_f20_m_point_load_missing_load_asks(span):
    _asks("beam_moment_point_load", {"span_m": span}, "point_load_kn", ("max_moment_kn_m",))


@pytest.mark.parametrize("p, span, a", [
    (50, 6, 0),   # load on the left support: a = 0, M = P x 0 x 6 / 6 = 0
    (80, 4, 4),   # load on the right support: b = 0, M = 0
])
def test_f20_b_point_load_on_a_support_is_zero(p, span, a):
    r = _ok("beam_moment_point_load", {"point_load_kn": p, "span_m": span, "distance_from_left_m": a})
    assert r["max_moment_kn_m"] == 0


@pytest.mark.parametrize("p, span, a", [(50, 6, 6.5), (20, 3, -1)])
def test_f20_b2_point_load_off_the_span_refused(p, span, a):
    _refuses("beam_moment_point_load",
             {"point_load_kn": p, "span_m": span, "distance_from_left_m": a},
             ("max_moment_kn_m",))


# ----------------------------------------------------- F21 beam, max shear
# Simply supported: V = w L / 2 + P / 2 (P central), at the support.

@pytest.mark.parametrize("params, shear", [
    ({"udl_w_kn_m": 20, "span_m": 6}, 60.0),                               # 20 x 6 / 2 = 60
    ({"udl_w_kn_m": 10, "span_m": 4, "central_point_load_kn": 30}, 35.0),  # 10 x 4 / 2 + 30 / 2 = 20 + 15
    ({"central_point_load_kn": 50, "span_m": 6}, 25.0),                    # 50 / 2 = 25
])
def test_f21_t_shear_typical(params, shear):
    r = _ok("beam_shear_simple", params)
    assert r["max_shear_kn"] == pytest.approx(shear, rel=_REL)


@pytest.mark.parametrize("w, span, shear", [
    (20, "20 ft", 60.96),  # 6.096 m; 20 x 6.096 / 2 = 60.96
    (10, "10 ft", 15.24),  # 3.048 m; 10 x 3.048 / 2 = 15.24
])
def test_f21_u_shear_span_in_feet(w, span, shear):
    r = _ok("beam_shear_simple", {"udl_w_kn_m": w, "span_m": span})
    assert r["max_shear_kn"] == pytest.approx(shear, rel=_REL)


@pytest.mark.parametrize("w", [20, 7])
def test_f21_m_shear_missing_span_asks(w):
    _asks("beam_shear_simple", {"udl_w_kn_m": w}, "span_m", ("max_shear_kn",))


@pytest.mark.parametrize("span", [6, 11])
def test_f21_m2_shear_missing_every_load_asks(span):
    # neither the uniform load nor the point load: the shear cannot be known
    env = run_calculation("beam_shear_simple", {"span_m": span})
    assert env.get("status") in ("needs_input", "error"), env
    assert (env.get("result") or {}).get("max_shear_kn") is None, env


@pytest.mark.parametrize("params", [
    {"udl_w_kn_m": -5, "span_m": 6},
    {"udl_w_kn_m": 5, "span_m": 6, "central_point_load_kn": -10},
])
def test_f21_b_negative_load_refused(params):
    _refuses("beam_shear_simple", params, ("max_shear_kn",))


# ------------------------------------------ F22 fixed-ended beam, uniform load
# Support (hogging) M = w L^2 / 12 (the maximum); midspan (sagging) w L^2 / 24.

@pytest.mark.parametrize("w, span, support, mid", [
    (24, 6, 72.0, 36.0),    # 24 x 36 / 12 = 72; 24 x 36 / 24 = 36
    (12, 10, 100.0, 50.0),  # 12 x 100 / 12 = 100; 12 x 100 / 24 = 50
])
def test_f22_t_fixed_udl_typical(w, span, support, mid):
    r = _ok("beam_moment_fixed_udl", {"udl_w_kn_m": w, "span_m": span})
    assert r["support_moment_kn_m"] == pytest.approx(support, rel=_REL)
    assert r["midspan_moment_kn_m"] == pytest.approx(mid, rel=_REL)
    assert r["max_moment_kn_m"] == pytest.approx(support, rel=_REL)


@pytest.mark.parametrize("w, span, support", [
    (20, "20 ft", 61.935),  # 6.096^2 = 37.1612; 20 x 37.1612 / 12 = 61.935
    (10, "10 ft", 7.742),   # 3.048^2 = 9.290304; 10 x 9.290304 / 12 = 7.742
])
def test_f22_u_fixed_udl_span_in_feet(w, span, support):
    r = _ok("beam_moment_fixed_udl", {"udl_w_kn_m": w, "span_m": span})
    assert r["support_moment_kn_m"] == pytest.approx(support, rel=_REL)


@pytest.mark.parametrize("span", [6, 8])
def test_f22_m_fixed_udl_missing_load_asks(span):
    _asks("beam_moment_fixed_udl", {"span_m": span}, "udl_w_kn_m",
          ("support_moment_kn_m", "max_moment_kn_m"))


@pytest.mark.parametrize("w", [24, 3])
def test_f22_b_fixed_udl_zero_span_refused(w):
    _refuses("beam_moment_fixed_udl", {"udl_w_kn_m": w, "span_m": 0},
             ("support_moment_kn_m", "max_moment_kn_m"))


@pytest.mark.parametrize("span", [6, 2.5])
def test_f22_b2_fixed_udl_zero_load_is_zero(span):
    r = _ok("beam_moment_fixed_udl", {"udl_w_kn_m": 0, "span_m": span})
    assert r["max_moment_kn_m"] == 0


# ------------------------------------------------ F23 steel tension capacity
# AISC 360-16 D2 (LRFD): min(0.90 Fy Ag, 0.75 Fu An).
# EN 1993-1-1 6.2.3: min(A fy / 1.0, 0.9 Anet fu / 1.25).

@pytest.mark.parametrize("params, cap, governs", [
    # yield 0.9 x 345 x 2000 = 621.0 kN; rupture 0.75 x 450 x 1700 = 573.75 kN
    ({"gross_area_mm2": 2000, "net_area_mm2": 1700, "fy_mpa": 345, "fu_mpa": 450}, 573.75, "rupture"),
    # yield 0.9 x 250 x 3000 = 675.0 kN; rupture 0.75 x 400 x 2900 = 870.0 kN
    ({"gross_area_mm2": 3000, "net_area_mm2": 2900, "fy_mpa": 250, "fu_mpa": 400}, 675.0, "yield"),
])
def test_f23_t_tension_aisc_typical(params, cap, governs):
    r = _ok("steel_tension_capacity", params)
    assert r["capacity_kn"] == pytest.approx(cap, rel=_REL)
    assert r["governing_limit_state"] == governs


@pytest.mark.parametrize("params, cap, governs", [
    # yield 2000 x 355 / 1.0 = 710.0; rupture 0.9 x 1700 x 510 / 1.25 = 624.24
    ({"gross_area_mm2": 2000, "net_area_mm2": 1700, "fy_mpa": 355, "fu_mpa": 510}, 624.24, "rupture"),
    # yield 1000 x 275 = 275.0; rupture 0.9 x 1000 x 430 / 1.25 = 309.6
    ({"gross_area_mm2": 1000, "net_area_mm2": 1000, "fy_mpa": 275, "fu_mpa": 430}, 275.0, "yield"),
])
def test_f23_t2_tension_eurocode_typical(params, cap, governs):
    r = _ok("steel_tension_capacity", dict(params, code="EC3"))
    assert r["capacity_kn"] == pytest.approx(cap, rel=_REL)
    assert r["governing_limit_state"] == governs


@pytest.mark.parametrize("area, fy, fu, cap", [
    # 3.1 in2 = 2000.0 mm2; net = gross; yield 0.9 x 345 x 2000 = 621.0 < rupture 675.0
    ("3.1 in2", 345, 450, 621.0),
    # 2 in2 = 1290.32 mm2; yield 0.9 x 250 x 1290.32 = 290.32 < rupture 387.10
    ("2 in2", 250, 400, 290.32),
])
def test_f23_u_tension_area_in_square_inches(area, fy, fu, cap):
    r = _ok("steel_tension_capacity", {"gross_area_mm2": area, "fy_mpa": fy, "fu_mpa": fu})
    assert r["capacity_kn"] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("fy", [345, 250])
def test_f23_m_tension_missing_area_asks(fy):
    _asks("steel_tension_capacity", {"fy_mpa": fy, "fu_mpa": 450}, "gross_area_mm2",
          ("capacity_kn",))


@pytest.mark.parametrize("area", [2000, 800])
def test_f23_m2_tension_default_strengths_are_named(area):
    # No fy / fu given: a code default may be used, but it must be named.
    r = _ok("steel_tension_capacity", {"gross_area_mm2": area})
    _flagged_default(r, "fy_mpa")


@pytest.mark.parametrize("area, fy, fu, cap", [
    (1000, 250, 300, 225.0),  # 0.9 x 250 = 225 = 0.75 x 300: both 225.0 kN, a tie
    (500, 300, 360, 135.0),   # 0.9 x 300 x 500 = 135,000 N = 0.75 x 360 x 500: tie
])
def test_f23_b_tension_tie_between_limit_states(area, fy, fu, cap):
    r = _ok("steel_tension_capacity", {"gross_area_mm2": area, "fy_mpa": fy, "fu_mpa": fu})
    assert r["capacity_kn"] == pytest.approx(cap, rel=_REL)
    assert r["governing_limit_state"] in ("yield", "rupture")


# ------------------------------------------------------- F24 backfill volume
# void = excavation - structure; loose to import = void x (1 + swell).

@pytest.mark.parametrize("bank, structure, swell, void, loose", [
    (500, 120, 0.20, 380.0, 456.0),    # 500 - 120 = 380; 380 x 1.2 = 456
    (1000, 250, 0.25, 750.0, 937.5),   # 1000 - 250 = 750; 750 x 1.25 = 937.5
])
def test_f24_t_backfill_typical(bank, structure, swell, void, loose):
    r = _ok("backfill_volume", {"excavation_bank_m3": bank, "structure_volume_m3": structure,
                                "swell_factor": swell})
    assert r["void_volume_m3"] == pytest.approx(void, rel=_REL)
    assert r["loose_backfill_needed_m3"] == pytest.approx(loose, rel=_REL)


@pytest.mark.parametrize("bank, structure, swell, void, loose", [
    # 100 yd3 = 76.4555 m3; 76.4555 - 20 = 56.4555; x 1.1 = 62.101
    ("100 yd3", 20, 0.10, 56.4555, 62.101),
    # 200 yd3 = 152.9110 m3; 152.9110 - 50 = 102.9110; x 1.2 = 123.493
    ("200 yd3", 50, 0.20, 102.911, 123.493),
])
def test_f24_u_backfill_bank_in_cubic_yards(bank, structure, swell, void, loose):
    r = _ok("backfill_volume", {"excavation_bank_m3": bank, "structure_volume_m3": structure,
                                "swell_factor": swell})
    assert r["void_volume_m3"] == pytest.approx(void, rel=_REL)
    assert r["loose_backfill_needed_m3"] == pytest.approx(loose, rel=_REL)


@pytest.mark.parametrize("bank", [500, 64])
def test_f24_m_backfill_missing_structure_asks(bank):
    _asks("backfill_volume", {"excavation_bank_m3": bank, "swell_factor": 0.2},
          "structure_volume_m3", ("void_volume_m3", "loose_backfill_needed_m3"))


@pytest.mark.parametrize("bank, structure", [(500, 120), (90, 30)])
def test_f24_m2_backfill_default_swell_is_named(bank, structure):
    r = _ok("backfill_volume", {"excavation_bank_m3": bank, "structure_volume_m3": structure})
    _flagged_default(r, "swell_factor")


@pytest.mark.parametrize("vol", [300, 42.5])
def test_f24_b_backfill_structure_fills_the_hole(vol):
    # excavation = structure: void 0, nothing to import
    r = _ok("backfill_volume", {"excavation_bank_m3": vol, "structure_volume_m3": vol,
                                "swell_factor": 0.2})
    assert r["void_volume_m3"] == 0
    assert r["loose_backfill_needed_m3"] == 0


@pytest.mark.parametrize("bank, structure", [(100, 150), (10, 10.5)])
def test_f24_b2_backfill_never_negative(bank, structure):
    # a structure larger than its excavation: never a negative volume
    env = run_calculation("backfill_volume", {"excavation_bank_m3": bank,
                                              "structure_volume_m3": structure,
                                              "swell_factor": 0.2})
    if env.get("status") == "success":
        assert env["result"]["void_volume_m3"] >= 0
        assert env["result"]["loose_backfill_needed_m3"] >= 0


# --------------------------------------------- F25 line extension qty x rate

@pytest.mark.parametrize("qty, rate, total", [
    (120, 45.5, 5460.00),    # 120 x 45.5 = 5,460.00
    (3, 1999.99, 5999.97),   # 3 x 1,999.99 = 5,999.97
])
def test_f25_t_line_total_typical(qty, rate, total):
    r = _ok("unit_cost_total", {"quantity": qty, "unit_rate": rate})
    assert r["total_cost"] == pytest.approx(total, abs=0.005)


@pytest.mark.parametrize("qty, rate, total", [
    ("2,400", 12.5, 30000.00),  # thousands separator: 2,400 x 12.5 = 30,000.00
    ("1,000", 7.25, 7250.00),   # 1,000 x 7.25 = 7,250.00
])
def test_f25_u_line_total_quantity_with_separator(qty, rate, total):
    r = _ok("unit_cost_total", {"quantity": qty, "unit_rate": rate})
    assert r["total_cost"] == pytest.approx(total, abs=0.005)


@pytest.mark.parametrize("qty", [120, 7])
def test_f25_m_line_total_missing_rate_asks(qty):
    _asks("unit_cost_total", {"quantity": qty}, "unit_rate", ("total_cost",))


@pytest.mark.parametrize("qty, rate, total", [
    (0, 45.5, 0.0),       # nothing measured: 0
    (3, 33.333, 100.0),   # 99.999 -> 100.00 at the currency's minor unit
    (6, 0.1666, 1.0),     # 0.9996 -> 1.00
])
def test_f25_b_line_total_zero_and_minor_unit(qty, rate, total):
    r = _ok("unit_cost_total", {"quantity": qty, "unit_rate": rate})
    assert r["total_cost"] == pytest.approx(total, abs=0.005)
