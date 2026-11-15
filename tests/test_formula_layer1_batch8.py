"""Formula Layer 1, batch 8 (exam 5): masonry, shrinkage, curing, temporary works.

Same rules as batches 1-7 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation from the published equation,
written in the comment beside its parametrize row:
  * TMS 402 8.2/8.3 (allowable stress)    Pa = (0.25 f'm An + 0.65 Ast Fs) R,
                                          R = 1 - (h/(140 r))^2 when
                                          h/r <= 99, else (70 r/h)^2;
                                          r = t/sqrt(12) for a solid wall
  * EN 1996-1-1 6.1.2 + Annex G           NRd = Phi_m A fk/gamma_M,
                                          Phi_m = A1 exp(-u^2/2),
                                          A1 = 1 - 2 emk/t,
                                          u = (lam - 0.063)/(0.73 - 1.17 emk/t),
                                          lam = hef/tef sqrt(fk/E), E = KE fk
  * ACI 209R-92 Eq. 2-9 (moist cured)     esh(t) = t/(35 + t) esh,u
  * ACI 209R-92 Eq. 2-1 inverted          f(t)/f28 = t/(a + b t)
                                          => t = a f/(1 - b f)
  * dynamic pressure, rho_air 1.226       q = 0.5 rho V^2 = 0.613 V^2 (Pa);
                                          F = q A Cf; M = F h/2;
                                          sigma = M/(w h^2/6)
  * OSHA 1926.451(a)(1) + Appendix A      capacity = 4 x intended load;
                                          light / medium / heavy duty
                                          25 / 50 / 75 psf
  * free thermal strain                   dT = esh/alpha
None was copied from the product's output. A case where the product
disagrees with the hand value is marked ``xfail(strict=True)`` with a
one-line reason and is a suspected defect for the advisor; the expected
value is never moved to match. Every value test runs on two inputs worked
by hand the same way.

No M case for F52: every input has a stated default (fraction 0.70,
a = 4.0, b = 0.85 -- the ACI 209R-92 moist-cured type I constants), so
nothing can be missing; instead ``test_f52_defaults_disclosed`` checks that
each input left out is named in ``defaults_used`` and each input given is
not, the same class as F68 and F71 in batch 11.

Scaffold tolerance: OSHA states the duty loads in psf (25 psf =
1.19701 kPa); a kPa table that keeps one decimal is within 0.3 %, so the
F54 value cases use 0.5 % and say so. Every other case uses 0.1 %.

Unit factors (exact by definition): 1 ft = 0.3048 m; 1 in = 25.4 mm;
1 psi = 6.894757293 kPa; 1 km/h = 1/3.6 m/s; 1 psf = 0.04788025898 kN/m2.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 1e-3   # plan tolerance: 0.1 % unless a case states otherwise
_REL_PSF = 5e-3


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


# ---------------------------------------------------------------- F50 masonry
F50 = "masonry_wall_capacity"
_P = "capacity_kn"


@pytest.mark.parametrize("fm, an, h, t, ast, fs, pa, R", [
    # r = 190/sqrt12 = 54.848; h/r = 54.70; R = 1-(3000/7678.7)^2 = 0.84736;
    # Pa = 0.25*10*190000*0.84736 = 402.50 kN
    (10, 190000, 3000, 190, 0, 0, 402.497, 0.84736),
    # r = 40.415; h/r = 69.28; R = 1-(2800/5658.1)^2 = 0.75510;
    # axial = 0.25*13.8*140000 + 0.65*400*165 = 483000 + 42900 = 525900 N;
    # Pa = 525.9*0.75510 = 397.11 kN
    (13.8, 140000, 2800, 140, 400, 165, 397.108, 0.75510),
])
def test_f50_t_tms(fm, an, h, t, ast, fs, pa, R):
    r = _ok(F50, {"masonry_strength_mpa": fm, "net_area_mm2": an, "height_mm": h,
                  "thickness_mm": t, "steel_area_mm2": ast, "steel_stress_mpa": fs})
    assert r[_P] == pytest.approx(pa, rel=_REL)
    assert r["slenderness_reduction_R"] == pytest.approx(R, rel=_REL)


@pytest.mark.parametrize("fk, a, h, t, e, nrd, phi", [
    # e = 0.05 t = 10; A1 = 0.9; lam = 15 sqrt(1/1000) = 0.47434;
    # u = 0.41134/0.6715 = 0.61257; Phi = 0.9 exp(-0.18762) = 0.74604;
    # NRd = 0.74604*200000*6/2.7 = 331.57 kN
    (6, 200000, 3000, 200, None, 331.571, 0.74604),
    # e = 25, e/t = 0.1; A1 = 0.8; lam = 10.8/31.623 = 0.34153;
    # u = 0.27853/0.613 = 0.45437; Phi = 0.8 exp(-0.10323) = 0.72154;
    # NRd = 0.72154*250000*8/2.7 = 534.47 kN
    (8, 250000, 2700, 250, 25, 534.474, 0.72154),
])
def test_f50_t_eurocode(fk, a, h, t, e, nrd, phi):
    params = {"masonry_strength_mpa": fk, "net_area_mm2": a, "height_mm": h,
              "thickness_mm": t, "code": "eurocode"}
    if e is not None:
        params["eccentricity_mm"] = e
    r = _ok(F50, params)
    assert r[_P] == pytest.approx(nrd, rel=_REL)
    assert r["phi_reduction"] == pytest.approx(phi, rel=_REL)


@pytest.mark.parametrize("params", [
    # 3000 mm and 190 mm written in feet and inches -> 402.50 kN (row 1 above)
    {"masonry_strength_mpa": 10, "net_area_mm2": 190000,
     "height_mm": "9.842519685 ft", "thickness_mm": "7.480314961 in"},
    # f'm = 10 MPa written in psi (10/0.006894757 = 1450.377 psi)
    {"masonry_strength_mpa": "1450.377 psi", "net_area_mm2": 190000,
     "height_mm": 3000, "thickness_mm": 190},
])
def test_f50_u(params):
    assert _ok(F50, params)[_P] == pytest.approx(402.497, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"net_area_mm2": 190000, "height_mm": 3000, "thickness_mm": 190},
     "masonry_strength"),
    ({"masonry_strength_mpa": 10, "height_mm": 3000, "thickness_mm": 190},
     "net_area"),
])
def test_f50_m(params, missing):
    _asks(F50, params, missing, (_P,))


@pytest.mark.parametrize("h, r, R", [
    # h/r = 98 <= 99: R = 1 - (98/140)^2 = 0.51
    (3920, 40, 0.51),
    # h/r = 100 > 99: R = (70/100)^2 = 0.49
    (4000, 40, 0.49),
])
def test_f50_b_slenderness_branch(h, r, R):
    res = _ok(F50, {"masonry_strength_mpa": 12, "net_area_mm2": 200000, "height_mm": h,
                    "thickness_mm": 200, "radius_gyration_mm": r})
    assert res["slenderness_reduction_R"] == pytest.approx(R, rel=_REL)
    # Pa = 0.25*12*200000*R
    assert res[_P] == pytest.approx(600.0 * R, rel=_REL)


@pytest.mark.parametrize("params", [
    {"masonry_strength_mpa": 10, "net_area_mm2": 190000, "height_mm": 3000, "thickness_mm": 0},
    {"masonry_strength_mpa": 10, "net_area_mm2": 190000, "height_mm": 0, "thickness_mm": 190},
])
def test_f50_b_zero_dimension_refused(params):
    _refuses(F50, params, (_P,))


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: a code named by its standard number ('EN 1996', "
    "'EN 1996-1-1') is not recognised and the wall is silently checked to "
    "TMS 402 instead of EN 1996 Annex G"))
@pytest.mark.parametrize("code", ["EN 1996", "EN 1996-1-1"])
def test_f50_b_code_named_by_standard(code):
    # e = 0 -> A1 = 1; u = 0.41134/0.73 = 0.56348; Phi = exp(-0.15875) = 0.85321;
    # NRd = 0.85321*200000*6/2.7 = 379.20 kN
    r = _ok(F50, {"masonry_strength_mpa": 6, "net_area_mm2": 200000, "height_mm": 3000,
                  "thickness_mm": 200, "eccentricity_mm": 0, "code": code})
    assert r[_P] == pytest.approx(379.202, rel=_REL)


# ---------------------------------------------------------------- F51 shrinkage
F51 = "concrete_shrinkage"
_E = "shrinkage_microstrain"


@pytest.mark.parametrize("t, eu, esh", [
    (28, 780, 346.667),   # 28/63 * 780
    (90, 600, 432.0),     # 90/125 * 600
])
def test_f51_t(t, eu, esh):
    r = _ok(F51, {"time_days": t, "ultimate_shrinkage_microstrain": eu})
    assert r[_E] == pytest.approx(esh, rel=_REL)


@pytest.mark.parametrize("params, esh", [
    ({"time_days": "4 weeks", "ultimate_shrinkage_microstrain": 780}, 346.667),
    ({"time_days": "2160 h", "ultimate_shrinkage_microstrain": 600}, 432.0),   # 90 d
])
def test_f51_u(params, esh):
    assert _ok(F51, params)[_E] == pytest.approx(esh, rel=_REL)


@pytest.mark.parametrize("params", [
    {"ultimate_shrinkage_microstrain": 780},
    {"ultimate_shrinkage_microstrain": 600, "time_constant_days": 35},
])
def test_f51_m(params):
    _asks(F51, params, "time_days", (_E,))


@pytest.mark.parametrize("t, eu, esh", [
    (0, 780, 0.0),     # no drying yet
    (35, 780, 390.0),  # t = f: exactly half the ultimate
])
def test_f51_b(t, eu, esh):
    assert _ok(F51, {"time_days": t, "ultimate_shrinkage_microstrain": eu})[_E] == \
        pytest.approx(esh, abs=1e-6, rel=_REL)


@pytest.mark.parametrize("t", [-1, -28])
def test_f51_b_negative_age_refused(t):
    _refuses(F51, {"time_days": t}, (_E,))


# ---------------------------------------------------------------- F52 curing time
F52 = "concrete_curing_time"
_D = "days_to_target"


@pytest.mark.parametrize("f, a, b, days", [
    (0.7, 4.0, 0.85, 6.91358),   # 2.8/0.405
    (0.9, 2.3, 0.92, 12.03488),  # 2.07/0.172
])
def test_f52_t(f, a, b, days):
    r = _ok(F52, {"target_strength_fraction": f, "gain_a": a, "gain_b": b})
    assert r[_D] == pytest.approx(days, rel=_REL)


@pytest.mark.parametrize("frac, days", [
    ("70 %", 6.91358),
    ("50 %", 3.47826),   # 2/0.575
])
def test_f52_u(frac, days):
    r = _ok(F52, {"target_strength_fraction": frac, "gain_a": 4.0, "gain_b": 0.85})
    assert r[_D] == pytest.approx(days, rel=_REL)


@pytest.mark.parametrize("f, days", [
    (1.0, 26.66667),   # 4/0.15: the curve reaches f28 at 28 d only to within
                       # rounding of a and b, so the inverse gives 26.67 d
    (0.5, 3.47826),
])
def test_f52_b_full_strength(f, days):
    r = _ok(F52, {"target_strength_fraction": f, "gain_a": 4.0, "gain_b": 0.85})
    assert r[_D] == pytest.approx(days, rel=_REL)


@pytest.mark.parametrize("params", [
    {"target_strength_fraction": 0, "gain_a": 4.0, "gain_b": 0.85},
    {"target_strength_fraction": 0.5, "gain_a": 4.0, "gain_b": 2.0},
])
def test_f52_b_refused(params):
    _refuses(F52, params, (_D,))


@pytest.mark.parametrize("params, defaulted, days", [
    # f = 0.9 given, a and b default: 3.6/(1 - 0.765) = 3.6/0.235
    ({"target_strength_fraction": 0.9}, {"gain_a", "gain_b"}, 15.31915),
    # a and b given, f defaults to 0.70: 1.61/(1 - 0.644) = 1.61/0.356
    ({"gain_a": 2.3, "gain_b": 0.92}, {"target_strength_fraction"}, 4.52247),
])
def test_f52_defaults_disclosed(params, defaulted, days):
    r = _ok(F52, params)
    listed = {row["parameter"] for row in r.get("defaults_used") or []}
    assert listed == defaulted, r
    assert r[_D] == pytest.approx(days, rel=_REL)


# ---------------------------------------------------------------- F53 wind on formwork
F53 = "wind_load_on_formwork"
_W = "wind_force_kn"


@pytest.mark.parametrize("V, w, h, cf, q, F, M, s", [
    # q = 0.613*400 = 245.2 Pa; F = 245.2*18 = 4413.6 N; M = 4413.6*1.5 = 6620.4 N m;
    # Z = 6*9/6 = 9 m3; sigma = 735.6 N/m2
    (20, 6, 3, 1.0, 0.2452, 4.4136, 6.6204, 735.6),
    # q = 551.7 Pa; F = 551.7*20*1.3 = 14344.2 N; M = 35860.5 N m; Z = 16.667;
    # sigma = 2151.63 N/m2
    (30, 4, 5, 1.3, 0.5517, 14.3442, 35.8605, 2151.63),
])
def test_f53_t(V, w, h, cf, q, F, M, s):
    r = _ok(F53, {"wind_velocity_m_s": V, "formwork_width_m": w,
                  "formwork_height_m": h, "shape_factor": cf})
    assert r["wind_pressure_kpa"] == pytest.approx(q, rel=_REL)
    assert r[_W] == pytest.approx(F, rel=_REL)
    assert r["overturning_moment_kn_m"] == pytest.approx(M, rel=_REL)
    assert r["bending_stress_n_m2"] == pytest.approx(s, rel=_REL)


@pytest.mark.parametrize("params", [
    {"wind_velocity_m_s": "72 km/h", "formwork_width_m": 6, "formwork_height_m": 3,
     "shape_factor": 1.0},
    {"wind_velocity_m_s": 20, "formwork_width_m": "19.685039 ft",
     "formwork_height_m": "9.8425197 ft", "shape_factor": 1.0},
])
def test_f53_u(params):
    assert _ok(F53, params)[_W] == pytest.approx(4.4136, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"formwork_width_m": 6, "formwork_height_m": 3}, "wind_velocity"),
    ({"wind_velocity_m_s": 20, "formwork_height_m": 3}, "area"),
])
def test_f53_m(params, missing):
    _asks(F53, params, missing, (_W, "overturning_moment_kn_m"))


@pytest.mark.parametrize("V, F", [
    (0, 0.0),        # still air
    (40, 17.6544),   # 0.613*1600*18 = 17654.4 N: force is quadratic in V
])
def test_f53_b(V, F):
    r = _ok(F53, {"wind_velocity_m_s": V, "formwork_width_m": 6, "formwork_height_m": 3,
                  "shape_factor": 1.0})
    assert r[_W] == pytest.approx(F, abs=1e-9, rel=_REL)


# ---------------------------------------------------------------- F54 scaffold
F54 = "scaffold_load_capacity"
_C = "required_capacity_kn"
_PSF = 0.04788025898


@pytest.mark.parametrize("duty, psf, area", [
    ("light", 25, 10.0),   # 25 psf = 1.19701 kPa; 11.970 kN; 4x = 47.881 kN
    ("heavy", 75, 7.5),    # 75 psf = 3.59102 kPa; 26.933 kN; 4x = 107.731 kN
])
def test_f54_t(duty, psf, area):
    r = _ok(F54, {"platform_area_m2": area, "duty": duty, "safety_factor": 4.0})
    intended = psf * _PSF * area
    assert r["intended_load_kn"] == pytest.approx(intended, rel=_REL_PSF)
    assert r[_C] == pytest.approx(4.0 * intended, rel=_REL_PSF)


@pytest.mark.parametrize("area, psf", [
    ("80.729328 ft2", 75),   # 7.5 m2 in ft2, heavy
    ("107.639104 ft2", 50),  # 10 m2 in ft2, medium
])
def test_f54_u(area, psf):
    duty = {50: "medium", 75: "heavy"}[psf]
    m2 = float(area.split()[0]) * 0.3048 ** 2
    r = _ok(F54, {"platform_area_m2": area, "duty": duty, "safety_factor": 4.0})
    assert r[_C] == pytest.approx(4.0 * psf * _PSF * m2, rel=_REL_PSF)


@pytest.mark.parametrize("params", [
    {"duty": "light"},
    {"duty": "heavy", "safety_factor": 4.0},
])
def test_f54_m(params):
    _asks(F54, params, "platform_area", (_C, "intended_load_kn"))


@pytest.mark.parametrize("sf, ratio", [(4.0, 4.0), (6.0, 6.0)])
def test_f54_b_factor_is_the_input(sf, ratio):
    r = _ok(F54, {"platform_area_m2": 10, "duty": "medium", "safety_factor": sf})
    assert r[_C] / r["intended_load_kn"] == pytest.approx(ratio, rel=1e-9)


@pytest.mark.parametrize("duty, sf", [
    ("light", 4.0),   # 0 m2 x 25 psf = 0 kN; 4 x 0 = 0
    ("heavy", 6.0),   # 0 m2 x 75 psf = 0 kN; 6 x 0 = 0
])
def test_f54_b_zero_area(duty, sf):
    r = _ok(F54, {"platform_area_m2": 0, "duty": duty, "safety_factor": sf})
    assert r["intended_load_kn"] == 0.0
    assert r[_C] == 0.0


@pytest.mark.parametrize("params", [
    {"platform_area_m2": 10, "duty": "extreme"},
    {"platform_area_m2": -5, "duty": "light"},
])
def test_f54_b_refused(params):
    _refuses(F54, params, (_C,))


# ---------------------------------------------------------------- F55 shrinkage as dT
F55 = "thermal_shrinkage_equivalence"
_T = "equivalent_temp_drop_c"


@pytest.mark.parametrize("e, a, dT", [
    (0.0002, 10e-6, 20.0),
    (0.0003, 12e-6, 25.0),
])
def test_f55_t(e, a, dT):
    assert _ok(F55, {"shrinkage_strain": e, "alpha_c": a})[_T] == pytest.approx(dT, rel=_REL)


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: microstrain is not read as a unit of strain, so "
    "'200 microstrain' is refused as 200 (a ratio) instead of 0.0002"))
@pytest.mark.parametrize("e, dT", [("200 microstrain", 20.0), ("450 microstrain", 50.0)])
def test_f55_u(e, dT):
    assert _ok(F55, {"shrinkage_strain": e, "alpha_c": 10e-6 if dT == 20.0 else 9e-6})[_T] == \
        pytest.approx(dT, rel=_REL)


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: the strain is the quantity being converted, yet a "
    "missing strain is filled with a 0.0002 default and a temperature is returned"))
@pytest.mark.parametrize("params", [{"alpha_c": 10e-6}, {"alpha_c": 12e-6}])
def test_f55_m(params):
    _asks(F55, params, "shrinkage_strain", (_T,))


@pytest.mark.parametrize("e, a, dT", [
    (0.0, 10e-6, 0.0),        # no shrinkage, no equivalent drop
    (0.00045, 9e-6, 50.0),
])
def test_f55_b(e, a, dT):
    assert _ok(F55, {"shrinkage_strain": e, "alpha_c": a})[_T] == \
        pytest.approx(dT, abs=1e-9, rel=_REL)


@pytest.mark.parametrize("a", [0, -1e-5])
def test_f55_b_zero_alpha_refused(a):
    _refuses(F55, {"shrinkage_strain": 0.0002, "alpha_c": a}, (_T,))
