"""Formula Layer 1, batch 14 (exam 5): concrete beam bending capacity,
concrete section shear capacity, reinforcement lap length, nominal shear
stress, bolt shear capacity, fillet weld capacity, precast member erection
demand, plumbing installation programme, BIM clash tolerance (more
formulas follow in later parts of this batch).

Same rules as batches 1-13 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation, written in the comment beside
its parametrize row:
  * beam bending capacity        ACI 318-19 22.2: a = As fy / (0.85 fc b);
                                 phi Mn = 0.9 As fy (d - a/2);
                                 EN 1992-1-1 6.1: fcd = fc/1.5, fyd = fy/1.15,
                                 x = As fyd / (0.8 b fcd), z = d - 0.4 x,
                                 MRd = As fyd z
  * section shear capacity       ACI 318-19 Table 22.5.5.1 (Av < Av,min):
                                 phi Vc = 0.75 x 0.66 ls rho^(1/3) sqrt(fc)
                                 bw d, ls = sqrt(2/(1 + 0.004 d)) <= 1,
                                 sqrt(fc) <= 8.3 MPa;
                                 EN 1992-1-1 6.2.2: VRd,c = 0.12 k
                                 (100 rho fck)^(1/3) bw d, k = 1 +
                                 sqrt(200/d) <= 2, rho <= 0.02
  * lap length                   ACI 318-19 25.4.2: ld = fy ps pg / (1.1
                                 sqrt(fc) (cb+Ktr)/db) x db, ps = 0.8 for
                                 db <= 19 mm else 1.0, pg (Table 25.4.2.5)
                                 1.0 to 420 MPa, 1.15 to 550, 1.3 above;
                                 Class B lap = 1.3 ld;
                                 EN 1992-1-1 8.4/8.7: fctd = 0.7 x 0.3
                                 fck^(2/3) / 1.5, fbd = 2.25 n2 fctd, n2 = 1
                                 for db <= 32 else (132 - db)/100, lb,rqd =
                                 db/4 x fyd/fbd, l0 = 1.5 lb,rqd
  * nominal shear stress         v = V / (b d), kN -> N: x 1000
None was copied from the product's output. A case where the product
disagrees with the hand value would be marked ``xfail(strict=True)`` with a
one-line reason and reported as a suspected defect; the expected value is
never moved to match. Every value test runs on two inputs worked by hand
the same way.

The design code is a word, not a unit, so the U cases convert lengths,
areas and stresses only.

Unit factors (exact by definition): 1 in = 25.4 mm; 1 cm = 10 mm;
1 cm2 = 100 mm2; 1 ksi = 6.894757 MPa; 1 kN = 1000 N.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation


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


# ---------------------------------------------------------------- F103 RC bending
F103 = "rc_beam_moment_capacity"
_F103_OUT = ("moment_capacity_kn_m",)
_F103_BASE = {"steel_area_mm2": 1500, "fy_mpa": 420, "width_mm": 300,
              "eff_depth_mm": 500, "fc_mpa": 30}


@pytest.mark.parametrize("params, cap", [
    # a = 1500 x 420 / (0.85 x 30 x 300) = 630000 / 7650 = 82.353;
    # 0.9 x 1500 x 420 x (500 - 41.176) = 567000 x 458.824 = 260.153e6 N.mm
    (_F103_BASE, 260.15),
    # a = 2000 x 500 / (0.85 x 40 x 400) = 1e6 / 13600 = 73.529;
    # 0.9 x 2000 x 500 x (600 - 36.765) = 900000 x 563.235 = 506.912e6
    ({"steel_area_mm2": 2000, "fy_mpa": 500, "width_mm": 400, "eff_depth_mm": 600,
      "fc_mpa": 40}, 506.91),
    # EC: fcd 20, fyd 434.783; x = 652174 / 4800 = 135.870; z = 445.652;
    # 652174 x 445.652 = 290.643e6
    ({**_F103_BASE, "fy_mpa": 500, "code": "EC"}, 290.64),
])
def test_f103_t(params, cap):
    assert _ok(F103, params)["moment_capacity_kn_m"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("params", [
    # 15 cm2 = 1500 mm2, 30 cm = 300 mm, 50 cm = 500 mm -> 260.15 as above
    {**_F103_BASE, "steel_area_mm2": "15 cm2", "width_mm": "30 cm", "eff_depth_mm": "50 cm"},
    # 60.915852 ksi = 420.000 MPa; 19.685039 in = 500.000 mm -> 260.15
    {**_F103_BASE, "fy_mpa": "60.915852 ksi", "eff_depth_mm": "19.685039 in"},
])
def test_f103_u(params):
    assert _ok(F103, params)["moment_capacity_kn_m"] == pytest.approx(260.15, abs=0.02)


@pytest.mark.parametrize("missing", ["steel_area_mm2", "eff_depth_mm"])
def test_f103_m(missing):
    params = {k: v for k, v in _F103_BASE.items() if k != missing}
    _asks(F103, params, missing, _F103_OUT)


@pytest.mark.parametrize("area, cap", [
    (0, 0.0),        # no steel: a = 0, phi Mn = 0 (the declared floor 0 mm2)
    # a = 42000 / 7650 = 5.490; 0.9 x 100 x 420 x (500 - 2.745) = 37800 x
    # 497.255 = 18.796e6 N.mm
    (100, 18.80),
])
def test_f103_b_steel_floor(area, cap):
    r = _ok(F103, {**_F103_BASE, "steel_area_mm2": area})
    assert r["moment_capacity_kn_m"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("area", [-1, -1500])
def test_f103_b_negative_steel_refused(area):
    _refuses(F103, {**_F103_BASE, "steel_area_mm2": area}, _F103_OUT)


# ---------------------------------------------------------------- F104 RC shear
F104 = "rc_beam_shear_capacity"
_F104_OUT = ("shear_capacity_kn",)
_F104_BASE = {"width_mm": 300, "eff_depth_mm": 500, "fc_mpa": 30, "rho_l": 0.01}


@pytest.mark.parametrize("params, cap", [
    # ls = sqrt(2/3) = 0.8165; 0.66 x 0.8165 x 0.21544 x 5.4772 = 0.63591;
    # 0.75 x 0.63591 x 150000 = 71539 N
    (_F104_BASE, 71.54),
    # ls = sqrt(2/3.4) = 0.76696; 0.015^(1/3) = 0.24662; sqrt 40 = 6.3246;
    # 0.66 x 0.76696 x 0.24662 x 6.3246 = 0.78955; 0.75 x 0.78955 x 240000
    ({"width_mm": 400, "eff_depth_mm": 600, "fc_mpa": 40, "rho_l": 0.015}, 142.12),
    # EC: k = 1 + sqrt(0.4) = 1.63246; (100 x 0.01 x 30)^(1/3) = 3.10723;
    # 0.12 x 1.63246 x 3.10723 x 150000 = 91303 N (vmin 59.98 kN lower)
    ({**_F104_BASE, "code": "EC"}, 91.30),
])
def test_f104_t(params, cap):
    assert _ok(F104, params)["shear_capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("params", [
    {**_F104_BASE, "width_mm": "30 cm", "eff_depth_mm": "0.5 m"},           # 300, 500
    {**_F104_BASE, "width_mm": "11.811024 in", "eff_depth_mm": "50 cm"},    # 300.000, 500
])
def test_f104_u(params):
    assert _ok(F104, params)["shear_capacity_kn"] == pytest.approx(71.54, abs=0.02)


@pytest.mark.parametrize("missing", ["width_mm", "eff_depth_mm"])
def test_f104_m(missing):
    params = {k: v for k, v in _F104_BASE.items() if k != missing}
    _asks(F104, params, missing, _F104_OUT)


@pytest.mark.parametrize("fc", [68.89, 80])
def test_f104_b_sqrt_fc_cap(fc):
    # sqrt(68.89) = 8.3 exactly, the cap; 80 is past it and is held at 8.3:
    # 0.75 x 0.66 x 0.8165 x 0.21544 x 8.3 x 150000 = 108408 N for both
    r = _ok(F104, {**_F104_BASE, "fc_mpa": fc})
    assert r["shear_capacity_kn"] == pytest.approx(108.41, abs=0.01)


@pytest.mark.parametrize("d, cap", [
    (250, 43.81),   # ls = sqrt(2/2) = 1, the limit: 0.75 x 0.66 x 0.21544 x 5.4772 x 75000
    (200, 35.05),   # ls = sqrt(2/1.8) = 1.054 held at 1: same stress x 60000
])
def test_f104_b_size_factor_limit(d, cap):
    r = _ok(F104, {**_F104_BASE, "eff_depth_mm": d})
    assert r["shear_capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("rho", [0.02, 0.03])
def test_f104_b_ec_rho_cap(rho):
    # rho held at 0.02: 0.12 x 1.63246 x (60)^(1/3) = 3.91487 x 150000 = 115035 N
    r = _ok(F104, {**_F104_BASE, "rho_l": rho, "code": "EC"})
    assert r["shear_capacity_kn"] == pytest.approx(115.04, abs=0.01)


# ---------------------------------------------------------------- F105 lap length
F105 = "rebar_lap_length"
_F105_OUT = ("lap_length_mm",)
_F105_BASE = {"bar_diameter_mm": 25, "fy_mpa": 420, "fc_mpa": 30}


@pytest.mark.parametrize("params, lap", [
    # db 16 (ps 0.8): 420 x 0.8 / (1.1 x 5.47723 x 1.5) x 16 = 594.860; x 1.3
    ({**_F105_BASE, "bar_diameter_mm": 16}, 773.3),
    # db 25 (ps 1.0): 420 / 9.03742 x 25 = 1161.836; x 1.3 = 1510.39
    (_F105_BASE, 1510.4),
    # EC db 16, fy 500, fck 30: fctd = 0.7 x 0.3 x 9.65489 / 1.5 = 1.35168;
    # fbd = 3.04129; lb = 4 x 434.783 / 3.04129 = 571.84; x 1.5 = 857.76
    ({"bar_diameter_mm": 16, "fy_mpa": 500, "fc_mpa": 30, "code": "EC"}, 857.8),
])
def test_f105_t(params, lap):
    assert _ok(F105, params)["lap_length_mm"] == pytest.approx(lap, abs=0.1)


@pytest.mark.parametrize("params, lap", [
    # 0.625 in = 15.875 mm; 4 ksi = 27.579 MPa (sqrt 5.25158):
    # 420 x 0.8 / (1.1 x 5.25158 x 1.5) x 15.875 = 615.573; x 1.3 = 800.24
    ({**_F105_BASE, "bar_diameter_mm": "0.625 in", "fc_mpa": "4 ksi"}, 800.2),
    # 60 ksi = 413.685 MPa (pg 1.0): 773.318 x 413.685 / 420 = 761.69
    ({**_F105_BASE, "bar_diameter_mm": 16, "fy_mpa": "60 ksi"}, 761.7),
])
def test_f105_u(params, lap):
    assert _ok(F105, params)["lap_length_mm"] == pytest.approx(lap, abs=0.1)


@pytest.mark.parametrize("missing", ["bar_diameter_mm", "fy_mpa"])
def test_f105_m(missing):
    params = {k: v for k, v in _F105_BASE.items() if k != missing}
    _asks(F105, params, missing, _F105_OUT)


@pytest.mark.parametrize("db, lap", [
    (19, 918.3),    # ps 0.8 on the limit: 420 x 0.8 / 9.03742 x 19 x 1.3
    (20, 1208.3),   # ps 1.0 past it: 420 / 9.03742 x 20 x 1.3
])
def test_f105_b_bar_size_factor(db, lap):
    r = _ok(F105, {**_F105_BASE, "bar_diameter_mm": db})
    assert r["lap_length_mm"] == pytest.approx(lap, abs=0.1)


@pytest.mark.parametrize("fy, lap", [
    (420, 1510.4),   # pg 1.0 on the limit
    (421, 1741.1),   # pg 1.15: 421 x 1.15 / 9.03742 x 25 x 1.3
    (550, 2274.6),   # pg 1.15 on the next limit: 550 x 1.15 / 9.03742 x 32.5
    (551, 2575.9),   # pg 1.3: 551 x 1.3 / 9.03742 x 32.5
])
def test_f105_b_grade_factor_steps(fy, lap):
    r = _ok(F105, {**_F105_BASE, "fy_mpa": fy})
    assert r["lap_length_mm"] == pytest.approx(lap, abs=0.1)


@pytest.mark.parametrize("db, lap", [
    (32, 1715.5),   # n2 = 1 on the limit: 8 x 434.783 / 3.04129 x 1.5
    (40, 2330.9),   # n2 = 0.92: fbd 2.79799; 10 x 434.783 / 2.79799 x 1.5
])
def test_f105_b_ec_large_bar(db, lap):
    r = _ok(F105, {**_F105_BASE, "bar_diameter_mm": db, "fy_mpa": 500, "code": "EC"})
    assert r["lap_length_mm"] == pytest.approx(lap, abs=0.1)


@pytest.mark.parametrize("conf, lap", [
    (2.5, 906.2),    # the declared cap: 420 / (1.1 x 5.47723 x 2.5) x 25 x 1.3
    (2.0, 1132.8),   # under it: 420 / (1.1 x 5.47723 x 2.0) x 25 x 1.3 = 1132.79
])
def test_f105_b_confinement_on_cap(conf, lap):
    r = _ok(F105, {**_F105_BASE, "confinement_ratio": conf})
    assert r["lap_length_mm"] == pytest.approx(lap, abs=0.1)


@pytest.mark.parametrize("params", [
    {**_F105_BASE, "confinement_ratio": 3.0},   # past the 2.5 cap: ask, never stretch
    {**_F105_BASE, "bar_diameter_mm": 3},       # under the 4 mm floor
])
def test_f105_b_past_declared_range_refused(params):
    _refuses(F105, params, _F105_OUT)


# ---------------------------------------------------------------- F106 shear stress
F106 = "shear_stress_check"
_F106_OUT = ("shear_stress_n_mm2",)


@pytest.mark.parametrize("params, v", [
    ({"v_kn": 150, "b_mm": 300, "d_mm": 500}, 1.0),     # 150000 / 150000
    ({"v_kn": 300, "b_mm": 400, "d_mm": 600}, 1.25),    # 300000 / 240000
])
def test_f106_t(params, v):
    assert _ok(F106, params)["shear_stress_n_mm2"] == pytest.approx(v, abs=1e-3)


@pytest.mark.parametrize("params", [
    {"v_kn": "150000 N", "b_mm": "30 cm", "d_mm": "0.5 m"},              # 150 kN, 300, 500
    {"v_kn": 150, "b_mm": "11.811024 in", "d_mm": "19.685039 in"},       # 300.000, 500.000
])
def test_f106_u(params):
    assert _ok(F106, params)["shear_stress_n_mm2"] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("params, missing", [
    ({"v_kn": 150, "b_mm": 300}, "d_mm"),
    ({"b_mm": 300, "d_mm": 500}, "v_kn"),
])
def test_f106_m(params, missing):
    _asks(F106, params, missing, _F106_OUT)


@pytest.mark.parametrize("v_kn, v", [
    (0, 0.0),      # no shear force: zero stress
    (75, 0.5),     # 75000 / 150000
])
def test_f106_b_zero_force(v_kn, v):
    r = _ok(F106, {"v_kn": v_kn, "b_mm": 300, "d_mm": 500})
    assert r["shear_stress_n_mm2"] == pytest.approx(v, abs=1e-3)


@pytest.mark.parametrize("b, v", [
    (20, 10.0),   # the declared floor: 100000 / (20 x 500)
    (25, 8.0),    # just above it: 100000 / (25 x 500)
])
def test_f106_b_width_on_floor(b, v):
    assert _ok(F106, {"v_kn": 100, "b_mm": b, "d_mm": 500})["shear_stress_n_mm2"] == pytest.approx(v)


@pytest.mark.parametrize("b", [19, 0])
def test_f106_b_width_under_floor_refused(b):
    # under the 20 mm floor (0 would divide by zero): refused, no stress
    _refuses(F106, {"v_kn": 100, "b_mm": b, "d_mm": 500}, _F106_OUT)


# ---------------------------------------------------------------- F107 bolt shear
# Hand source: AISC 360-16 J3.6 (LRFD): phi Rn = 0.75 Fnv Ab n;
# EN 1993-1-8 Table 3.4: Fv,Rd = av fub A n / 1.25, av 0.6 for classes
# 4.6/5.6/8.8 and 0.5 for 4.8/5.8/6.8/10.9/12.9; ISO 898-1: M20 tensile
# stress area 245 mm2, fub 800 (8.8) and 1000 (10.9).
F107 = "bolt_shear_capacity"
_F107_OUT = ("capacity_kn",)


@pytest.mark.parametrize("params, cap", [
    ({"bolt_area_mm2": 300}, 83.70),                                 # 0.75 x 372 x 300 (stated Fnv)
    ({"bolt_area_mm2": 300, "n_shear_planes": 2}, 167.40),           # x 2 planes
    ({"bolt_area_mm2": "M20", "shear_strength_mpa": 469}, 86.18),    # 0.75 x 469 x 245
    ({"bolt_area_mm2": "M20", "code": "EC", "grade": "8.8"}, 94.08),   # 0.6 x 800 x 245 / 1.25
    ({"bolt_area_mm2": "M20", "code": "EC", "grade": "10.9"}, 98.0),   # 0.5 x 1000 x 245 / 1.25
])
def test_f107_t(params, cap):
    assert _ok(F107, params)["capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("params, cap", [
    # 53.954 ksi = 371.9997 MPa: 0.75 x 372.0 x 300 = 83.70
    ({"bolt_area_mm2": 300, "shear_strength_mpa": "53.954 ksi"}, 83.70),
    # 68.0215 ksi = 468.992 MPa: 0.75 x 468.992 x 245 = 86.18
    ({"bolt_area_mm2": "M20", "shear_strength_mpa": "68.0215 ksi"}, 86.18),
])
def test_f107_u_strength(params, cap):
    assert _ok(F107, params)["capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: a bolt area given with a unit is refused as 'not a "
    "tensile stress area or a nominal metric size'; other area inputs convert it"))
@pytest.mark.parametrize("area", [
    "3 cm2",       # = 300 mm2 -> 0.75 x 372 x 300 = 83.70
    "0.465 in2",   # x 645.16 = 299.999 mm2 -> 83.70
])
def test_f107_u_area_with_unit(area):
    assert _ok(F107, {"bolt_area_mm2": area})["capacity_kn"] == pytest.approx(83.70, abs=0.01)


@pytest.mark.parametrize("params", [{}, {"shear_strength_mpa": 372, "n_shear_planes": 2}])
def test_f107_m(params):
    _asks(F107, params, "bolt_area_mm2", _F107_OUT)


@pytest.mark.parametrize("fub, cap", [
    # 1000 MPa is class 10.9 only: av 0.5 -> 0.5 x 1000 x 245 / 1.25 = 98.0
    (1000, 98.0),
    # 400 MPa is class 4.6 (0.6) and class 4.8 (0.5): no class given, the
    # stated 0.6 stays -> 0.6 x 400 x 245 / 1.25 = 47.04
    (400, 47.04),
])
def test_f107_b_ec_class_from_fub(fub, cap):
    r = _ok(F107, {"bolt_area_mm2": "M20", "code": "EC", "shear_strength_mpa": fub})
    assert r["capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("planes, cap", [
    (4, 334.80),   # the declared limit: 0.75 x 372 x 300 x 4
    (3, 251.10),   # under it: 0.75 x 372 x 300 x 3
])
def test_f107_b_four_planes_on_limit(planes, cap):
    r = _ok(F107, {"bolt_area_mm2": 300, "n_shear_planes": planes})
    assert r["capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("params", [
    {"bolt_area_mm2": 300, "n_shear_planes": 5},          # past the 4-plane limit
    {"bolt_area_mm2": 0},                                 # no area
    {"bolt_area_mm2": "M21"},                             # not an ISO coarse size
    {"bolt_area_mm2": "M20", "code": "EC", "grade": "9.9"},   # class not in the table
])
def test_f107_b_refused(params):
    _refuses(F107, params, _F107_OUT)


# ---------------------------------------------------------------- F108 fillet weld
# Hand source: throat = 0.707 x leg (equal-leg fillet);
# AISC 360-16 J2.4 (LRFD): phi Rn = 0.75 x 0.60 FEXX x throat x L (stated
# FEXX 490 MPa when none is given);
# EN 1993-1-8 4.5.3.3: fvw,d = fu / (sqrt 3 x bw x 1.25); EN 1993-1-1
# Table 3.1 / EN 1993-1-8 Table 4.1: S235 (360, 0.80), S275 (430, 0.85),
# S355 (510, 0.90).
F108 = "weld_capacity"
_F108_OUT = ("capacity_kn", "capacity_per_mm_kn")


@pytest.mark.parametrize("params, cap", [
    # throat 5.656; 0.45 x 490 x 5.656 = 1247.148 N/mm x 200 = 249.43 kN
    ({"leg_size_mm": 8, "length_mm": 200}, 249.43),
    # throat 4.242; 0.45 x 480 x 4.242 = 916.272 x 150 = 137.44
    ({"leg_size_mm": 6, "length_mm": 150, "electrode_strength_mpa": 480}, 137.44),
    # EC: 510 / (1.73205 x 0.9 x 1.25) = 261.729 x 5.656 x 200 = 296.07
    ({"leg_size_mm": 8, "length_mm": 200, "code": "EC", "electrode_strength_mpa": 510,
      "beta_w": 0.9}, 296.07),
])
def test_f108_t(params, cap):
    assert _ok(F108, params)["capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("params", [
    {"leg_size_mm": "0.8 cm", "length_mm": "20 cm"},       # 8 mm, 200 mm
    {"leg_size_mm": 8, "length_mm": "7.874016 in"},        # 200.000 mm
])
def test_f108_u(params):
    assert _ok(F108, params)["capacity_kn"] == pytest.approx(249.43, abs=0.01)


@pytest.mark.parametrize("params, missing", [
    ({"length_mm": 200}, "leg_size_mm"),
    ({"leg_size_mm": 8}, "length_mm"),
])
def test_f108_m(params, missing):
    _asks(F108, params, missing, _F108_OUT)


@pytest.mark.parametrize("grade, cap", [
    # S275: 430 / (1.73205 x 0.85 x 1.25) = 233.659 x 5.656 x 200 = 264.31
    ("S275", 264.31),
    # S235: 360 / (1.73205 x 0.80 x 1.25) = 207.846 x 5.656 x 200 = 235.12
    ("S235", 235.12),
])
def test_f108_b_ec_parent_metal_table(grade, cap):
    r = _ok(F108, {"leg_size_mm": 8, "length_mm": 200, "code": "EC", "steel_grade": grade})
    assert r["capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("leg, cap", [
    (2, 62.36),   # the declared floor: 0.45 x 490 x 1.414 x 200
    (3, 93.54),   # above it: 0.45 x 490 x 2.121 x 200 = 93.536
])
def test_f108_b_leg_on_floor(leg, cap):
    r = _ok(F108, {"leg_size_mm": leg, "length_mm": 200})
    assert r["capacity_kn"] == pytest.approx(cap, abs=0.01)


@pytest.mark.parametrize("params", [
    {"leg_size_mm": 1.9, "length_mm": 200},                                  # under the 2 mm floor
    {"leg_size_mm": 8, "length_mm": 0},                                      # no length
    {"leg_size_mm": 8, "length_mm": 200, "code": "EC", "steel_grade": "S300"},  # grade not tabulated
])
def test_f108_b_refused(params):
    _refuses(F108, params, _F108_OUT)


# ---------------------------------------------------------------- F109 precast erection
# Lift planning practice (BS 7121-1 / ASME B30.5 pick charts): effective
# hook load = (member weight + rigging) x dynamic allowance; utilisation =
# effective / allowable pick weight at the radius x 100; safe while <= 100 %.
F109 = "precast_beam_erection_check"
_F109_OUT = ("effective_lift_weight_t", "utilization_pct", "is_safe")
_F109_BASE = {"beam_weight_t": 12, "beam_length_m": 10, "crane_capacity_t": 25,
              "lift_radius_m": 15}


@pytest.mark.parametrize("params, eff, util", [
    # (12 + 0.5) x 1.25 = 15.625 t; 15.625 / 25 = 62.5 %
    (_F109_BASE, 15.625, 62.5),
    # (20 + 1.0) x 1.1 = 23.1 t; 23.1 / 30 = 77.0 %
    ({"beam_weight_t": 20, "beam_length_m": 14, "crane_capacity_t": 30,
      "lift_radius_m": 18, "rigging_weight_t": 1.0, "dynamic_factor": 1.1}, 23.1, 77.0),
])
def test_f109_t(params, eff, util):
    r = _ok(F109, params)
    assert r["effective_lift_weight_t"] == pytest.approx(eff, abs=0.01)
    assert r["utilization_pct"] == pytest.approx(util, abs=0.1)
    assert r["is_safe"] is True


@pytest.mark.parametrize("params", [
    dict(_F109_BASE, beam_weight_t="12000 kg"),                       # 12 t
    dict(_F109_BASE, crane_capacity_t="25000 kg", beam_length_m="393.7 in"),  # 25 t, 10.0 m
])
def test_f109_u(params):
    r = _ok(F109, params)
    assert r["effective_lift_weight_t"] == pytest.approx(15.625, abs=0.01)
    assert r["utilization_pct"] == pytest.approx(62.5, abs=0.1)


@pytest.mark.parametrize("missing", ["crane_capacity_t", "beam_weight_t"])
def test_f109_m(missing):
    params = {k: v for k, v in _F109_BASE.items() if k != missing}
    _asks(F109, params, missing, _F109_OUT)


@pytest.mark.parametrize("weight, util, safe", [
    # (19.5 + 0.5) x 1.25 = 25.0 t = 100.0 % of 25 t: on the limit, still safe
    (19.5, 100.0, True),
    # (19.6 + 0.5) x 1.25 = 25.125 t = 100.5 %: past it, not safe
    (19.6, 100.5, False),
])
def test_f109_b_on_and_past_capacity(weight, util, safe):
    r = _ok(F109, dict(_F109_BASE, beam_weight_t=weight))
    assert r["utilization_pct"] == pytest.approx(util, abs=0.1)
    assert r["is_safe"] is safe


@pytest.mark.parametrize("params", [
    dict(_F109_BASE, crane_capacity_t=0),       # no allowable pick weight
    dict(_F109_BASE, dynamic_factor=0.9),       # an allowance below 1 lightens the load
])
def test_f109_b_refused(params):
    _refuses(F109, params, _F109_OUT)


# ---------------------------------------------------------------- F110 plumbing programme
# The calculator's declared sequence: submittal 7 d, procurement 21 d,
# 1st fix ceil(A/40), 2nd fix ceil(A/60), final fix ceil(A/80), water
# connection 7 d, testing ceil(A/50), handover 3 d, each worked stage at
# least 1 d; A = floor area x floors. Fixed stages: 7 + 21 + 7 + 3 = 38 d.
F110 = "plumbing_flow_programme"
_F110_OUT = ("total_days", "stages")


@pytest.mark.parametrize("params, total", [
    # A 2400: 60 + 40 + 30 + 48 + 38 = 216
    ({"floor_area_m2": 1200, "num_floors": 2}, 216),
    # A 900: ceil 22.5 = 23, 15, ceil 11.25 = 12, 18; 23 + 15 + 12 + 18 + 38 = 106
    ({"floor_area_m2": 300, "num_floors": 3}, 106),
])
def test_f110_t(params, total):
    r = _ok(F110, params)
    assert r["total_days"] == total
    last = list(r["stages"].values())[-1]
    assert last["cumulative_days"] == total


@pytest.mark.parametrize("params", [
    # 12000 ft2 x 0.09290304 = 1114.836 m2: 28 + 19 + 14 + 23 + 38 = 122
    {"floor_area_m2": "12000 ft2"},
    # 0.1114836 ha = 1114.836 m2, same programme
    {"floor_area_m2": "0.1114836 ha"},
])
def test_f110_u(params):
    assert _ok(F110, params)["total_days"] == 122


@pytest.mark.parametrize("params", [{}, {"num_floors": 4}])
def test_f110_m(params):
    _asks(F110, params, "floor_area_m2", _F110_OUT)


@pytest.mark.parametrize("area, total", [
    (40, 42),   # every worked stage exactly 1 d: 4 + 38
    (41, 43),   # 1st fix past one day's output: ceil(41/40) = 2; 5 + 38
])
def test_f110_b_one_day_step(area, total):
    assert _ok(F110, {"floor_area_m2": area})["total_days"] == total


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: zero floor area returns a 42-day programme with four "
    "1-day installation stages instead of refusing; no area means no work"))
@pytest.mark.parametrize("params", [
    {"floor_area_m2": 0},
    {"floor_area_m2": 0, "num_floors": 5},
])
def test_f110_b_zero_area_refused(params):
    _refuses(F110, params, _F110_OUT)


# ---------------------------------------------------------------- F111 clash tolerance
# No standard publishes a clash tolerance per LOD: the calculator says the
# project BEP governs. So no tolerance VALUE is asserted here. What is
# checked has a source: the LOD scale itself (BIMForum LOD Specification:
# 100, 200, 300, 350, 400, 500) and that a more developed model is checked
# to a tolerance no looser than a less developed one.
F111 = "bim_clash_tolerance"
_F111_OUT = ("clash_tolerance_mm",)


@pytest.mark.parametrize("lower, higher", [(200, 300), (350, 400)])
def test_f111_t_tolerance_does_not_loosen_with_development(lower, higher):
    lo = _ok(F111, {"lod": lower})["clash_tolerance_mm"]
    hi = _ok(F111, {"lod": higher})["clash_tolerance_mm"]
    assert lo is not None and hi is not None
    assert hi <= lo


@pytest.mark.parametrize("lod", [300, 500])
def test_f111_t_lod_echoed(lod):
    assert _ok(F111, {"lod": lod})["lod"] == lod


def test_f111_m_default_is_disclosed():
    # lod has a default; when omitted the result must say a default was used
    r = _ok(F111, {})
    assert any(d.get("parameter") == "lod" for d in r.get("defaults_used") or [])
    r2 = _ok(F111, {"lod": 400})
    assert not any(d.get("parameter") == "lod" for d in r2.get("defaults_used") or [])


@pytest.mark.parametrize("lod", [
    250,   # inside the range but not a level on the LOD scale
    600,   # past the top of the scale
])
def test_f111_b_refused(lod):
    _refuses(F111, {"lod": lod}, _F111_OUT)
