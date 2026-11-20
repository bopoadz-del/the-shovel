"""Formula Layer 1, batch 6 (exam 5): six structural capacity formulas.

Same rules as batches 1-5 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation from the published equation,
written in the comment beside its parametrize row:
  * ACI 318-19 22.2 (rectangular block)   a = As fy / (0.85 fc b);
                                          phi Mn = 0.9 As fy (d - a/2)
  * EN 1992-1-1 3.1.7 / 6.1               fcd = fck/1.5, fyd = fyk/1.15,
                                          x = As fyd / (0.8 b fcd),
                                          MRd = As fyd (d - 0.4 x)
  * ACI 318-19 Table 22.5.5.1 (c)         phi Vc = 0.75 x 0.66 lambda_s
                                          rho^(1/3) sqrt(fc) bw d,
                                          lambda_s = sqrt(2/(1+0.004 d)) <= 1,
                                          sqrt(fc) <= 8.3
  * EN 1992-1-1 6.2.2 (6.2a, 6.2b, 6.3N)  VRd,c = 0.12 k (100 rho fck)^(1/3)
                                          bw d >= 0.035 k^1.5 sqrt(fck) bw d,
                                          k = 1 + sqrt(200/d) <= 2,
                                          rho <= 0.02
  * ACI 318-19 25.4.2.4 + 25.5.2 (B)      ld = fy psi_s psi_g /
                                          (1.1 sqrt(fc) (cb+Ktr)/db) db,
                                          (cb+Ktr)/db <= 2.5; lap = 1.3 ld;
                                          Table 25.4.2.5: psi_s = 0.8 for
                                          No. 19 and smaller, psi_g = 1.0 /
                                          1.15 / 1.3 for fy <= 420 / <= 550 /
                                          above
  * EN 1992-1-1 8.4.2-8.4.3, 8.7.3        fbd = 2.25 eta1 eta2 fctd,
                                          fctd = 0.7 x 0.3 fck^(2/3) / 1.5,
                                          eta2 = (132 - db)/100 above 32 mm,
                                          lb,rqd = db/4 x fyd/fbd,
                                          l0 = 1.5 lb,rqd (alpha6, >50% lapped)
  * AISC 360-16 J3.6 (LRFD)               phi Rn = 0.75 Fnv Ab n
  * EN 1993-1-8 Table 3.4                 Fv,Rd = alpha_v fub A n / 1.25;
                                          8.8: fub 800, alpha_v 0.6;
                                          10.9: fub 1000, alpha_v 0.5
  * ISO 898-1 tensile stress area         M20 245 mm2, M24 353 mm2
  * AISC 360-16 J2.4 (LRFD)               phi Rn = 0.75 x 0.60 FEXX x
                                          0.707 leg x L
  * EN 1993-1-8 4.5.3.3                   fvw,d = fu / (sqrt3 beta_w 1.25)
  * ACI 318-19 22.4.2                     P0 = 0.85 fc (Ag - Ast) + fy Ast;
                                          phi Pn,max = 0.65 x 0.80 P0 tied,
                                          0.75 x 0.85 P0 spiral
  * EN 1992-1-1 6.1                       NRd = fcd Ac + fyd As
None was copied from the product's output. A case where the product
disagrees with the hand value is marked ``xfail(strict=True)`` with a
one-line reason and is a suspected defect for the advisor; the expected
value is never moved to match. Every value test runs on two inputs worked
by hand the same way.

Unit factors (exact by definition): 1 in = 25.4 mm; 1 psi = 6.894757293 kPa;
1 ksi = 6.894757293 MPa.

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


# ------------------------------------------ F38 RC beam bending capacity

F38 = "rc_beam_moment_capacity"
_M = "moment_capacity_kn_m"


@pytest.mark.parametrize("As, fy, b, d, fc, m", [
    # a = 1500x420/(0.85x30x300) = 630000/7650 = 82.353;
    # 0.9 x 630000 x (500 - 41.176) = 260.153 kN.m
    (1500, 420, 300, 500, 30, 260.153),
    # a = 1000x500/(0.85x25x250) = 500000/5312.5 = 94.118;
    # 0.9 x 500000 x (450 - 47.059) = 181.324 kN.m
    (1000, 500, 250, 450, 25, 181.324),
])
def test_f38_t_aci(As, fy, b, d, fc, m):
    r = _ok(F38, {"steel_area_mm2": As, "fy_mpa": fy, "width_mm": b,
                  "eff_depth_mm": d, "fc_mpa": fc, "code": "aci"})
    assert r[_M] == pytest.approx(m, rel=_REL)


@pytest.mark.parametrize("As, b, d, fc, m", [
    # fcd 20, fyd 434.783; x = 652174/(0.8x300x20) = 135.870;
    # z = 500 - 54.348 = 445.652; MRd = 652174 x 445.652 = 290.643 kN.m
    (1500, 300, 500, 30, 290.643),
    # fcd 16.667, fyd 434.783; x = 521739/(0.8x250x16.667) = 156.522;
    # z = 450 - 62.609 = 387.391; MRd = 202.117 kN.m
    (1200, 250, 450, 25, 202.117),
])
def test_f38_t_eurocode(As, b, d, fc, m):
    r = _ok(F38, {"steel_area_mm2": As, "fy_mpa": 500, "width_mm": b,
                  "eff_depth_mm": d, "fc_mpa": fc, "code": "eurocode"})
    assert r[_M] == pytest.approx(m, rel=_REL)


@pytest.mark.parametrize("params, m", [
    # b = 12 in = 304.8 mm, d = 20 in = 508 mm; a = 630000/(0.85x30x304.8)
    # = 81.057; 0.9 x 630000 x (508 - 40.529) = 265.057 kN.m
    ({"steel_area_mm2": 1500, "fy_mpa": 420, "width_mm": "12 in",
      "eff_depth_mm": "20 in", "fc_mpa": 30}, 265.057),
    # b 10 in = 254, d 18 in = 457.2, fc 4000 psi = 27.579 MPa;
    # a = 420000/(0.85x27.579x254) = 70.535; 0.9x420000x(457.2-35.268) = 159.490
    ({"steel_area_mm2": 1000, "fy_mpa": 420, "width_mm": "10 in",
      "eff_depth_mm": "18 in", "fc_mpa": "4000 psi"}, 159.490),
])
def test_f38_u_us_units(params, m):
    r = _ok(F38, {**params, "code": "aci"})
    assert r[_M] == pytest.approx(m, rel=_REL)


@pytest.mark.parametrize("drop, word", [("fc_mpa", "fc"), ("eff_depth_mm", "depth")])
def test_f38_m_missing_input_asks(drop, word):
    p = {"steel_area_mm2": 1500, "fy_mpa": 420, "width_mm": 300,
         "eff_depth_mm": 500, "fc_mpa": 30, "code": "aci"}
    p.pop(drop)
    _asks(F38, p, word, (_M,))


@pytest.mark.parametrize("code", ["aci", "eurocode"])
def test_f38_b_no_steel_gives_zero(code):
    # As = 0: no tension force, so a = 0 / x = 0 and the capacity is 0
    r = _ok(F38, {"steel_area_mm2": 0, "fy_mpa": 420, "width_mm": 300,
                  "eff_depth_mm": 500, "fc_mpa": 30, "code": code})
    assert r[_M] == pytest.approx(0.0, abs=0.005)


# ---------------------------------------- F39 RC section shear capacity

F39 = "rc_beam_shear_capacity"
_V = "shear_capacity_kn"


@pytest.mark.parametrize("b, d, fc, rho, v", [
    # lambda_s = sqrt(2/3) = 0.81650; 0.01^(1/3) = 0.21544; sqrt30 = 5.47723;
    # 0.75 x 0.66 x 0.8165 x 0.21544 x 5.47723 x 300 x 500 = 71.539 kN
    (300, 500, 30, 0.01, 71.539),
    # lambda_s = sqrt(2/2.6) = 0.87706; 0.015^(1/3) = 0.24662; sqrt25 = 5;
    # x 250 x 400 x 0.495 = 53.535 kN
    (250, 400, 25, 0.015, 53.535),
])
def test_f39_t_aci(b, d, fc, rho, v):
    r = _ok(F39, {"width_mm": b, "eff_depth_mm": d, "fc_mpa": fc,
                  "rho_l": rho, "code": "aci"})
    assert r[_V] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("b, d, fc, rho, v", [
    # k = 1 + sqrt(0.4) = 1.63246; (100x0.01x30)^(1/3) = 3.10723;
    # 0.12 x 1.63246 x 3.10723 = 0.60869 MPa (> vmin 0.39984) x 150000 = 91.304 kN
    (300, 500, 30, 0.01, 91.304),
    # k = 1 + sqrt(0.5) = 1.70711; (37.5)^(1/3) = 3.34716; v = 0.68568
    # (> vmin 0.39033) x 100000 = 68.568 kN
    (250, 400, 25, 0.015, 68.568),
])
def test_f39_t_eurocode(b, d, fc, rho, v):
    r = _ok(F39, {"width_mm": b, "eff_depth_mm": d, "fc_mpa": fc,
                  "rho_l": rho, "code": "eurocode"})
    assert r[_V] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("params, v", [
    # bw 12 in = 304.8, d 20 in = 508; lambda_s = sqrt(2/3.032) = 0.81218;
    # 0.495 x 0.81218 x 0.21544 x 5.47723 x 304.8 x 508 = 73.456 kN
    ({"width_mm": "12 in", "eff_depth_mm": "20 in", "fc_mpa": 30}, 73.456),
    # fc 4000 psi = 27.579 MPa, sqrt = 5.25158; 71.539 x 5.25158/5.47723 = 68.592
    ({"width_mm": 300, "eff_depth_mm": 500, "fc_mpa": "4000 psi"}, 68.592),
])
def test_f39_u_us_units(params, v):
    r = _ok(F39, {**params, "rho_l": 0.01, "code": "aci"})
    assert r[_V] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("drop, word", [("width_mm", "width"), ("fc_mpa", "fc")])
def test_f39_m_missing_input_asks(drop, word):
    p = {"width_mm": 300, "eff_depth_mm": 500, "fc_mpa": 30, "code": "aci"}
    p.pop(drop)
    _asks(F39, p, word, (_V,))


@pytest.mark.parametrize("d, fc, v", [
    # d = 250: 1 + 0.004 x 250 = 2, lambda_s = 1 exactly (on the cap);
    # 0.495 x 0.21544 x 5.47723 x 300 x 250 = 43.809 kN
    (250, 30, 43.809),
    # fc = 100: sqrt = 10 capped at 8.3; 0.495 x 0.8165 x 0.21544 x 8.3
    # x 300 x 500 = 108.408 kN
    (500, 100, 108.408),
])
def test_f39_b_aci_caps(d, fc, v):
    r = _ok(F39, {"width_mm": 300, "eff_depth_mm": d, "fc_mpa": fc,
                  "rho_l": 0.01, "code": "aci"})
    assert r[_V] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("b, d, rho, v", [
    # d = 150: 1 + sqrt(200/150) = 2.155 capped at k = 2.0;
    # 0.12 x 2 x 3.10723 = 0.74574 MPa x 300 x 150 = 33.558 kN
    (300, 150, 0.01, 33.558),
    # rho 0.03 capped at 0.02: (100x0.02x30)^(1/3) = 3.91487;
    # 0.12 x 1.63246 x 3.91487 = 0.76690 x 150000 = 115.035 kN
    (300, 500, 0.03, 115.035),
    # rho 0.001: 0.12 x 1.63246 x 3^(1/3)=1.44225 -> 0.28253 < vmin
    # 0.035 x 1.63246^1.5 x 5.47723 = 0.39984; 0.39984 x 1000 x 500 = 199.922 kN
    (1000, 500, 0.001, 199.922),
])
def test_f39_b_eurocode_caps_and_minimum(b, d, rho, v):
    r = _ok(F39, {"width_mm": b, "eff_depth_mm": d, "fc_mpa": 30,
                  "rho_l": rho, "code": "eurocode"})
    assert r[_V] == pytest.approx(v, rel=_REL)


# ------------------------------------------------- F40 rebar lap length

F40 = "rebar_lap_length"
_L = "lap_length_mm"


@pytest.mark.parametrize("db, fc, conf, lap", [
    # psi_s 1.0 (No. 20), psi_g 1.0 (fy 420): ld = 420/(1.1 x 5.29150 x 1.5)
    # x 20 = 962.09; lap = 1.3 ld = 1250.72 mm
    (20, 28, 1.5, 1250.72),
    # ld = 420/(1.1 x 5.91608 x 2.0) x 25 = 806.74; lap = 1048.76 mm
    (25, 35, 2.0, 1048.76),
])
def test_f40_t_aci(db, fc, conf, lap):
    r = _ok(F40, {"bar_diameter_mm": db, "fy_mpa": 420, "fc_mpa": fc,
                  "confinement_ratio": conf, "code": "aci"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


@pytest.mark.parametrize("db, fc, lap", [
    # fctd = 0.7 x 0.3 x 30^(2/3) / 1.5 = 1.35168; fbd = 2.25 x that = 3.04129;
    # lb = 20/4 x 434.783/3.04129 = 714.80; l0 = 1.5 lb = 1072.20 mm
    (20, 30, 1072.20),
    # fctd(25) = 1.19698; fbd = 2.69321; lb = 4 x 434.783/2.69321 = 645.75;
    # l0 = 968.62 mm
    (16, 25, 968.62),
])
def test_f40_t_eurocode(db, fc, lap):
    r = _ok(F40, {"bar_diameter_mm": db, "fy_mpa": 500, "fc_mpa": fc,
                  "code": "eurocode"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


@pytest.mark.parametrize("params, lap", [
    # fc 4000 psi = 27.579 MPa, sqrt 5.25158; ld = 420/(1.1x5.25158x1.5)x20
    # = 969.41; lap 1260.23 mm
    ({"bar_diameter_mm": 20, "fy_mpa": 420, "fc_mpa": "4000 psi"}, 1260.23),
    # fy 60 ksi = 413.685 MPa (psi_g 1.0), db 25, fc 28:
    # ld = 413.685/(1.1x5.29150x1.5) x 25 = 1184.53; lap 1539.89 mm
    ({"bar_diameter_mm": 25, "fy_mpa": "60 ksi", "fc_mpa": 28}, 1539.89),
])
def test_f40_u_us_units(params, lap):
    r = _ok(F40, {**params, "confinement_ratio": 1.5, "code": "aci"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


@pytest.mark.parametrize("drop, word", [("bar_diameter_mm", "diameter"), ("fc_mpa", "fc")])
def test_f40_m_missing_input_asks(drop, word):
    p = {"bar_diameter_mm": 20, "fy_mpa": 420, "fc_mpa": 28, "code": "aci"}
    p.pop(drop)
    _asks(F40, p, word, (_L,))


@pytest.mark.parametrize("fy, lap", [
    # fy 500 (420 < fy <= 550): psi_g 1.15; ld = 500x1.15/(1.1x5.2915x1.5)
    # x 20 = 1317.15; lap 1712.29 mm
    (500, 1712.29),
    # fy 600 (> 550): psi_g 1.3; ld = 600x1.3/(8.73098) x 20 = 1786.74;
    # lap 2322.76 mm
    (600, 2322.76),
])
def test_f40_b_aci_grade_factor_steps(fy, lap):
    r = _ok(F40, {"bar_diameter_mm": 20, "fy_mpa": fy, "fc_mpa": 28,
                  "confinement_ratio": 1.5, "code": "aci"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


@pytest.mark.parametrize("conf, lap", [
    # on the cap: ld = 420/(1.1 x 5.2915 x 2.5) x 20 = 577.25; lap 750.43 mm
    (2.5, 750.43),
    # below it: ld = 420/(1.1 x 5.2915 x 2.0) x 20 = 721.57; lap 938.04 mm
    (2.0, 938.04),
])
def test_f40_b_aci_confinement_on_and_below_the_cap(conf, lap):
    r = _ok(F40, {"bar_diameter_mm": 20, "fy_mpa": 420, "fc_mpa": 28,
                  "confinement_ratio": conf, "code": "aci"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


@pytest.mark.parametrize("conf", [2.6, 3.0])
def test_f40_b_aci_confinement_past_the_cap_is_never_credited(conf):
    # (cb+Ktr)/db shall not exceed 2.5: a larger term may be refused or
    # used as 2.5, but the lap may never be shorter than the 2.5 lap,
    # 750.43 mm (hand value above)
    env = run_calculation(F40, {"bar_diameter_mm": 20, "fy_mpa": 420, "fc_mpa": 28,
                                "confinement_ratio": conf, "code": "aci"})
    lap = (env.get("result") or {}).get(_L)
    if env.get("status") == "success":
        assert lap >= 750.43 * (1 - _REL), env
    else:
        assert lap is None, env


@pytest.mark.parametrize("db, lap", [
    # db 16 (well below No. 19): psi_s 0.8; ld = 0.8 x 962.09 x 16/20
    # = 615.74; lap 800.46 mm
    (16, 800.46),
    # db 12: ld = 420/8.73098 = 48.1046 per mm x 0.8 x 12 = 461.80; lap 600.34
    (12, 600.34),
])
def test_f40_b_aci_small_bar_factor(db, lap):
    r = _ok(F40, {"bar_diameter_mm": db, "fy_mpa": 420, "fc_mpa": 28,
                  "confinement_ratio": 1.5, "code": "aci"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: psi_s = 0.8 is applied only to db <= 19.0 mm, but ACI "
    "Table 25.4.2.5 applies it to No. 19 bars, whose nominal diameter is "
    "19.1 mm (0.750 in); a No. 19 bar gets psi_s 1.0 and a 25 % longer lap"))
@pytest.mark.parametrize("db, lap", [
    # No. 19 nominal 19.1 mm, psi_s 0.8: ld = 420x0.8/(8.73098) x 19.1
    # = 735.03; lap 955.55 mm
    (19.1, 955.55),
    # #6 = 0.75 in = 19.05 mm, psi_s 0.8: ld = 733.11; lap 953.05 mm
    ("0.75 in", 953.05),
])
def test_f40_b_aci_no19_bar_takes_small_bar_factor(db, lap):
    r = _ok(F40, {"bar_diameter_mm": db, "fy_mpa": 420, "fc_mpa": 28,
                  "confinement_ratio": 1.5, "code": "aci"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


@pytest.mark.parametrize("db, lap", [
    # db 32 (on the limit): eta2 = 1; lb = 8 x 434.783/3.04129 = 1143.68;
    # l0 = 1715.52 mm
    (32, 1715.52),
    # db 40 (past it): eta2 = (132-40)/100 = 0.92; fbd = 2.79799;
    # lb = 10 x 434.783/2.79799 = 1553.91; l0 = 2330.87 mm
    (40, 2330.87),
])
def test_f40_b_eurocode_large_bar_bond(db, lap):
    r = _ok(F40, {"bar_diameter_mm": db, "fy_mpa": 500, "fc_mpa": 30,
                  "code": "eurocode"})
    assert r[_L] == pytest.approx(lap, rel=_REL)


# ---------------------------------------------- F41 bolt shear capacity

F41 = "bolt_shear_capacity"
_C = "capacity_kn"


@pytest.mark.parametrize("A, fnv, n, cap", [
    # 0.75 x 372 x 245 x 1 = 68,355 N = 68.355 kN
    (245, 372, 1, 68.355),
    # 0.75 x 469 x 353 x 2 = 248,335 N = 248.335 kN
    (353, 469, 2, 248.335),
])
def test_f41_t_aisc(A, fnv, n, cap):
    r = _ok(F41, {"bolt_area_mm2": A, "shear_strength_mpa": fnv,
                  "n_shear_planes": n, "code": "aci"})
    assert r[_C] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("A, grade, n, cap", [
    # class 8.8: fub 800, alpha_v 0.6; 0.6 x 800 x 245 / 1.25 = 94.080 kN
    (245, "8.8", 1, 94.080),
    # class 10.9: fub 1000, alpha_v 0.5; 0.5 x 1000 x 353 x 2 / 1.25 = 282.400 kN
    (353, "10.9", 2, 282.400),
])
def test_f41_t_eurocode_property_class(A, grade, n, cap):
    r = _ok(F41, {"bolt_area_mm2": A, "grade": grade, "n_shear_planes": n,
                  "code": "eurocode"})
    assert r[_C] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("size, code, extra, cap", [
    # M20 tensile stress area 245 mm2 (ISO 898-1): 0.75 x 372 x 245 = 68.355 kN
    ("M20", "aci", {"shear_strength_mpa": 372}, 68.355),
    # M24 = 353 mm2, class 8.8: 0.6 x 800 x 353 / 1.25 = 135.552 kN
    ("M24", "eurocode", {"grade": "8.8"}, 135.552),
])
def test_f41_u_nominal_metric_size(size, code, extra, cap):
    r = _ok(F41, {"bolt_area_mm2": size, "n_shear_planes": 1, "code": code, **extra})
    assert r[_C] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("code", ["aci", "eurocode"])
def test_f41_m_no_bolt_area_asks(code):
    _asks(F41, {"shear_strength_mpa": 400, "n_shear_planes": 1, "code": code},
          "area", (_C,))


@pytest.mark.parametrize("params", [
    {"bolt_area_mm2": 245, "n_shear_planes": 0},
    {"bolt_area_mm2": 0, "n_shear_planes": 1},
])
def test_f41_b_zero_area_or_planes_refused(params):
    _refuses(F41, {**params, "shear_strength_mpa": 372, "code": "aci"}, (_C,))


@pytest.mark.parametrize("fub, cap", [
    # fub 1000 belongs only to 10.9 -> alpha_v 0.5: 0.5x1000x245/1.25 = 98.000
    (1000, 98.000),
    # fub 400 is shared by 4.6 (0.6) and 4.8 (0.5): keep 0.6;
    # 0.6 x 400 x 245 / 1.25 = 47.040 kN
    (400, 47.040),
])
def test_f41_b_eurocode_alpha_from_fub_when_class_not_given(fub, cap):
    r = _ok(F41, {"bolt_area_mm2": 245, "shear_strength_mpa": fub,
                  "n_shear_planes": 1, "code": "eurocode"})
    assert r[_C] == pytest.approx(cap, rel=_REL)


# ---------------------------------------------- F42 fillet weld capacity

F42 = "weld_capacity"


@pytest.mark.parametrize("leg, L, fexx, cap", [
    # throat 0.707 x 8 = 5.656; 0.75 x 0.6 x 490 x 5.656 x 100 = 124.715 kN
    (8, 100, 490, 124.715),
    # throat 4.242; 0.45 x 480 x 4.242 x 250 = 229.068 kN
    (6, 250, 480, 229.068),
])
def test_f42_t_aisc(leg, L, fexx, cap):
    r = _ok(F42, {"leg_size_mm": leg, "length_mm": L,
                  "electrode_strength_mpa": fexx, "code": "aci"})
    assert r[_C] == pytest.approx(cap, rel=_REL)
    assert r["effective_throat_mm"] == pytest.approx(0.707 * leg, rel=_REL)


@pytest.mark.parametrize("leg, L, fu, beta, cap", [
    # fvw,d = 510/(1.73205 x 0.9 x 1.25) = 261.732 MPa; x 5.656 x 100 = 148.036 kN
    (8, 100, 510, 0.9, 148.036),
    # fvw,d = 360/(1.73205 x 0.8 x 1.25) = 207.846; x 3.535 x 200 = 146.947 kN
    (5, 200, 360, 0.8, 146.947),
])
def test_f42_t_eurocode(leg, L, fu, beta, cap):
    r = _ok(F42, {"leg_size_mm": leg, "length_mm": L, "electrode_strength_mpa": fu,
                  "beta_w": beta, "code": "eurocode"})
    assert r[_C] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("leg, L, cap", [
    # 0.3125 in = 7.9375 mm, 4 in = 101.6 mm;
    # 0.45 x 490 x 0.707 x 7.9375 x 101.6 = 125.720 kN
    ("0.3125 in", "4 in", 125.720),
    # 8 mm leg, 10 in = 254 mm; 1.24715 kN/mm x 254 = 316.776 kN
    (8, "10 in", 316.776),
])
def test_f42_u_inches(leg, L, cap):
    r = _ok(F42, {"leg_size_mm": leg, "length_mm": L,
                  "electrode_strength_mpa": 490, "code": "aci"})
    assert r[_C] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("drop, word", [("leg_size_mm", "leg"), ("length_mm", "length")])
def test_f42_m_missing_input_asks(drop, word):
    p = {"leg_size_mm": 8, "length_mm": 100, "electrode_strength_mpa": 490, "code": "aci"}
    p.pop(drop)
    _asks(F42, p, word, (_C,))


@pytest.mark.parametrize("L, ratio", [(100, 1.0), (300, 3.0)])
def test_f42_b_capacity_is_linear_in_length(L, ratio):
    # same weld per mm: 1.24715 kN/mm; 100 mm -> 124.715, 300 mm -> 374.145
    r = _ok(F42, {"leg_size_mm": 8, "length_mm": L,
                  "electrode_strength_mpa": 490, "code": "aci"})
    assert r["capacity_per_mm_kn"] == pytest.approx(1.24715, rel=_REL)
    assert r[_C] == pytest.approx(124.715 * ratio, rel=_REL)


# ------------------------------------- F43 RC column axial capacity

F43 = "column_axial_capacity"
_N = "axial_capacity_kn"


@pytest.mark.parametrize("Ag, As, fc, tie, cap", [
    # P0 = 0.85x30x157000 + 420x3000 = 4,003,500 + 1,260,000 = 5,263,500 N;
    # tied 0.65 x 0.80 x P0 = 2737.02 kN
    (160000, 3000, 30, "tied", 2737.02),
    # spiral 0.75 x 0.85 x 5,263,500 = 3355.48 kN
    (160000, 3000, 30, "spiral", 3355.48),
    # P0 = 0.85x40x245000 + 420x5000 = 8,330,000 + 2,100,000 = 10,430,000;
    # tied 5423.60 kN
    (250000, 5000, 40, "tied", 5423.60),
])
def test_f43_t_aci(Ag, As, fc, tie, cap):
    r = _ok(F43, {"gross_area_mm2": Ag, "steel_area_mm2": As, "fc_mpa": fc,
                  "fy_mpa": 420, "tie_type": tie, "code": "aci"})
    assert r[_N] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("Ag, As, fc, cap", [
    # 20 x 157000 + 434.783 x 3000 = 3,140,000 + 1,304,348 = 4444.35 kN
    (160000, 3000, 30, 4444.35),
    # 26.667 x 245000 + 434.783 x 5000 = 6,533,333 + 2,173,913 = 8707.25 kN
    (250000, 5000, 40, 8707.25),
])
def test_f43_t_eurocode(Ag, As, fc, cap):
    r = _ok(F43, {"gross_area_mm2": Ag, "steel_area_mm2": As, "fc_mpa": fc,
                  "fy_mpa": 500, "code": "eurocode"})
    assert r[_N] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("fc, fy, cap", [
    # fc 4000 psi = 27.579 MPa: P0 = 0.85x27.579x157000 + 1,260,000
    # = 4,940,421; x 0.52 = 2569.02 kN
    ("4000 psi", 420, 2569.02),
    # fy 60 ksi = 413.685 MPa: P0 = 4,003,500 + 1,241,056 = 5,244,556;
    # x 0.52 = 2727.17 kN
    (30, "60 ksi", 2727.17),
])
def test_f43_u_us_units(fc, fy, cap):
    r = _ok(F43, {"gross_area_mm2": 160000, "steel_area_mm2": 3000, "fc_mpa": fc,
                  "fy_mpa": fy, "code": "aci"})
    assert r[_N] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("drop, word", [("gross_area_mm2", "gross"), ("fy_mpa", "fy")])
def test_f43_m_missing_input_asks(drop, word):
    p = {"gross_area_mm2": 160000, "steel_area_mm2": 3000, "fc_mpa": 30,
         "fy_mpa": 420, "code": "aci"}
    p.pop(drop)
    _asks(F43, p, word, (_N,))


@pytest.mark.parametrize("Ag, As", [(160000, 160000), (90000, 120000)])
def test_f43_b_steel_not_less_than_gross_refused(Ag, As):
    _refuses(F43, {"gross_area_mm2": Ag, "steel_area_mm2": As, "fc_mpa": 30,
                   "fy_mpa": 420, "code": "aci"}, (_N,))


@pytest.mark.parametrize("Ag, fc, cap", [
    # As = 0: plain section; 0.52 x 0.85 x 30 x 160000 = 2121.60 kN
    (160000, 30, 2121.60),
    # 0.52 x 0.85 x 25 x 90000 = 994.50 kN
    (90000, 25, 994.50),
])
def test_f43_b_no_steel_is_the_concrete_term_only(Ag, fc, cap):
    r = _ok(F43, {"gross_area_mm2": Ag, "steel_area_mm2": 0, "fc_mpa": fc,
                  "fy_mpa": 420, "code": "aci"})
    assert r[_N] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("r_gyr, slender", [
    # k lu / r = 1 x 3000 / 120 = 25.0 > 22 -> slender
    (120, True),
    # 3000 / 150 = 20.0 <= 22 -> short
    (150, False),
])
def test_f43_b_slenderness_either_side_of_22(r_gyr, slender):
    r = _ok(F43, {"gross_area_mm2": 160000, "steel_area_mm2": 3000, "fc_mpa": 30,
                  "fy_mpa": 420, "unsupported_length_mm": 3000,
                  "radius_gyration_mm": r_gyr, "code": "aci"})
    assert r["slenderness_ratio"] == pytest.approx(3000 / r_gyr, rel=_REL)
    assert r["is_slender"] is slender
