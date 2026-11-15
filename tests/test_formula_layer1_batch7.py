"""Formula Layer 1, batch 7 (exam 5): loads, geotechnics, curing, sections.

Same rules as batches 1-6 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation from the published equation,
written in the comment beside its parametrize row:
  * ASCE 7-16 4.7.2 (SI, Eq. 4.7-1)       L = L0 (0.25 + 4.57/sqrt(KLL AT)),
                                          only when KLL AT >= 37.2 m2,
                                          not below 0.5 L0 (0.4 L0 when
                                          the member supports two floors+)
  * EN 1991-1-1 6.3.1.2 (6.1)             alphaA = 5/7 psi0 + A0/A <= 1.0,
                                          A0 = 10 m2
  * infinite slope, dry                   FoS = c/(gamma z sin b cos b)
                                                + tan(phi)/tan(b)
  * hydrostatic uplift on a raft          u = gamma_w h;
                                          W = gamma_c (t_raft + n t_floor);
                                          FoS = W/u; stop when FoS >= 1.25
  * Nurse-Saul maturity (ASTM C1074)      M = sum max(0, T - T0) dt,
                                          te = M/(Tr - T0), T0 = -10, Tr = 20
  * ACI 209R-92 Eq. 2-1 (moist, Type I)   f(t) = t/(4 + 0.85 t) f28
  * nominal shear stress                  v = V/(b d)
  * EN 1994-1-1 6.7.3.2 (6.30)            Npl,Rd = Aa fy/1.0
                                                 + 0.85 Ac fck/1.5
                                                 + As fsk/1.15,
                                          Ac = pi D^2/4 - Aa - As
None was copied from the product's output. A case where the product
disagrees with the hand value is marked ``xfail(strict=True)`` with a
one-line reason and is a suspected defect for the advisor; the expected
value is never moved to match. Every value test runs on two inputs worked
by hand the same way.

Unit factors (exact by definition): 1 ft = 0.3048 m; 1 in = 25.4 mm;
1 psf = 0.04788025898 kN/m2; 1 psi = 6.894757293 kPa;
1 ksi = 6.894757293 MPa; 1 kip = 4.4482216152605 kN.

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


# ------------------------------------------ F44 live load reduction

F44 = "live_load_reduction"
_L = "reduced_live_load_kn_m2"


@pytest.mark.parametrize("L0, at, kll, L", [
    # KLL AT = 80; 0.25 + 4.57/8.94427 = 0.76094; 2.4 x 0.76094 = 1.82626
    (2.4, 20, 4, 1.82626),
    # KLL AT = 60; 0.25 + 4.57/7.74597 = 0.83998; 4.79 x 0.83998 = 4.02353
    (4.79, 30, 2, 4.02353),
])
def test_f44_t_asce(L0, at, kll, L):
    r = _ok(F44, {"base_live_load_kn_m2": L0, "tributary_area_m2": at,
                  "kll": kll, "code": "aci"})
    assert r[_L] == pytest.approx(L, rel=_REL)


@pytest.mark.parametrize("L0, a, L", [
    # alphaA = 5/7 x 0.7 + 10/50 = 0.5 + 0.2 = 0.7; 3 x 0.7 = 2.1
    (3.0, 50, 2.1),
    # alphaA = 0.5 + 10/25 = 0.9; 5 x 0.9 = 4.5
    (5.0, 25, 4.5),
])
def test_f44_t_eurocode(L0, a, L):
    r = _ok(F44, {"base_live_load_kn_m2": L0, "tributary_area_m2": a,
                  "psi0": 0.7, "code": "eurocode"})
    assert r[_L] == pytest.approx(L, rel=_REL)


@pytest.mark.parametrize("params, L", [
    # 50 psf = 2.39401 kN/m2; 400 ft2 = 37.1612 m2; KLL AT = 148.645;
    # 0.25 + 4.57/12.1920 = 0.62484; L = 1.49587
    ({"base_live_load_kn_m2": "50 psf", "tributary_area_m2": "400 ft2",
      "kll": 4}, 1.49587),
    # 100 psf = 4.78803; 1000 ft2 = 92.9030 m2; KLL AT = 185.806;
    # 0.25 + 4.57/13.6311 = 0.58526; L = 2.80226
    ({"base_live_load_kn_m2": "100 psf", "tributary_area_m2": "1000 ft2",
      "kll": 2}, 2.80226),
])
def test_f44_u(params, L):
    r = _ok(F44, {**params, "code": "aci"})
    assert r[_L] == pytest.approx(L, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"tributary_area_m2": 40}, "base_live_load_kn_m2"),
    ({"base_live_load_kn_m2": 3}, "tributary_area_m2"),
])
def test_f44_m(params, missing):
    _asks(F44, params, missing, (_L,))


@pytest.mark.parametrize("at, L", [
    # KLL AT = 36 < 37.2: no reduction, L = L0 = 3.0
    (9.0, 3.0),
    # KLL AT = 37.2 (on the limit): 0.25 + 4.57/6.09918 = 0.99928; L = 2.99784
    (9.3, 2.99784),
])
def test_f44_b_reduction_threshold(at, L):
    r = _ok(F44, {"base_live_load_kn_m2": 3.0, "tributary_area_m2": at,
                  "kll": 4, "code": "aci"})
    assert r[_L] == pytest.approx(L, rel=_REL)


@pytest.mark.parametrize("floor, L", [
    # KLL AT = 1600; 0.25 + 4.57/40 = 0.36425, below the floor:
    # one floor 0.5 x 3 = 1.5; two floors+ 0.4 x 3 = 1.2
    (0.5, 1.5),
    (0.4, 1.2),
])
def test_f44_b_minimum_factor(floor, L):
    r = _ok(F44, {"base_live_load_kn_m2": 3.0, "tributary_area_m2": 400,
                  "kll": 4, "min_factor": floor, "code": "aci"})
    assert r[_L] == pytest.approx(L, rel=_REL)


@pytest.mark.parametrize("a", [
    5,    # 0.5 + 10/5 = 2.5 -> capped at 1.0; L = L0
    10,   # 0.5 + 10/10 = 1.5 -> capped at 1.0; L = L0
])
def test_f44_b_eurocode_cap(a):
    r = _ok(F44, {"base_live_load_kn_m2": 4.0, "tributary_area_m2": a,
                  "psi0": 0.7, "code": "eurocode"})
    assert r[_L] == pytest.approx(4.0, rel=_REL)


@pytest.mark.parametrize("params", [
    # a negative tributary area is not an area, under either code
    {"base_live_load_kn_m2": 3, "tributary_area_m2": -20, "code": "aci"},
    {"base_live_load_kn_m2": 3, "tributary_area_m2": -20, "code": "eurocode"},
    # A0/A with A = 0 has no value: no reduced load may be returned
    {"base_live_load_kn_m2": 3, "tributary_area_m2": 0, "code": "eurocode"},
])
def test_f44_b_refused(params):
    _refuses(F44, params, (_L,))


# ------------------------------------------ F45 infinite-slope factor of safety

F45 = "slope_fos_simple"
_F = "factor_of_safety"


@pytest.mark.parametrize("phi, beta, fos", [
    # tan30/tan20 = 0.577350/0.363970 = 1.58626
    (30, 20, 1.58626),
    # tan35/tan25 = 0.700208/0.466308 = 1.50160
    (35, 25, 1.50160),
])
def test_f45_t_cohesionless(phi, beta, fos):
    r = _ok(F45, {"friction_angle_deg": phi, "slope_angle_deg": beta})
    assert r[_F] == pytest.approx(fos, rel=_REL)


@pytest.mark.parametrize("phi, beta, c, g, z, fos", [
    # sin20 cos20 = 0.321394; 10/(18x3x0.321394) = 0.576194; + 1.58626 = 2.16245
    (30, 20, 10, 18, 3, 2.16245),
    # sin30 cos30 = 0.433013; 5/(20x2x0.433013) = 0.288675;
    # tan25/tan30 = 0.807670; FoS = 1.09634
    (25, 30, 5, 20, 2, 1.09634),
])
def test_f45_t_cohesive(phi, beta, c, g, z, fos):
    r = _ok(F45, {"friction_angle_deg": phi, "slope_angle_deg": beta,
                  "cohesion_kpa": c, "unit_weight_kn_m3": g, "depth_m": z})
    assert r[_F] == pytest.approx(fos, rel=_REL)


@pytest.mark.parametrize("params, fos", [
    # z = 10 ft = 3.048 m; 10/(18x3.048x0.321394) = 0.567118; FoS = 2.15338
    ({"friction_angle_deg": 30, "slope_angle_deg": 20, "cohesion_kpa": 10,
      "unit_weight_kn_m3": 18, "depth_m": "10 ft"}, 2.15338),
    # c = 200 psf = 9.57605 kPa; 9.57605/(18x2x0.321394) = 0.827649;
    # FoS = 2.41391
    ({"friction_angle_deg": 30, "slope_angle_deg": 20, "cohesion_kpa": "200 psf",
      "unit_weight_kn_m3": 18, "depth_m": 2}, 2.41391),
])
def test_f45_u(params, fos):
    r = _ok(F45, params)
    assert r[_F] == pytest.approx(fos, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"friction_angle_deg": 30}, "slope_angle_deg"),
    ({"slope_angle_deg": 20}, "friction_angle_deg"),
])
def test_f45_m(params, missing):
    _asks(F45, params, missing, (_F,))


@pytest.mark.parametrize("angle", [
    20,   # phi = beta: tan/tan = 1.0, limit equilibrium
    30,
])
def test_f45_b_limit_equilibrium(angle):
    r = _ok(F45, {"friction_angle_deg": angle, "slope_angle_deg": angle})
    assert r[_F] == pytest.approx(1.0, rel=_REL)


@pytest.mark.parametrize("phi", [30, 20])
def test_f45_b_flat_ground_refused(phi):
    # beta = 0: tan(beta) = 0, there is no slope to fail -- no FoS figure
    _refuses(F45, {"friction_angle_deg": phi, "slope_angle_deg": 0}, (_F,))


# ------------------------------------------ F46 uplift check to stop dewatering
# required_fos is passed explicitly in every value case: the 11:17Z ruling
# moves the product's default to the course value, and these cases test the
# equation, not a default.

F46 = "dewatering_uplift_check"


@pytest.mark.parametrize("h, raft, n, up, cw, fos, stop, nmin", [
    # u = 8; W = 1.2x2.5 + 3x0.3x2.5 = 3.0 + 2.25 = 5.25; FoS = 0.65625;
    # floors to reach 1.25 x 8 = 10: ceil((10 - 3)/0.75) = ceil(9.33) = 10
    (8, 1.2, 3, 8.0, 5.25, 0.65625, False, 10),
    # u = 4; W = 3.75 + 1.5 = 5.25; FoS = 1.3125;
    # ceil((5 - 3.75)/0.75) = ceil(1.667) = 2
    (4, 1.5, 2, 4.0, 5.25, 1.3125, True, 2),
])
def test_f46_t(h, raft, n, up, cw, fos, stop, nmin):
    r = _ok(F46, {"water_depth": h, "raft_thickness": raft, "floor_count": n,
                  "required_fos": 1.25})
    assert r["uplift_force_t_m2"] == pytest.approx(up, rel=_REL)
    assert r["counter_weight_t_m2"] == pytest.approx(cw, rel=_REL)
    assert r["fos"] == pytest.approx(fos, rel=_REL)
    assert r["can_stop"] is stop
    assert r["needs_tension_piles"] is (not stop)
    assert r["min_floors_for_stop"] == nmin


@pytest.mark.parametrize("params, fos, stop", [
    # h = 10 ft = 3.048 m; W = 2.5 + 1.5 = 4.0; FoS = 1.31234
    ({"water_depth": "10 ft", "raft_thickness": 1, "floor_count": 2}, 1.31234, True),
    # raft = 3 ft = 0.9144 m; W = 2.286 + 4x0.75 = 5.286; FoS = 5.286/5 = 1.0572
    ({"water_depth": 5, "raft_thickness": "3 ft", "floor_count": 4}, 1.0572, False),
])
def test_f46_u(params, fos, stop):
    r = _ok(F46, {**params, "required_fos": 1.25})
    assert r["fos"] == pytest.approx(fos, rel=_REL)
    assert r["can_stop"] is stop


@pytest.mark.parametrize("params, missing", [
    ({"raft_thickness": 1, "floor_count": 2}, "water_depth"),
    ({"water_depth": 4, "floor_count": 2}, "raft_thickness"),
])
def test_f46_m(params, missing):
    _asks(F46, params, missing, ("fos", "can_stop"))


@pytest.mark.parametrize("h, stop", [
    # W = 4.0; 4.0/3.2 = 1.25, on the limit -> may stop
    (3.2, True),
    # 4.0/3.21 = 1.24611, just below -> may not stop
    (3.21, False),
])
def test_f46_b_on_required_fos(h, stop):
    r = _ok(F46, {"water_depth": h, "raft_thickness": 1, "floor_count": 2,
                  "required_fos": 1.25})
    assert r["can_stop"] is stop


@pytest.mark.parametrize("required, stop", [
    # W = 4.0, h = 3.25: FoS = 1.23077 -- passes a 1.2 requirement,
    # fails a 1.25 one. The threshold is the caller's input, not a constant.
    (1.2, True),
    (1.25, False),
])
def test_f46_b_threshold_is_the_input(required, stop):
    r = _ok(F46, {"water_depth": 3.25, "raft_thickness": 1, "floor_count": 2,
                  "required_fos": required})
    assert r["fos"] == pytest.approx(1.23077, rel=_REL)
    assert r["can_stop"] is stop


@pytest.mark.parametrize("raft", [1.0, 0.5])
def test_f46_b_no_head(raft):
    # h = 0: no uplift; dewatering may stop with no added floor
    r = _ok(F46, {"water_depth": 0, "raft_thickness": raft, "floor_count": 2})
    assert r["uplift_force_t_m2"] == 0
    assert r["can_stop"] is True
    assert r["min_floors_for_stop"] == 0


@pytest.mark.parametrize("h", [-1, -5])
def test_f46_b_negative_head_refused(h):
    _refuses(F46, {"water_depth": h, "raft_thickness": 1, "floor_count": 2},
             ("fos", "can_stop"))


# ------------------------------------------ F47 strength by maturity

F47 = "concrete_maturity_strength"
_S = "predicted_strength_n_mm2"
_ABS = 0.05   # the product reports strength to 0.1 N/mm2


@pytest.mark.parametrize("temps, hours, f28, mi, te, s", [
    # M = 30x48 = 1440; te = 1440/30 = 48 h = 2 d; 2/(4+1.7) = 0.350877;
    # 40 x 0.350877 = 14.035
    ([20, 20], [24, 24], 40, 1440, 2.0, 14.035),
    # M = 40x24 + 20x24 + 30x48 = 2880; te = 96 h = 4 d; 4/7.4 = 0.540541;
    # 30 x 0.540541 = 16.216
    ([30, 10, 20], [24, 24, 48], 30, 2880, 4.0, 16.216),
])
def test_f47_t(temps, hours, f28, mi, te, s):
    r = _ok(F47, {"temperature_history_c": temps, "time_intervals_hours": hours,
                  "strength_28d_n_mm2": f28})
    assert r["maturity_index_c_hrs"] == pytest.approx(mi, rel=_REL)
    assert r["equivalent_age_days"] == pytest.approx(te, rel=_REL)
    assert r[_S] == pytest.approx(s, abs=_ABS)


@pytest.mark.parametrize("temps, hours, mi, s", [
    # -15 is below the datum: contributes 0; M = 30x24 = 720; te = 1 d;
    # 1/4.85 = 0.206186; 40 x 0.206186 = 8.247
    ([-15, 20], [24, 24], 720, 8.247),
    # -10 on the datum: contributes 0; M = 15x36 = 540; te = 0.75 d;
    # 0.75/4.6375 = 0.161725; 40 x 0.161725 = 6.469
    ([-10, 5], [12, 36], 540, 6.469),
])
def test_f47_b_at_or_below_datum(temps, hours, mi, s):
    r = _ok(F47, {"temperature_history_c": temps, "time_intervals_hours": hours,
                  "strength_28d_n_mm2": 40})
    assert r["maturity_index_c_hrs"] == pytest.approx(mi, rel=_REL)
    assert r[_S] == pytest.approx(s, abs=_ABS)


@pytest.mark.parametrize("f28, s", [
    # 5000 psi = 34.4738 MPa; x 0.350877 = 12.096
    ("5000 psi", 12.096),
    # 4 ksi = 27.5790 MPa; x 0.350877 = 9.677
    ("4 ksi", 9.677),
])
def test_f47_u(f28, s):
    r = _ok(F47, {"temperature_history_c": [20, 20], "time_intervals_hours": [24, 24],
                  "strength_28d_n_mm2": f28})
    assert r[_S] == pytest.approx(s, abs=_ABS)


@pytest.mark.parametrize("params, missing", [
    ({"time_intervals_hours": [24]}, "temperature_history_c"),
    ({"temperature_history_c": [20]}, "time_intervals_hours"),
])
def test_f47_m(params, missing):
    _asks(F47, params, missing, (_S,))


@pytest.mark.parametrize("params", [
    # one interval for two temperatures: the readings do not pair up
    {"temperature_history_c": [20, 20], "time_intervals_hours": [24]},
    # a zero-length interval
    {"temperature_history_c": [20, 20], "time_intervals_hours": [24, 0]},
])
def test_f47_b_refused(params):
    _refuses(F47, params, (_S,))


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: the product caps f(t) at f28, but ACI 209R-92 Eq. 2-1 "
    "keeps rising past 28 d toward f28/0.85 (28 d gives 1.0072 f28)"))
@pytest.mark.parametrize("hours, s", [
    # M = 30 x 672 = 20160; te = 28 d; 28/(4+23.8) = 1.007194; 40 x = 40.288
    ([672], 40.288),
    # M = 30 x 1344 = 40320; te = 56 d; 56/51.6 = 1.085271; 40 x = 43.411
    ([1344], 43.411),
])
def test_f47_b_past_28_days(hours, s):
    r = _ok(F47, {"temperature_history_c": [20], "time_intervals_hours": hours,
                  "strength_28d_n_mm2": 40})
    assert r[_S] == pytest.approx(s, abs=_ABS)


# ------------------------------------------ F48 nominal shear stress

F48 = "shear_stress_check"
_V = "shear_stress_n_mm2"


@pytest.mark.parametrize("V, b, d, v", [
    (150, 300, 450, 1.11111),    # 150000/135000
    (250, 400, 600, 1.041667),   # 250000/240000
])
def test_f48_t(V, b, d, v):
    r = _ok(F48, {"v_kn": V, "b_mm": b, "d_mm": d})
    assert r[_V] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("params, v", [
    # b = 12 in = 304.8 mm; 100000/(304.8x500) = 0.656168
    ({"v_kn": 100, "b_mm": "12 in", "d_mm": 500}, 0.656168),
    # V = 50 kip = 222.411 kN; 222411/150000 = 1.48274
    ({"v_kn": "50 kip", "b_mm": 300, "d_mm": 500}, 1.48274),
])
def test_f48_u(params, v):
    r = _ok(F48, params)
    assert r[_V] == pytest.approx(v, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"v_kn": 100, "b_mm": 300}, "d_mm"),
    ({"v_kn": 100, "d_mm": 500}, "b_mm"),
])
def test_f48_m(params, missing):
    _asks(F48, params, missing, (_V,))


@pytest.mark.parametrize("b, d", [(300, 450), (400, 600)])
def test_f48_b_zero_shear(b, d):
    r = _ok(F48, {"v_kn": 0, "b_mm": b, "d_mm": d})
    assert r[_V] == 0


@pytest.mark.parametrize("params", [
    {"v_kn": 100, "b_mm": 0, "d_mm": 500},   # no width: v has no value
    {"v_kn": 100, "b_mm": 300, "d_mm": 0},   # no depth: v has no value
])
def test_f48_b_zero_section_refused(params):
    _refuses(F48, params, (_V,))


# ------------------------------------------ F49 composite column squash load

F49 = "composite_column_design"


def _cap(r: dict, option: str) -> float:
    return r[option]["capacity_kn"]


@pytest.mark.parametrize("D, fck, Aa, As, a, b", [
    # Ag = pi 300^2 = 282743.3; Ac = 269743.3;
    # A = 10000x355 + 0.85x269743.3x26.667 + 3000x434.783
    #   = 3550000 + 6114182 + 1304348 = 10968.5 kN
    # B (no steel section): Ac = 279743.3; 6340848 + 1304348 = 7645.2 kN
    (600, 40, 10000, 3000, 10968.5, 7645.2),
    # Ag = 196349.5; Ac = 186349.5;
    # A = 2840000 + 0.85x186349.5x20 + 2000x434.783 = 6877.5 kN
    # B: Ac = 194349.5; 3303942 + 869565 = 4173.5 kN
    (500, 30, 8000, 2000, 6877.5, 4173.5),
])
def test_f49_t(D, fck, Aa, As, a, b):
    r = _ok(F49, {"axial_load_kn": 5000, "column_diameter_mm": D,
                  "concrete_grade_n_mm2": fck, "steel_area_mm2": Aa,
                  "rebar_area_mm2": As})
    assert _cap(r, "option_a") == pytest.approx(a, rel=_REL)
    assert _cap(r, "option_b") == pytest.approx(b, rel=_REL)


@pytest.mark.parametrize("load, safe_a, safe_b", [
    # against A 6877.5 / B 4173.5 kN
    (5000, True, False),
    (7000, False, False),
])
def test_f49_t_is_safe(load, safe_a, safe_b):
    r = _ok(F49, {"axial_load_kn": load, "column_diameter_mm": 500,
                  "concrete_grade_n_mm2": 30, "steel_area_mm2": 8000,
                  "rebar_area_mm2": 2000})
    assert r["option_a"]["is_safe"] is safe_a
    assert r["option_b"]["is_safe"] is safe_b


@pytest.mark.parametrize("extra, a", [
    # D = 24 in = 609.6 mm; Ag = 291864.5; Ac = 278864.5;
    # 3550000 + 6320928 + 1304348 = 11175.3 kN
    ({"column_diameter_mm": "24 in"}, 11175.3),
    # fy = 50 ksi = 344.738 MPa; 3447379 + 6114182 + 1304348 = 10865.9 kN
    ({"column_diameter_mm": 600, "steel_fy_mpa": "50 ksi"}, 10865.9),
])
def test_f49_u(extra, a):
    r = _ok(F49, {"axial_load_kn": 5000, "concrete_grade_n_mm2": 40,
                  "steel_area_mm2": 10000, "rebar_area_mm2": 3000, **extra})
    assert _cap(r, "option_a") == pytest.approx(a, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"column_diameter_mm": 600}, "axial_load_kn"),
    ({"axial_load_kn": 5000}, "column_diameter_mm"),
])
def test_f49_m(params, missing):
    _asks(F49, params, missing, ("option_a", "option_b"))


@pytest.mark.parametrize("D, fck, n", [
    # no steel at all: both options are plain 0.85 Ag fcd
    # 0.85 x 282743.3 x 26.667 = 6408.8 kN
    (600, 40, 6408.8),
    # 0.85 x 125663.7 x 20 = 2136.3 kN
    (400, 30, 2136.3),
])
def test_f49_b_no_steel(D, fck, n):
    r = _ok(F49, {"axial_load_kn": 1000, "column_diameter_mm": D,
                  "concrete_grade_n_mm2": fck, "steel_area_mm2": 0,
                  "rebar_area_mm2": 0})
    assert _cap(r, "option_a") == pytest.approx(n, rel=_REL)
    assert _cap(r, "option_b") == pytest.approx(n, rel=_REL)
