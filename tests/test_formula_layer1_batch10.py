"""Formula Layer 1, batch 10 (exam 5): dewatering, prestress, staffing, progress, risk.

Same rules as batches 1-9 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation, written in the comment beside
its parametrize row:
  * well-point dewatering        stages = ceil(drawdown / 5 m), the
                                 calculator's stated 5 m lift per stage and
                                 3-stage (15 m) limit; spacing by the
                                 calculator's stated permeability bands
                                 (> 1e-3 m/s 1.5 m; > 1e-4 1.2 m;
                                 > 1e-5 0.9 m; else 0.6 m)
  * post-tensioning (load        w = 0.75 DL (DL = 25 kN/m3 x h when not
    balancing)                   given); e = ratio x h; P = w L^2 / (8 e);
                                 A = P / f; strand = 0.78 pi (d/2)^2;
                                 strands = ceil(A / strand)
  * supervision manpower         supervisors = ceil(quantity / ratio) per
                                 trade; HSE = ceil(total / ratio);
                                 document controllers = ceil(total / ratio)
  * progress and quantity        planned % = Pq/T x 100; actual % = Aq/T x 100;
                                 remaining = T - Aq; variance = actual - planned
  * progress KPI                 planned % = Pc/T x 100; actual % = Ac/T x 100;
                                 SPI (progress) = Ac/Pc
  * risk score                   score = P x I on a 1-5 x 1-5 matrix; band
                                 by the calculator's stated procedure
                                 (<= 4 GREEN, 5-9 AMBER, >= 10 RED)
None was copied from the product's output. A case where the product
disagrees with the hand value is marked ``xfail(strict=True)`` with a
one-line reason and is a suspected defect for the advisor; the expected
value is never moved to match. Every value test runs on two inputs worked
by hand the same way.

No U case for F64-F67: supervision ratios are in the project's own units
(quantity and ratio share one unit by contract), and progress, KPI and risk
are counts, percents and ratings.

Unit factors (exact by definition): 1 ft = 0.3048 m; 1 mm = 0.001 m.

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


# ---------------------------------------------------------------- F62 well points
F62 = "dewatering_well_point_spacing"
_S = "stages_needed"
_SP = "well_point_spacing_m"


@pytest.mark.parametrize("k, dd, spacing, stages, cap", [
    # 5e-4 m/s is in (1e-4, 1e-3] -> 1.2 m; ceil(7/5) = 2 stages; 2 x 5 = 10 m
    (5e-4, 7, 1.2, 2, 10.0),
    # 2e-3 m/s > 1e-3 -> 1.5 m; ceil(12/5) = 3 stages; 3 x 5 = 15 m
    (2e-3, 12, 1.5, 3, 15.0),
    # 5e-5 m/s is in (1e-5, 1e-4] -> 0.9 m; ceil(4/5) = 1 stage; 5 m
    (5e-5, 4, 0.9, 1, 5.0),
    # 5e-6 m/s <= 1e-5 -> 0.6 m; ceil(9.5/5) = 2 stages; 10 m
    (5e-6, 9.5, 0.6, 2, 10.0),
])
def test_f62_t(k, dd, spacing, stages, cap):
    r = _ok(F62, {"soil_permeability_m_s": k, "required_drawdown_m": dd})
    assert r[_SP] == pytest.approx(spacing, rel=_REL)
    assert r[_S] == stages
    assert r["max_depth_capacity_m"] == pytest.approx(cap, rel=_REL)


@pytest.mark.parametrize("dd, stages", [
    # 7 m written in feet (7/0.3048 = 22.96588 ft) -> ceil(7/5) = 2
    ("22.96587926 ft", 2),
    # 12 m written in mm -> ceil(12/5) = 3
    ("12000 mm", 3),
])
def test_f62_u(dd, stages):
    r = _ok(F62, {"soil_permeability_m_s": 5e-4, "required_drawdown_m": dd})
    assert r[_S] == stages


@pytest.mark.parametrize("params, missing", [
    ({"required_drawdown_m": 7}, "soil_permeability"),
    ({"soil_permeability_m_s": 5e-4}, "required_drawdown"),
])
def test_f62_m(params, missing):
    _asks(F62, params, missing, (_S, _SP))


@pytest.mark.parametrize("dd, stages", [
    (5.0, 1),    # exactly one 5 m lift: ceil(5/5) = 1
    (5.01, 2),   # just past one lift: ceil(5.01/5) = 2
])
def test_f62_b_stage_edge(dd, stages):
    r = _ok(F62, {"soil_permeability_m_s": 5e-4, "required_drawdown_m": dd})
    assert r[_S] == stages


@pytest.mark.parametrize("dd, stages, deep_well", [
    (15.0, 3, False),   # on the 3-stage (15 m) limit: ceil(15/5) = 3, no warning
    (15.01, 4, True),   # past it: ceil(15.01/5) = 4 -> deep well system named
])
def test_f62_b_three_stage_limit(dd, stages, deep_well):
    r = _ok(F62, {"soil_permeability_m_s": 5e-4, "required_drawdown_m": dd})
    assert r[_S] == stages
    assert any("deep well" in n.lower() for n in r["notes"]) is deep_well


@pytest.mark.parametrize("k, expected", [
    (1e-4, 0.9),   # exactly on the 1e-4 band edge: not > 1e-4 -> 0.9 m
    (1e-5, 0.6),   # exactly on the 1e-5 band edge: not > 1e-5 -> 0.6 m
])
def test_f62_b_band_edge(k, expected):
    r = _ok(F62, {"soil_permeability_m_s": k, "required_drawdown_m": 6})
    assert r[_SP] == pytest.approx(expected, rel=_REL)


@pytest.mark.parametrize("k", [0, -1e-4])
def test_f62_b_permeability_not_positive_refused(k):
    _refuses(F62, {"soil_permeability_m_s": k, "required_drawdown_m": 7}, (_S, _SP))


@pytest.mark.parametrize("dd", [-1, -0.5])
def test_f62_b_negative_drawdown_refused(dd):
    _refuses(F62, {"soil_permeability_m_s": 5e-4, "required_drawdown_m": dd}, (_S, _SP))


# ---------------------------------------------------------------- F63 post-tensioning
F63 = "post_tensioning_force"
_P = "tendon_force_kn"


@pytest.mark.parametrize("params, w, e_mm, P, A, n", [
    # DL = 25 x 0.25 = 6.25; w = 0.75 x 6.25 = 4.6875; e = 0.15 x 0.25 = 0.0375 m
    # P = 4.6875 x 10^2 / (8 x 0.0375) = 1562.5 kN; A = 1562500/1300 = 1201.92 mm2
    # strand = 0.78 pi 6.35^2 = 98.81 mm2; 1201.92/98.81 = 12.16 -> 13
    ({"span_m": 10, "slab_thickness_m": 0.25, "live_load_kn_m2": 3},
     4.6875, 37.5, 1562.5, 1201.92, 13),
    # w = 0.75 x 7.0 = 5.25; e = 0.2 x 0.2 = 0.04 m; P = 5.25 x 64 / 0.32 = 1050 kN
    # A = 1050000/1400 = 750 mm2; strand = 0.78 pi 7.6^2 = 141.54 mm2;
    # 750/141.54 = 5.30 -> 6
    ({"span_m": 8, "slab_thickness_m": 0.2, "live_load_kn_m2": 2.5,
      "dead_load_kn_m2": 7.0, "eccentricity_ratio": 0.2,
      "tendon_stress_n_mm2": 1400, "tendon_diameter_mm": 15.2},
     5.25, 40.0, 1050.0, 750.0, 6),
])
def test_f63_t(params, w, e_mm, P, A, n):
    r = _ok(F63, params)
    assert r["balanced_load_kn_m"] == pytest.approx(w, abs=0.01)
    assert r["eccentricity_mm"] == pytest.approx(e_mm, rel=_REL)
    assert r[_P] == pytest.approx(P, rel=_REL)
    assert r["tendon_area_mm2"] == pytest.approx(A, rel=_REL)
    assert r["num_strands"] == n


@pytest.mark.parametrize("params", [
    # 10 m written in feet (10/0.3048 = 32.80840 ft), 0.25 m in mm -> 1562.5 kN
    {"span_m": "32.80839895 ft", "slab_thickness_m": "250 mm", "live_load_kn_m2": 3},
    # 0.25 m written in feet (0.25/0.3048 = 0.8202100 ft) -> 1562.5 kN
    {"span_m": 10, "slab_thickness_m": "0.8202099738 ft", "live_load_kn_m2": 3},
])
def test_f63_u(params):
    assert _ok(F63, params)[_P] == pytest.approx(1562.5, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"slab_thickness_m": 0.25, "live_load_kn_m2": 3}, "span"),
    ({"span_m": 10, "live_load_kn_m2": 3}, "slab_thickness"),
])
def test_f63_m(params, missing):
    _asks(F63, params, missing, (_P, "num_strands"))


@pytest.mark.parametrize("h", [0.25, 0.3])
def test_f63_b_zero_span(h):
    # L = 0 -> P = w x 0 / (8e) = 0 kN, no strands
    r = _ok(F63, {"span_m": 0, "slab_thickness_m": h, "live_load_kn_m2": 3})
    assert r[_P] == pytest.approx(0.0, abs=1e-9)
    assert r["num_strands"] == 0


@pytest.mark.parametrize("dl, P", [
    # P scales with w: DL 6.25 -> 1562.5 kN (above); DL 12.5 -> 3125 kN
    (6.25, 1562.5),
    (12.5, 3125.0),
])
def test_f63_b_force_linear_in_dead_load(dl, P):
    r = _ok(F63, {"span_m": 10, "slab_thickness_m": 0.25, "live_load_kn_m2": 3,
                  "dead_load_kn_m2": dl})
    assert r[_P] == pytest.approx(P, rel=_REL)


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: with no tendon drape (e = 0) P = wL^2/(8e) has no "
    "value, but the calculator returns 0 kN (no prestress needed)"))
@pytest.mark.parametrize("span, dl", [(10, 6.25), (8, 7.0)])
def test_f63_b_zero_eccentricity_refused(span, dl):
    _refuses(F63, {"span_m": span, "slab_thickness_m": 0.25, "live_load_kn_m2": 3,
                   "dead_load_kn_m2": dl, "eccentricity_ratio": 0}, (_P, "num_strands"))


# ---------------------------------------------------------------- F64 supervision
F64 = "supervision_ratio"
_RATIOS = {"supervisors_per_hse_officer": 3, "staff_per_document_controller": 5}


@pytest.mark.parametrize("params, by_trade, total, hse, docs, staff", [
    # civil ceil(5000/1500) = 4; HSE ceil(4/3) = 2; docs ceil(4/5) = 1; 4+2+1 = 7
    ({"concrete_m3": 5000, "concrete_m3_per_supervisor": 1500, **_RATIOS},
     {"civil_supervisors": 4}, 4, 2, 1, 7),
    # steel ceil(1200/250) = 5; general ceil(30000/10000) = 3; total 8;
    # HSE ceil(8/4) = 2; docs ceil(8/6) = 2; 8+2+2 = 12
    ({"structural_steel_t": 1200, "steel_t_per_supervisor": 250,
      "area_m2": 30000, "area_m2_per_supervisor": 10000,
      "supervisors_per_hse_officer": 4, "staff_per_document_controller": 6},
     {"structural_supervisors": 5, "general_supervisors": 3}, 8, 2, 2, 12),
])
def test_f64_t(params, by_trade, total, hse, docs, staff):
    r = _ok(F64, params)
    for key, value in by_trade.items():
        assert r[key] == value
    assert r["total_supervisors"] == total
    assert r["hse_officers"] == hse
    assert r["document_controllers"] == docs
    assert r["total_supervision_staff"] == staff


@pytest.mark.parametrize("params, missing", [
    ({"concrete_m3": 5000, **_RATIOS}, "concrete per civil supervisor"),
    ({"concrete_m3": 5000, "concrete_m3_per_supervisor": 1500,
      "staff_per_document_controller": 5}, "supervisors per hse officer"),
])
def test_f64_m(params, missing):
    _asks(F64, params, missing, ("total_supervisors", "total_supervision_staff"))


@pytest.mark.parametrize("q, civil", [
    (4500, 3),   # exact multiple: 4500/1500 = 3 -> 3, not rounded up
    (4501, 4),   # one unit past it: ceil(4501/1500) = 4
])
def test_f64_b_ceiling_edge(q, civil):
    r = _ok(F64, {"concrete_m3": q, "concrete_m3_per_supervisor": 1500, **_RATIOS})
    assert r["civil_supervisors"] == civil


@pytest.mark.parametrize("params", [
    {},
    {"concrete_m3": 0, "area_m2": 0},
])
def test_f64_b_no_quantity_no_staff(params):
    # nothing to supervise -> 0 supervisors, 0 HSE, 0 document controllers
    r = _ok(F64, params)
    assert r["total_supervisors"] == 0
    assert r["total_supervision_staff"] == 0


# ---------------------------------------------------------------- F65 progress quantity
F65 = "progress_quantity"


@pytest.mark.parametrize("T, Pq, Aq, pp, ap, rem, var", [
    # 250/400 = 62.5 %; 200/400 = 50 %; 400 - 200 = 200; 50 - 62.5 = -12.5
    (400, 250, 200, 62.5, 50.0, 200.0, -12.5),
    # 300/1250 = 24 %; 375/1250 = 30 %; 1250 - 375 = 875; 30 - 24 = +6
    (1250, 300, 375, 24.0, 30.0, 875.0, 6.0),
])
def test_f65_t(T, Pq, Aq, pp, ap, rem, var):
    r = _ok(F65, {"total_qty": T, "planned_qty": Pq, "actual_qty": Aq})
    assert r["planned_percent"] == pytest.approx(pp, rel=_REL)
    assert r["actual_percent"] == pytest.approx(ap, rel=_REL)
    assert r["remaining_qty"] == pytest.approx(rem, rel=_REL)
    assert r["progress_variance_percent"] == pytest.approx(var, rel=_REL)


@pytest.mark.parametrize("T", [400, 1250])
def test_f65_m_no_progress_quantity(T):
    _asks(F65, {"total_qty": T}, "planned_qty",
          ("planned_percent", "actual_percent", "remaining_qty"))


@pytest.mark.parametrize("Aq, ap, rem", [
    (400, 100.0, 0.0),   # complete: 400/400 = 100 %, nothing left
    (0, 0.0, 400.0),     # not started: 0 %, all 400 left
])
def test_f65_b_ends(Aq, ap, rem):
    r = _ok(F65, {"total_qty": 400, "actual_qty": Aq})
    assert r["actual_percent"] == pytest.approx(ap, abs=1e-9)
    assert r["remaining_qty"] == pytest.approx(rem, abs=1e-9)
    assert r.get("progress_variance_percent") is None   # no planned qty given


@pytest.mark.parametrize("T", [0, -5])
def test_f65_b_total_not_positive_refused(T):
    _refuses(F65, {"total_qty": T, "actual_qty": 1}, ("actual_percent", "remaining_qty"))


# ---------------------------------------------------------------- F66 progress KPI
F66 = "progress_kpi"


@pytest.mark.parametrize("Pc, Ac, T, pp, ap, var, spi", [
    # 250/400 = 62.5 %; 200/400 = 50 %; -12.5; SPI 200/250 = 0.8
    (250, 200, 400, 62.5, 50.0, -12.5, 0.8),
    # 120/600 = 20 %; 150/600 = 25 %; +5; SPI 150/120 = 1.25
    (120, 150, 600, 20.0, 25.0, 5.0, 1.25),
])
def test_f66_t(Pc, Ac, T, pp, ap, var, spi):
    r = _ok(F66, {"planned_cum": Pc, "actual_cum": Ac, "planned_total": T})
    assert r["planned_pct"] == pytest.approx(pp, rel=_REL)
    assert r["actual_pct"] == pytest.approx(ap, rel=_REL)
    assert r["variance_pct"] == pytest.approx(var, rel=_REL)
    assert r["spi_progress"] == pytest.approx(spi, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"planned_cum": 250}, "actual_cum"),
    ({"actual_cum": 200}, "planned_cum"),
])
def test_f66_m(params, missing):
    _asks(F66, params, missing, ("spi_progress", "actual_pct"))


@pytest.mark.parametrize("Pc, Ac, spi", [
    (250, 200, 0.8),    # no total: SPI 200/250 = 0.8 only, no percents
    (120, 150, 1.25),   # no total: SPI 150/120 = 1.25 only, no percents
])
def test_f66_m_no_total_spi_only(Pc, Ac, spi):
    r = _ok(F66, {"planned_cum": Pc, "actual_cum": Ac})
    assert r["spi_progress"] == pytest.approx(spi, rel=_REL)
    assert r["planned_pct"] is None and r["actual_pct"] is None


@pytest.mark.parametrize("Ac", [10, 0])
def test_f66_b_nothing_planned_no_spi(Ac):
    # Pc = 0 -> Ac/Pc has no value: SPI withheld, not 0 and not infinite
    r = _ok(F66, {"planned_cum": 0, "actual_cum": Ac})
    assert r["spi_progress"] is None


@pytest.mark.parametrize("Pc, Ac", [(250, 200), (120, 150)])
def test_f66_b_zero_total_refused(Pc, Ac):
    _refuses(F66, {"planned_cum": Pc, "actual_cum": Ac, "planned_total": 0},
             ("planned_pct", "actual_pct"))


# ---------------------------------------------------------------- F67 risk score
F67 = "score_risk"


@pytest.mark.parametrize("p, i, score, band, action", [
    (2, 2, 4, "GREEN", False),    # 2 x 2 = 4 <= 4
    (3, 3, 9, "AMBER", True),     # 3 x 3 = 9 in 5-9
    (4, 5, 20, "RED", True),      # 4 x 5 = 20 >= 10
])
def test_f67_t(p, i, score, band, action):
    r = _ok(F67, {"probability": p, "impact": i})
    assert r["score"] == score
    assert r["band"] == band
    assert r["requires_action"] is action


@pytest.mark.parametrize("params, missing", [
    ({"probability": 2}, "impact"),
    ({"impact": 3}, "probability"),
])
def test_f67_m(params, missing):
    _asks(F67, params, missing, ("score", "band"))


@pytest.mark.parametrize("p, i, band", [
    (1, 4, "GREEN"),   # 4: top of GREEN
    (1, 5, "AMBER"),   # 5: first AMBER
    (3, 3, "AMBER"),   # 9: top of AMBER
    (2, 5, "RED"),     # 10: first RED
])
def test_f67_b_band_edges(p, i, band):
    assert _ok(F67, {"probability": p, "impact": i})["band"] == band


@pytest.mark.parametrize("p, i", [(0, 2), (6, 2), (3, 0), (3, 6)])
def test_f67_b_off_matrix_refused(p, i):
    _refuses(F67, {"probability": p, "impact": i}, ("score", "band"))
