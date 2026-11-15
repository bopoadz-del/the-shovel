"""Formula Layer 1, batch 13 (exam 5): beam moments, payment, cash flow,
concrete mix design, mix proportions, slip-form mix, mass-concrete thermal
check, crane count, electrical programme (more follow in this batch).

Same rules as batches 1-12 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation, written in the comment beside
its parametrize row:
  * beam governing moment        standard beam-table results:
                                 simply supported UDL wL^2/8 at L/2;
                                 simply supported point load Pab/L under
                                 the load; cantilever UDL wL^2/2 at the
                                 fixed end (hogging); fixed-fixed UDL
                                 wL^2/12 at the supports (hogging);
                                 propped cantilever UDL wL^2/8 at the fixed
                                 end (hogging)
  * payment due on a claim       retention = certified x rate; net =
                                 certified - retention; disputed = claimed
                                 - certified; cumulative = previous +
                                 certified; % complete = cumulative /
                                 contract x 100
  * contractor cash flow         advance = a x sum V at month 0; each month
                                 certified = V (1 - a - r), retention
                                 released in the last month; cash in lags
                                 ceil(lag days / 30) months; running
                                 balance starts at the advance; peak
                                 funding = the deepest negative balance
  * concrete mix (absolute       1 m3 = W/1 + C/SGc + F/SGf + Cr/SGcr, with
    volume)                      W = (w/c) C, F = f C, Cr = c C (C in t);
                                 so C = 1 / (w/c + 1/SGc + f/SGf + c/SGcr)
  * concrete mix proportions     dry = wet x 1.54 (x (1 + waste));
                                 constituent = dry x parts / sum of parts
  * slip-form mix                the absolute-volume rule above, on the mix
                                 the calculator states (1:2:2.6, w/c 0.42)
  * mass concrete thermal check  stated limits: core <= 70 C and core -
                                 surface <= 20 C; F to C: (F - 32) / 1.8
  * crane count                  lifts/h = 60 / cycle; t/day = capacity x
                                 utilisation x lifts/h x h x shifts; cranes
                                 = ceil(demand / (t/day x working days));
                                 cost = cranes x the caller's rate
  * electrical programme         days per stage = ceil(area x floors /
                                 that stage's m2 per day); total = sum
None was copied from the product's output. A case where the product
disagrees with the hand value would be marked ``xfail(strict=True)`` with a
one-line reason and reported as a suspected defect; the expected value is
never moved to match. Every value test runs on two inputs worked by hand
the same way.

No U case for the payment, cash flow or mix design calculators: they take
money in the project's own currency, or dimensionless ratios. No M or U
case for the slip-form mix (it takes no input). A bare "158 F" is refused
as an unrecognised unit (an honest ask, not a defect); the U case writes
"degF".

Unit factors (exact by definition): 1 ft = 0.3048 m; 1 cm = 0.01 m;
1 ft3 = 0.028316846592 m3.

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


# ---------------------------------------------------------------- F80 beam moment
F80 = "beam_governing_moment"
_F80_OUT = ("moment_kn_m", "position_m")


def _udl(w: float) -> list:
    return [{"kind": "udl", "w": w}]


@pytest.mark.parametrize("support, span, w, moment, pos, sense", [
    ("simply_supported", 6, 10, 45.0, 3.0, "sagging"),     # 10 x 36 / 8 = 45 at 3
    ("simply_supported", 8, 20, 160.0, 4.0, "sagging"),    # 20 x 64 / 8 = 160 at 4
    ("cantilever", 4, 10, -80.0, 0.0, "hogging"),          # 10 x 16 / 2 = 80 at the fixed end
    ("cantilever", 6, 5, -90.0, 0.0, "hogging"),           # 5 x 36 / 2 = 90
    ("fixed_fixed", 6, 12, -36.0, 0.0, "hogging"),         # 12 x 36 / 12 = 36 at a support
    ("fixed_fixed", 6, 10, -30.0, 0.0, "hogging"),         # 10 x 36 / 12 = 30
    ("propped_cantilever", 5, 8, -25.0, 0.0, "hogging"),   # 8 x 25 / 8 = 25 at the fixed end
    ("propped_cantilever", 6, 10, -45.0, 0.0, "hogging"),  # 10 x 36 / 8 = 45
])
def test_f80_t_udl(support, span, w, moment, pos, sense):
    r = _ok(F80, {"span_m": span, "support": support, "loads": _udl(w)})
    assert r["moment_kn_m"] == pytest.approx(moment, rel=_REL)
    # the position of a support moment may be either end; read it by magnitude
    if sense == "sagging":
        assert r["position_m"] == pytest.approx(pos, rel=_REL)
    else:
        assert r["position_m"] in (pytest.approx(0.0, abs=1e-6), pytest.approx(span, rel=_REL))
    assert r["sense"] == sense


@pytest.mark.parametrize("span, P, a, moment", [
    (8, 50, 2, 75.0),   # 50 x 2 x 6 / 8 = 75 under the load
    (5, 40, 3, 48.0),   # 40 x 3 x 2 / 5 = 48 under the load
])
def test_f80_t_point(span, P, a, moment):
    r = _ok(F80, {"span_m": span, "loads": [{"kind": "point", "P": P, "a": a}]})
    assert r["moment_kn_m"] == pytest.approx(moment, rel=_REL)
    assert r["position_m"] == pytest.approx(a, rel=_REL)


@pytest.mark.parametrize("span, moment, pos", [
    ("20 ft", 46.4515, 3.048),   # 6.096 m; 10 x 6.096^2 / 8 = 46.4515 at 3.048
    ("600 cm", 45.0, 3.0),       # 6 m; 10 x 36 / 8 = 45 at 3
])
def test_f80_u_span(span, moment, pos):
    r = _ok(F80, {"span_m": span, "loads": _udl(10)})
    assert r["moment_kn_m"] == pytest.approx(moment, rel=_REL)
    assert r["position_m"] == pytest.approx(pos, rel=_REL)


@pytest.mark.parametrize("params", [
    {"loads": _udl(10)},
    {"support": "cantilever", "loads": _udl(5)},
])
def test_f80_m_span(params):
    _asks(F80, params, "span", _F80_OUT)


@pytest.mark.parametrize("span", [0, -3])
def test_f80_b_no_span_refused(span):
    _refuses(F80, {"span_m": span, "loads": _udl(10)}, _F80_OUT)


@pytest.mark.parametrize("w", [10, 4])
def test_f80_b_overhang_needs_support_positions(w):
    # the calculator states it does not guess support positions for an overhang
    _refuses(F80, {"span_m": 6, "support": "overhang", "loads": _udl(w)}, _F80_OUT)


# ---------------------------------------------------------------- F81 payment
F81 = "calculate_payment"
_F81_OUT = ("retention_held", "net_payment_due")


@pytest.mark.parametrize("claimed, cert, rate, prev, contract, ret, net, disp, cum, pct", [
    # 100000 x 0.05 = 5000; 95000; 120000 - 100000 = 20000;
    # 300000 + 100000 = 400000; / 1000000 = 40.0 %
    (120000, 100000, 0.05, 300000, 1000000, 5000, 95000, 20000, 400000, 40.0),
    # 48000 x 0.10 = 4800; 43200; 2000; 0 + 48000 = 48000; / 240000 = 20.0 %
    (50000, 48000, 0.10, 0, 240000, 4800, 43200, 2000, 48000, 20.0),
])
def test_f81_t(claimed, cert, rate, prev, contract, ret, net, disp, cum, pct):
    r = _ok(F81, {"claimed_amount": claimed, "certified_amount": cert,
                  "retention_rate": rate, "cumulative_previous_certified": prev,
                  "contract_value": contract})
    assert r["retention_held"] == pytest.approx(ret, rel=_REL)
    assert r["net_payment_due"] == pytest.approx(net, rel=_REL)
    assert r["disputed_amount"] == pytest.approx(disp, rel=_REL)
    assert r["cumulative_certified"] == pytest.approx(cum, rel=_REL)
    assert r["percent_complete"] == pytest.approx(pct, rel=_REL)


@pytest.mark.parametrize("params", [
    {"claimed_amount": 120000, "retention_rate": 0.05},
    {"claimed_amount": 50000, "retention_rate": 0.10, "contract_value": 240000},
])
def test_f81_m_certified(params):
    _asks(F81, params, "certified", _F81_OUT)


@pytest.mark.parametrize("cert, rate, net", [
    (80000, 0.0, 80000),   # no retention: net = certified
    (80000, 1.0, 0),       # all retained: net = 0
])
def test_f81_b_retention_limits(cert, rate, net):
    r = _ok(F81, {"claimed_amount": cert, "certified_amount": cert, "retention_rate": rate})
    assert r["net_payment_due"] == pytest.approx(net, abs=0.01)
    assert r["disputed_amount"] == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize("params", [
    {"claimed_amount": 1000, "certified_amount": 1000, "retention_rate": 1.5},   # over 100 %
    {"claimed_amount": 1000, "certified_amount": -1, "retention_rate": 0.05},    # negative certificate
])
def test_f81_b_out_of_range_refused(params):
    _refuses(F81, params, _F81_OUT)


# ---------------------------------------------------------------- F82 cash flow
F82 = "cash_flow"
_F82_OUT = ("peak_funding_requirement", "check_valuation")


@pytest.mark.parametrize("V, C, a, r, lag, contract, adv, ret, valuation, peak", [
    # 300; advance 30; retention 15; certified 85, 85, 85 + 15 = 100 -> 270;
    # lag 15 d = 1 month: cash in 0, 85, 85; net -90, -5, -5;
    # balance 30 -> -60 -> -65 -> -70: peak 70
    ([100, 100, 100], [90, 90, 90], 0.10, 0.05, 15, 300, 30, 15, 270, 70),
    # 600; advance 0; retention 60; certified 180, 360 + 60 = 420 -> 600;
    # lag 30 d = 1 month: cash in 0, 180; net -150, -120;
    # balance 0 -> -150 -> -270: peak 270
    ([200, 400], [150, 300], 0.0, 0.10, 30, 600, 0, 60, 600, 270),
])
def test_f82_t(V, C, a, r, lag, contract, adv, ret, valuation, peak):
    res = _ok(F82, {"monthly_value": V, "monthly_cost": C, "advance_pct": a,
                    "retention_pct": r, "payment_lag_days": lag})
    assert res["contract_value"] == pytest.approx(contract, rel=_REL)
    assert res["advance_payment"] == pytest.approx(adv, abs=0.01)
    assert res["retention_total"] == pytest.approx(ret, rel=_REL)
    assert res["check_valuation"] == pytest.approx(valuation, rel=_REL)
    assert res["peak_funding_requirement"] == pytest.approx(peak, rel=_REL)


@pytest.mark.parametrize("V", [[100, 100, 100], [200, 400]])
def test_f82_m_cost(V):
    _asks(F82, {"monthly_value": V}, "monthly_cost", _F82_OUT)


@pytest.mark.parametrize("lag, peak", [
    (30, 70),    # ceil(30/30) = 1 month: as the first T case, peak 70
    (31, 155),   # ceil(31/30) = 2 months: cash in 0, 0, 85; net -90, -90, -5;
                 # balance 30 -> -60 -> -150 -> -155: peak 155
])
def test_f82_b_lag_month_edge(lag, peak):
    res = _ok(F82, {"monthly_value": [100, 100, 100], "monthly_cost": [90, 90, 90],
                    "advance_pct": 0.10, "retention_pct": 0.05, "payment_lag_days": lag})
    assert res["peak_funding_requirement"] == pytest.approx(peak, rel=_REL)


@pytest.mark.parametrize("V, C", [
    ([100, 100, 100], [90, 90]),   # one cost month short
    ([200, 400], [150, 300, 10]),  # one cost month too many
])
def test_f82_b_series_length_mismatch_refused(V, C):
    _refuses(F82, {"monthly_value": V, "monthly_cost": C}, _F82_OUT)


# ---------------------------------------------------------------- F83 mix design (SG)
F83 = "concrete_mix_design_sg"
_F83_OUT = ("cement_kg_m3", "water_litres_m3")


@pytest.mark.parametrize("wc, sgc, sgf, sgcr, f, c, cem, wat, fine, coarse", [
    # 0.5 + 1/3.15 + 2/2.67 + 4/2.54 = 0.5 + 0.31746 + 0.74906 + 1.57480
    # = 3.14132; C = 318.3 -> 318 kg; W 159; F 636; Cr 1272
    (0.5, 3.15, 2.67, 2.54, 2.0, 4.0, 318, 159, 636, 1272),
    # 0.45 + 1/3.15 + 1.6/2.6 + 3.2/2.7 = 0.45 + 0.31746 + 0.61538 + 1.18519
    # = 2.56803; C = 389.4 -> 389 kg; W 175.05 -> 175; F 622.4 -> 622;
    # Cr 1244.8 -> 1245
    (0.45, 3.15, 2.6, 2.7, 1.6, 3.2, 389, 175, 622, 1245),
])
def test_f83_t(wc, sgc, sgf, sgcr, f, c, cem, wat, fine, coarse):
    r = _ok(F83, {"w_c_ratio": wc, "cement_sg": sgc, "fine_agg_sg": sgf,
                  "coarse_agg_sg": sgcr, "fine_agg_ratio": f, "coarse_agg_ratio": c})
    # each mass is stated to the whole kilogram
    assert r["cement_kg_m3"] == pytest.approx(cem, abs=1)
    assert r["water_litres_m3"] == pytest.approx(wat, abs=1)
    assert r["fine_aggregate_kg_m3"] == pytest.approx(fine, abs=1)
    assert r["coarse_aggregate_kg_m3"] == pytest.approx(coarse, abs=1)
    assert r["total_weight_kg_m3"] == pytest.approx(cem + wat + fine + coarse, abs=2)


@pytest.mark.parametrize("params", [{}, {"cement_sg": 3.15, "fine_agg_ratio": 2}])
def test_f83_m_wc(params):
    _asks(F83, params, "w_c_ratio", _F83_OUT)


@pytest.mark.parametrize("wc", [0.1, 1.2])
def test_f83_b_wc_outside_range_refused(wc):
    # the calculator states its range: w/c 0.2 to 1
    _refuses(F83, {"w_c_ratio": wc}, _F83_OUT)


@pytest.mark.parametrize("dune, ok", [(0, True), (10, False)])
def test_f83_b_dune_sand_not_applied(dune, ok):
    # no dune-sand SG is defined, so any share other than 0 is refused
    params = {"w_c_ratio": 0.5, "dune_sand_pct": dune}
    if ok:
        assert _ok(F83, params)["cement_kg_m3"] == pytest.approx(318, abs=1)
    else:
        _refuses(F83, params, _F83_OUT)


# ---------------------------------------------------------------- F84 mix proportions
F84 = "concrete_mix_proportions"
_F84_OUT = ("dry_volume", "cement_volume", "sand_volume", "aggregate_volume")


@pytest.mark.parametrize("wet, c, s, a, waste, dry, cem, sand, agg", [
    (10, 1, 2, 4, None, 15.4, 2.2, 4.4, 8.8),     # 10 x 1.54 = 15.4; / 7 parts
    (5, 1, 1.5, 3, None, 7.7, 1.4, 2.1, 4.2),     # 5 x 1.54 = 7.7; / 5.5 parts
    (10, 1, 2, 4, 0.05, 16.17, 2.31, 4.62, 9.24),  # 15.4 x 1.05 = 16.17; / 7
])
def test_f84_t(wet, c, s, a, waste, dry, cem, sand, agg):
    params = {"wet_volume": wet, "cement_parts": c, "sand_parts": s, "aggregate_parts": a}
    if waste is not None:
        params["waste_factor"] = waste
    r = _ok(F84, params)
    assert r["dry_volume"] == pytest.approx(dry, rel=_REL)
    assert r["cement_volume"] == pytest.approx(cem, rel=_REL)
    assert r["sand_volume"] == pytest.approx(sand, rel=_REL)
    assert r["aggregate_volume"] == pytest.approx(agg, rel=_REL)


@pytest.mark.parametrize("wet, cem", [
    ("353.1467 ft3", 2.2),    # 353.1467 x 0.0283168 = 10.000 m3 -> as 10 m3
    ("176.5733 ft3", 1.1),    # 5.000 m3; 5 x 1.54 / 7 = 1.1
])
def test_f84_u(wet, cem):
    r = _ok(F84, {"wet_volume": wet, "cement_parts": 1, "sand_parts": 2, "aggregate_parts": 4})
    assert r["cement_volume"] == pytest.approx(cem, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"wet_volume": 10, "cement_parts": 1, "aggregate_parts": 4}, "sand_parts"),
    ({"cement_parts": 1, "sand_parts": 2, "aggregate_parts": 4}, "wet_volume"),
])
def test_f84_m(params, missing):
    _asks(F84, params, missing, _F84_OUT)


@pytest.mark.parametrize("params", [
    {"wet_volume": 0, "cement_parts": 1, "sand_parts": 2, "aggregate_parts": 4},
    {"wet_volume": 10, "cement_parts": 0, "sand_parts": 0, "aggregate_parts": 0},
])
def test_f84_b_nothing_to_mix_refused(params):
    _refuses(F84, params, _F84_OUT)


# ---------------------------------------------------------------- F85 slip-form mix
F85 = "concrete_mix_slip_form"


@pytest.mark.parametrize("key, kg", [
    # stated mix 1:2:2.6, w/c 0.42, default SGs 3.15 / 2.67 / 2.54:
    # 0.42 + 1/3.15 + 2/2.67 + 2.6/2.54 = 0.42 + 0.31746 + 0.74906 + 1.02362
    # = 2.51014; C = 398.4 -> 398 kg
    ("cement_kg_m3", 398),
    ("water_litres_m3", 167),            # 398 x 0.42 = 167.2 -> 167
    ("fine_aggregate_kg_m3", 796),       # 398 x 2 = 796
    ("coarse_aggregate_kg_m3", 1035),    # 398 x 2.6 = 1034.8 -> 1035
])
def test_f85_t(key, kg):
    assert _ok(F85, {})[key] == pytest.approx(kg, abs=1)


@pytest.mark.parametrize("key, value", [
    ("retarder_20c_lit_m3", 3.8),   # stated in the calculator
    ("retarder_10c_lit_m3", 2.9),
    ("max_concrete_temp_c", 32),
])
def test_f85_t_stated_constants(key, value):
    assert _ok(F85, {})[key] == pytest.approx(value, rel=_REL)


@pytest.mark.parametrize("extra", [{"slump_mm": 200}, {"w_c_ratio": 0.6}])
def test_f85_b_documented_mix_is_locked(extra):
    # the calculator states its constants are locked; a stray input does
    # not move the mix (398 kg as above)
    assert _ok(F85, extra)["cement_kg_m3"] == pytest.approx(398, abs=1)


# ---------------------------------------------------------------- F86 thermal check
F86 = "concrete_thermal_cracking_check"
_F86_OUT = ("delta_t_c", "thermal_cracking_risk")


@pytest.mark.parametrize("core, surface, delta, core_ok, delta_ok", [
    (65, 50, 15, True, True),     # 65 <= 70; 15 <= 20
    (55, 42, 13, True, True),     # 55 <= 70; 13 <= 20
    (75, 60, 15, False, True),    # core over 70
    (60, 35, 25, True, False),    # difference over 20
])
def test_f86_t(core, surface, delta, core_ok, delta_ok):
    r = _ok(F86, {"core_temp_c": core, "surface_temp_c": surface})
    assert r["delta_t_c"] == pytest.approx(delta, abs=0.05)
    assert r["core_ok"] is core_ok
    assert r["delta_ok"] is delta_ok
    assert r["thermal_cracking_risk"] is not (core_ok and delta_ok)


@pytest.mark.parametrize("core, surface, delta, risk", [
    ("158 degF", "122 degF", 20.0, False),   # (158-32)/1.8 = 70; (122-32)/1.8 = 50
    ("140 degF", "86 degF", 30.0, True),     # 60 and 30 C: difference 30 > 20
])
def test_f86_u_fahrenheit(core, surface, delta, risk):
    r = _ok(F86, {"core_temp_c": core, "surface_temp_c": surface})
    assert r["delta_t_c"] == pytest.approx(delta, abs=0.05)
    assert r["thermal_cracking_risk"] is risk


@pytest.mark.parametrize("params", [{"core_temp_c": 65}, {"core_temp_c": 72}])
def test_f86_m_surface(params):
    _asks(F86, params, "surface", _F86_OUT)


@pytest.mark.parametrize("core, surface, risk", [
    (70, 50, False),      # core on the 70 limit, difference on the 20 limit
    (70.1, 60, True),     # core just past 70
    (50, 30, False),      # difference on 20
    (50, 29.5, True),     # difference 20.5, just past 20
])
def test_f86_b_limits(core, surface, risk):
    r = _ok(F86, {"core_temp_c": core, "surface_temp_c": surface})
    assert r["thermal_cracking_risk"] is risk


# ---------------------------------------------------------------- F87 crane count
F87 = "crane_planning"
_F87_OUT = ("cranes_required", "daily_tonnage_per_crane")
_F87_SLOW = {"utilization_pct": 50, "shifts_per_day": 1, "hours_per_shift": 10,
             "cycle_time_minutes": 30, "working_days": 25}


@pytest.mark.parametrize("params, lph, daily, n, cost", [
    # defaults 65 %, 2 x 9.5 h, 20 min, 26 d: 60/20 = 3 lifts/h;
    # 10 x 0.65 = 6.5 t/lift; 3 x 9.5 x 2 = 57 lifts; 370.5 t/day;
    # x 26 = 9633 t/month; ceil(10000 / 9633) = 2; 2 x 30000 = 60000
    ({"total_lift_demand_tons": 10000, "crane_capacity_tons": 10,
      "dry_hire_rate_per_month": 30000}, 3.0, 370.5, 2, 60000),
    # 60/30 = 2 lifts/h; 20 x 0.5 = 10 t; 2 x 10 x 1 = 20 lifts; 200 t/day;
    # x 25 = 5000 t/month; ceil(12000 / 5000) = 3; 3 x 25000 = 75000
    ({"total_lift_demand_tons": 12000, "crane_capacity_tons": 20,
      "dry_hire_rate_per_month": 25000, **_F87_SLOW}, 2.0, 200.0, 3, 75000),
])
def test_f87_t(params, lph, daily, n, cost):
    r = _ok(F87, params)
    assert r["lifts_per_hour_per_crane"] == pytest.approx(lph, rel=_REL)
    assert r["daily_tonnage_per_crane"] == pytest.approx(daily, rel=_REL)
    assert r["cranes_required"] == n
    assert r["total_monthly_cost"] == pytest.approx(cost, rel=_REL)


@pytest.mark.parametrize("cap, daily", [
    ("10000 kg", 370.5),   # 10 t: as the first T case
    ("20000 kg", 741.0),   # 20 t: 20 x 0.65 x 57 = 741
])
def test_f87_u_capacity(cap, daily):
    r = _ok(F87, {"total_lift_demand_tons": 10000, "crane_capacity_tons": cap})
    assert r["daily_tonnage_per_crane"] == pytest.approx(daily, rel=_REL)


@pytest.mark.parametrize("params", [
    {"total_lift_demand_tons": 10000},
    {"total_lift_demand_tons": 12000, **_F87_SLOW},
])
def test_f87_m_capacity(params):
    _asks(F87, params, "crane_capacity", _F87_OUT)


@pytest.mark.parametrize("demand", [10000, 12000])
def test_f87_m_rate_gives_count_without_cost(demand):
    # no rate is built in: the count stands, the cost is withheld
    r = _ok(F87, {"total_lift_demand_tons": demand, "crane_capacity_tons": 10})
    assert r["cranes_required"] == 2      # ceil(10000 / 9633) = ceil(12000 / 9633) = 2
    assert r["total_monthly_cost"] is None


@pytest.mark.parametrize("demand, n", [
    (5000, 1),   # exactly one crane-month at 5000 t: ceil(1.0) = 1
    (5001, 2),   # one tonne past it: ceil(1.0002) = 2
])
def test_f87_b_capacity_edge(demand, n):
    r = _ok(F87, {"total_lift_demand_tons": demand, "crane_capacity_tons": 20, **_F87_SLOW})
    assert r["cranes_required"] == n


@pytest.mark.parametrize("cap", [0, -5])
def test_f87_b_no_capacity_refused(cap):
    _refuses(F87, {"total_lift_demand_tons": 10000, "crane_capacity_tons": cap}, _F87_OUT)


# ---------------------------------------------------------------- F88 electrical programme
F88 = "electrical_installation_sequence"
_F88_OUT = ("stages", "total_days")


@pytest.mark.parametrize("area, floors, rates, stages, total", [
    # 1000 x 5 = 5000 m2; 5000/250 = 20, /500 = 10, /1000 = 5; 35 days
    (1000, 5, {"first_fix": 250, "second_fix": 500, "testing": 1000},
     {"first_fix": 20, "second_fix": 10, "testing": 5}, 35),
    # 800 x 3 = 2400 m2; ceil(2400/300) = 8, ceil(2400/700 = 3.43) = 4; 12
    (800, 3, {"containment": 300, "terminations": 700},
     {"containment": 8, "terminations": 4}, 12),
])
def test_f88_t(area, floors, rates, stages, total):
    r = _ok(F88, {"floor_area_m2": area, "num_floors": floors, "stage_output_rates": rates})
    assert r["total_area_m2"] == pytest.approx(area * floors, rel=_REL)
    assert r["stages"] == stages
    assert r["total_days"] == total


@pytest.mark.parametrize("area, days", [
    ("10763.9104 ft2", 20),   # x 0.09290304 = 1000.0 m2; x 5 / 250 = 20
    ("5381.9552 ft2", 10),    # 500.0 m2; x 5 / 250 = 10
])
def test_f88_u_area(area, days):
    r = _ok(F88, {"floor_area_m2": area, "num_floors": 5, "stage_output_rates": {"first_fix": 250}})
    assert r["total_days"] == days


@pytest.mark.parametrize("params, missing", [
    ({"floor_area_m2": 1000, "num_floors": 5}, "stage_output_rates"),
    ({"num_floors": 5, "stage_output_rates": {"first_fix": 250}}, "floor_area"),
])
def test_f88_m(params, missing):
    _asks(F88, params, missing, _F88_OUT)


@pytest.mark.parametrize("rate, days", [
    (1000, 1),   # 1000 / 1000 = 1 day exactly
    (999, 2),    # 1000 / 999 = 1.001 -> a part day is a whole day: 2
])
def test_f88_b_part_day_rounds_up(rate, days):
    r = _ok(F88, {"floor_area_m2": 1000, "num_floors": 1, "stage_output_rates": {"stage": rate}})
    assert r["total_days"] == days


@pytest.mark.xfail(strict=True, reason=(
    "suspected defect: a stage whose output rate is 0 is dropped from the "
    "programme and the total is reported without it; a stage with no rate "
    "cannot be programmed and must be asked for"))
@pytest.mark.parametrize("rates", [
    {"first_fix": 250, "second_fix": 0},
    {"containment": 500, "testing": 0, "terminations": 100},
])
def test_f88_b_zero_rate_stage_is_not_dropped(rates):
    env = run_calculation(F88, {"floor_area_m2": 1000, "num_floors": 1, "stage_output_rates": rates})
    assert env.get("status") != "success", env
