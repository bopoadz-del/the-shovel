"""Formula Layer 1, batch 9 (exam 5): geometry, reinforcement, money, checks.

Same rules as batches 1-8 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation, written in the comment beside
its parametrize row:
  * panel volume (prism)            V = L t D per panel; total = n V;
                                    with tremie waste = 1.10 total (the
                                    factor is the calculator's stated
                                    allowance, read from its output)
  * bar mass per metre              m = pi/4 d^2 rho, rho = 7850 kg/m3;
                                    bars per metre = 1000/s;
                                    length = (1000/s) A ways
  * return on investment            ROI = (gain - cost)/cost x 100
  * tolerance band                  low = S - minus, high = S + plus
                                    (minus = plus when not given);
                                    within iff low <= m <= high;
                                    margin = min(m - low, high - m)
  * productivity                    rate = Q/man-hours;
                                    crew-hours = man-hours/crew;
                                    per crew-hour = Q/crew-hours
  * de-rated crane chart            net = chart - deductions;
                                    utilisation = load/net;
                                    passes iff load <= net and
                                    utilisation <= the stated limit
None was copied from the product's output. A case where the product
disagrees with the hand value is marked ``xfail(strict=True)`` with a
one-line reason and is a suspected defect for the advisor; the expected
value is never moved to match. Every value test runs on two inputs worked
by hand the same way.

No U case for F58 (ROI) or F59 (tolerance): both are unit-free by their
own contract (currency in, currency out; "same unit as specified").

Unit factors (exact by definition): 1 ft = 0.3048 m; 1 in = 25.4 mm;
1 ft2 = 0.09290304 m2; 1 lb = 0.45359237 kg.

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


# ---------------------------------------------------------------- F56 diaphragm wall
F56 = "diaphragm_wall_panel_volume"
_V = "total_volume_m3"


@pytest.mark.parametrize("L, t, D, n, per, total, waste", [
    # 6.0 x 0.8 x 25 = 120 m3; x 4 = 480; x 1.10 = 528
    (6.0, 0.8, 25, 4, 120.0, 480.0, 528.0),
    # 2.8 x 1.2 x 40.5 = 136.08 m3; x 1 = 136.08; x 1.10 = 149.688
    (2.8, 1.2, 40.5, 1, 136.08, 136.08, 149.688),
])
def test_f56_t(L, t, D, n, per, total, waste):
    r = _ok(F56, {"panel_length": L, "wall_thickness": t, "excavation_depth": D,
                  "panel_count": n})
    assert r["volume_per_panel_m3"] == pytest.approx(per, rel=_REL)
    assert r[_V] == pytest.approx(total, rel=_REL)
    assert r["volume_with_waste_m3"] == pytest.approx(waste, rel=_REL)


@pytest.mark.parametrize("params", [
    # 6.0 m written in feet (6/0.3048 = 19.68504 ft), 0.8 m in mm -> 120 m3
    {"panel_length": "19.68503937 ft", "wall_thickness": "800 mm",
     "excavation_depth": 25, "panel_count": 1},
    # 25 m written in feet (25/0.3048 = 82.02100 ft) -> 120 m3
    {"panel_length": 6.0, "wall_thickness": 0.8,
     "excavation_depth": "82.02099738 ft", "panel_count": 1},
])
def test_f56_u(params):
    assert _ok(F56, params)[_V] == pytest.approx(120.0, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"panel_length": 6.0, "excavation_depth": 25}, "wall_thickness"),
    ({"panel_length": 6.0, "wall_thickness": 0.8}, "excavation_depth"),
])
def test_f56_m(params, missing):
    _asks(F56, params, missing, (_V, "volume_with_waste_m3"))


@pytest.mark.parametrize("params", [
    {"panel_length": 0, "wall_thickness": 0.8, "excavation_depth": 25},
    {"panel_length": 6.0, "wall_thickness": 0, "excavation_depth": 25},
])
def test_f56_b_zero_dimension_refused(params):
    _refuses(F56, params, (_V,))


@pytest.mark.parametrize("n", [0, -2])
def test_f56_b_no_panels_refused(n):
    _refuses(F56, {"panel_length": 6.0, "wall_thickness": 0.8, "excavation_depth": 25,
                   "panel_count": n}, (_V,))


# ---------------------------------------------------------------- F57 rebar by area
F57 = "rebar_by_area"
_M = "total_mass_kg"


@pytest.mark.parametrize("A, s, d, both, bars, length, mass", [
    # m = pi/4 x 0.012^2 x 7850 = 0.887814 kg/m; 1000/200 = 5 bars/m;
    # 5 x 50 = 250 m; 250 x 0.887814 = 221.954 kg
    (50, 200, 12, False, 5.0, 250.0, 221.954),
    # m = pi/4 x 0.016^2 x 7850 = 1.578336 kg/m; 1000/150 = 6.6667 bars/m;
    # 6.6667 x 20 x 2 = 266.667 m; x 1.578336 = 420.890 kg
    (20, 150, 16, True, 6.66667, 266.667, 420.890),
])
def test_f57_t(A, s, d, both, bars, length, mass):
    r = _ok(F57, {"area_m2": A, "spacing_mm": s, "bar_diameter_mm": d, "both_ways": both})
    assert r["bars_per_m"] == pytest.approx(bars, rel=_REL)
    assert r["total_bar_length_m"] == pytest.approx(length, rel=_REL)
    assert r[_M] == pytest.approx(mass, rel=_REL)


@pytest.mark.parametrize("params", [
    # 50 m2 written in ft2 (50/0.09290304 = 538.1955 ft2) -> 221.954 kg
    {"area_m2": "538.1955208 ft2", "spacing_mm": 200, "bar_diameter_mm": 12},
    # 200 mm written in inches (200/25.4 = 7.874016 in) -> 221.954 kg
    {"area_m2": 50, "spacing_mm": "7.874015748 in", "bar_diameter_mm": 12},
])
def test_f57_u(params):
    assert _ok(F57, params)[_M] == pytest.approx(221.954, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"area_m2": 50, "bar_diameter_mm": 12}, "spacing"),
    ({"area_m2": 50, "spacing_mm": 200}, "bar_diameter"),
])
def test_f57_m(params, missing):
    _asks(F57, params, missing, (_M,))


@pytest.mark.parametrize("d", [10, 25])
def test_f57_b_zero_area(d):
    r = _ok(F57, {"area_m2": 0, "spacing_mm": 200, "bar_diameter_mm": d})
    assert r[_M] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("d, one_way", [
    # 10 mm @ 250 over 35 m2: m = 0.616538 kg/m; 4 x 35 = 140 m -> 86.315 kg
    (10, 86.3153),
    # 12 mm @ 250 over 35 m2: 140 m x 0.887814 = 124.294 kg
    (12, 124.294),
])
def test_f57_b_both_ways_doubles(d, one_way):
    one = _ok(F57, {"area_m2": 35, "spacing_mm": 250, "bar_diameter_mm": d})[_M]
    two = _ok(F57, {"area_m2": 35, "spacing_mm": 250, "bar_diameter_mm": d,
                    "both_ways": True})[_M]
    assert one == pytest.approx(one_way, rel=_REL)
    assert two == pytest.approx(2 * one_way, rel=_REL)


@pytest.mark.parametrize("s", [0, -150])
def test_f57_b_spacing_not_positive_refused(s):
    _refuses(F57, {"area_m2": 50, "spacing_mm": s, "bar_diameter_mm": 12}, (_M,))


# ---------------------------------------------------------------- F58 ROI
F58 = "roi_calculator"
_R = "roi_percent"


@pytest.mark.parametrize("gain, cost, net, roi", [
    (1_250_000, 1_000_000, 250_000, 25.0),   # 250000/1000000 x 100
    (800, 1000, -200, -20.0),                # a loss: -200/1000 x 100
])
def test_f58_t(gain, cost, net, roi):
    r = _ok(F58, {"gain": gain, "cost": cost})
    assert r["net_profit"] == pytest.approx(net, rel=_REL)
    assert r[_R] == pytest.approx(roi, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"gain": 1_250_000}, "cost"),
    ({"cost": 1_000_000}, "gain"),
])
def test_f58_m(params, missing):
    _asks(F58, params, missing, (_R,))


@pytest.mark.parametrize("amount", [1000, 37.5])
def test_f58_b_break_even(amount):
    r = _ok(F58, {"gain": amount, "cost": amount})
    assert r[_R] == pytest.approx(0.0, abs=1e-9)
    assert r["net_profit"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("gain", [100, 0])
def test_f58_b_zero_cost_refused(gain):
    _refuses(F58, {"gain": gain, "cost": 0}, (_R,))


# ---------------------------------------------------------------- F59 tolerance
F59 = "tolerance_check"
_W = "within_tolerance"


@pytest.mark.parametrize("params, within, dev, low, high, margin", [
    # 100 +/- 5 -> [95, 105]; 102 is in; dev +2; margin min(7, 3) = 3
    ({"measured": 102, "specified": 100, "tolerance_plus": 5},
     True, 2.0, 95.0, 105.0, 3.0),
    # 100 +3/-2 -> [98, 103]; 96.5 is out; dev -3.5; margin min(-1.5, 6.5) = -1.5
    ({"measured": 96.5, "specified": 100, "tolerance_plus": 3, "tolerance_minus": 2},
     False, -3.5, 98.0, 103.0, -1.5),
])
def test_f59_t(params, within, dev, low, high, margin):
    r = _ok(F59, params)
    assert r[_W] is within
    assert r["deviation"] == pytest.approx(dev, rel=_REL)
    assert r["lower_limit"] == pytest.approx(low, rel=_REL)
    assert r["upper_limit"] == pytest.approx(high, rel=_REL)
    assert r["margin"] == pytest.approx(margin, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"specified": 100, "tolerance_plus": 5}, "measured"),
    ({"measured": 102, "specified": 100}, "tolerance_plus"),
])
def test_f59_m(params, missing):
    _asks(F59, params, missing, (_W,))


@pytest.mark.parametrize("measured", [105, 95])
def test_f59_b_on_the_limit_is_within(measured):
    # 100 +/- 5: both ends of the band are inside it, margin exactly 0
    r = _ok(F59, {"measured": measured, "specified": 100, "tolerance_plus": 5})
    assert r[_W] is True
    assert r["margin"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("measured", [105.01, 94.99])
def test_f59_b_just_past_the_limit_is_outside(measured):
    r = _ok(F59, {"measured": measured, "specified": 100, "tolerance_plus": 5})
    assert r[_W] is False
    assert r["margin"] == pytest.approx(-0.01, rel=1e-2)


# ---------------------------------------------------------------- F60 productivity
F60 = "productivity_rate"
_P = "rate_per_hour"


@pytest.mark.parametrize("Q, mh, crew, rate, crew_h, per_crew", [
    (120, 40, 4, 3.0, 10.0, 12.0),     # 120/40; 40/4; 120/10
    (450, 60, 5, 7.5, 12.0, 37.5),     # 450/60; 60/5; 450/12
])
def test_f60_t(Q, mh, crew, rate, crew_h, per_crew):
    r = _ok(F60, {"output_quantity": Q, "labor_hours": mh, "crew_size": crew})
    assert r[_P] == pytest.approx(rate, rel=_REL)
    assert r["crew_hours"] == pytest.approx(crew_h, rel=_REL)
    assert r["rate_per_crew_hour"] == pytest.approx(per_crew, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"output_quantity": 120, "crew_size": 4}, "labor_hours"),
    ({"labor_hours": 40, "crew_size": 4}, "output_quantity"),
])
def test_f60_m(params, missing):
    _asks(F60, params, missing, (_P,))


@pytest.mark.parametrize("Q, mh, rate", [(120, 40, 3.0), (450, 60, 7.5)])
def test_f60_m_no_crew_withholds_crew_figures(Q, mh, rate):
    # The rate per man-hour needs no crew; the per-crew figures do, and
    # must not be filled by an assumed crew.
    r = _ok(F60, {"output_quantity": Q, "labor_hours": mh})
    assert r[_P] == pytest.approx(rate, rel=_REL)
    assert r.get("rate_per_crew_hour") is None
    assert r.get("crew_hours") is None
    assert "crew_size" in (r.get("needs_input") or [])


@pytest.mark.parametrize("mh", [40, 7.25])
def test_f60_b_zero_output(mh):
    assert _ok(F60, {"output_quantity": 0, "labor_hours": mh})[_P] == \
        pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("params", [
    {"output_quantity": 120, "labor_hours": 0, "crew_size": 4},
    {"output_quantity": 120, "labor_hours": 40, "crew_size": 0},
])
def test_f60_b_zero_hours_or_crew_refused(params):
    _refuses(F60, params, ("rate_per_crew_hour",))


# ---------------------------------------------------------------- F61 crane lift
F61 = "crane_lift_capacity"
_N = "net_capacity_t"


@pytest.mark.parametrize("chart, ded, load, lim, net, util, passed", [
    # 50 - 2.5 = 47.5 t; 30/47.5 = 63.158 % <= 75 % -> passes
    (50, 2.5, 30, 0.75, 47.5, 63.158, True),
    # 20 - 1.2 = 18.8 t; 17/18.8 = 90.426 % > 85 % -> fails
    (20, 1.2, 17, 0.85, 18.8, 90.426, False),
])
def test_f61_t(chart, ded, load, lim, net, util, passed):
    r = _ok(F61, {"chart_capacity_t": chart, "deductions_t": ded, "load_t": load,
                  "max_utilization": lim})
    assert r[_N] == pytest.approx(net, rel=_REL)
    assert r["utilization_percent"] == pytest.approx(util, rel=_REL)
    assert r["passed"] is passed


@pytest.mark.parametrize("params", [
    # 50 t written in kg -> net 47.5 t
    {"chart_capacity_t": "50000 kg", "deductions_t": 2.5, "load_t": 30,
     "max_utilization": 0.75},
    # 50 t written in lb (50000/0.45359237 = 110231.131 lb) -> net 47.5 t
    {"chart_capacity_t": "110231.1311 lb", "deductions_t": 2.5, "load_t": 30,
     "max_utilization": 0.75},
])
def test_f61_u(params):
    r = _ok(F61, params)
    assert r[_N] == pytest.approx(47.5, rel=_REL)
    assert r["utilization_percent"] == pytest.approx(63.158, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"chart_capacity_t": 50, "deductions_t": 2.5}, "load_t"),
    ({"deductions_t": 2.5, "load_t": 30}, "chart_capacity"),
])
def test_f61_m(params, missing):
    _asks(F61, params, missing, ("utilization_percent", "passed"))


@pytest.mark.parametrize("lim, passed", [
    # 40 - 0 = 40 t; 32/40 = 80 %: on an 80 % limit it passes,
    # on a 79 % limit it fails -- the limit is read from the input
    (0.80, True),
    (0.79, False),
])
def test_f61_b_limit_read_from_input(lim, passed):
    r = _ok(F61, {"chart_capacity_t": 40, "deductions_t": 0, "load_t": 32,
                  "max_utilization": lim})
    assert r["utilization_percent"] == pytest.approx(80.0, rel=_REL)
    assert r["passed"] is passed


@pytest.mark.parametrize("load", [47.6, 60])
def test_f61_b_load_over_net_fails(load):
    # net 47.5 t; any load above it fails even on a 100 % limit
    r = _ok(F61, {"chart_capacity_t": 50, "deductions_t": 2.5, "load_t": load,
                  "max_utilization": 1.0})
    assert r["passed"] is False


@pytest.mark.parametrize("ded", [50, 55])
def test_f61_b_deductions_eat_the_chart_refused(ded):
    _refuses(F61, {"chart_capacity_t": 50, "deductions_t": ded, "load_t": 10,
                   "max_utilization": 0.85}, (_N, "passed"))
