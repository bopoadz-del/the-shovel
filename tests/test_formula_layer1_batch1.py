"""Formula Layer 1, batch 1: nine formulas, four case kinds each.

Case kinds (formula test plan, Layer 1):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it (status
     ``needs_input`` or ``error`` naming the input) and must NOT return the
     figure it would have needed that input for;
  B  boundary -- zero, a value on the limit, rounding direction.

Every expected value is a HAND calculation from the published definition,
written in the comment beside its parametrize row. None was copied from the
product's output. A case where the product disagrees with the hand value is
marked ``xfail(strict=True)`` with a one-line reason and is a suspected
formula defect for the advisor; the expected value is never moved to match.
Every value test runs on two inputs: the plan's case and a second invented
case worked by hand the same way.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import math
import re

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


# ---------------------------------------------------------------- F1 beam
# Simply supported, uniform load: M = w L^2 / 8.

@pytest.mark.parametrize("w, span, moment", [
    (20, 6, 90.0),    # 20 x 36 / 8 = 90 kN.m
    (10, 4, 20.0),    # 10 x 16 / 8 = 20 kN.m
])
def test_f1_t_beam_moment_typical(w, span, moment):
    r = _ok("beam_moment_simple", {"udl_w_kn_m": w, "span_m": span})
    assert r["max_moment_kn_m"] == pytest.approx(moment, rel=_REL)


@pytest.mark.parametrize("w, span, moment", [
    (20, "20 ft", 92.903),  # 20 x 0.3048 = 6.096 m; 6.096^2 = 37.1612; 20 x 37.1612 / 8 = 92.903
    (10, "10 ft", 11.613),  # 3.048 m; 3.048^2 = 9.290304; 10 x 9.290304 / 8 = 11.6129
])
def test_f1_u_beam_moment_span_in_feet(w, span, moment):
    r = _ok("beam_moment_simple", {"udl_w_kn_m": w, "span_m": span})
    assert r["max_moment_kn_m"] == pytest.approx(moment, rel=_REL)


@pytest.mark.parametrize("span", [6, 4])
def test_f1_m_beam_moment_missing_load_asks(span):
    _asks("beam_moment_simple", {"span_m": span}, "udl_w_kn_m", ("max_moment_kn_m",))


@pytest.mark.parametrize("w", [20, 5])
def test_f1_b_beam_moment_zero_span_no_nan(w):
    # L = 0: w x 0 / 8 = 0, or a clear refusal; never NaN and never a crash.
    env = run_calculation("beam_moment_simple", {"udl_w_kn_m": w, "span_m": 0})
    if env.get("status") == "success":
        assert env["result"]["max_moment_kn_m"] == 0
    else:
        assert (env.get("result") or {}).get("max_moment_kn_m") is None, env


@pytest.mark.parametrize("span", [6, 4])
def test_f1_b2_beam_moment_zero_load(span):
    # w = 0: 0 x L^2 / 8 = 0 exactly
    r = _ok("beam_moment_simple", {"udl_w_kn_m": 0, "span_m": span})
    assert r["max_moment_kn_m"] == 0


# ------------------------------------------------------- F2 interim payment
# retention = gross x rate; net = gross - retention.

@pytest.mark.parametrize("gross, pct, retention, net", [
    (1_250_000, 5, 62_500, 1_187_500),  # 1,250,000 x 0.05 = 62,500; 1,250,000 - 62,500
    (800_000, 3, 24_000, 776_000),      # 800,000 x 0.03 = 24,000; 800,000 - 24,000
])
def test_f2_t_interim_payment_typical(gross, pct, retention, net):
    r = _ok("calculate_interim_payment", {"gross_valuation": gross, "retention_percent": pct})
    assert r["retention_amount"] == pytest.approx(retention, abs=0.01)
    assert r["net_payment"] == pytest.approx(net, abs=0.01)


@pytest.mark.parametrize("gross, pct, retention, net", [
    ("1.25 million", "5 percent", 62_500, 1_187_500),   # 1,250,000 x 0.05; 1,250,000 - 62,500
    ("2.4 million", "2.5 percent", 60_000, 2_340_000),  # 2,400,000 x 0.025; 2,400,000 - 60,000
])
def test_f2_u_interim_payment_written_figures(gross, pct, retention, net):
    r = _ok("calculate_interim_payment", {"gross_valuation": gross, "retention_percent": pct})
    assert r["retention_amount"] == pytest.approx(retention, abs=0.01)
    assert r["net_payment"] == pytest.approx(net, abs=0.01)


@pytest.mark.xfail(strict=True, reason="suspected defect: retention_percent silently defaults to 10 (plan F2-M, ask-not-guess)")
@pytest.mark.parametrize("gross", [1_250_000, 800_000])
def test_f2_m_interim_payment_missing_retention_asks(gross):
    # A silent 10 % would give 1,250,000 - 125,000 = 1,125,000: must ask instead.
    _asks("calculate_interim_payment", {"gross_valuation": gross},
          "retention", ("net_payment", "retention_amount"))


@pytest.mark.parametrize("pct, net", [
    (0, 1_250_000),  # 0 %: net = gross
    (100, 0),        # 100 %: 1,250,000 - 1,250,000 = 0
])
def test_f2_b_interim_payment_retention_limits(pct, net):
    r = _ok("calculate_interim_payment", {"gross_valuation": 1_250_000, "retention_percent": pct})
    assert r["net_payment"] == pytest.approx(net, abs=0.01)


@pytest.mark.parametrize("pct", [150, 100.5])
def test_f2_b2_interim_payment_retention_over_100_refused(pct):
    # Above 100 % is not a retention rate: refuse, no net figure.
    _refuses("calculate_interim_payment", {"gross_valuation": 1_250_000, "retention_percent": pct},
             ("net_payment",))


# ------------------------------------------------------------ F3 crew sizing
# crews = Q / (R x D); round UP to whole crews.

@pytest.mark.parametrize("q, rate, days, crews, whole", [
    (2400, 30, 20, 4.0, 4),    # 2,400 / (30 x 20) = 2,400 / 600 = 4.0
    (1800, 15, 10, 12.0, 12),  # 1,800 / (15 x 10) = 1,800 / 150 = 12.0
])
def test_f3_t_crew_sizing_typical(q, rate, days, crews, whole):
    r = _ok("crew_sizing", {"quantity": q, "rate_per_crew_day": rate, "duration_days": days})
    assert r["crews_required"] == pytest.approx(crews, rel=_REL)
    assert r["crews_required_rounded_up"] == whole


@pytest.mark.xfail(strict=True, reason="suspected defect: quantity in ft2 is not converted to the rate's m2 (25,833.4 ft2 gives 43.056 = 25,833.4 / 600; hand value 4.0)")
@pytest.mark.parametrize("q, crews", [
    ("25833.4 ft2", 4.0),  # 25,833.4 / 10.7639 = 2,400.0 m2; 2,400 / (30 x 20) = 4.0
    ("12916.7 ft2", 2.0),  # 12,916.7 / 10.7639 = 1,200.0 m2; 1,200 / 600 = 2.0
])
def test_f3_u_crew_sizing_quantity_in_square_feet(q, crews):
    r = _ok("crew_sizing", {"quantity": q, "rate_per_crew_day": "30 m2/d", "duration_days": 20})
    assert r["crews_required"] == pytest.approx(crews, abs=0.001)


@pytest.mark.parametrize("q, rate, days_for_3, whole_days", [
    (2400, 30, 26.667, 27),  # 2,400 / (30 x 3) = 26.667 -> 27 working days
    (1000, 25, 13.333, 14),  # 1,000 / (25 x 3) = 13.333 -> 14 working days
])
def test_f3_u2_crew_sizing_duration_for_given_crews(q, rate, days_for_3, whole_days):
    r = _ok("crew_sizing", {"quantity": q, "rate_per_crew_day": rate, "duration_days": 20,
                            "crew_size": 3})
    assert r["duration_for_crews"] == pytest.approx(days_for_3, rel=_REL)
    assert math.ceil(r["duration_for_crews"]) == whole_days


@pytest.mark.parametrize("q", [2400, 1800])
def test_f3_m_crew_sizing_missing_rate_asks(q):
    _asks("crew_sizing", {"quantity": q, "duration_days": 20}, "rate_per_crew_day",
          ("crews_required", "crews_required_rounded_up"))


@pytest.mark.parametrize("q, crews, whole", [
    (2410, 4.0167, 5),  # 2,410 / 600 = 4.0167 -> 5 whole crews (never 4)
    (601, 1.0017, 2),   # 601 / 600 = 1.00167 -> 2 whole crews (never 1)
])
def test_f3_b_crew_sizing_rounds_up(q, crews, whole):
    r = _ok("crew_sizing", {"quantity": q, "rate_per_crew_day": 30, "duration_days": 20})
    assert r["crews_required"] == pytest.approx(crews, rel=_REL)
    assert r["crews_required_rounded_up"] == whole


@pytest.mark.parametrize("rate, days", [(30, 20), (15, 10)])
def test_f3_b2_crew_sizing_zero_quantity_is_refused(rate, days):
    # Advisor ruling (15 Nov): refusing quantity 0 is acceptable; no crews are
    # computed and the refusal names the input. No wrong number, no crash.
    env = run_calculation("crew_sizing", {"quantity": 0, "rate_per_crew_day": rate, "duration_days": days})
    assert env.get("status") == "error", env
    assert "quantity" in env.get("error", "")
    assert "result" not in env


# -------------------------------------------------------- F4 excavation volume
# bank = L x W x D; loose = bank x (1 + bulking).

@pytest.mark.parametrize("dims, bulk, bank, loose", [
    ((10, 6, 3), 0.25, 180.0, 225.0),  # 10 x 6 x 3 = 180; 180 x 1.25 = 225
    ((5, 4, 2), 0.30, 40.0, 52.0),     # 5 x 4 x 2 = 40; 40 x 1.30 = 52
])
def test_f4_t_excavation_typical(dims, bulk, bank, loose):
    length, width, depth = dims
    r = _ok("excavation_volume", {"length_m": length, "width_m": width, "depth_m": depth,
                                  "bulking_factor": bulk})
    assert r["bank_volume_m3"] == pytest.approx(bank, rel=_REL)
    assert r["loose_volume_m3"] == pytest.approx(loose, rel=_REL)


@pytest.mark.parametrize("dims, bulk, bank, loose", [
    (("30 ft", "20 ft", "10 ft"), 0.25, 169.901, 212.376),  # 6,000 ft3 x 0.0283168 = 169.901; x 1.25
    (("10 ft", "10 ft", "10 ft"), 0.20, 28.317, 33.980),    # 1,000 ft3 x 0.0283168 = 28.3168; x 1.20
])
def test_f4_u_excavation_in_feet(dims, bulk, bank, loose):
    length, width, depth = dims
    r = _ok("excavation_volume", {"length_m": length, "width_m": width, "depth_m": depth,
                                  "bulking_factor": bulk})
    assert r["bank_volume_m3"] == pytest.approx(bank, rel=_REL)
    assert r["loose_volume_m3"] == pytest.approx(loose, rel=_REL)


@pytest.mark.parametrize("length, width", [(10, 6), (5, 4)])
def test_f4_m_excavation_missing_depth_asks(length, width):
    _asks("excavation_volume", {"length_m": length, "width_m": width, "bulking_factor": 0.25},
          "depth", ("bank_volume_m3", "loose_volume_m3"))


@pytest.mark.parametrize("length, width", [(10, 6), (5, 4)])
def test_f4_b_excavation_zero_depth(length, width):
    # L x W x 0 = 0 m3 bank, and 0 x (1 + bulking) = 0 m3 loose
    r = _ok("excavation_volume", {"length_m": length, "width_m": width, "depth_m": 0,
                                  "bulking_factor": 0.25})
    assert r["bank_volume_m3"] == 0
    assert r["loose_volume_m3"] == 0


# ---------------------------------------------------------- F5 concrete volume
# rectangular: V = L x W x t; with waste V x (1 + w).

@pytest.mark.parametrize("dims, waste, net, gross", [
    ((12, 5, 0.25), 0.05, 15.0, 15.75),  # 12 x 5 x 0.25 = 15; 15 x 1.05 = 15.75
    ((8, 4, 0.2), 0.10, 6.4, 7.04),      # 8 x 4 x 0.2 = 6.4; 6.4 x 1.10 = 7.04
])
def test_f5_t_concrete_typical(dims, waste, net, gross):
    length, width, thick = dims
    r = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick,
                                "waste_factor": waste})
    assert r["net_volume_m3"] == pytest.approx(net, rel=_REL)
    assert r["volume_with_waste_m3"] == pytest.approx(gross, rel=_REL)


@pytest.mark.parametrize("dims, net", [
    (("10 ft", "20 ft", "6 in"), 2.8317),  # 10 x 20 x 0.5 ft = 100 ft3 x 0.0283168 = 2.8317
    (("12 ft", "10 ft", "4 in"), 1.1327),  # 12 x 10 x (4/12) ft = 40 ft3 x 0.0283168 = 1.1327
])
def test_f5_u_concrete_feet_and_inches(dims, net):
    length, width, thick = dims
    r = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick,
                                "waste_factor": 0})
    assert r["net_volume_m3"] == pytest.approx(net, rel=_REL)


@pytest.mark.xfail(strict=True, reason="suspected defect: missing thickness_m silently defaults to 0 and returns 0 m3 instead of asking")
@pytest.mark.parametrize("length, width", [(12, 5), (8, 4)])
def test_f5_m_concrete_missing_thickness_asks(length, width):
    _asks("concrete_volume", {"length_m": length, "width_m": width, "waste_factor": 0.05},
          "thickness", ("net_volume_m3", "volume_with_waste_m3", "volume_m3"))


@pytest.mark.parametrize("dims, net", [
    ((12, 5, 0.25), 15.0),  # waste 0: with-waste = net = 12 x 5 x 0.25 = 15
    ((8, 4, 0.2), 6.4),     # waste 0: with-waste = net = 8 x 4 x 0.2 = 6.4
])
def test_f5_b_concrete_zero_waste(dims, net):
    length, width, thick = dims
    r = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick,
                                "waste_factor": 0})
    assert r["net_volume_m3"] == pytest.approx(net, rel=_REL)
    assert r["volume_with_waste_m3"] == pytest.approx(net, rel=_REL)


@pytest.mark.parametrize("length, width", [(12, 5), (8, 4)])
def test_f5_b2_concrete_zero_thickness(length, width):
    # thickness stated as 0: L x W x 0 = 0, or a refusal; never a positive figure
    env = run_calculation("concrete_volume", {"length_m": length, "width_m": width,
                                              "thickness_m": 0, "waste_factor": 0.05})
    if env.get("status") == "success":
        assert env["result"]["net_volume_m3"] == 0
    else:
        assert (env.get("result") or {}).get("net_volume_m3") is None, env


# ------------------------------------------------------------ F6 unit convert
# exact factors: 1 in = 25.4 mm; 1 psi = 6.894757 kPa; 1 m3 = 35.3147 ft3.

@pytest.mark.parametrize("value, out", [
    (2.5, 63.5),   # 2.5 x 25.4 = 63.5 mm
    (12, 304.8),   # 12 x 25.4 = 304.8 mm
])
def test_f6_t_convert_inches_to_mm(value, out):
    r = _ok("pe_unit_convert", {"value": value, "from_unit": "in", "to_unit": "mm"})
    assert r["value_out"] == pytest.approx(out, rel=1e-6)


@pytest.mark.parametrize("value, src, dst, out", [
    (100, "psi", "kPa", 689.4757),  # 100 x 6.894757 = 689.4757 kPa
    (1, "m3", "ft3", 35.3147),      # 1 / 0.0283168 = 35.3147 ft3
])
def test_f6_u_convert_pressure_and_volume(value, src, dst, out):
    r = _ok("pe_unit_convert", {"value": value, "from_unit": src, "to_unit": dst})
    assert r["value_out"] == pytest.approx(out, rel=_REL)


@pytest.mark.parametrize("value, dst", [(50, "kPa"), (12, "mm")])
def test_f6_m_convert_missing_source_unit_asks(value, dst):
    _asks("pe_unit_convert", {"value": value, "to_unit": dst}, "from_unit", ("value_out",))


@pytest.mark.parametrize("src, dst", [("in", "mm"), ("psi", "kPa")])
def test_f6_b_convert_zero(src, dst):
    # 0 x any factor = 0
    r = _ok("pe_unit_convert", {"value": 0, "from_unit": src, "to_unit": dst})
    assert r["value_out"] == 0


@pytest.mark.parametrize("value, src, dst", [(5, "m", "kg"), (3, "kPa", "mm")])
def test_f6_b2_convert_dimension_mismatch_refused(value, src, dst):
    # no factor exists between different dimensions: refuse, no number
    _refuses("pe_unit_convert", {"value": value, "from_unit": src, "to_unit": dst},
             ("value_out",))


# -------------------------------------------------------------------- F7 EVM
# CV = EV - AC; SV = EV - PV; CPI = EV / AC; SPI = EV / PV; EAC = BAC / CPI.

@pytest.mark.parametrize("bac, ev, ac, pv, cv, sv, cpi, spi, eac", [
    # CV 400k-500k; SV 400k-450k; CPI 400/500; SPI 400/450 = 0.8889; EAC 1,000,000 / 0.8
    (1_000_000, 400_000, 500_000, 450_000, -100_000, -50_000, 0.800, 0.889, 1_250_000),
    # CV 600k-480k; SV 600k-750k; CPI 600/480 = 1.25; SPI 600/750 = 0.8; EAC 2,000,000 / 1.25
    (2_000_000, 600_000, 480_000, 750_000, 120_000, -150_000, 1.250, 0.800, 1_600_000),
])
def test_f7_t_evm_typical(bac, ev, ac, pv, cv, sv, cpi, spi, eac):
    r = _ok("calculate_evm", {"bac": bac, "ev": ev, "ac": ac, "pv": pv})
    assert r["CV"] == pytest.approx(cv, abs=0.01)
    assert r["SV"] == pytest.approx(sv, abs=0.01)
    assert r["CPI"] == pytest.approx(cpi, abs=0.001)
    assert r["SPI"] == pytest.approx(spi, abs=0.001)
    assert r["EAC"] == pytest.approx(eac, rel=_REL)


@pytest.mark.xfail(strict=True, reason='plan F7-U: money written as "AED 1.0 m" is refused and asked for, not read as 1,000,000 (no wrong number)')
@pytest.mark.parametrize("bac, ev, ac, pv, cpi, spi, eac", [
    ("AED 1.0 m", "AED 0.4 m", "AED 0.5 m", "AED 0.45 m", 0.800, 0.889, 1_250_000),   # as F7-T row 1
    ("AED 2.0 m", "AED 0.6 m", "AED 0.48 m", "AED 0.75 m", 1.250, 0.800, 1_600_000),  # as F7-T row 2
])
def test_f7_u_evm_figures_written_in_millions(bac, ev, ac, pv, cpi, spi, eac):
    r = _ok("calculate_evm", {"bac": bac, "ev": ev, "ac": ac, "pv": pv})
    assert r["CPI"] == pytest.approx(cpi, abs=0.001)
    assert r["SPI"] == pytest.approx(spi, abs=0.001)
    assert r["EAC"] == pytest.approx(eac, rel=_REL)


@pytest.mark.parametrize("bac, ev, pv", [(1_000_000, 400_000, 450_000),
                                         (2_000_000, 600_000, 750_000)])
def test_f7_m_evm_missing_actual_cost_asks(bac, ev, pv):
    _asks("calculate_evm", {"bac": bac, "ev": ev, "pv": pv}, "AC", ("CPI", "EAC"))


@pytest.mark.parametrize("ev, pv", [(400_000, 450_000), (600_000, 750_000)])
def test_f7_b_evm_zero_actual_cost(ev, pv):
    # AC = 0: CPI = EV / 0 is undefined -> no number, no infinity, no crash
    env = run_calculation("calculate_evm", {"bac": 1_000_000, "ev": ev, "ac": 0, "pv": pv})
    result = env.get("result") or {}
    for key in ("CPI", "EAC"):
        val = result.get(key)
        assert val is None or isinstance(val, str), (key, env)


# --------------------------------------------------------- F8 critical float
# TF = LS - ES = LF - EF; critical when TF = 0.

@pytest.mark.parametrize("es, ef, ls, lf, tf", [
    (5, 12, 8, 15, 3.0),  # 8 - 5 = 3 = 15 - 12
    (0, 10, 4, 14, 4.0),  # 4 - 0 = 4 = 14 - 10
])
def test_f8_t_float_typical(es, ef, ls, lf, tf):
    r = _ok("critical_path_float", {"early_start": es, "early_finish": ef,
                                    "late_start": ls, "late_finish": lf})
    assert r["total_float"] == pytest.approx(tf)
    assert r["is_critical"] is False


@pytest.mark.xfail(strict=True, reason="plan F8-U: dates given in weeks are refused as not a count, not passed through (no wrong number)")
@pytest.mark.parametrize("es, ef, ls, lf, tf", [
    ("5 weeks", "12 weeks", "8 weeks", "15 weeks", 3.0),  # 8 - 5 = 3 weeks
    ("0 weeks", "10 weeks", "4 weeks", "14 weeks", 4.0),  # 4 - 0 = 4 weeks
])
def test_f8_u_float_in_weeks_keeps_value(es, ef, ls, lf, tf):
    r = _ok("critical_path_float", {"early_start": es, "early_finish": ef,
                                    "late_start": ls, "late_finish": lf})
    assert r["total_float"] == pytest.approx(tf)


@pytest.mark.parametrize("es, ef, ls, tf", [
    (5, 12, 8, 3.0),  # LS - ES = 8 - 5 = 3
    (0, 10, 4, 4.0),  # LS - ES = 4 - 0 = 4
])
def test_f8_m_float_missing_late_finish(es, ef, ls, tf):
    # Plan allows either: ask for LF, or answer LS - ES and say which formula.
    env = run_calculation("critical_path_float", {"early_start": es, "early_finish": ef,
                                                  "late_start": ls})
    if env.get("status") == "success":
        assert env["result"]["total_float"] == pytest.approx(tf)
        assert "LS - ES" in str(env["result"].get("note", ""))
    else:
        text = " ".join(str(env.get(k) or "") for k in ("error", "question")).lower()
        assert "late_finish" in text or "late finish" in text, env


@pytest.mark.parametrize("es, ef", [(5, 12), (0, 10)])
def test_f8_b_float_zero_is_critical(es, ef):
    # LS = ES and LF = EF -> TF = 0, critical
    r = _ok("critical_path_float", {"early_start": es, "early_finish": ef,
                                    "late_start": es, "late_finish": ef})
    assert r["total_float"] == 0
    assert r["is_critical"] is True


@pytest.mark.xfail(strict=True, reason="suspected defect: a set with LF-EF != LS-ES is reported, not flagged as inconsistent (plan F8-B)")
@pytest.mark.parametrize("es, ef, ls, lf", [
    (5, 12, 8, 14),  # LS - ES = 3 but LF - EF = 2
    (0, 10, 4, 13),  # LS - ES = 4 but LF - EF = 3
])
def test_f8_b2_float_inconsistent_set_is_flagged(es, ef, ls, lf):
    env = run_calculation("critical_path_float", {"early_start": es, "early_finish": ef,
                                                  "late_start": ls, "late_finish": lf})
    assert env.get("status") != "success" or re.search(r"inconsisten", str(env).lower()), env


# ---------------------------------------------- F9 modulus of elasticity (ACI)
# ACI 318-19 Eq. 19.2.2.1b (SI): Ec = 4700 sqrt(f'c) MPa.

@pytest.mark.parametrize("fck, ec", [
    (30, 25_743),  # sqrt(30) = 5.47723; x 4700 = 25,742.96 MPa
    (40, 29_725),  # sqrt(40) = 6.32456; x 4700 = 29,725.4 MPa
])
def test_f9_t_modulus_aci_typical(fck, ec):
    r = _ok("modulus_of_elasticity_concrete", {"fck_n_mm2": fck, "code": "aci"})
    assert r["unit"] == "MPa"
    assert r["value"] == pytest.approx(ec, rel=_REL)


@pytest.mark.parametrize("fck, ec", [
    # 4,350 psi x 0.006894757 = 29.99 MPa -> 4700 sqrt = 25,739; imperial
    # 57,000 sqrt(4,350) = 3.759e6 psi = 25,920 MPa; both within 1 % of 25,900
    ("4350 psi", 25_900),
    # 5,000 psi = 34.474 MPa -> 4700 sqrt = 27,596; imperial 57,000 sqrt(5,000)
    # = 4.0305e6 psi = 27,790 MPa; both within 1 % of 27,700
    ("5000 psi", 27_700),
])
def test_f9_u_modulus_strength_in_psi(fck, ec):
    r = _ok("modulus_of_elasticity_concrete", {"fck_n_mm2": fck, "code": "aci"})
    assert r["value"] == pytest.approx(ec, rel=0.01)


@pytest.mark.parametrize("code", ["aci", "ACI 318"])
def test_f9_m_modulus_missing_strength_asks(code):
    _asks("modulus_of_elasticity_concrete", {"code": code}, "fck", ("value", "value_mpa"))


@pytest.mark.parametrize("fck", [0, -30])
def test_f9_b_modulus_non_positive_strength_refused(fck):
    # sqrt of a zero or negative strength is not a concrete: refuse, no number
    _refuses("modulus_of_elasticity_concrete", {"fck_n_mm2": fck, "code": "aci"}, ("value",))
