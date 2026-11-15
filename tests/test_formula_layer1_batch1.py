"""Formula Layer 1, batch 1: nine formulas, four case kinds each.

Case kinds (formula test plan, Layer 1):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it (status
     ``needs_input`` or ``error`` naming the input) and must NOT return the
     figure it would have needed that input for;
  B  boundary -- zero, a value on the limit, rounding direction.

Every expected value is a HAND calculation from the published definition,
written in the comment above the assertion. None was copied from the
product's output. A case where the product disagrees with the hand value is
marked ``xfail(strict=True)`` with a one-line reason and is a suspected
formula defect for the advisor; the expected value is never moved to match.

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

def test_f1_t_beam_moment_typical():
    # 20 x 6^2 / 8 = 20 x 36 / 8 = 90 kN.m
    r = _ok("beam_moment_simple", {"udl_w_kn_m": 20, "span_m": 6})
    assert r["max_moment_kn_m"] == pytest.approx(90.0, rel=_REL)
    # second input: a different span moves the answer by L^2 (w=10, L=4 -> 20)
    r = _ok("beam_moment_simple", {"udl_w_kn_m": 10, "span_m": 4})
    assert r["max_moment_kn_m"] == pytest.approx(20.0, rel=_REL)


def test_f1_u_beam_moment_span_in_feet():
    # L = 20 ft x 0.3048 = 6.096 m; 6.096^2 = 37.1612; 20 x 37.1612 / 8 = 92.903 kN.m
    r = _ok("beam_moment_simple", {"udl_w_kn_m": 20, "span_m": "20 ft"})
    assert r["max_moment_kn_m"] == pytest.approx(92.903, rel=_REL)


def test_f1_m_beam_moment_missing_load_asks():
    _asks("beam_moment_simple", {"span_m": 6}, "udl_w_kn_m", ("max_moment_kn_m",))


def test_f1_b_beam_moment_zero_span_no_nan():
    # L = 0: w L^2 / 8 = 0, or a clear refusal; never NaN and never a crash.
    env = run_calculation("beam_moment_simple", {"udl_w_kn_m": 20, "span_m": 0})
    if env.get("status") == "success":
        assert env["result"]["max_moment_kn_m"] == 0
    else:
        assert (env.get("result") or {}).get("max_moment_kn_m") is None, env
    # zero load on a real span: 0 x 36 / 8 = 0 exactly
    r = _ok("beam_moment_simple", {"udl_w_kn_m": 0, "span_m": 6})
    assert r["max_moment_kn_m"] == 0


# ------------------------------------------------------- F2 interim payment
# retention = gross x rate; net = gross - retention.

def test_f2_t_interim_payment_typical():
    # 1,250,000 x 0.05 = 62,500; 1,250,000 - 62,500 = 1,187,500
    r = _ok("calculate_interim_payment", {"gross_valuation": 1_250_000, "retention_percent": 5})
    assert r["retention_amount"] == pytest.approx(62_500, abs=0.01)
    assert r["net_payment"] == pytest.approx(1_187_500, abs=0.01)
    # 800,000 x 0.03 = 24,000; net 776,000
    r = _ok("calculate_interim_payment", {"gross_valuation": 800_000, "retention_percent": 3})
    assert r["net_payment"] == pytest.approx(776_000, abs=0.01)


def test_f2_u_interim_payment_written_figures():
    # "1.25 million" = 1,250,000; "5 percent" = 5 %: same 62,500 / 1,187,500
    r = _ok("calculate_interim_payment",
            {"gross_valuation": "1.25 million", "retention_percent": "5 percent"})
    assert r["retention_amount"] == pytest.approx(62_500, abs=0.01)
    assert r["net_payment"] == pytest.approx(1_187_500, abs=0.01)


@pytest.mark.xfail(strict=True, reason="suspected defect: retention_percent silently defaults to 10 (plan F2-M, ask-not-guess)")
def test_f2_m_interim_payment_missing_retention_asks():
    # A silent 10 % would give 1,250,000 - 125,000 = 1,125,000: must ask instead.
    _asks("calculate_interim_payment", {"gross_valuation": 1_250_000},
          "retention", ("net_payment", "retention_amount"))


def test_f2_b_interim_payment_retention_limits():
    # 0 %: net = gross = 1,250,000
    r = _ok("calculate_interim_payment", {"gross_valuation": 1_250_000, "retention_percent": 0})
    assert r["net_payment"] == pytest.approx(1_250_000, abs=0.01)
    # 100 %: net = 0
    r = _ok("calculate_interim_payment", {"gross_valuation": 1_250_000, "retention_percent": 100})
    assert r["net_payment"] == pytest.approx(0, abs=0.01)
    # 150 %: not a retention rate -> refuse, no net figure
    _refuses("calculate_interim_payment", {"gross_valuation": 1_250_000, "retention_percent": 150},
             ("net_payment",))


# ------------------------------------------------------------ F3 crew sizing
# crews = Q / (R x D); round UP to whole crews.

def test_f3_t_crew_sizing_typical():
    # 2,400 / (30 x 20) = 2,400 / 600 = 4.0 -> 4
    r = _ok("crew_sizing", {"quantity": 2400, "rate_per_crew_day": 30, "duration_days": 20})
    assert r["crews_required"] == pytest.approx(4.0, rel=_REL)
    assert r["crews_required_rounded_up"] == 4


@pytest.mark.xfail(strict=True, reason="suspected defect: quantity in ft2 is not converted to the rate's m2 (returns 43.056 = 25,833.4 / 600, hand value 4.0)")
def test_f3_u_crew_sizing_quantity_in_square_feet():
    # 2,400 m2 x 10.7639 = 25,833.4 ft2; back to m2 = 2,400; 2,400 / 600 = 4.0 (+-0.001)
    r = _ok("crew_sizing", {"quantity": "25833.4 ft2", "rate_per_crew_day": "30 m2/d",
                            "duration_days": 20})
    assert r["crews_required"] == pytest.approx(4.0, abs=0.001)


def test_f3_u2_crew_sizing_duration_for_given_crews():
    # 3 crews: 2,400 / (30 x 3) = 26.667 days (27 working days rounded up)
    r = _ok("crew_sizing", {"quantity": 2400, "rate_per_crew_day": 30, "duration_days": 20,
                            "crew_size": 3})
    assert r["duration_for_crews"] == pytest.approx(26.667, rel=_REL)
    assert math.ceil(r["duration_for_crews"]) == 27


def test_f3_m_crew_sizing_missing_rate_asks():
    _asks("crew_sizing", {"quantity": 2400, "duration_days": 20}, "rate_per_crew_day",
          ("crews_required", "crews_required_rounded_up"))


def test_f3_b_crew_sizing_rounds_up():
    # 2,410 / 600 = 4.0167 -> 5 whole crews (never 4)
    r = _ok("crew_sizing", {"quantity": 2410, "rate_per_crew_day": 30, "duration_days": 20})
    assert r["crews_required"] == pytest.approx(4.0167, rel=_REL)
    assert r["crews_required_rounded_up"] == 5


@pytest.mark.xfail(strict=True, reason="plan F3-B2 wants 0 crews for quantity 0; product refuses quantity <= 0")
def test_f3_b2_crew_sizing_zero_quantity():
    # 0 / 600 = 0 crews, no crash
    r = _ok("crew_sizing", {"quantity": 0, "rate_per_crew_day": 30, "duration_days": 20})
    assert r["crews_required"] == 0
    assert r["crews_required_rounded_up"] == 0


# -------------------------------------------------------- F4 excavation volume
# bank = L x W x D; loose = bank x (1 + bulking).

def test_f4_t_excavation_typical():
    # 10 x 6 x 3 = 180 m3 bank; 180 x 1.25 = 225 m3 loose
    r = _ok("excavation_volume", {"length_m": 10, "width_m": 6, "depth_m": 3,
                                  "bulking_factor": 0.25})
    assert r["bank_volume_m3"] == pytest.approx(180.0, rel=_REL)
    assert r["loose_volume_m3"] == pytest.approx(225.0, rel=_REL)


def test_f4_u_excavation_in_feet():
    # 30 x 20 x 10 ft = 6,000 ft3 x 0.0283168 = 169.901 m3; x 1.25 = 212.376 m3
    r = _ok("excavation_volume", {"length_m": "30 ft", "width_m": "20 ft", "depth_m": "10 ft",
                                  "bulking_factor": 0.25})
    assert r["bank_volume_m3"] == pytest.approx(169.901, rel=_REL)
    assert r["loose_volume_m3"] == pytest.approx(212.376, rel=_REL)


def test_f4_m_excavation_missing_depth_asks():
    _asks("excavation_volume", {"length_m": 10, "width_m": 6, "bulking_factor": 0.25},
          "depth", ("bank_volume_m3", "loose_volume_m3"))


def test_f4_b_excavation_zero_depth():
    # 10 x 6 x 0 = 0 m3 bank and 0 m3 loose
    r = _ok("excavation_volume", {"length_m": 10, "width_m": 6, "depth_m": 0,
                                  "bulking_factor": 0.25})
    assert r["bank_volume_m3"] == 0
    assert r["loose_volume_m3"] == 0


# ---------------------------------------------------------- F5 concrete volume
# rectangular: V = L x W x t; with waste V x (1 + w).

def test_f5_t_concrete_typical():
    # 12 x 5 x 0.25 = 15.0 m3 net; 15 x 1.05 = 15.75 m3 with waste
    r = _ok("concrete_volume", {"length_m": 12, "width_m": 5, "thickness_m": 0.25,
                                "waste_factor": 0.05})
    assert r["net_volume_m3"] == pytest.approx(15.0, rel=_REL)
    assert r["volume_with_waste_m3"] == pytest.approx(15.75, rel=_REL)


def test_f5_u_concrete_feet_and_inches():
    # 10 ft x 20 ft x 6 in = 10 x 20 x 0.5 ft = 100 ft3 x 0.0283168 = 2.8317 m3
    r = _ok("concrete_volume", {"length_m": "10 ft", "width_m": "20 ft", "thickness_m": "6 in",
                                "waste_factor": 0})
    assert r["net_volume_m3"] == pytest.approx(2.8317, rel=_REL)


@pytest.mark.xfail(strict=True, reason='suspected defect: missing thickness_m silently defaults to 0 and returns 0 m3 instead of asking')
def test_f5_m_concrete_missing_thickness_asks():
    _asks("concrete_volume", {"length_m": 12, "width_m": 5, "waste_factor": 0.05},
          "thickness", ("net_volume_m3", "volume_with_waste_m3", "volume_m3"))


def test_f5_b_concrete_zero_waste_and_zero_thickness():
    # waste 0: with-waste = net = 12 x 5 x 0.25 = 15.0
    r = _ok("concrete_volume", {"length_m": 12, "width_m": 5, "thickness_m": 0.25,
                                "waste_factor": 0})
    assert r["volume_with_waste_m3"] == pytest.approx(r["net_volume_m3"], abs=1e-9)
    assert r["net_volume_m3"] == pytest.approx(15.0, rel=_REL)
    # thickness stated as 0: 12 x 5 x 0 = 0, or a refusal; never a positive figure
    env = run_calculation("concrete_volume", {"length_m": 12, "width_m": 5, "thickness_m": 0,
                                              "waste_factor": 0.05})
    if env.get("status") == "success":
        assert env["result"]["net_volume_m3"] == 0
    else:
        assert (env.get("result") or {}).get("net_volume_m3") is None, env


# ------------------------------------------------------------ F6 unit convert
# exact factors: 1 in = 25.4 mm; 1 psi = 6.894757 kPa; 1 m3 = 35.3147 ft3.

def test_f6_t_convert_inches_to_mm():
    # 2.5 x 25.4 = 63.5 mm
    r = _ok("pe_unit_convert", {"value": 2.5, "from_unit": "in", "to_unit": "mm"})
    assert r["value_out"] == pytest.approx(63.5, rel=1e-6)


def test_f6_u_convert_pressure_and_volume():
    # 100 x 6.894757 = 689.4757 kPa
    r = _ok("pe_unit_convert", {"value": 100, "from_unit": "psi", "to_unit": "kPa"})
    assert r["value_out"] == pytest.approx(689.4757, rel=_REL)
    # 1 m3 = 1 / 0.0283168 = 35.3147 ft3
    r = _ok("pe_unit_convert", {"value": 1, "from_unit": "m3", "to_unit": "ft3"})
    assert r["value_out"] == pytest.approx(35.3147, rel=_REL)


def test_f6_m_convert_missing_source_unit_asks():
    _asks("pe_unit_convert", {"value": 50, "to_unit": "kPa"}, "from_unit", ("value_out",))


def test_f6_b_convert_zero_and_dimension_mismatch():
    # 0 in x 25.4 = 0 mm
    r = _ok("pe_unit_convert", {"value": 0, "from_unit": "in", "to_unit": "mm"})
    assert r["value_out"] == 0
    # length to mass: no factor exists -> refuse, no number
    _refuses("pe_unit_convert", {"value": 5, "from_unit": "m", "to_unit": "kg"}, ("value_out",))


# -------------------------------------------------------------------- F7 EVM
# CV = EV - AC; SV = EV - PV; CPI = EV / AC; SPI = EV / PV; EAC = BAC / CPI.

def test_f7_t_evm_typical():
    # CV 400,000 - 500,000 = -100,000; SV 400,000 - 450,000 = -50,000
    # CPI 400/500 = 0.800; SPI 400/450 = 0.8889; EAC 1,000,000 / 0.8 = 1,250,000
    r = _ok("calculate_evm", {"bac": 1_000_000, "ev": 400_000, "ac": 500_000, "pv": 450_000})
    assert r["CV"] == pytest.approx(-100_000, abs=0.01)
    assert r["SV"] == pytest.approx(-50_000, abs=0.01)
    assert r["CPI"] == pytest.approx(0.800, abs=0.001)
    assert r["SPI"] == pytest.approx(0.889, abs=0.001)
    assert r["EAC"] == pytest.approx(1_250_000, rel=_REL)


@pytest.mark.xfail(strict=True, reason='plan F7-U: "AED 1.0 m" money figures are refused and asked for, not read as 1,000,000 (no wrong number)')
def test_f7_u_evm_figures_written_in_millions():
    # "AED 1.0 m" = 1,000,000 etc.: same indices as F7-T
    r = _ok("calculate_evm", {"bac": "AED 1.0 m", "ev": "AED 0.4 m", "ac": "AED 0.5 m",
                              "pv": "AED 0.45 m"})
    assert r["CPI"] == pytest.approx(0.800, abs=0.001)
    assert r["SPI"] == pytest.approx(0.889, abs=0.001)
    assert r["EAC"] == pytest.approx(1_250_000, rel=_REL)


def test_f7_m_evm_missing_actual_cost_asks():
    _asks("calculate_evm", {"bac": 1_000_000, "ev": 400_000, "pv": 450_000}, "AC",
          ("CPI", "EAC"))


def test_f7_b_evm_zero_actual_cost():
    # AC = 0: CPI = 400,000 / 0 is undefined -> not a number, not infinity, no crash
    env = run_calculation("calculate_evm", {"bac": 1_000_000, "ev": 400_000, "ac": 0,
                                            "pv": 450_000})
    result = env.get("result") or {}
    cpi = result.get("CPI")
    assert cpi is None or (isinstance(cpi, str)), env
    eac = result.get("EAC")
    assert eac is None or (isinstance(eac, str)), env


# --------------------------------------------------------- F8 critical float
# TF = LS - ES = LF - EF; critical when TF = 0.

def test_f8_t_float_typical():
    # 8 - 5 = 3 = 15 - 12 -> float 3, not critical
    r = _ok("critical_path_float", {"early_start": 5, "early_finish": 12,
                                    "late_start": 8, "late_finish": 15})
    assert r["total_float"] == pytest.approx(3.0)
    assert r["is_critical"] is False


@pytest.mark.xfail(strict=True, reason='plan F8-U: dates given in weeks are refused as not a count, not passed through (no wrong number)')
def test_f8_u_float_in_weeks_keeps_value():
    # weeks in, weeks out: 8 - 5 = 3 weeks
    r = _ok("critical_path_float", {"early_start": "5 weeks", "early_finish": "12 weeks",
                                    "late_start": "8 weeks", "late_finish": "15 weeks"})
    assert r["total_float"] == pytest.approx(3.0)


def test_f8_m_float_missing_late_finish():
    # Plan allows either: ask for LF, or answer LS - ES = 8 - 5 = 3 and say so.
    env = run_calculation("critical_path_float", {"early_start": 5, "early_finish": 12,
                                                  "late_start": 8})
    if env.get("status") == "success":
        assert env["result"]["total_float"] == pytest.approx(3.0)
        assert "LS - ES" in str(env["result"].get("note", ""))
    else:
        text = " ".join(str(env.get(k) or "") for k in ("error", "question"))
        assert "late_finish" in text.lower(), env


def test_f8_b_float_zero_is_critical():
    # LS = ES = 5, LF = EF = 12 -> TF 0, critical
    r = _ok("critical_path_float", {"early_start": 5, "early_finish": 12,
                                    "late_start": 5, "late_finish": 12})
    assert r["total_float"] == 0
    assert r["is_critical"] is True


@pytest.mark.xfail(strict=True, reason="suspected defect: LF-EF != LS-ES is reported, not flagged as inconsistent (plan F8-B)")
def test_f8_b2_float_inconsistent_set_is_flagged():
    # LS - ES = 8 - 5 = 3 but LF - EF = 14 - 12 = 2: an inconsistent set must be flagged.
    env = run_calculation("critical_path_float", {"early_start": 5, "early_finish": 12,
                                                  "late_start": 8, "late_finish": 14})
    blob = str(env).lower()
    assert env.get("status") != "success" or re.search(r"inconsisten", blob), env


# ---------------------------------------------- F9 modulus of elasticity (ACI)
# ACI 318-19 Eq. 19.2.2.1b (SI): Ec = 4700 sqrt(f'c) MPa.

def test_f9_t_modulus_aci_typical():
    # sqrt(30) = 5.47723; x 4700 = 25,742.96 MPa
    r = _ok("modulus_of_elasticity_concrete", {"fck_n_mm2": 30, "code": "aci"})
    assert r["unit"] == "MPa"
    assert r["value"] == pytest.approx(25_743, rel=_REL)
    # sqrt(40) = 6.32456; x 4700 = 29,725.4 MPa
    r = _ok("modulus_of_elasticity_concrete", {"fck_n_mm2": 40, "code": "aci"})
    assert r["value"] == pytest.approx(29_725, rel=_REL)


def test_f9_u_modulus_strength_in_psi():
    # 4,350 psi x 0.006894757 = 29.99 MPa -> 4700 sqrt(29.99) = 25,739 MPa;
    # imperial 57,000 sqrt(4,350) psi = 3.759e6 psi = 25,920 MPa; both within 1 % of 25,900
    r = _ok("modulus_of_elasticity_concrete", {"fck_n_mm2": "4350 psi", "code": "aci"})
    assert r["value"] == pytest.approx(25_900, rel=0.01)


def test_f9_m_modulus_missing_strength_asks():
    _asks("modulus_of_elasticity_concrete", {"code": "aci"}, "fck", ("value", "value_mpa"))


def test_f9_b_modulus_non_positive_strength_refused():
    # sqrt of a zero or negative strength is not a concrete: refuse, no number
    _refuses("modulus_of_elasticity_concrete", {"fck_n_mm2": 0, "code": "aci"}, ("value",))
    _refuses("modulus_of_elasticity_concrete", {"fck_n_mm2": -30, "code": "aci"}, ("value",))
