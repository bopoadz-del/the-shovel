"""EVERY registered construction formula must execute -- and compute RIGHT.

Live-fire campaign, formula sweep 2026-08-15. The registry exposes 168
executable formulas; until F19c they were unreachable through the
orchestrator dispatch path, so "the tests pass" said nothing about the
library actually running. This fence does two jobs:

1. **Sweep**: execute every formula in ``executable_formula_ids()`` with
   signature-driven plausible inputs. A formula that cannot run on sane
   inputs (or a new formula registered without a working binding) fails
   here by name, instead of staying dormant until a live user hits it.
   First live run: 168/168 executed, 0 errors.

2. **Hand-checked arithmetic**: a battery across modules with expected
   values computed INDEPENDENTLY (statics, geometry, BS 8666 bar mass,
   EVM, CPM, soil bulking, sieve analysis, Nurse-Saul maturity). A formula
   that runs but computes wrong fails loudly. This battery caught a real
   one: EAC was derived from the ROUNDED 3dp CPI, drifting 82 per 211k
   (200000/0.947 vs 200000/(90000/95000)) -- display rounding was leaking
   into arithmetic.
"""
from __future__ import annotations

import inspect

import pytest

from app.agents import formulas as F

# name-fragment -> representative value (order matters, first hit wins)
_NUM_GUESS = [
    ("percent", 10.0), ("ratio", 0.5), ("factor", 1.2), ("efficiency", 0.85),
    ("slope", 0.02),
    ("length", 12.0), ("width", 8.0), ("breadth", 8.0), ("depth", 0.5),
    ("height", 3.0), ("thickness", 0.25), ("span", 6.0), ("dia", 0.02),
    ("radius", 0.5), ("spacing", 0.2), ("cover", 0.04),
    ("area", 50.0), ("volume", 100.0), ("perimeter", 40.0),
    ("load", 25.0), ("moment", 50.0), ("force", 100.0), ("shear", 80.0),
    ("udl", 10.0), ("pressure", 150.0), ("stress", 200.0),
    ("fck", 30.0), ("fcu", 30.0), ("fy", 460.0), ("fu", 400.0),
    ("strength", 30.0), ("modulus", 200000.0), ("density", 2400.0),
    ("unit_weight", 24.0), ("weight", 500.0), ("mass", 1000.0),
    ("qty", 100.0), ("quantity", 100.0), ("count", 10), ("number", 10),
    ("num", 10),
    ("rate", 50.0), ("cost", 10000.0), ("price", 100.0), ("budget", 100000.0),
    ("value", 50000.0), ("amount", 50000.0),
    ("bcws", 100000.0), ("bcwp", 90000.0), ("acwp", 95000.0),
    ("pv", 100000.0), ("ev", 90000.0), ("ac", 95000.0),
    ("duration", 30.0), ("days", 30), ("hours", 8.0), ("time", 10.0),
    ("temp", 25.0), ("slump", 100.0), ("wc", 0.45),
    ("bearing", 150.0), ("capacity", 200.0), ("phi", 30.0), ("angle", 30.0),
    ("cohesion", 25.0), ("safety", 2.0), ("fos", 2.0),
]
_STR_GUESS = [
    ("type", "standard"), ("grade", "C30"), ("class", "A"), ("unit", "m3"),
    ("category", "general"), ("status", "open"), ("discipline", "CIV"),
    ("code", "GEN"), ("name", "item"), ("method", "default"),
]
# formulas whose params are series data -- name-fragment -> list value
_LIST_PARAMS = {
    "temperature_history_c": [20.0, 22.0, 25.0, 24.0],
    "time_intervals_hours": [12.0, 12.0, 12.0, 12.0],
    "sieve_retained_percentages": [5.0, 15.0, 30.0, 55.0, 75.0, 90.0],
}


def _guess(pname, ann):
    if pname in _LIST_PARAMS:
        return _LIST_PARAMS[pname]
    n = pname.lower()
    ann_s = str(ann)
    if ann is str or ann_s == "str":
        for frag, v in _STR_GUESS:
            if frag in n:
                return v
        return "general"
    for frag, v in _NUM_GUESS:
        if frag in n:
            return v
    if ann is int or ann_s == "int":
        return 10
    if ann is bool or ann_s == "bool":
        return False
    if "ict" in ann_s:
        return {}
    if "ist" in ann_s or "equence" in ann_s:
        return []
    return 10.0


def _params_for(fn):
    sig = inspect.signature(fn)
    return {
        pname: _guess(pname, p.annotation)
        for pname, p in sig.parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.default is p.empty
    }


def test_every_registered_formula_executes_on_plausible_inputs():
    ids = sorted(F.executable_formula_ids())
    assert len(ids) >= 168, f"registry shrank: {len(ids)}"
    failures = []
    for fid in ids:
        try:
            fn = F.resolve_binding_or_raise(fid)
            F.run_formula(fid, _params_for(fn))
        except Exception as e:  # noqa: BLE001 -- collecting, not hiding
            failures.append(f"{fid}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


# -- hand-checked arithmetic (expected values computed independently) -----

def run(fid, **params):
    return F.run_formula(fid, params)


def test_concrete_volume_slab():
    r = run("concrete_volume", length_m=12, width_m=8, thickness_m=0.25)
    assert r["net_volume_m3"] == pytest.approx(24.0)
    assert r["volume_with_waste_m3"] == pytest.approx(25.2)


def test_beam_statics():
    assert run("beam_moment_simple", udl_w_kn_m=10, span_m=6)[
        "max_moment_kn_m"] == pytest.approx(45.0)          # wL^2/8
    assert run("beam_shear_simple", udl_w_kn_m=10, span_m=6)[
        "max_shear_kn"] == pytest.approx(30.0)             # wL/2


def test_rebar_mass_bs8666():
    r = run("rebar_weight", bar_diameter_mm=16, total_length_m=100)
    assert r["unit_mass_kg_m"] == pytest.approx(1.578, abs=0.001)  # pi/4 d^2 rho
    assert r["total_mass_kg"] == pytest.approx(157.83, abs=0.01)


def test_excavation_bank_and_bulked():
    r = run("excavation_volume", length_m=10, width_m=5, depth_m=2)
    assert r["bank_volume_m3"] == pytest.approx(100.0)
    assert r["loose_volume_m3"] == pytest.approx(125.0)


def test_evm_indices_and_eac_from_unrounded_cpi():
    """RED before the fix: EAC = 200000/0.947 (rounded CPI) = 211,193.24;
    the correct EAC = 200000/(90000/95000) = 211,111.11."""
    r = run("calculate_evm", bac=200000, bcwp=90000, bcws=100000, acwp=95000)
    assert r["CPI"] == pytest.approx(0.947)
    assert r["SPI"] == pytest.approx(0.9)
    assert r["CV"] == pytest.approx(-5000)
    assert r["SV"] == pytest.approx(-10000)
    assert r["EAC"] == pytest.approx(211111.11, abs=0.02)
    assert r["ETC"] == pytest.approx(211111.11 - 95000, abs=0.02)
    assert r["VAC"] == pytest.approx(200000 - 211111.11, abs=0.02)
    assert r["status"]["cost"] == "OVER BUDGET"
    assert r["status"]["schedule"] == "BEHIND"


def test_cpm_total_float():
    r = run("critical_path_float", early_start=0, early_finish=10,
            late_start=5, late_finish=15)
    assert r["total_float"] == pytest.approx(5.0)
    assert r["is_critical"] is False
    r = run("critical_path_float", early_start=0, early_finish=10,
            late_start=0, late_finish=10)
    assert r["total_float"] == pytest.approx(0.0)
    assert r["is_critical"] is True


def test_materials_lab_formulas():
    # FM = sum(cumulative % retained on standard sieves) / 100
    assert run("fineness_modulus",
               sieve_retained_percentages=[5, 15, 30, 55, 75, 90]) == pytest.approx(2.7)
    # Nurse-Saul: sum((T - (-10)) * dt) = (30+32+35+34)*12 = 1572
    r = run("concrete_maturity_strength",
            temperature_history_c=[20, 22, 25, 24],
            time_intervals_hours=[12, 12, 12, 12])
    assert r["maturity_index_c_hrs"] == pytest.approx(1572.0)


def test_simple_cost_and_productivity():
    assert run("cost_per_area", total_cost=500000, area=250)[
        "cost_per_area"] == pytest.approx(2000.0)
    r = run("productivity_rate", output_quantity=100, labor_hours=50, crew_size=2)
    assert r["rate_per_hour"] == pytest.approx(2.0)
    assert r["rate_per_worker_hour"] == pytest.approx(2.0)


def test_wind_pressure_uses_the_asce_si_constant():
    # The constant is the point of this test. The factors are NOT defaulted any
    # more: Kd used to default to 0.85, so an operator asking for the basic
    # velocity pressure got a 15%-low figure with a site assumption they never
    # made printed in the working (live SET4 T14, 23 Sep 2026). Supplying Kd
    # explicitly still works and is asserted below.
    r = run("wind_pressure", wind_speed_m_s=40)
    assert r["velocity_pressure_pa"] == pytest.approx(0.613 * 40**2, abs=0.1)
    r_kd = run("wind_pressure", wind_speed_m_s=40, kd=0.85)
    assert r_kd["velocity_pressure_pa"] == pytest.approx(0.613 * 0.85 * 40**2, abs=0.1)
