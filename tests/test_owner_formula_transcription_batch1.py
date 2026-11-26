"""Transcription check, batch 1: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what the owner's stated form says.
The stated form is the owner's knowledge-base entry where one exists
(``app/knowledge/construction_kb.json``), otherwise the formula's own
docstring. Every expected value below is a hand calculation from that stated
form (or the owner's own worked example), written as a literal; none is
obtained by running the calculator. Inputs other than the owner's worked
examples are invented figures.
"""
from __future__ import annotations

import pytest

from app.agents.base.formulas.construction_formulas_earthwork import (
    backfill_volume,
    excavation_volume,
)
from app.agents.base.formulas.construction_formulas_quantities import concrete_volume
from app.agents.base.formulas.construction_formulas_commercial import (
    cost_per_area,
    unit_cost_total,
)
from app.agents.hats.qaqc.formulas.construction_formulas_earthwork import compaction_control
from app.agents.hats.quantities.formulas.construction_formulas_earthwork import cut_fill_balance
from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    rebar_by_area,
    rebar_weight,
)
from app.agents.hats.design.formulas.construction_formulas_reference_tables import (
    carbon_footprint_concrete,
)
from app.agents.hats.planning.formulas.project_control import crew_sizing

TOL = 0.011


def _close(got, want):
    assert got is not None
    assert abs(float(got) - float(want)) <= TOL, (got, want)


# bank = L*W*D ; loose = bank*(1+b)
@pytest.mark.parametrize("kw,bank,loose", [
    (dict(length_m=18, width_m=9, depth_m=2.5, bulking_factor=0.25), 405.0, 506.25),
    (dict(length_m=12, width_m=3, depth_m=1.5, bulking_factor=0.2), 54.0, 64.8),
])
def test_excavation_volume_is_bank_times_one_plus_bulking(kw, bank, loose):
    r = excavation_volume(**kw)
    _close(r["bank_volume_m3"], bank)
    _close(r["loose_volume_m3"], loose)


# void = excavation - structure ; loose to import = void*(1+swell); floored at 0
@pytest.mark.parametrize("kw,void,loose", [
    (dict(excavation_bank_m3=1200, structure_volume_m3=450, swell_factor=0.25), 750.0, 937.5),
    (dict(excavation_bank_m3=800, structure_volume_m3=300, swell_factor=0.2), 500.0, 600.0),
    (dict(excavation_bank_m3=300, structure_volume_m3=800, swell_factor=0.2), 0.0, 0.0),
])
def test_backfill_volume_is_void_times_one_plus_swell(kw, void, loose):
    r = backfill_volume(**kw)
    _close(r["void_volume_m3"], void)
    _close(r["loose_backfill_needed_m3"], loose)


# rectangular L*W*T*(1+waste) ; cylinder pi*(D/2)^2*H*(1+waste)
@pytest.mark.parametrize("kw,vol", [
    (dict(length_m=22, width_m=14, thickness_m=0.3, waste_factor=0.05), 97.02),
    (dict(shape="cylinder", diameter_m=0.6, height_m=10, waste_factor=0.05), 2.969),
    (dict(length_m=10, width_m=5, thickness_m=0.2, waste_factor=0.0), 10.0),
])
def test_concrete_volume_is_geometry_times_one_plus_waste(kw, vol):
    _close(concrete_volume(**kw)["volume_m3"], vol)


# owner's KB form: %compaction = field dry density / lab MDD * 100.
# The first two rows are the owner's own worked examples in that entry.
@pytest.mark.parametrize("kw,pct,passed", [
    (dict(field_dry_density=2.104, max_dry_density=2.131), 98.73, True),
    (dict(field_dry_density=2.12, max_dry_density=2.131), 99.48, True),
    (dict(field_dry_density=1.81, max_dry_density=1.95, required_compaction_percent=95), 92.82, False),
    (dict(field_dry_density=1.9, max_dry_density=2.0, required_compaction_percent=95), 95.0, True),
])
def test_compaction_control_is_field_over_lab_times_100(kw, pct, passed):
    r = compaction_control(**kw)
    _close(r["compaction_percent"], pct)
    assert r["passed"] is passed


# balance = cut - fill (bank) ; loose to haul = |balance|*(1+b)
@pytest.mark.parametrize("kw,bal,loose,status", [
    (dict(cut_volume_m3=2600, fill_volume_m3=1900, bulking_factor=0.25), 700.0, 875.0, "export_surplus"),
    (dict(cut_volume_m3=12000, fill_volume_m3=15000, bulking_factor=0.2), -3000.0, 3600.0, "import_deficit"),
    (dict(cut_volume_m3=500, fill_volume_m3=500, bulking_factor=0.2), 0.0, 0.0, "balanced"),
])
def test_cut_fill_balance_is_cut_minus_fill(kw, bal, loose, status):
    r = cut_fill_balance(**kw)
    _close(r["balance_bank_m3"], bal)
    _close(r["haul_loose_m3"], loose)
    assert r["status"] == status


# unit mass = pi/4*d^2*7850 kg/m (16 mm -> 1.578 kg/m, the docstring's table value)
@pytest.mark.parametrize("kw,mass", [
    (dict(bar_diameter_mm=20, total_length_m=4800), 11837.52),
    (dict(bar_diameter_mm=16, total_length_m=100), 157.83),
])
def test_rebar_weight_is_length_times_bar_unit_mass(kw, mass):
    _close(rebar_weight(**kw)["total_mass_kg"], mass)


# length = area * 1000/spacing * ways ; mass = length * pi/4*d^2*7850
@pytest.mark.parametrize("kw,length,mass", [
    (dict(area_m2=250, spacing_mm=200, bar_diameter_mm=16, both_ways=True), 2500.0, 3945.84),
    (dict(area_m2=100, spacing_mm=150, bar_diameter_mm=12), 666.67, 591.88),
])
def test_rebar_by_area_is_bars_per_metre_times_area(kw, length, mass):
    r = rebar_by_area(**kw)
    _close(r["total_bar_length_m"], length)
    _close(r["total_mass_kg"], mass)


# total = volume * factor ; C30 table factor 320 kgCO2e/m3, or the supplier's factor
@pytest.mark.parametrize("kw,kg", [
    (dict(volume_m3=97.02, grade="c30"), 31046.4),
    (dict(volume_m3=10, embodied_kgco2e_m3=275), 2750.0),
])
def test_carbon_footprint_is_volume_times_factor(kw, kg):
    r = carbon_footprint_concrete(**kw)
    _close(r["total_kgco2e"], kg)
    _close(r["total_tco2e"], kg / 1000.0)


# cost per area = total / area
@pytest.mark.parametrize("kw,rate", [
    (dict(total_cost=50000, area=320), 156.25),
    (dict(total_cost=100000, area=400), 250.0),
])
def test_cost_per_area_is_total_over_area(kw, rate):
    _close(cost_per_area(**kw)["cost_per_area"], rate)


# line extension: total = quantity * unit rate, to 2 dp (the docstring's form).
# Hand cases folded from #742: 12*4850, 37.5*212.40, 7*1.1111 = 7.7777 -> 7.78, 0*1250.
@pytest.mark.parametrize("kw,total", [
    (dict(quantity=12, unit_rate=4850), 58200.0),
    (dict(quantity=37.5, unit_rate=212.40), 7965.0),
    (dict(quantity=7, unit_rate=1.1111), 7.78),
    (dict(quantity=0, unit_rate=1250), 0.0),
])
def test_unit_cost_total_is_quantity_times_rate(kw, total):
    assert unit_cost_total(**kw)["total_cost"] == pytest.approx(total, abs=1e-9)


# crews = Q/(D*R), rounded up
@pytest.mark.parametrize("kw,crews", [
    (dict(quantity=2100, duration_days=4, rate_per_crew_day=250), 3),
    (dict(quantity=1800, duration_days=15, rate_per_crew_day=150, hours_per_day=10, crew_size=4), 1),
    (dict(quantity=3000, duration_days=5, rate_per_crew_day=300), 2),
])
def test_crew_sizing_is_quantity_over_duration_times_rate(kw, crews):
    assert crew_sizing(**kw)["crews_required_rounded_up"] == crews


# The same figures through the public route every caller uses.
@pytest.mark.parametrize("name,params,key,want", [
    ("excavation_volume", {"length_m": 18, "width_m": 9, "depth_m": 2.5, "bulking_factor": 0.25},
     "loose_volume_m3", 506.25),
    ("compaction_control", {"field_dry_density": 2.104, "max_dry_density": 2.131},
     "compaction_percent", 98.73),
    ("cost_per_area", {"total_cost": 50000, "area": 320}, "cost_per_area", 156.25),
    ("unit_cost_total", {"quantity": 24, "unit_rate": 1375.5}, "total_cost", 33012.0),
])
def test_public_route_returns_the_hand_value(name, params, key, want):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation(name, params)
    assert env["status"] == "success"
    _close(env["result"][key], want)


# Degenerate inputs are refused, never a confident wrong figure.
@pytest.mark.parametrize("fn,kw", [
    (excavation_volume, dict(length_m=-1, width_m=9, depth_m=2.5)),
    (backfill_volume, dict(excavation_bank_m3=-5, structure_volume_m3=1)),
    (compaction_control, dict(field_dry_density=1.8, max_dry_density=0)),
    (cut_fill_balance, dict(cut_volume_m3=-1, fill_volume_m3=10)),
    (cost_per_area, dict(total_cost=100, area=0)),
    (crew_sizing, dict(quantity=100, duration_days=0, rate_per_crew_day=10)),
])
def test_degenerate_input_is_refused(fn, kw):
    assert "error" in fn(**kw)
