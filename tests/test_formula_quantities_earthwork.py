"""Oracle tests for the quantities + earthwork additive formulas.

Geometry/arithmetic — values hand-derived and shown in comments.
"""
import pytest

from app.lib.construction_formulas import CALCULATORS


# ── concrete_volume ────────────────────────────────────────────────────────

def test_concrete_volume_rectangular():
    # 10*5*0.3 = 15 m3; +5% = 15.75.
    r = CALCULATORS["concrete_volume"](length_m=10, width_m=5, thickness_m=0.3)
    assert r["net_volume_m3"] == pytest.approx(15.0, abs=0.001)
    assert r["volume_with_waste_m3"] == pytest.approx(15.75, abs=0.001)


def test_concrete_volume_cylinder():
    # pi*(0.6/2)^2*10 = pi*0.09*10 = 2.827 m3.
    r = CALCULATORS["concrete_volume"](shape="cylinder", diameter_m=0.6, height_m=10)
    assert r["net_volume_m3"] == pytest.approx(2.827, abs=0.001)


def test_concrete_volume_trapezoidal():
    # ((2+4)/2*1.5)*10 = 4.5*10 = 45 m3; +5% = 47.25.
    r = CALCULATORS["concrete_volume"](shape="trapezoidal", top_width_m=2,
                                       bottom_width_m=4, depth_m=1.5, length_m=10)
    assert r["net_volume_m3"] == pytest.approx(45.0, abs=0.001)
    assert r["volume_with_waste_m3"] == pytest.approx(47.25, abs=0.001)


# ── rebar_weight / rebar_by_area ───────────────────────────────────────────

def test_rebar_weight_16mm():
    # unit mass = (pi/4)*(0.016)^2*7850 = 1.578 kg/m (matches BS 8666 table);
    # x12m x50 = 947.0 kg.
    r = CALCULATORS["rebar_weight"](bar_diameter_mm=16, total_length_m=12, quantity=50)
    assert r["unit_mass_kg_m"] == pytest.approx(1.578, abs=0.002)
    assert r["total_mass_kg"] == pytest.approx(947.0, abs=0.5)


def test_rebar_weight_to_length_12_tonnes_y16():
    """Live A2-2: 12 t of Y16 → metres run, not a 1 m demo."""
    r = CALCULATORS["rebar_weight"](
        bar_diameter_mm=16, total_weight_kg=12000, mode="weight_to_length",
    )
    assert r.get("error") is None
    metres = r.get("metres_run") or r["total_length_m"]
    assert metres == pytest.approx(12000 / r["unit_mass_kg_m"], abs=1.0)
    assert metres > 1000


def test_rebar_by_area_one_way():
    # d=12 -> unit mass 0.888 kg/m; 1000/200 = 5 bars/m; 5*20*1 = 100 m; *0.888 = 88.8 kg.
    r = CALCULATORS["rebar_by_area"](area_m2=20, spacing_mm=200, bar_diameter_mm=12)
    assert r["bars_per_m"] == pytest.approx(5.0, abs=0.001)
    assert r["total_mass_kg"] == pytest.approx(88.8, abs=0.3)


# ── earthwork ──────────────────────────────────────────────────────────────

def test_excavation_volume():
    # 20*10*3 = 600 bank; *1.25 = 750 loose.
    r = CALCULATORS["excavation_volume"](length_m=20, width_m=10, depth_m=3)
    assert r["bank_volume_m3"] == pytest.approx(600.0, abs=0.01)
    assert r["loose_volume_m3"] == pytest.approx(750.0, abs=0.01)


def test_backfill_volume():
    # void = 600 - 200 = 400; *1.20 = 480 loose.
    r = CALCULATORS["backfill_volume"](excavation_bank_m3=600, structure_volume_m3=200)
    assert r["void_volume_m3"] == pytest.approx(400.0, abs=0.01)
    assert r["loose_backfill_needed_m3"] == pytest.approx(480.0, abs=0.01)


def test_cut_fill_balance_surplus():
    # 5000 - 3500 = 1500 surplus; loose 1500*1.25 = 1875.
    r = CALCULATORS["cut_fill_balance"](cut_volume_m3=5000, fill_volume_m3=3500)
    assert r["balance_bank_m3"] == pytest.approx(1500.0, abs=0.01)
    assert r["status"] == "export_surplus"
    assert r["haul_loose_m3"] == pytest.approx(1875.0, abs=0.01)


def test_compaction_control_pass():
    # 1.90/2.00*100 = 95.0% -> pass at 95%.
    r = CALCULATORS["compaction_control"](field_dry_density=1.90, max_dry_density=2.00)
    assert r["compaction_percent"] == pytest.approx(95.0, abs=0.01)
    assert r["passed"] is True


def test_slope_fos_cohesionless():
    # tan(30)/tan(20) = 0.5774/0.3640 = 1.586.
    r = CALCULATORS["slope_fos_simple"](friction_angle_deg=30, slope_angle_deg=20)
    assert r["factor_of_safety"] == pytest.approx(1.586, abs=0.002)


def test_slope_fos_with_cohesion():
    # cohesive = 5/(18*3*sin20*cos20) = 5/17.355 = 0.288; + 1.586 = 1.874.
    r = CALCULATORS["slope_fos_simple"](friction_angle_deg=30, slope_angle_deg=20,
                                        cohesion_kpa=5, unit_weight_kn_m3=18, depth_m=3)
    assert r["cohesive_term"] == pytest.approx(0.288, abs=0.003)
    assert r["factor_of_safety"] == pytest.approx(1.874, abs=0.003)
