"""Production formula-pack F1–F5 against the live construction_calc registry.

These are the pending formula gates that must stay green after Render deploys.
They call ``run_calculation`` — the same dispatcher the agent ``construction_calc``
tool uses — not the hat LLM.
"""
from __future__ import annotations

from app.lib.construction_formulas import available_calculations, run_calculation


def test_formula_pack_ids_are_registered():
    names = set(available_calculations())
    for needed in (
        "beam_moment_point_load",
        "concrete_cylinders",
        "compaction_control",
        "rebar_weight",
        "calculate_interim_payment",
    ):
        assert needed in names, needed


def test_f1_midspan_point_load_moment():
    r = run_calculation(
        "beam_moment_point_load",
        {"point_load_kn": 80, "span_m": 8},
    )
    assert r["status"] == "success"
    assert r["result"]["max_moment_kn_m"] == 160.0


def test_f2_cylinder_compressive_strength():
    r = run_calculation(
        "concrete_cylinders",
        {"failure_load_kn": 680, "cylinder_diameter_mm": 150},
    )
    assert r["status"] == "success"
    assert r["result"]["compressive_strength_mpa"] == 38.48


def test_f3_compaction_does_not_pass():
    r = run_calculation(
        "compaction_control",
        {
            "field_dry_density": 1.82,
            "max_dry_density": 1.95,
            "required_compaction_percent": 95,
        },
    )
    assert r["status"] == "success"
    assert r["result"]["compaction_percent"] == 93.33
    assert r["result"]["passed"] is False
    assert "FAIL" in r["result"]["note"]


def test_f4_t20_rebar_mass():
    r = run_calculation(
        "rebar_weight",
        {"bar_diameter_mm": 20, "total_length_m": 120, "quantity": 8},
    )
    assert r["status"] == "success"
    mass = r["result"]["total_mass_kg"]
    assert abs(mass - 2367.7) < 1.0
    assert "2367" in str(mass)


def test_f5_interim_payment_net():
    r = run_calculation(
        "calculate_interim_payment",
        {"gross_valuation": 750000, "retention_percent": 5},
    )
    assert r["status"] == "success"
    assert r["result"]["net_payment"] == 712500.0
