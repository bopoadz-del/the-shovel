"""Bind engineer-language numbers from ask text into construction_calc.

Phase-2 re-probe of #43–84 scored missing-required after #639 because
named_calculator / predispatch passes ``text`` only. The model (or the
probe) writes As1500 / span 8m / W=10000 kN — those numbers never became
kwargs. Owner D7: extract only what the ask/tool args contain; never
invent a missing figure.

SHARED WITH AGENT C: extract + bind live in run_calculation (formula
envelope). The tool path is the same flatten #636/#639 already use.
"""
from __future__ import annotations

import json

import pytest

from app.lib import construction_formulas as _cf
from app.lib.construction_formulas import (
    CALCULATORS,
    _BIND_SEMANTIC_ALIASES,
    bind_calculation_params,
    extract_calculation_params_from_text,
    run_calculation,
)
from tests.test_construction_calc_tool import _agent, _run


def _ok(name, params):
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, params, env)
    return env["result"]


def _err(name, params):
    env = run_calculation(name, params)
    assert env.get("status") == "error", (name, params, env)
    return env


def _tool(calculation, extra):
    agent = _agent(["construction"])
    tc = {
        "id": "c1",
        "function": {
            "name": "construction_calc",
            "arguments": json.dumps({"calculation": calculation, **extra}),
        },
    }
    return _run(agent._run_tool_call(tc))


# ── rc_beam_moment_capacity (ACI oracle 288.50 kN·m) ───────────────────────

_RC_ASK = (
    "rc_beam_moment_capacity design ACI As1500, fy420, b300, d550, fc30"
)
_RC_KEY = "moment_capacity_kn_m"
_RC_VAL = 288.50
_RC_ABS = 0.3


def test_rc_beam_engineer_ask_text_binds_and_computes():
    """Predispatch shape: calculation + text, no numeric kwargs."""
    inner = _ok("rc_beam_moment_capacity", {"text": _RC_ASK})
    assert inner[_RC_KEY] == pytest.approx(_RC_VAL, abs=_RC_ABS)


def test_rc_beam_engineer_abbreviation_kwargs_bind():
    """Model may put As/fy/b/d/fc next to calculation (not inside params)."""
    inner = _ok("rc_beam_moment_capacity", {
        "As": 1500, "fy": 420, "b": 300, "d": 550, "fc": 30,
    })
    assert inner[_RC_KEY] == pytest.approx(_RC_VAL, abs=_RC_ABS)


@pytest.mark.parametrize("shape", ("top_level_text", "params", "input"))
def test_rc_beam_engineer_ask_three_shapes(shape):
    if shape == "top_level_text":
        extra = {"text": _RC_ASK}
    elif shape == "params":
        extra = {"params": {"text": _RC_ASK}}
    else:
        extra = {"input": {"text": _RC_ASK}}
    env = _tool("rc_beam_moment_capacity", extra)
    assert env.get("ok") is True, env
    assert env["result"]["result"][_RC_KEY] == pytest.approx(_RC_VAL, abs=_RC_ABS)


def test_rc_beam_partial_ask_does_not_invent_missing_dims():
    """D7: As+fy only — refuse; do not invent b=300 / d=550 / fc=30."""
    env = _err("rc_beam_moment_capacity", {
        "text": "rc_beam_moment_capacity As1500 fy420",
    })
    err = env["error"]
    assert "missing required" in err
    for name in ("width_mm", "eff_depth_mm", "fc_mpa"):
        assert name in env["missing"], env["missing"]
        assert name in err
    assert env.get("result") is None or _RC_KEY not in (env.get("result") or {})


# ── steel_tension_capacity (AISC oracle 860.63 kN) ─────────────────────────

_STEEL_ASK = (
    "steel_tension_capacity AISC Ag=3000, Ae=2550 mm2, Fy=345, Fu=450"
)
_STEEL_KEY = "capacity_kn"
_STEEL_VAL = 860.63


def test_steel_tension_engineer_ask_text_binds_and_computes():
    inner = _ok("steel_tension_capacity", {"text": _STEEL_ASK})
    assert inner[_STEEL_KEY] == pytest.approx(_STEEL_VAL, abs=0.05)


def test_steel_tension_engineer_abbreviation_kwargs_bind():
    inner = _ok("steel_tension_capacity", {
        "Ag": 3000, "Ae": 2550, "Fy": 345, "Fu": 450,
    })
    assert inner[_STEEL_KEY] == pytest.approx(_STEEL_VAL, abs=0.05)


def test_steel_tension_partial_ask_does_not_invent_gross_area():
    env = _err("steel_tension_capacity", {
        "text": "steel_tension_capacity Fy=345 Fu=450",
    })
    assert "gross_area_mm2" in env["missing"]
    assert "missing required" in env["error"]


# ── post_tensioning_force (8 m / 0.25 m → 1000 kN; LL unused) ─────────────

_PT_ASK = (
    "post_tensioning_force for span 8m, slab 0.25m, live load 5 kN/m2"
)
_PT_PARTIAL = "post_tensioning_force for span 8m and slab 0.25m"
_PT_KEY = "tendon_force_kn"


def test_post_tensioning_engineer_ask_text_binds_and_computes():
    inner = _ok("post_tensioning_force", {"text": _PT_ASK})
    assert inner[_PT_KEY] == pytest.approx(1000.0, abs=0.1)


def test_post_tensioning_span_and_slab_without_live_load_is_honest():
    """D7: span+slab in the ask still do not invent live_load_kn_m2=5."""
    env = _err("post_tensioning_force", {"text": _PT_PARTIAL})
    assert "live_load_kn_m2" in env["missing"]
    assert "span_m" not in env["missing"]
    assert "slab_thickness_m" not in env["missing"]
    assert "missing required" in env["error"]
    assert "live_load_kn_m2 (kN/m2)" in env["error"]


def test_post_tensioning_construction_calc_text_only():
    env = _tool("post_tensioning_force", {"text": _PT_ASK})
    assert env.get("ok") is True, env
    assert env["result"]["result"][_PT_KEY] == pytest.approx(1000.0, abs=0.1)


# ── seismic_base_shear (ASCE 1250 kN) + unit on success ────────────────────

_SEISMIC_ASK = (
    "seismic_base_shear ASCE W=10000 kN, SDS=1.0, R=8, Ie=1"
)
_SEISMIC_KEY = "base_shear_kn"


def test_seismic_engineer_ask_text_binds_and_stamps_unit():
    env = run_calculation("seismic_base_shear", {"text": _SEISMIC_ASK})
    assert env.get("status") == "success", env
    inner = env["result"]
    assert inner[_SEISMIC_KEY] == pytest.approx(1250.0, abs=0.05)
    assert inner.get("unit") == "kN"
    assert "kn" in str(inner.get("note") or "").lower()


def test_seismic_kwargs_success_stamps_unit_like_628():
    """#628 unitless/SAR pattern: a computed figure must carry a unit."""
    env = run_calculation("seismic_base_shear", {
        "seismic_weight_kn": 10000, "code": "aci",
        "sds": 1.0, "r": 8.0, "ie": 1.0,
    })
    assert env.get("status") == "success", env
    inner = env["result"]
    assert inner[_SEISMIC_KEY] == pytest.approx(1250.0, abs=0.05)
    assert inner.get("unit") == "kN"
    assert inner.get("value") == pytest.approx(1250.0, abs=0.05)


def test_seismic_construction_calc_text_only_has_unit():
    env = _tool("seismic_base_shear", {"text": _SEISMIC_ASK})
    assert env.get("ok") is True, env
    inner = env["result"]["result"]
    assert inner[_SEISMIC_KEY] == pytest.approx(1250.0, abs=0.05)
    assert inner.get("unit") == "kN"


# ── more #43–84 engineer asks (same extract path, no invented defaults) ────

def test_live_load_reduction_engineer_ask_binds():
    inner = _ok("live_load_reduction", {
        "text": "live_load_reduction L0=4.79 kN/m2, KLL=4, AT=40 m2",
    })
    assert inner["reduced_live_load_kn_m2"] == pytest.approx(2.928, abs=0.01)


def test_modulus_of_rupture_engineer_ask_binds():
    inner = _ok("modulus_of_rupture", {
        "text": "modulus of rupture for fck 40 N/mm2",
    })
    assert inner["modulus_of_rupture_n_mm2"] == pytest.approx(4.8, abs=0.005)


def test_rebar_by_area_engineer_ask_binds():
    inner = _ok("rebar_by_area", {
        "text": "rebar_by_area 20 m2, spacing 200 mm, d12",
    })
    assert inner["total_mass_kg"] == pytest.approx(88.8, abs=0.2)


def test_masonry_wall_engineer_ask_binds():
    inner = _ok("masonry_wall_capacity", {
        "text": "masonry_wall_capacity fm=10 MPa, An=190000 mm2, h=3000 mm, t=190 mm",
    })
    assert inner["capacity_kn"] == pytest.approx(402.5, abs=1.0)


def test_wind_load_on_formwork_engineer_ask_binds():
    inner = _ok("wind_load_on_formwork", {
        "text": "wind_load_on_formwork V=30 m/s, formwork area 50 m2, height 3 m, width 8 m",
    })
    assert inner["wind_force_kn"] == pytest.approx(27.585, abs=0.01)


def test_productivity_comma_list_does_not_eat_next_label():
    """'100, man_hours 50' must not swallow the comma and the leading m."""
    inner = _ok("productivity_manpower_duration", {
        "text": "productivity_manpower_duration quantity_executed 100, man_hours 50",
    })
    assert inner.get("productivity") == pytest.approx(2.0, abs=0.01)


def test_progress_quantity_engineer_ask_binds():
    inner = _ok("progress_quantity", {
        "text": "progress_quantity total_qty 1000, planned_qty 400, actual_qty 350",
    })
    assert inner["actual_percent"] == pytest.approx(35.0, abs=0.01)


def test_resource_line_cost_engineer_ask_binds():
    inner = _ok("resource_line_cost", {
        "text": "resource_line_cost quantity 100, daily_output 10, day_rate 280",
    })
    assert inner["labour_cost"] == pytest.approx(2800.0, abs=0.05)


def test_precast_beam_engineer_ask_binds():
    inner = _ok("precast_beam_erection_check", {
        "text": (
            "precast_beam_erection_check beam weight 20 t, beam length 15 m, "
            "crane capacity 50 t, lift radius 12 m"
        ),
    })
    assert inner["utilization_pct"] == pytest.approx(51.2, abs=0.15)


def test_empty_text_still_names_required_does_not_invent():
    env = _err("rc_beam_moment_capacity", {"text": "rc_beam_moment_capacity"})
    assert "missing required" in env["error"]
    assert "steel_area_mm2" in env["missing"]
    bound = bind_calculation_params(
        CALCULATORS["rc_beam_moment_capacity"],
        {"text": "rc_beam_moment_capacity"},
    )
    assert bound == {}


# C #663 (7d67f9e) dests that the B union must keep. Adding dests is OK;
# dropping a C dest (or the key) is a bind regression.
_C663_ALIAS_DESTS = {
    "ab": ("bolt_area_mm2",),
    "ac": ("ac", "acwp"),
    "aca": ("contract_amount",),
    "accepted_contract_amount": ("contract_amount",),
    "actual_cost": ("ac", "acwp"),
    "ae": ("net_area_mm2",),
    "ag": ("gross_area_mm2",),
    "age": ("time_days",),
    "age_days": ("time_days",),
    "an": ("net_area_mm2",),
    "anet": ("net_area_mm2",),
    "as": ("steel_area_mm2",),
    "ast": ("steel_area_mm2",),
    "at": ("tributary_area_m2", "area_m2"),
    "axial": ("axial_load_kn",),
    "axial_load": ("axial_load_kn",),
    "b": ("width_mm",),
    "bank": ("excavation_bank_m3",),
    "bank_volume": ("excavation_bank_m3",),
    "bolt_area": ("bolt_area_mm2",),
    "budget_at_completion": ("bac",),
    "bw": ("width_mm",),
    "cert": ("certified_amount",),
    "certified": ("certified_amount",),
    "claim": ("claimed_amount",),
    "claimed": ("claimed_amount",),
    "contract_value": ("contract_amount",),
    "d": ("eff_depth_mm", "bar_diameter_mm"),
    "days": ("time_days",),
    "delay_rate": ("rate_percent",),
    "dia": ("column_diameter_mm", "diameter_mm", "bolt_diameter_mm"),
    "diameter": ("column_diameter_mm", "diameter_mm", "bolt_diameter_mm"),
    "earned_value": ("ev", "bcwp"),
    "esh_ult": ("ultimate_shrinkage_microstrain",),
    "ev": ("ev", "bcwp"),
    "excavation": ("excavation_bank_m3",),
    "excavation_bank": ("excavation_bank_m3",),
    "fc": ("fc_mpa", "fck_n_mm2"),
    "fck": ("fck_n_mm2", "fc_mpa"),
    "fdd": ("field_dry_density",),
    "field": ("field_dry_density",),
    "field_density": ("field_dry_density",),
    "floors": ("floor_count",),
    "fm": ("masonry_strength_mpa",),
    "fu": ("fu_mpa",),
    "fy": ("fy_mpa",),
    "gross": ("gross_valuation",),
    "gross_value": ("gross_valuation",),
    "h": ("height_mm", "formwork_height_m"),
    "h_w": ("water_depth",),
    "height": ("height_mm", "formwork_height_m", "height_m"),
    "hw": ("water_depth",),
    "ie": ("ie",),
    "kll": ("kll",),
    "l": ("span_m", "length_m"),
    "l0": ("base_live_load_kn_m2",),
    "lab_density": ("max_dry_density",),
    "live_load": ("live_load_kn_m2",),
    "ll": ("live_load_kn_m2",),
    "max_density": ("max_dry_density",),
    "mdd": ("max_dry_density",),
    "n_floors": ("floor_count",),
    "nfloors": ("floor_count",),
    "p": ("central_point_load_kn", "axial_load_kn", "point_load_kn"),
    "planned_value": ("pv", "bcws"),
    "pv": ("pv", "bcws"),
    "raft": ("raft_thickness",),
    "rate": ("rate_percent",),
    "retention": ("retention_percent", "retention_rate"),
    "sds": ("sds",),
    "slab": ("slab_thickness_m",),
    "slab_thickness": ("slab_thickness_m",),
    "span": ("span_m",),
    "structure": ("structure_volume_m3",),
    "structure_volume": ("structure_volume_m3",),
    "t": ("time_days", "thickness_mm", "slab_thickness_m"),
    "time": ("time_days",),
    "traft": ("raft_thickness",),
    "udl": ("udl_w_kn_m",),
    "udl_w": ("udl_w_kn_m",),
    "ultimate_shrinkage": ("ultimate_shrinkage_microstrain",),
    "v": ("wind_velocity_m_s", "wind_speed_m_s"),
    "valuation": ("gross_valuation",),
    "vol": ("volume_m3",),
    "volume": ("volume_m3",),
    "w": ("udl_w_kn_m", "seismic_weight_kn", "formwork_width_m"),
    "water_head": ("water_depth",),
    "water_table": ("water_depth",),
    "width": ("width_mm", "formwork_width_m", "width_m"),
}


def test_bind_alias_map_is_union_of_c663_dests():
    """One `_BIND_SEMANTIC_ALIASES` — B dests may grow; C #663 dests stay."""
    assert len(_C663_ALIAS_DESTS) == 89
    missing_keys = sorted(set(_C663_ALIAS_DESTS) - set(_BIND_SEMANTIC_ALIASES))
    assert missing_keys == [], missing_keys
    drops = []
    for key, dests in _C663_ALIAS_DESTS.items():
        have = _BIND_SEMANTIC_ALIASES[key]
        lost = [d for d in dests if d not in have]
        if lost:
            drops.append((key, dests, have, lost))
    assert drops == [], drops


def test_extract_from_text_runs_on_empty_and_partial_kwargs(monkeypatch):
    """C #663 extract must run when kwargs are empty/partial and text is set."""
    calls: list[tuple[str, str]] = []
    real = extract_calculation_params_from_text

    def _spy(fn, text):
        calls.append((getattr(fn, "__name__", ""), str(text)))
        return real(fn, text)

    monkeypatch.setattr(_cf, "extract_calculation_params_from_text", _spy)
    empty = run_calculation("rc_beam_moment_capacity", {"text": _RC_ASK})
    assert empty.get("status") == "success", empty
    assert any(name == "rc_beam_moment_capacity" for name, _ in calls), calls
    calls.clear()
    partial = run_calculation("rc_beam_moment_capacity", {
        "text": _RC_ASK, "fy": 420,
    })
    assert partial.get("status") == "success", partial
    assert any(name == "rc_beam_moment_capacity" for name, _ in calls), calls
    assert "steel_area_mm2" in extract_calculation_params_from_text(
        CALCULATORS["rc_beam_moment_capacity"], _RC_ASK,
    )
