"""Lane registry and figures for the safety hat.

Generated from the hat's registered formulas. Every param has a plain
label, a unit or a closed choice, and a range. Every formula recomputes
a hand-checked figure.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app.lib.construction_formulas import run_calculation
from app.lib.formula_registry import NUMERIC_KINDS, all_specs, parameters

LANE = "safety"
_LANE_ROOT = Path(__file__).resolve().parents[3] / "app" / "agents" / "hats" / LANE
_UNIT_IN_LABEL = re.compile(
    r"\b\d+\s*(?:mm|m2|m3|m/s|n/mm2|kpa|kn|kg|h|m)\b",
    re.IGNORECASE,
)
_MEASURED = ("density", "strength", "fraction", "utilisation", "utilization")


def _lane_specs():
    return [spec for spec in all_specs() if spec.owner == LANE]


def _lane_params():
    return [(spec, name, p) for spec in _lane_specs() for name, p in parameters(spec).items()]


def _ids(rows):
    return [f"{spec.name}.{name}" for spec, name, _p in rows]


def _bakes_unit(label: str, unit: str) -> bool:
    words = (label or "").strip().lower().split()
    token = re.sub(r"[^a-z0-9%]", "", (unit or "").lower())
    if token and token not in {"-", "fraction"} and words and words[-1] == token:
        return True
    return bool(_UNIT_IN_LABEL.search(label or ""))


def _filler(spec, skip=()):
    values = {}
    for name, p in parameters(spec).items():
        if name in skip or not p.required:
            continue
        if p.kind == "series":
            lo = 0.0 if p.lo is None else p.lo
            hi = 10.0 if p.hi is None else p.hi
            values[name] = [lo, (lo + hi) / 2.0]
        elif p.kind == "integer":
            values[name] = int(p.default if p.default is not None else (p.lo or 1))
        elif p.kind in NUMERIC_KINDS:
            if p.lo is not None and p.hi is not None:
                values[name] = (p.lo + p.hi) / 2.0
            else:
                values[name] = 1.0
        elif p.kind == "flag":
            values[name] = False
        else:
            values[name] = p.default if p.default not in (None, "") else "x"
    return values


LANE_PARAMS = _lane_params()
TEXT_PARAMS = [(s, n, p) for s, n, p in LANE_PARAMS if p.kind == "text"]


@pytest.mark.parametrize("spec,name,p", LANE_PARAMS, ids=_ids(LANE_PARAMS))
def test_every_lane_param_has_a_plain_label(spec, name, p):
    label = (p.label or "").strip()
    assert label, f"{spec.name}.{name} has no display label"
    assert label != name
    assert "_" not in label
    assert not _bakes_unit(label, p.unit), (spec.name, name, label, p.unit)


@pytest.mark.parametrize("spec,name,p", LANE_PARAMS, ids=_ids(LANE_PARAMS))
def test_every_lane_param_has_a_unit_or_choice_and_a_range(spec, name, p):
    if p.kind == "text":
        return
    assert p.unit, f"{spec.name}.{name} declares no unit"
    if p.kind in NUMERIC_KINDS:
        assert p.lo is not None and p.hi is not None, f"{spec.name}.{name} has no range"
        assert p.lo < p.hi
    if any(word in name.lower() for word in _MEASURED):
        assert p.unit not in ("", "-"), f"{spec.name}.{name} is a measured input with no unit"


@pytest.mark.parametrize("spec,name,p", TEXT_PARAMS, ids=_ids(TEXT_PARAMS))
def test_every_lane_text_param_rejects_an_unknown_choice(spec, name, p):
    values = _filler(spec, skip=(name,))
    values[name] = "super-heavy"
    env = run_calculation(spec.name, values)
    assert env["status"] == "error", f"{spec.name}.{name} accepted an unknown choice"
    text = str(env.get("error") or env.get("question") or "").lower()
    assert "super-heavy" in text
    assert "light" in text and "medium" in text and "heavy" in text


def test_every_lane_formula_recomputes_a_hand_checked_figure():
    r = run_calculation("scaffold_load_capacity", {
        "platform_area_m2": 12, "duty": "medium",
    })["result"]
    assert r["intended_load_kn"] == pytest.approx(28.8, abs=0.01)
    assert r["required_capacity_kn"] == pytest.approx(115.2, abs=0.01)

    r = run_calculation("fall_arrest_force", {
        "worker_mass_kg": 100, "free_fall_m": 1.8, "deceleration_distance_m": 1.0,
    })["result"]
    assert r["max_arrest_force_kn"] == pytest.approx(2.75, abs=0.02)

    r = run_calculation("crane_lift_capacity", {
        "chart_capacity_t": 50, "deductions_t": 3.5, "load_t": 30,
    })["result"]
    assert r["utilization_percent"] == pytest.approx(64.5, abs=0.1)

    r = run_calculation("wind_load_on_formwork", {
        "wind_velocity_m_s": 25, "formwork_width_m": 5, "formwork_height_m": 6,
    })["result"]
    assert r["wind_pressure_kpa"] == pytest.approx(0.383, abs=0.001)
    assert r["wind_force_kn"] == pytest.approx(11.49, abs=0.01)
    assert r["overturning_moment_kn_m"] == pytest.approx(34.48, abs=0.01)

    r = run_calculation("guardrail_top_rail_height")["result"]
    assert r["top_rail_height_in"] == 42
    assert r["top_rail_height_mm"] == 1067
    note = str(r.get("note") or "")
    assert not re.search(r"\b(?:SABIC|Aramco|ADNOC|NEOM)\b", note)


def test_unknown_scaffold_duty_is_rejected_with_the_valid_ones():
    env = run_calculation("scaffold_load_capacity", {
        "platform_area_m2": 10, "duty": "super-heavy",
    })
    assert env["status"] == "error"
    text = str(env.get("error") or env.get("question") or "")
    assert "super-heavy" in text
    assert "light" in text.lower()
    assert "medium" in text.lower()
    assert "heavy" in text.lower()
    assert "[" not in text


def test_wind_load_derives_area_from_width_and_height():
    env = run_calculation("wind_load_on_formwork", {
        "text": "form 5 m wide by 6 m high, wind 25 m/s",
    })
    assert env["status"] == "success", env
    r = env["result"]
    assert r["wind_pressure_kpa"] == pytest.approx(0.383, abs=0.001)
    assert r["wind_force_kn"] == pytest.approx(11.49, abs=0.01)
    assert r["overturning_moment_kn_m"] == pytest.approx(34.48, abs=0.01)


def test_load_above_chart_fails_with_a_reason_and_no_silent_limit():
    env = run_calculation("crane_lift_capacity", {
        "chart_capacity_t": 10, "deductions_t": 0, "load_t": 15,
    })
    assert env["status"] == "success", env
    r = env["result"]
    assert r["passed"] is False
    reason = str(r.get("reason") or "")
    assert reason
    assert "15" in reason
    assert re.search(r"capacit", reason, re.I)
    assert "85" not in reason
    blob = " ".join(f"{key} {r.get(key)}" for key in r)
    if r.get("max_utilization_percent") is not None:
        assert "default" in blob.lower()


def test_lane_source_names_no_company_or_client():
    named = []
    for path in _LANE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if re.search(r"\b(?:SABIC|Aramco|ADNOC|NEOM)\b", node.value):
                named.append(f"{path.name}:{node.lineno}")
    assert not named, named
