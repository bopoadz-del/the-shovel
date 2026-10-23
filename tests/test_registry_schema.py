"""Every registry entry carries the full schema (F-CORE-ROUND-2 step 1/2).

Formulas and tools: display_name; every parameter: display label, type,
unit and range (numeric), required / optional. Existing gaps are listed in
tests/fixtures/registry_schema_gaps.json, which may only shrink: a new or
changed entry missing a field fails here, and a filled gap must leave the
baseline (scripts/registry_schema_gaps.py --shrink).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.lib import formula_registry as fr

BASELINE = Path(__file__).parent / "fixtures" / "registry_schema_gaps.json"


def _baseline() -> set:
    return set(json.loads(BASELINE.read_text(encoding="utf-8")))


def test_no_registry_entry_is_missing_a_field_outside_the_baseline():
    new = sorted(set(fr.schema_gaps()) - _baseline())
    assert new == [], f"declare these fields (do not list them): {new[:20]}"


def test_the_baseline_lists_only_gaps_that_still_exist():
    filled = sorted(_baseline() - set(fr.schema_gaps()))
    assert filled == [], f"run scripts/registry_schema_gaps.py --shrink; filled: {filled[:20]}"


def _synthetic(**inputs):
    def calc(width_m: float, count: int = 1):
        return {"result": width_m * count}
    calc.__name__ = "syn_schema_probe"
    fr._REGISTRY.pop("syn_schema_probe", None)
    fr.formula(owner="base", description="Synthetic probe.", display_name="Synthetic probe",
               inputs=inputs, outputs={"result": "m"})(calc)
    return fr.get("syn_schema_probe")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    fr._REGISTRY.pop("syn_schema_probe", None)


def test_a_new_formula_missing_a_label_or_unit_is_a_gap():
    _synthetic(width_m=fr.Param(unit="m", lo=0.01, hi=100.0), count="-")
    gaps = [g for g in fr.schema_gaps() if g.startswith("formula:syn_schema_probe")]
    assert "formula:syn_schema_probe.width_m:label" in gaps
    assert "formula:syn_schema_probe.count:unit" in gaps
    assert "formula:syn_schema_probe.count:range" in gaps
    assert sorted(set(gaps) - _baseline()) == gaps  # never covered by the baseline


def test_a_complete_formula_has_no_gaps():
    _synthetic(width_m=fr.Param(unit="m", lo=0.01, hi=100.0, label="width"),
               count=fr.Param(unit="count", lo=1, hi=1000, label="number of items"))
    assert [g for g in fr.schema_gaps() if g.startswith("formula:syn_schema_probe")] == []
    p = fr.parameters(fr.get("syn_schema_probe"))
    assert p["width_m"].display_label == "width" and p["width_m"].required
    assert p["count"].kind == "integer" and not p["count"].required


def test_a_tool_declares_its_parameters_with_kind_and_required():
    from app.agents.core import tool_registry as tr

    async def handle(call):
        return {}
    try:
        tr.tool("syn_tool_probe", owner="base", display_name="Synthetic tool")(handle)
        assert "tool:syn_tool_probe:params" in fr.schema_gaps()
        tr._TOOLS.pop("syn_tool_probe")
        tr.tool("syn_tool_probe", owner="base", display_name="Synthetic tool", params={
            "amount": fr.Param(unit="currency", lo=0, hi=1e12, label="amount", kind="number", required=True),
            "note": fr.Param(label="note", kind="text"),
        })(handle)
        gaps = [g for g in fr.schema_gaps() if "syn_tool_probe" in g]
        assert gaps == ["tool:syn_tool_probe.note:required"]
    finally:
        tr._TOOLS.pop("syn_tool_probe", None)
