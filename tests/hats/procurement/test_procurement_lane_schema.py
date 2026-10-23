"""Procurement hat registry schema: every formula and tool parameter is declared."""
from __future__ import annotations

from app.agents.base.formulas.construction_formulas_planning import parse_unit
from app.agents.core.tool_registry import get as get_tool
from app.agents.core.tool_registry import tools_of
from app.lib.formula_registry import NUMERIC_KINDS, TOOL_KINDS, all_specs, parameters, schema_gaps

LANE = "procurement"


def _assert_plain(name: str, label: str) -> None:
    text = (label or "").strip()
    assert text, name
    assert text != name, name
    assert "_" not in text, name


def test_every_procurement_formula_and_tool_declares_its_schema():
    gaps = schema_gaps()
    for spec in all_specs():
        if spec.owner != LANE:
            continue
        assert spec.display_name.strip(), spec.name
        for name, param in parameters(spec).items():
            _assert_plain(f"{spec.name}.{name}", param.label)
            assert param.kind, name
            assert isinstance(param.required, bool), name
            if param.kind in NUMERIC_KINDS:
                assert param.unit, name
                if param.unit != "-":
                    assert parse_unit(param.unit) is not None, name
                assert param.lo is not None and param.hi is not None, name
                assert param.lo < param.hi, name
        own = [
            g for g in gaps
            if g.startswith(f"formula:{spec.name}.") or g == f"formula:{spec.name}:display_name"
        ]
        allowed = {
            f"formula:{spec.name}.{name}:unit"
            for name, param in parameters(spec).items()
            if param.kind in NUMERIC_KINDS and param.unit == "-"
        }
        assert set(own) <= allowed, (spec.name, sorted(set(own) - allowed))
    for tool_name in tools_of(LANE):
        spec = get_tool(tool_name)
        assert spec is not None and spec.display_name.strip()
        assert spec.params, tool_name
        for name, param in spec.params.items():
            _assert_plain(f"{tool_name}.{name}", param.label)
            assert param.kind in TOOL_KINDS, name
            assert param.required is not None, name
            if param.kind in NUMERIC_KINDS:
                assert param.unit and param.unit != "-", name
                assert param.lo is not None and param.hi is not None, name
                assert param.lo < param.hi, name
        assert not [g for g in gaps if g.startswith(f"tool:{tool_name}")]
