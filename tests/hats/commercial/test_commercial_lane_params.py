"""Commercial hat: every formula and tool parameter is declared."""
from __future__ import annotations

from app.agents.core import tool_registry
from app.lib.formula_registry import all_specs, parameters, schema_gaps

LANE = "commercial"


def test_commercial_formulas_and_tools_have_no_schema_gaps():
    names = {spec.name for spec in all_specs() if spec.owner == LANE}
    tool_registry.load()
    tools = {name for name, spec in tool_registry._TOOLS.items() if spec.owner == LANE}
    assert names and tools
    gaps = [
        gap for gap in schema_gaps()
        if any(gap.startswith(f"formula:{name}") for name in names)
        or any(gap.startswith(f"tool:{name}") for name in tools)
    ]
    assert gaps == []


def test_commercial_formula_params_declare_label_unit_and_range():
    for spec in all_specs():
        if spec.owner != LANE:
            continue
        assert spec.display_name.strip()
        for name, param in parameters(spec).items():
            assert param.label.strip(), name
            assert param.required in (True, False)
            if param.kind in ("number", "integer", "series"):
                assert param.unit and param.unit != "-", name
                assert param.lo is not None and param.hi is not None, name
                assert param.lo < param.hi, name
