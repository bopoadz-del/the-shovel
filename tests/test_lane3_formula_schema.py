"""Every design and quantities formula declares a complete registry schema.

A complete entry has a display name, and each input has a label, a type,
and required or optional taken from its signature. A numeric input also
has a unit the converter can read and a lower and upper limit.
"""
from __future__ import annotations

from app.agents.base.formulas.construction_formulas_planning import parse_unit
from app.lib.formula_registry import NUMERIC_KINDS, all_specs, parameters, schema_gaps

OWNERS = ("design", "quantities")


def test_every_lane3_formula_declares_a_complete_schema():
    gaps = schema_gaps()
    seen = 0
    for spec in all_specs():
        if spec.owner not in OWNERS:
            continue
        seen += 1
        assert spec.display_name.strip(), spec.name
        own = [
            g for g in gaps
            if g.startswith(f"formula:{spec.name}.") or g == f"formula:{spec.name}:display_name"
        ]
        assert own == [], (spec.name, own)
        for name, param in parameters(spec).items():
            where = f"{spec.name}.{name}"
            assert param.label.strip(), where
            assert param.kind, where
            assert isinstance(param.required, bool), where
            if param.kind not in NUMERIC_KINDS:
                continue
            assert param.unit and param.unit != "-", where
            assert parse_unit(param.unit) is not None, where
            assert param.lo is not None and param.hi is not None, where
            assert param.lo < param.hi, where
    assert seen > 0
