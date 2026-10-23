"""CORE schema: a dimensionless numeric input is a count or a ratio, and
the converter knows both."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import Param, registered


@pytest.mark.xfail(strict=True, reason="CORE: 19 schema_gaps asks a dimensionless input to declare count/ratio; the converter knows neither")
def test_a_dimensionless_input_declares_count_or_ratio():
    from app.agents.base.formulas.construction_formulas_planning import parse_unit
    from app.lib.formula_registry import schema_gaps

    name = "lane5_schema_tally"

    def fn(tally_n: float) -> dict:
        return {"value": tally_n}

    with registered(name, fn, "Qxnet tally gate", {
        "tally_n": Param("-", lo=0, hi=100, label="tally"),
    }, {"value": "-"}):
        ours = [gap for gap in schema_gaps() if name in gap]
        told = any("count" in gap or "ratio" in gap for gap in ours)
        known = (parse_unit("count") is not None, parse_unit("ratio") is not None)
    assert ours, "the dimensionless input produced no schema gap"
    assert (told, known) == (True, (True, True))
