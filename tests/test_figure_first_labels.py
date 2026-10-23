"""A figure written before its label binds to that label: "120,000 m3 of
concrete", "6,000 t steel", in any unit of the same kind. Synthetic
calculator; optional inputs, so no unit-only rule can bind them instead."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import Param, registered


def _crew(concrete_m3: float = 0.0, structural_steel_t: float = 0.0) -> dict:
    return {"value": concrete_m3 / 1000 + structural_steel_t / 100}


_INPUTS = {
    "concrete_m3": Param("m3", lo=0, hi=1e9, label="concrete", required=False),
    "structural_steel_t": Param("t", lo=0, hi=1e9, label="structural steel", required=False),
}


@pytest.mark.parametrize("ask", [
    "lane5 crew check for 120,000 m3 concrete and 6,000 t steel",
    "lane5 crew check for 120,000 m3 of concrete and 6,000,000 kg of structural steel",
])
def test_a_figure_written_before_its_label_binds_to_it(ask):
    from app.lib.construction_formulas import run_calculation

    with registered("lane5_fig_first", _crew, "Lane5 crew check", _INPUTS, {"value": "count"}):
        env = run_calculation("lane5_fig_first", {"text": ask})
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(120 + 60)


def test_a_figure_of_another_kind_does_not_bind_on_a_word():
    from app.lib.construction_formulas import run_calculation

    with registered("lane5_fig_first", _crew, "Lane5 crew check", _INPUTS, {"value": "count"}):
        env = run_calculation("lane5_fig_first", {"text": "lane5 crew check for 12 m concrete"})
    assert (env.get("result") or {}).get("value", 0) == 0
