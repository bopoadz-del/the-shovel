"""1c regression: a calculator that declares a source keeps its citation; a
bare computation still strips an invented attribution."""
from __future__ import annotations

import json

from app.agents.citation_provenance import gate
from app.lib.construction_formulas import run_calculation


def _calc_messages(calculation: str, params: dict) -> list[dict]:
    envelope = run_calculation(calculation, params)
    args = json.dumps({"calculation": calculation, "params": params})
    return [
        {"role": "user", "content": "fixture ask"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "construction_calc", "arguments": args},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "construction_calc",
            "content": json.dumps(envelope),
        },
    ]


def test_a_calculator_with_a_declared_standard_keeps_its_source_citation():
    msgs = _calc_messages("crew_sizing",
                          {"quantity": 3000, "duration_days": 15,
                           "rate_per_crew_day": 12})
    answer = "Crews: 17.\n\nSource: owner-supplied project-control template, 09 Nov 2026."
    out = gate(answer, None, msgs)
    assert "owner-supplied project-control template" in out, out


def test_a_reference_table_calc_keeps_its_provenance_citation():
    msgs = _calc_messages("productivity_benchmark", {"work_item": "blockwork walls"})
    answer = "Blockwork: 12 m2/crew-day.\n\nSource: owner-supplied planner template, unverified."
    out = gate(answer, None, msgs)
    assert "owner-supplied planner template" in out, out


def test_a_bare_calculation_still_strips_an_invented_attribution():
    msgs = _calc_messages("dewatering_uplift_check",
                          {"water_depth": 23, "raft_thickness": 2,
                           "floor_count": 5})
    answer = "Uplift: safe.\n\nSource: Some made-up standard XYZ-123."
    out = gate(answer, None, msgs)
    assert "Some made-up standard" not in out, out
