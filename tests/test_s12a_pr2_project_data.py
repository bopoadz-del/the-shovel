"""S-12a PR-2: an operand composed from the project's contract data is stamped
project_data at compose time and carried through the CREDITED calculator turn,
so figure provenance shows "project_data" -- never "unresolved" -- on the
project-rate path. The user-text path is unchanged. Synthetic only.
Full-dotted imports so the mutation gate selects this.
"""
from __future__ import annotations

import app.agents.runtime as rt
from app.agents import provenance_trail as pt
from app.agents.citation_provenance import figure_provenance

import pytest


def _calc_inputs(prov):
    cals = [e for e in prov if e.get("source") == pt.SOURCE_CALCULATOR and e.get("inputs")]
    assert cals, prov
    return {r["name"]: r for r in cals[0]["inputs"]}


@pytest.mark.parametrize("rate,aca", [(0.137, 39098392.98), (0.015, 12500000.0)])
def test_project_rate_operands_are_stamped_project_data(rate, aca):
    daily = round(aca * rate / 100.0, 2)
    messages = [{"role": "user", "content": "What are the delay damages per day?"}]
    env = {
        "calculation": "delay_damages_daily", "status": "success",
        "result": {"daily_amount": daily, "rate_percent": rate,
                   "contract_amount": aca, "currency": "SAR"},
    }
    rt._record_calculator_tool_turn(
        messages, "delay_damages_daily",
        {"rate_percent": rate, "contract_amount": aca}, env,
        input_sources={"rate_percent": "project_data", "contract_amount": "project_data"},
    )
    _out, prov = figure_provenance(
        f"The delay damages are {daily} per day.", None, messages, enforce=False)
    rows = _calc_inputs(prov)
    assert rows["rate_percent"]["source"] == "project_data", rows
    assert rows["contract_amount"]["source"] == "project_data", rows
    # unresolved == 0 on the exercised project-rate path
    assert all(r["source"] != "unresolved" for r in rows.values()), rows


@pytest.mark.parametrize("value", [250.0, 480.0])
def test_user_text_operand_is_unchanged_without_a_stamp(value):
    # No _input_sources stamp -> a value the operator stated stays user_text
    # (the credited-turn plumbing does not disturb the user-text path).
    messages = [{"role": "user", "content": f"Delay damages at {value} per day over the job?"}]
    env = {
        "calculation": "delay_damages_daily", "status": "success",
        "result": {"daily_amount": value, "currency": "SAR"},
    }
    rt._record_calculator_tool_turn(
        messages, "delay_damages_daily", {"daily_amount": value}, env)
    _out, prov = figure_provenance(
        f"The delay damages are {value} per day.", None, messages, enforce=False)
    rows = _calc_inputs(prov)
    assert rows["daily_amount"]["source"] == "user_text", rows
