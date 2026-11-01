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
        f"The delay damages are SAR {daily} per day.", None, messages, enforce=False)
    rows = _calc_inputs(prov)
    assert rows["rate_percent"]["source"] == "project_data", rows
    assert rows["contract_amount"]["source"] == "project_data", rows
    # unresolved == 0 on the exercised project-rate path
    assert all(r["source"] != "unresolved" for r in rows.values()), rows


@pytest.mark.parametrize("rate", [0.1, 0.25])
def test_an_operator_stated_operand_stays_user_text(rate):
    # The operator states the rate; the Accepted Contract Amount is stamped
    # project_data. The figure is the COMPUTED daily amount (not a value the
    # operator typed) so it credits to the calculator, and the rate operand --
    # whose value is in the operator's words and carries no stamp -- stays
    # user_text. One calc exercises user_text AND project_data together.
    aca = 10_000_000.0
    daily = round(aca * rate / 100.0, 2)
    messages = [{"role": "user", "content": f"Delay damages at {rate}% per day?"}]
    env = {
        "calculation": "delay_damages_daily", "status": "success",
        "result": {"daily_amount": daily, "rate_percent": rate,
                   "contract_amount": aca, "currency": "SAR"},
    }
    rt._record_calculator_tool_turn(
        messages, "delay_damages_daily",
        {"rate_percent": rate, "contract_amount": aca}, env,
        input_sources={"contract_amount": "project_data"})
    _out, prov = figure_provenance(
        f"The delay damages are SAR {daily} per day.", None, messages, enforce=False)
    rows = _calc_inputs(prov)
    assert rows["rate_percent"]["source"] == "user_text", rows
    assert rows["contract_amount"]["source"] == "project_data", rows
