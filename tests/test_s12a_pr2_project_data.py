"""S-12a PR-2: a composed delay_damages_daily figure (rate x Accepted Contract
Amount, both read from the project's contract data) is mirrored as a CREDITED
calculator turn so its operands carry source "project_data" -- never
"unresolved" -- while the display the user sees is byte-identical. Synthetic
only. Full-dotted imports so the mutation gate selects this.
"""
from __future__ import annotations

import json

import pytest

import app.agents.runtime as rt
from app.agents import provenance_trail as pt
from app.agents.citation_provenance import figure_provenance


def _flat_payload(rate, aca, daily):
    """The flat DISPLAY payload the delay_damages_daily graft appends."""
    return json.dumps({
        "calculation": "delay_damages_daily", "daily_amount": daily,
        "rate_percent": rate, "contract_amount": aca, "currency": "SAR",
        "note": f"{rate}% of SAR {aca} = SAR {daily}/day",
    })


def _calc_inputs(prov):
    cals = [e for e in prov if e.get("source") == pt.SOURCE_CALCULATOR and e.get("inputs")]
    assert cals, prov
    return {r["name"]: r for r in cals[0]["inputs"]}


@pytest.mark.parametrize("rate,aca", [(0.1, 10_000_000.0), (0.015, 39_098_392.98)])
def test_composed_rate_payload_credits_project_data(rate, aca):
    daily = round(aca * rate / 100.0, 2)
    messages = [
        {"role": "user", "content": "What are the delay damages per day?"},
        {"role": "tool", "content": _flat_payload(rate, aca, daily)},
    ]
    _out, prov = figure_provenance(
        f"The delay damages are SAR {daily} per day.", None, messages, enforce=False)
    rows = _calc_inputs(prov)
    assert rows["rate_percent"]["source"] == "project_data", rows
    assert rows["contract_amount"]["source"] == "project_data", rows
    # unresolved == 0 on the exercised project-rate path
    assert all(r["source"] != "unresolved" for r in rows.values()), rows


@pytest.mark.parametrize("calc", ["excavation_volume", "crane_cost_estimate"])
def test_a_non_delay_payload_is_not_mirrored(calc):
    # Kills the calculation-name guard: a different calc that happens to carry
    # rate_percent/contract_amount must NOT be mirrored as delay_damages_daily.
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": json.dumps(
            {"calculation": calc, "rate_percent": 0.1, "contract_amount": 5000.0})},
    ]
    rt.credit_composed_rate_turns(messages)
    assert not any(
        isinstance(m, dict) and m.get("role") == "assistant"
        and "delay_damages_daily" in str(m.get("tool_calls") or "")
        for m in messages
    ), messages


@pytest.mark.parametrize("rate,aca", [(0.1, 10_000_000.0), (0.2, 5_000_000.0)])
def test_display_answer_is_byte_identical(rate, aca):
    # The mirror only ADDS provenance; the credited figure survives enforcement
    # and the answer text the user sees is unchanged.
    daily = round(aca * rate / 100.0, 2)
    answer = f"The delay damages are SAR {daily} per day."
    messages = [
        {"role": "user", "content": "delay damages per day?"},
        {"role": "tool", "content": _flat_payload(rate, aca, daily)},
    ]
    out, _trail = figure_provenance(answer, None, messages, enforce=True)
    assert out == answer, out
