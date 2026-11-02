"""S-12a PR-2 follow-up: the composed-rate grafts beyond delay_damages_daily --
user_priced_takeoff and delay_damages_over_period -- each append a FLAT display
payload whose project-sourced operands (the priced unit rate; the rate schedule
and Accepted Contract Amount) must be credited "project_data", never
"unresolved", while the answer the user sees stays byte-identical. Driven
table-wise by credit_composed_rate_turns over _COMPOSED_RATE_CREDITS. Synthetic
only; full dotted imports so the mutation gate selects this module.
"""
from __future__ import annotations

import json

import pytest

import app.agents.runtime as rt
from app.agents import provenance_trail as pt
from app.agents.citation_provenance import figure_provenance


def _delay_daily():
    rate, aca = 0.1, 10_000_000.0
    daily = round(aca * rate / 100.0, 2)
    return (
        {"calculation": "delay_damages_daily", "daily_amount": daily,
         "rate_percent": rate, "contract_amount": aca, "currency": "SAR",
         "note": f"{rate}% of SAR {aca} = SAR {daily}/day"},
        f"The delay damages are SAR {daily} per day.",
        ("rate_percent", "contract_amount"),
    )


def _priced_takeoff():
    unit_rate, net, vww = 350.0, 100.0, 105.0
    base, total = round(net * unit_rate, 2), round(vww * unit_rate, 2)
    return (
        {"calculation": "user_priced_takeoff", "net_volume_m3": net,
         "volume_with_waste_m3": vww, "unit_rate": unit_rate, "currency": "SAR",
         "waste_percent": 5, "contingency_percent": 0, "base_cost": base,
         "total_cost": total, "note": f"{vww} m3 x SAR {unit_rate} = SAR {total}"},
        f"The priced take-off totals SAR {total}.",
        ("unit_rate",),
    )


def _delay_over_period():
    amount, aca = 50_000.0, 5_000_000.0
    rates = [{"from_day": 0, "to_day": 10, "rate_percent": 0.1}]
    return (
        {"calculation": "delay_damages_over_period", "amount": amount, "days": 10,
         "rates": rates, "milestones": None, "contract_amount": aca,
         "currency": "SAR", "note": f"SAR {amount} over 10 days"},
        f"Delay damages over the period are SAR {amount}.",
        # "rates" is a list of bands (not a scalar input row); the credited
        # project-sourced scalar is the Accepted Contract Amount.
        ("contract_amount",),
    )


CASES = [_delay_daily(), _priced_takeoff(), _delay_over_period()]


def _calc_inputs(prov):
    cals = [e for e in prov if e.get("source") == pt.SOURCE_CALCULATOR and e.get("inputs")]
    assert cals, prov
    return {r["name"]: r for r in cals[0]["inputs"]}


@pytest.mark.parametrize("payload,answer,operands", CASES)
def test_each_composed_rate_payload_credits_its_operands_project_data(payload, answer, operands):
    messages = [
        {"role": "user", "content": "What is the figure?"},
        {"role": "tool", "content": json.dumps(payload)},
    ]
    _out, prov = figure_provenance(answer, None, messages, enforce=False)
    rows = _calc_inputs(prov)
    for op in operands:
        assert rows[op]["source"] == "project_data", (op, rows)
    # the credited turn's inputs are exactly the project operands -> none unresolved
    assert all(r["source"] != "unresolved" for r in rows.values()), rows
    assert all(r["source"] != "derived" for r in rows.values()), rows  # ruling A


@pytest.mark.parametrize("payload,answer,operands", CASES)
def test_display_answer_is_byte_identical(payload, answer, operands):
    # the mirror only ADDS a credited turn; the flat display payload is left in
    # place, so the enforced answer the user sees is unchanged.
    messages = [
        {"role": "user", "content": "figure?"},
        {"role": "tool", "content": json.dumps(payload)},
    ]
    out, _trail = figure_provenance(answer, None, messages, enforce=True)
    assert out == answer, out


@pytest.mark.parametrize("calc", ["excavation_volume", "crane_cost_estimate"])
def test_a_non_composed_payload_is_not_mirrored(calc):
    # Kills the table/calculation guard: a calc NOT in _COMPOSED_RATE_CREDITS
    # that happens to carry the same operand fields must not be mirrored.
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": json.dumps(
            {"calculation": calc, "rate_percent": 0.1, "contract_amount": 5000.0,
             "unit_rate": 350.0, "rates": [{"rate_percent": 0.1}]})},
    ]
    rt.credit_composed_rate_turns(messages)
    assert not any(
        isinstance(m, dict) and m.get("role") == "assistant"
        and calc in str(m.get("tool_calls") or "")
        for m in messages
    ), messages
