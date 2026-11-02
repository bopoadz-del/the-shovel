"""A list-valued calculator operand (a rate schedule of bands) surfaces as ONE
provenance input row -- {name: "rate schedule", value: "<n> bands", unit: "",
source: project_data} -- instead of being dropped silently by _governing_inputs
(ORDERS_02NOV_0556Z). The composed delay_damages_over_period display payload is
left byte-identical; only provenance gains the row.

The band figures themselves are never credited as answer figures: the schedule
is one row carrying its count, nothing more. Synthetic payload, no client data.
"""
from __future__ import annotations

import json

import pytest

from app.agents.citation_provenance import _governing_inputs, figure_provenance


def _flat_payload(rates: list[float], *, contract_amount: float = 1_000_000.0,
                  days: int = 10) -> dict:
    """The FLAT display payload a rate-composing graft appends (no result dict),
    exactly the shape credit_composed_rate_turns mirrors."""
    amount = round(sum(contract_amount * (r / 100.0) * days for r in rates), 2)
    return {
        "calculation": "delay_damages_over_period",
        "amount": amount,
        "days": days,
        "rates": list(rates),
        "milestones": [],
        "contract_amount": contract_amount,
        "currency": "USD",
        "note": "composed",
    }


def _messages(payload: dict) -> list[dict]:
    return [
        {"role": "user", "content": "What are the combined delay damages?"},
        {"role": "tool", "name": "construction_calc", "tool_call_id": "disp",
         "content": json.dumps(payload)},
    ]


def _credited_inputs(payload: dict) -> tuple[str, list[dict] | None]:
    """Run the provenance builder over the composed payload and return
    (answer_text_out, the delay-damages calc's input rows)."""
    text = f"The combined delay damages are USD {payload['amount']:,.2f}."
    out, entries = figure_provenance(text, None, _messages(payload), enforce=False)
    inputs = next(
        (e.get("inputs") for e in entries
         if e.get("formula") == "delay_damages_over_period" and e.get("inputs")),
        None,
    )
    return out, inputs


@pytest.mark.parametrize("rates,expected_bands", [
    ([0.1, 0.05], "2 bands"),            # two-band schedule
    ([0.1, 0.05, 0.2], "3 bands"),       # three-band schedule -> a different count
])
def test_rate_schedule_list_is_one_input_row_with_its_band_count(rates, expected_bands):
    payload = _flat_payload(rates)
    out, inputs = _credited_inputs(payload)

    assert inputs is not None, "the composed delay-damages figure was not credited"
    rows = {r["name"]: r for r in inputs}

    # the list operand collapses to exactly one row carrying its count
    assert "rate schedule" in rows, rows
    sched = rows["rate schedule"]
    assert sched["value"] == expected_bands
    assert sched["unit"] == ""               # a schedule has no scalar unit
    assert sched["source"] == "project_data"  # stamped at compose time

    # the scalar operand is still its own row, and no raw list leaks as a value
    assert "contract_amount" in rows
    assert all(not isinstance(r["value"], list) for r in inputs)

    # the answer text the user reads is byte-identical -- provenance only
    assert out == f"The combined delay damages are USD {payload['amount']:,.2f}."


@pytest.mark.parametrize("raw,expected", [
    # the "params" shape: scalars kept, a list-of-scalars operand kept (so it
    # can become one row), a bool and a nested dict dropped
    (json.dumps({"params": {"contract_amount": 1_000_000.0, "rates": [0.1, 0.05],
                            "flag": True, "meta": {"x": 1}}}),
     {"contract_amount": 1_000_000.0, "rates": [0.1, 0.05]}),
    # the "input" shape (no "params" key): the obj["input"] fallback must run,
    # or every operand is lost -- a different payload with a single scalar
    (json.dumps({"input": {"unit_rate": 42.0}}),
     {"unit_rate": 42.0}),
])
def test_governing_inputs_keeps_scalars_and_lists_from_either_payload_shape(raw, expected):
    assert _governing_inputs(raw) == expected
