"""First cash-flow / S-curve ask must not go predefined generate_wbs.

Live tip 53b8b29 (SO4 #669): cash_flow_forecast ask1 tools=['generate_wbs'];
ask2 PASS with cash_flow_forecast.

#657 forced the underscore tool in the agent loop. #669 steal-guarded
SmartOrchestrator scores so "schedule" / "programme" no longer outrank
cash_flow_forecast. Chat predefined dispatch still uses understand_intent:
ask1 says "Synthetic schedule/costs" + "Produce", the LLM returns
workflow=schedule, and the turn ships generate_wbs (204 activities).

Locks the look_ahead pattern: UNDERSTAND must short-circuit cash-flow /
S-curve before the schedule LLM hop, including the first ask.
"""
from __future__ import annotations

import asyncio

import pytest

from app.core.action_router import message_wants_cash_flow


LIVE_CASH_FLOW_ASK1 = (
    "Use cash_flow_forecast. Synthetic schedule/costs: 6 months starting Oct 2026; "
    "monthly costs AED: 400k, 650k, 900k, 750k, 500k, 300k. "
    "Produce simple monthly cashflow and cumulative."
)

LIVE_CASH_FLOW_ASK2 = (
    "cash_flow_forecast for synthetic 4-month programme: M1 AED 1.2M, M2 1.8M, "
    "M3 1.5M, M4 0.9M. Show monthly outflow and cumulative S-curve values."
)

CASH_FLOW_ASKS = [
    LIVE_CASH_FLOW_ASK1,
    LIVE_CASH_FLOW_ASK2,
    "Build a cash-flow S-curve for the next 6 months",
    "Give me the project cash flow forecast and cumulative drawdown",
]


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("message", CASH_FLOW_ASKS)
def test_understand_intent_does_not_map_cash_flow_to_generate_wbs(message, monkeypatch):
    from app.core import dynamic_reasoning as dr

    called = {"n": 0}

    async def fake_schedule(*a, **k):
        called["n"] += 1
        return {"workflow": "schedule", "mode": "produce", "params": {"target_count": 200}}

    monkeypatch.setattr(dr, "complete_json", fake_schedule)
    out = _run(dr.understand_intent(message))
    assert message_wants_cash_flow(message), message
    assert out["action"] != "generate_wbs", (
        f"understand_intent mapped cash-flow ask1 {message!r} to generate_wbs; got {out}"
    )
    assert out["action"] == "cash_flow_forecast", out
    assert out["workflow"] == "cash_flow_forecast", out
    assert out["deliverable"] is True
    assert called["n"] == 0, (
        "cash-flow / S-curve must short-circuit before the schedule LLM hop"
    )


def test_understand_intent_still_maps_real_wbs_produce(monkeypatch):
    from app.core import dynamic_reasoning as dr

    async def fake_schedule(*a, **k):
        return {"workflow": "schedule", "mode": "produce", "params": {"target_count": 200}}

    monkeypatch.setattr(dr, "complete_json", fake_schedule)
    out = _run(dr.understand_intent("Create L2 schedule with 200 activities for the data center."))
    assert out["action"] == "generate_wbs"
    assert out["deliverable"] is True
    assert not message_wants_cash_flow(
        "Create L2 schedule with 200 activities for the data center."
    )
