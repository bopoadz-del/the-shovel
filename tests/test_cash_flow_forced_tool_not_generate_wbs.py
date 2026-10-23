"""cash_flow_forecast asks must force cash_flow_forecast, not generate_wbs.

Live exit 78bd9ca ask1: "Use cash_flow_forecast. Synthetic … monthly costs"
routed to generate_wbs (204 activities) because `_message_wants_cash_flow`
only matched hyphen/space forms (`cash-flow` / `cash flow`), not the
underscore tool name `cash_flow_forecast` that operators paste from the
toolkit. Ask2 with "cash_flow_forecast for synthetic 4-month programme"
sometimes still worked via other paths — lock the forced-tool gate.
"""
from __future__ import annotations

from app.agents.runtime import _forced_specific_tool, _message_wants_cash_flow


CASH_FLOW_ASKS = [
    "Use cash_flow_forecast. Synthetic schedule/costs: 6 months starting Oct 2026; "
    "monthly costs AED: 400k, 650k, 900k, 750k, 500k, 300k. Produce monthly cashflow.",
    "cash_flow_forecast for synthetic 4-month programme: M1 AED 1.2M, M2 1.8M, M3 1.5M, M4 0.9M.",
    "Build a cash-flow S-curve for the next 6 months",
    "Give me the project cash flow forecast and cumulative drawdown",
]


def test_message_wants_cash_flow_matches_underscore_tool_name():
    assert _message_wants_cash_flow("Use cash_flow_forecast on synthetic monthly costs")
    assert _message_wants_cash_flow("cash_flow please")
    assert _message_wants_cash_flow("cash-flow S-curve")
    assert not _message_wants_cash_flow("generate a WBS for a tower")


def test_forced_tool_is_cash_flow_not_generate_wbs():
    available = {"generate_wbs", "cash_flow_forecast", "look_ahead", "construction_calc"}
    for message in CASH_FLOW_ASKS:
        msgs = [{"role": "user", "content": message}]
        assert _forced_specific_tool(msgs, available) == "cash_flow_forecast", message
