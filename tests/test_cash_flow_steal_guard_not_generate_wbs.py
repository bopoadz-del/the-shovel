"""cash_flow_forecast ask1 must not classify as generate_wbs.

Live tip 50c37f ask1: "Use cash_flow_forecast. Synthetic schedule/costs…"
went predefined workflow=generate_wbs (204 activities). #657 forced the
underscore tool name in `_forced_specific_tool`, but the chat predefined
path uses SmartOrchestrator keyword scores — "schedule" outscored cash
flow because keywords lacked `cash_flow_forecast` / `cash_flow`.
"""
from __future__ import annotations

from app.blocks.smart_orchestrator import SmartOrchestratorBlock
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


def test_message_wants_cash_flow_matches_live_asks():
    assert message_wants_cash_flow(LIVE_CASH_FLOW_ASK1)
    assert message_wants_cash_flow(LIVE_CASH_FLOW_ASK2)
    assert not message_wants_cash_flow("generate a WBS for a 10-floor tower")


def test_orchestrator_promotes_cash_flow_over_generate_wbs():
    block = SmartOrchestratorBlock()
    for q in (LIVE_CASH_FLOW_ASK1, LIVE_CASH_FLOW_ASK2):
        out = block._match_actions(q, None)  # noqa: SLF001 — unit under test
        assert out, q
        assert out[0]["action"] == "cash_flow_forecast", (q, out[:3])
        assert not any(r["action"] == "generate_wbs" for r in out), (q, out[:5])
