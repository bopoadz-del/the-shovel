"""Live tip 4ab55613: variation_order_manager ask2 must draft VO-D-002, not
steal to sympy_reasoning / change_order_impact.

Exit evidence ask2: tools=['sympy_reasoning','construction']; answer about
change_order_impact argument shape; no drafted ADD/OMIT lines.
"""
from __future__ import annotations

from app.core.site_vocab import message_wants_vo_draft
from app.agents.runtime import _conflicting_tools_after_predispatch


ASK2 = (
    "variation_order_manager: DRAFT VO-D-002 content (must include drafted clauses/lines, "
    "not merely 'Status: Success'). ADD: additional drainage 45 m @ AED 180/m; "
    "OMIT: omit feature lighting 12 fittings @ AED 2500 each. Show AED totals."
)


def test_ask2_message_wants_vo_draft():
    assert message_wants_vo_draft(ASK2)


def test_vo_predispatch_steals_sympy_and_impact():
    steal = _conflicting_tools_after_predispatch("variation_order_manager")
    assert "sympy_reasoning" in steal
    assert "change_order_impact" in steal
