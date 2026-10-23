"""Live tip 0a95d03: VO ask2 must not call construction / sympy_reasoning.

#662 broadened draft detection and steal-after-predispatch, but ask2 on tip
0a95d03 still ran tools=['construction','sympy_reasoning'] with a
change_order_impact argument error — predispatch miss left the steal set
unapplied. Hard-exclude those tools on VO-draft intent alone.
"""
from __future__ import annotations

from app.agents.runtime import (
    _forced_specific_tool,
    _message_wants_vo_draft,
    _vo_draft_hard_excludes,
)
from app.core.site_vocab import message_wants_vo_draft

LIVE_ASK2 = (
    "variation_order_manager: DRAFT VO-D-002 content (must include drafted clauses/lines, "
    "not merely 'Status: Success'). ADD: additional drainage 45 m @ AED 180/m; "
    "OMIT: omit feature lighting 12 fittings @ AED 2500 each. Show AED totals."
)


def test_live_ask2_wants_vo_draft():
    assert message_wants_vo_draft(LIVE_ASK2)
    assert _message_wants_vo_draft(LIVE_ASK2)


def test_live_ask2_hard_excludes_construction_and_sympy():
    ex = _vo_draft_hard_excludes(LIVE_ASK2)
    assert "construction" in ex
    assert "sympy_reasoning" in ex
    assert "change_order_impact" in ex


def test_live_ask2_forces_variation_order_manager():
    avail = {
        "variation_order_manager",
        "construction",
        "sympy_reasoning",
        "change_order_impact",
        "construction_calc",
    }
    assert (
        _forced_specific_tool([{"role": "user", "content": LIVE_ASK2}], avail)
        == "variation_order_manager"
    )
