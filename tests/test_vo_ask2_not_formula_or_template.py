"""Live tip db20f40: VO ask2 must not steal to formula/template.

#671 / G2 excluded construction + sympy on VO-draft intent. Ask1 PASSes on
variation_order_manager. Ask2 still ran tools=['formula_executor_v2',
'recommendation_template'] with an insufficient VO draft (no ADD/OMIT lines).
Hard-exclude those tools on draft intent; steal them after VO predispatch.
"""
from __future__ import annotations

from app.agents.runtime import (
    _conflicting_tools_after_predispatch,
    _forced_specific_tool,
    _message_wants_vo_draft,
    _should_handoff_unmatched_calc,
    _vo_draft_hard_excludes,
)
from app.core.site_vocab import message_wants_vo_draft

LIVE_ASK2 = (
    "variation_order_manager: DRAFT VO-D-002 content (must include drafted clauses/lines, "
    "not merely 'Status: Success'). ADD: additional drainage 45 m @ AED 180/m; "
    "OMIT: omit feature lighting 12 fittings @ AED 2500 each. Show AED totals."
)

STEAL_TOOLS = ("formula_executor_v2", "recommendation_template")


def test_live_ask2_wants_vo_draft():
    assert message_wants_vo_draft(LIVE_ASK2)
    assert _message_wants_vo_draft(LIVE_ASK2)


def test_live_ask2_hard_excludes_formula_and_template():
    ex = _vo_draft_hard_excludes(LIVE_ASK2)
    for name in STEAL_TOOLS:
        assert name in ex, name


def test_live_ask2_predispatch_steals_formula_and_template():
    steal = _conflicting_tools_after_predispatch("variation_order_manager")
    for name in STEAL_TOOLS:
        assert name in steal, name


def test_live_ask2_forces_variation_order_manager_not_formula_or_template():
    avail = {
        "variation_order_manager",
        "formula_executor_v2",
        "formula_executor",
        "recommendation_template",
        "construction",
        "construction_calc",
    }
    assert (
        _forced_specific_tool([{"role": "user", "content": LIVE_ASK2}], avail)
        == "variation_order_manager"
    )


def test_live_ask2_does_not_handoff_to_self_coding():
    assert _should_handoff_unmatched_calc(LIVE_ASK2) is False
