"""Live tip 9942b8e: VO ask2 must not steal to validation/delegate.

#681 / SO6 excluded formula_executor_v2 + recommendation_template on
VO-draft intent. Ask1 PASSes on variation_order_manager with a draft.
Ask2 still ran tools=['validation_pipeline','delegate_to_agent'] with a
status-only answer / "construction container is not a delegate target"
and no drafted ADD/OMIT lines. VO×2 flakes on ask2 (r1a2 sometimes
PASS, r2a2 FAIL).

Hard-exclude those tools on draft intent; steal them after VO
predispatch; keep the prior formula/template excludes. Second and later
VO draft asks must stay on variation_order_manager.
"""
from __future__ import annotations

from app.agents.runtime import (
    _conflicting_tools_after_predispatch,
    _forced_specific_tool,
    _message_wants_vo_draft,
    _vo_draft_hard_excludes,
)
from app.core.site_vocab import message_wants_vo_draft

LIVE_ASK1 = (
    "variation_order_manager: DRAFT a priced variation order. "
    "ADD: extra blockwork 20 m2 @ AED 120/m2; "
    "OMIT: omit plaster 8 m2 @ AED 40/m2. Show AED totals."
)

LIVE_ASK2 = (
    "variation_order_manager: DRAFT VO-D-002 content (must include drafted clauses/lines, "
    "not merely 'Status: Success'). ADD: additional drainage 45 m @ AED 180/m; "
    "OMIT: omit feature lighting 12 fittings @ AED 2500 each. Show AED totals."
)

STEAL_TOOLS = ("validation_pipeline", "delegate_to_agent")
PRIOR_EXCLUDES = ("formula_executor_v2", "recommendation_template")

ASK2_AVAIL = {
    "variation_order_manager",
    "validation_pipeline",
    "delegate_to_agent",
    "formula_executor_v2",
    "formula_executor",
    "recommendation_template",
    "construction",
    "construction_calc",
}


def test_live_ask2_wants_vo_draft():
    assert message_wants_vo_draft(LIVE_ASK2)
    assert _message_wants_vo_draft(LIVE_ASK2)


def test_live_ask2_hard_excludes_validation_and_delegate():
    ex = _vo_draft_hard_excludes(LIVE_ASK2)
    for name in STEAL_TOOLS:
        assert name in ex, name
    for name in PRIOR_EXCLUDES:
        assert name in ex, name


def test_live_ask2_predispatch_steals_validation_and_delegate():
    steal = _conflicting_tools_after_predispatch("variation_order_manager")
    for name in STEAL_TOOLS:
        assert name in steal, name
    for name in PRIOR_EXCLUDES:
        assert name in steal, name


def test_live_ask2_forces_variation_order_manager_not_validation_or_delegate():
    assert (
        _forced_specific_tool([{"role": "user", "content": LIVE_ASK2}], ASK2_AVAIL)
        == "variation_order_manager"
    )


def test_second_and_later_vo_draft_asks_stay_on_variation_order_manager():
    """Ask1 already drafted; ask2 must not flip to validation/delegate."""
    history = [
        {"role": "user", "content": LIVE_ASK1},
        {
            "role": "assistant",
            "content": "Status: Success. Drafted VO-D-001 with ADD/OMIT lines.",
        },
        {"role": "user", "content": LIVE_ASK2},
    ]
    assert _message_wants_vo_draft(LIVE_ASK2)
    assert (
        _forced_specific_tool(history, ASK2_AVAIL) == "variation_order_manager"
    )
    later = _vo_draft_hard_excludes(LIVE_ASK2)
    for name in STEAL_TOOLS:
        assert name in later, name
