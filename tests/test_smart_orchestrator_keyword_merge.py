"""smart_orchestrator 1.1.0 — keyword-merge regression guards.

Pre-1.1.0 the matcher built ``ACTION_PATTERNS = PROCEDURE_ROUTING_ADDITIONS +
<smart_orchestrator's own list>`` then iterated with a ``seen_actions`` set
that SKIPPED any duplicate action name. PROCEDURE_ROUTING wins the action
name (it's prepended), so smart_orchestrator's keywords for the same action
silently never matched.

Six action names appear in both lists:

  - safety_compliance_audit
  - tender_bid_analysis
  - change_order_impact
  - commissioning_checklist
  - payment_certificate
  - risk_register_auto_populate   (all keywords overlap — no real gap)

These tests pick one previously-dropped keyword per affected action and
assert it routes after the 1.1.0 merge fix. Plus the static descriptor
checks: version bump, "53-action" label, deduplicated merged_patterns
count matches what we expect.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from app.blocks.smart_orchestrator import (
    ACTION_PATTERNS,
    SmartOrchestratorBlock,
)


def _run(message: str) -> dict:
    block = SmartOrchestratorBlock()
    return asyncio.run(
        block.process({"user_message": message})
    )


def _top_action(result: dict) -> Optional[str]:
    matched = result.get("matched_actions") or []
    return matched[0]["action"] if matched else None


# ── Version / label / static shape ──────────────────────────────────────────


def test_version_is_1_1_0():
    assert SmartOrchestratorBlock.version == "1.1.0"


def test_description_says_58_actions():
    # look_ahead + evm_calculate make 58 unique actions; the label must track it.
    assert "58-action" in SmartOrchestratorBlock.description
    assert "56-action" not in SmartOrchestratorBlock.description
    assert "55-action" not in SmartOrchestratorBlock.description
    assert "54-action" not in SmartOrchestratorBlock.description
    assert "53-action" not in SmartOrchestratorBlock.description
    assert "52-action" not in SmartOrchestratorBlock.description
    assert "39-action" not in SmartOrchestratorBlock.description


def test_unique_action_count_is_58():
    """Net unique action names in ACTION_PATTERNS (PROCEDURE + smart_orch lists
    combined) must equal 58: 17 PROCEDURE + 47 smart_orch − 6 collisions.
    (look_ahead + evm_calculate; earlier cde_poll_events.)"""
    seen = set()
    for action, _ in ACTION_PATTERNS:
        seen.add(action)
    assert len(seen) == 58, f"expected 58 unique actions, got {len(seen)}"


# ── Regression guards: previously-dropped keywords must now route ───────────


@pytest.mark.parametrize(
    "phrase,expected_action",
    [
        # safety_compliance_audit — pre-1.1.0 lost: safety, hse, osha, ppe,
        # toolbox, hazard, risk assessment. Confidence = words * 0.2 per
        # match; pick phrases that clear the 0.3 default threshold.
        # "safety hazard" combines two smart_orch-unique single-word
        # keywords (0.2 + 0.2 = 0.4) — neither would match without 1.1.0.
        ("Flag any safety hazard on the lift slab today.", "safety_compliance_audit"),
        ("Schedule the toolbox talk for tomorrow morning.", "safety_compliance_audit"),
        # tender_bid_analysis — pre-1.1.0 lost: tender, bid, proposal,
        # quote comparison, contractor bid. "contractor bid" is the
        # multi-word smart_orch-unique keyword (0.4 alone); pair with
        # "proposal" for redundancy.
        ("Review the contractor bid and tender proposal.", "tender_bid_analysis"),
        ("Compare the contractor bid amounts and tell me who is lowest.",
         "tender_bid_analysis"),
        # change_order_impact — pre-1.1.0 lost: change order, scope change,
        # amendment.
        ("The owner asked for a scope change to add two more rooms.",
         "change_order_impact"),
        # commissioning_checklist — pre-1.1.0 lost: commissioning, handover,
        # startup checklist. ("handover" is owned by handover_management;
        # we test the smart_orch-unique "startup checklist" here.)
        ("Generate the startup checklist for the chiller plant.",
         "commissioning_checklist"),
        # payment_certificate — pre-1.1.0 lost: valuation, progress payment,
        # invoice, certificate.
        ("When is the next progress payment due to the subcontractor?",
         "payment_certificate"),
    ],
)
def test_previously_dropped_keywords_now_route(phrase: str, expected_action: str):
    result = _run(phrase)
    matched_actions = [m["action"] for m in (result.get("matched_actions") or [])]
    assert expected_action in matched_actions, (
        f"expected {expected_action!r} in matched_actions, got {matched_actions}; "
        f"action_queue={result.get('action_queue')}"
    )


def test_safety_top_match_includes_both_prc_and_everyday_keywords():
    """A message that fires BOTH a PROCEDURE keyword (hse audit) and a
    smart_orch-unique keyword (toolbox) must surface both in
    keywords_matched — proving the merged_patterns dict carries the
    full union, not just the first source's list."""
    result = _run("Reviewing the hse audit toolbox talk findings from yesterday.")
    matched = result.get("matched_actions") or []
    safety = next((m for m in matched if m["action"] == "safety_compliance_audit"), None)
    assert safety is not None, f"safety_compliance_audit missing from {matched}"
    kws = set(safety["keywords_matched"])
    assert "hse audit" in kws, f"PROCEDURE keyword 'hse audit' not in {kws}"
    assert "toolbox" in kws, f"smart_orch keyword toolbox not in {kws}"


def test_risk_register_no_keyword_regression():
    """risk_register_auto_populate is the only collision where all
    smart_orch keywords already exist in PROCEDURE — confirm it still
    routes for the canonical phrase so the merge didn't accidentally
    drop a working path."""
    result = _run("Populate the risk register with the new findings.")
    matched_actions = [m["action"] for m in (result.get("matched_actions") or [])]
    assert "risk_register_auto_populate" in matched_actions


def test_a_typed_procedure_code_routes_by_the_live_documents_kind(monkeypatch):
    """No document code is shipped in routing. A code the user types is
    resolved at run time against the project's own document names and
    routes by that document's kind."""
    from app.core import procedure_catalogue

    names = ["XYZ-101_Change Control.pdf", "XYZ-240 Design Review.docx"]
    monkeypatch.setattr(procedure_catalogue, "live_document_names", lambda pid: names if pid == "p_syn" else [])
    block = SmartOrchestratorBlock()
    result = asyncio.run(block.process({
        "user_message": "Summarise XYZ-101 for the site team",
        "session_context": {"project_id": "p_syn"},
    }))
    actions = [m["action"] for m in (result.get("matched_actions") or [])]
    assert "change_order_impact" in actions, actions
    # Without the project's documents the bare code means nothing.
    result = asyncio.run(block.process({"user_message": "Summarise XYZ-101 for the site team"}))
    actions = [m["action"] for m in (result.get("matched_actions") or [])]
    assert "change_order_impact" not in actions, actions


def test_routing_ships_no_document_codes():
    import re

    for _action, keywords in ACTION_PATTERNS:
        for kw in keywords:
            assert not re.fullmatch(r"[A-Z]{2,6}-\d{3,4}[A-Z]?", kw), kw
