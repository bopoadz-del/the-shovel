"""PR #80 — per-action gate-1 threshold for GENERATIVE_INTENTS.

Pre-PR-#80 the matcher used a single 0.3 confidence_threshold for every
action. The operator brief's literal query ``"generate a WBS for a
10-floor tower"`` only hit the bare ``"wbs"`` keyword (1 word, 0.2
score) -- below the global gate, so generate_wbs never showed up in
matched_actions and needs_planning never fired.

PR #80 splits the gate by intent class: actions in GENERATIVE_INTENTS
(generate_wbs, bim_analysis, drawing_qto, ...) clear gate-1 at 0.2;
everything else stays at the historical 0.3 to suppress thin matches.

This test locks in:
  1. The literal "generate a WBS for X" phrasings reach matched_actions
     with action='generate_wbs' (so action_router can decide if they
     route).
  2. A non-generative action (boq_process is NOT in GENERATIVE_INTENTS)
     still requires 0.3 -- the relaxed gate does not leak.
  3. The bare "wbs" by itself still reports as generate_wbs candidate
     (no regression of the matcher's existing surface).

Note: needs_planning has a SEPARATE routing gate at 0.4
(ROUTING_CONFIDENCE_THRESHOLD in action_router.py). 0.2 scores reach
matched_actions but do NOT trigger the heavy-reasoning redirect on
their own -- they're surfaced so callers (e.g. the chat router's hint
path) can use them.
"""
from __future__ import annotations

import asyncio

import pytest

from app.blocks.smart_orchestrator import SmartOrchestratorBlock


def _run(message: str) -> dict:
    block = SmartOrchestratorBlock()
    return asyncio.run(
        block.process({"user_message": message})
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "generate a WBS for a 10-floor tower",
        "Can you create a WBS for the substation?",
        "Build a WBS covering the demolition phase.",
        "Draft a WBS for the early works package.",
        "we need a wbs now",
    ],
)
def test_generative_intent_clears_relaxed_gate(phrase: str):
    """Operator-brief literal phrasings: only the bare "wbs" keyword
    matches (0.2 score). With the per-action gate-1 relaxed to 0.2 for
    GENERATIVE_INTENTS, generate_wbs must surface in matched_actions."""
    result = _run(phrase)
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    assert "generate_wbs" in actions, (
        f"generate_wbs missing from matched_actions for {phrase!r}; "
        f"got actions={actions}, action_queue={result.get('action_queue')}"
    )
    gw = next(m for m in matched if m["action"] == "generate_wbs")
    assert gw["confidence"] >= 0.2, (
        f"{phrase!r} produced confidence={gw['confidence']} -- below the "
        f"relaxed 0.2 gate (impossible if it's in matched_actions)"
    )


def test_non_generative_action_still_requires_0_3():
    """boq_process is NOT in GENERATIVE_INTENTS (BOQ Q&A is RAG-based by
    design). A thin single-word "boq" match (score 0.2) must NOT leak
    through gate-1 -- the relaxed threshold is generative-only."""
    result = _run("boq")
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    # If boq_process IS present, its score must be >= 0.3 (the historical
    # global gate). It will not be -- bare "boq" scores 0.2 only.
    if "boq_process" in actions:
        bp = next(m for m in matched if m["action"] == "boq_process")
        assert bp["confidence"] >= 0.3, (
            f"boq_process surfaced at confidence={bp['confidence']} -- the "
            f"relaxed 0.2 gate must apply ONLY to GENERATIVE_INTENTS"
        )


def test_clean_keyword_query_still_routes_at_high_confidence():
    """Regression guard: the strong multi-keyword query that worked pre-PR-#80
    must still produce a high-confidence generate_wbs match. The gate
    change must not REGRESS scoring for queries that were already routing."""
    result = _run("Create L2 schedule with 200 activities for a 10-floor tower.")
    matched = result.get("matched_actions") or []
    top = matched[0] if matched else None
    assert top is not None and top["action"] == "generate_wbs", (
        f"expected generate_wbs as top match, got matched={matched}"
    )
    assert top["confidence"] >= 0.4, (
        f"clean-keyword query confidence dropped to {top['confidence']} -- "
        f"the gate change introduced a regression"
    )
