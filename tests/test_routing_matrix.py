"""W5 — routing matrix contract.

Locks the intent -> path routing documented in docs/routing_matrix.md so a future
edit to GENERATIVE_INTENTS / thresholds can't silently change routing. Complements
the phrasing-level tests (test_smart_orchestrator_*.py) with the decision-function
contract + the ORCHESTRATOR_PREDEFINED noop-when-off guarantee.
"""
import pytest

from app.core import action_router as ar


# ── every generative intent routes to the planning path at the 0.2 gate ────

@pytest.mark.parametrize("action", sorted(ar.GENERATIVE_INTENTS))
def test_generative_intent_routes_to_planning_at_gate(action):
    assert ar.needs_planning(action, 0.2) is True
    # Just below the generative gate -> fast path.
    assert ar.needs_planning(action, 0.19) is False


# ── non-generative actions never take the planning path, even at high conf ─

@pytest.mark.parametrize("action", [
    "boq_process", "spec_analyze", "qa_qc_inspection", "commissioning_checklist",
    "progress_tracker", "safety_compliance_audit", "process_document",
])
def test_non_generative_stays_fast_even_at_high_confidence(action):
    assert action not in ar.GENERATIVE_INTENTS
    assert ar.needs_planning(action, 0.99) is False


def test_none_action_never_plans():
    assert ar.needs_planning(None, 1.0) is False


# ── thresholds are the documented values ───────────────────────────────────

def test_thresholds_match_matrix_doc():
    assert ar.GENERATIVE_ROUTING_THRESHOLD == 0.2
    assert ar.HINT_CONFIDENCE_THRESHOLD == 0.4
    assert ar.ROUTING_CONFIDENCE_THRESHOLD == 0.4


# ── best_action extraction ─────────────────────────────────────────────────

def test_best_action_picks_top_match():
    result = {"matched_actions": [
        {"action": "generate_wbs", "confidence": 0.6, "keywords_matched": ["wbs"]},
        {"action": "estimate_costs", "confidence": 0.3, "keywords_matched": ["cost"]},
    ]}
    assert ar.best_action(result) == ("generate_wbs", 0.6)


def test_best_action_empty_is_none():
    assert ar.best_action({"matched_actions": []}) == (None, 0.0)
    assert ar.best_action({}) == (None, 0.0)


# ── hints: generative + non-generative hint-registered actions resolve ─────

@pytest.mark.parametrize("action", [
    "drawing_qto", "parse_primavera_schedule", "boq_process", "spec_analyze",
])
def test_hint_registered_actions_have_a_hint(action):
    assert ar.hint_for_action(action), f"{action} should have a domain hint"


def test_hint_gated_by_confidence():
    # A hint-registered action below 0.4 yields no hint sentence.
    low = {"matched_actions": [{"action": "boq_process", "confidence": 0.3}]}
    assert ar.hint_for_orchestrator_result(low) is None
    high = {"matched_actions": [{"action": "boq_process", "confidence": 0.5}]}
    assert ar.hint_for_orchestrator_result(high) is not None


# ── ORCHESTRATOR_PREDEFINED reconciliation: noop when off ──────────────────

def test_predefined_flag_default_off_is_noop(monkeypatch):
    from app.routers import chat as chat_router
    monkeypatch.delenv("ORCHESTRATOR_PREDEFINED", raising=False)
    assert chat_router._predefined_enabled() is False
    for v in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", v)
        assert chat_router._predefined_enabled() is False


def test_predefined_flag_on_values(monkeypatch):
    for v in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", v)
        assert chat_predefined_enabled() is True


def chat_predefined_enabled() -> bool:
    from app.routers import chat as chat_router
    return chat_router._predefined_enabled()
