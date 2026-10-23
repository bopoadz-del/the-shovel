"""Unit tests for app.core.action_router — orchestrator → LLM domain hints."""

from app.core.action_router import (
    HINT_CONFIDENCE_THRESHOLD,
    best_action,
    hint_for_action,
    hint_for_orchestrator_result,
)


def test_best_action_returns_top_match_with_confidence():
    result = {
        "matched_actions": [
            {"action": "boq_process", "confidence": 0.9, "keywords_matched": ["boq"]},
            {"action": "estimate_costs", "confidence": 0.4, "keywords_matched": ["cost"]},
        ]
    }
    action, conf = best_action(result)
    assert action == "boq_process"
    assert conf == 0.9


def test_best_action_handles_empty_match_list():
    assert best_action({"matched_actions": []}) == (None, 0.0)
    assert best_action({}) == (None, 0.0)


def test_hint_for_known_action_returns_domain_prompt():
    hint = hint_for_action("boq_process")
    assert hint is not None
    assert "Bill of Quantities" in hint


def test_hint_for_unknown_action_returns_none():
    assert hint_for_action("not_a_real_action") is None
    assert hint_for_action("") is None


def test_hint_for_orchestrator_result_below_threshold_returns_none():
    """A weak keyword match shouldn't bias the LLM."""
    weak = {
        "matched_actions": [
            {"action": "boq_process", "confidence": HINT_CONFIDENCE_THRESHOLD - 0.1,
             "keywords_matched": []},
        ]
    }
    assert hint_for_orchestrator_result(weak) is None


def test_hint_for_orchestrator_result_above_threshold_returns_hint():
    strong = {
        "matched_actions": [
            {"action": "spec_analyze", "confidence": HINT_CONFIDENCE_THRESHOLD + 0.1,
             "keywords_matched": ["specification"]},
        ]
    }
    hint = hint_for_orchestrator_result(strong)
    assert hint is not None
    assert "specification" in hint.lower()


def test_hint_folds_cm_prompt_inject_and_follow_ups():
    """Reasoner inject + follow-ups must reach the chat hint path."""
    result = {
        "matched_actions": [
            {"action": "generate_wbs", "confidence": 0.8, "keywords_matched": ["wbs"]},
        ],
        "cm_prompt_inject": "[Workflow Template Match: New Project Setup]",
        "cm_follow_up_tools": ["procurement_list_generator", "cash_flow_forecast"],
        "cm_queue_appended": ["procurement_list_generator"],
        "cm_matched_template": "new_project_setup",
        "cm_workflow_plan": {"template_id": "new_project_setup", "step_count": 5},
    }
    hint = hint_for_orchestrator_result(result)
    assert hint is not None
    assert "Workflow Template Match" in hint
    assert "procurement_list_generator" in hint
    assert "new_project_setup" in hint


def test_hint_returns_none_when_no_matched_actions():
    """SmartOrchestratorBlock returns fallback action_queue when nothing matches
    — but matched_actions is the empty list. The hint should be None then."""
    no_match = {
        "action_queue": ["intelligent_workflow"],
        "matched_actions": [],
        "fallback_agent": "intelligent_workflow",
    }
    assert hint_for_orchestrator_result(no_match) is None
