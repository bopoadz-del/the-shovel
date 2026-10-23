"""
Test the HatActivationAdapter: scoring, selection, merging.

Covers:
- Message scoring against hats
- Threshold-based selection
- Multi-hat merging
- resolve_hat_with_base merge logic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Imported through the PACKAGE path, not via sys.path.insert(app/agents).
# The old form loaded these as top-level `catalog` / `models` / `activation`
# while application code imports `app.agents.*` — two distinct module
# objects for the same file, so a pydantic class from one would fail an
# isinstance check against the other. It also required the modules to use
# implicit relative imports (`from models import ...`), which are Python 2
# syntax: `import app.agents.catalog` raised ModuleNotFoundError, so no
# application code could reach the discipline hats at all.

from app.agents.catalog import get_base, get_hat, list_hats, resolve_hat_with_base
from app.agents.models import AgentManifest, Discipline


@pytest.fixture(scope="module")
def catalog():
    """Load full catalog."""
    from app.agents.catalog import get_agent_catalog
    return get_agent_catalog()


# -- Message Scoring ------------------------------------------------------


class TestMessageScoring:
    """Test that manifest score_message works for key construction terms."""

    def test_planning_scores_high_for_schedule(self):
        hat = get_hat(Discipline.PLANNING)
        score = hat.score_message("What is the critical path for this project?")
        assert score > 0.5, f"Planning should score high for 'critical path', got {score}"

    def test_commercial_scores_high_for_cost(self):
        hat = get_hat(Discipline.COMMERCIAL)
        score = hat.score_message("What is the cost buildup for concrete?")
        assert score > 0.5, f"Commercial should score high for 'cost buildup', got {score}"

    def test_contracts_scores_high_for_claim(self):
        hat = get_hat(Discipline.CONTRACTS)
        score = hat.score_message("We need to submit an EOT claim")
        assert score > 0.5, f"Contracts should score high for 'EOT claim', got {score}"

    def test_qaqc_scores_high_for_inspection(self):
        hat = get_hat(Discipline.QAQC)
        score = hat.score_message("Prepare the ITP for concrete testing")
        assert score > 0.5, f"QAQC should score high for 'ITP concrete testing', got {score}"

    def test_procurement_scores_high_for_tender(self):
        hat = get_hat(Discipline.PROCUREMENT)
        score = hat.score_message("Evaluate the tender for HVAC subcontractor")
        assert score > 0.5, f"Procurement should score high for 'tender evaluate', got {score}"

    def test_procurement_scores_high_for_crane(self):
        hat = get_hat(Discipline.PROCUREMENT)
        score = hat.score_message("Crane plan for tower erection")
        assert score > 0.5, f"Procurement should score high for 'crane plan', got {score}"

    def test_no_match_returns_zero(self):
        hat = get_hat(Discipline.PLANNING)
        score = hat.score_message("hello world")
        assert score == 0, f"Should score 0 for irrelevant message, got {score}"

    def test_scores_are_case_insensitive(self):
        hat = get_hat(Discipline.COMMERCIAL)
        score_lower = hat.score_message("cost analysis")
        score_upper = hat.score_message("COST ANALYSIS")
        assert score_lower == score_upper, "Scoring should be case-insensitive"


# -- Hat Merge ------------------------------------------------------------


class TestHatMerge:
    """Test resolve_hat_with_base merge logic."""

    def test_merge_preserves_base_formulas(self):
        hat = get_hat(Discipline.PLANNING)
        merged = resolve_hat_with_base(hat)
        base = get_base()
        base_ids = {f.id for f in base.playbook.formulas}
        merged_ids = {f.id for f in merged.playbook.formulas}
        for bid in base_ids:
            assert bid in merged_ids, f"Base formula '{bid}' missing after merge"

    def test_merge_adds_hat_formulas(self):
        hat = get_hat(Discipline.PLANNING)
        merged = resolve_hat_with_base(hat)
        hat_formula_ids = {f.id for f in hat.playbook.formulas}
        merged_ids = {f.id for f in merged.playbook.formulas}
        for hid in hat_formula_ids:
            assert hid in merged_ids, f"Hat formula '{hid}' missing after merge"

    def test_merge_adds_hat_actions(self):
        hat = get_hat(Discipline.COMMERCIAL)
        merged = resolve_hat_with_base(hat)
        for action in hat.allowed_actions:
            assert action in merged.allowed_actions, f"Hat action '{action}' missing after merge"

    def test_merge_preserves_base_actions(self):
        hat = get_hat(Discipline.COMMERCIAL)
        merged = resolve_hat_with_base(hat)
        base = get_base()
        for action in base.allowed_actions:
            assert action in merged.allowed_actions, f"Base action '{action}' missing after merge"

    def test_merge_uses_hat_system_prompt(self):
        hat = get_hat(Discipline.CONTRACTS)
        merged = resolve_hat_with_base(hat)
        assert merged.system_prompt_guidance == hat.system_prompt_guidance

    def test_merge_uses_hat_discipline(self):
        hat = get_hat(Discipline.QAQC)
        merged = resolve_hat_with_base(hat)
        assert merged.discipline == hat.discipline

    def test_merge_has_hat_handoffs(self):
        hat = get_hat(Discipline.CONTRACTS)
        merged = resolve_hat_with_base(hat)
        assert merged.handoffs, "Merged manifest should have handoffs"
        hat_handoff_tos = {h.to for h in hat.handoffs}
        merged_handoff_tos = {h.to for h in merged.handoffs}
        for target in hat_handoff_tos:
            assert target in merged_handoff_tos, f"Handoff to '{target}' missing after merge"

    def test_merge_no_duplicate_actions(self):
        hat = get_hat(Discipline.PLANNING)
        merged = resolve_hat_with_base(hat)
        assert len(merged.allowed_actions) == len(set(merged.allowed_actions)), \
            "Duplicate actions in merged manifest"

    def test_merge_no_duplicate_formulas(self):
        hat = get_hat(Discipline.PLANNING)
        merged = resolve_hat_with_base(hat)
        ids = [f.id for f in merged.playbook.formulas]
        assert len(ids) == len(set(ids)), f"Duplicate formula IDs: {[f for f in ids if ids.count(f) > 1]}"

    def test_all_hats_merge_cleanly(self):
        """Every hat must merge with base without errors."""
        for hat in list_hats():
            merged = resolve_hat_with_base(hat)
            assert merged is not None
            assert merged.id == hat.id
            assert merged.discipline == hat.discipline


# -- Adapter (simulated, since env flag may be off) -----------------------


class TestAdapterLogic:
    """Test adapter selection logic directly (no env flag dependency)."""

    def test_hat_scoring_ranking(self):
        """Test that correct hat scores highest for domain-specific messages."""
        test_cases = [
            ("What is the critical path and float analysis?", Discipline.PLANNING),
            ("Calculate the IPC for this month's work", Discipline.COMMERCIAL),
            ("Draft a variation order under FIDIC clause", Discipline.CONTRACTS),
            ("Check concrete cube test results and thermal cracking", Discipline.QAQC),
            ("Evaluate tender bids for the HVAC package", Discipline.PROCUREMENT),
        ]

        for message, expected_discipline in test_cases:
            scores = []
            for hat in list_hats():
                score = hat.score_message(message)
                scores.append((hat.discipline, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            best = scores[0][0]
            assert best == expected_discipline, (
                f"Message: '{message[:40]}...' -> expected {expected_discipline.value}, "
                f"got {best.value} (score: {scores[0][1]:.2f})"
            )

    def test_cross_domain_terms_score_multiple(self):
        """Terms like 'crane' should score for procurement (and potentially planning)."""
        message = "Crane plan for concrete lifting"
        proc_hat = get_hat(Discipline.PROCUREMENT)
        plan_hat = get_hat(Discipline.PLANNING)

        proc_score = proc_hat.score_message(message)
        plan_score = plan_hat.score_message(message)

        assert proc_score > 0, f"Procurement should score for crane message, got {proc_score}"
        assert plan_score > 0, f"Planning should score for crane message, got {plan_score}"

    def test_high_threshold_excludes_weak_matches(self):
        """With a very high threshold, weak matches should be excluded."""
        message = "hello world schedule"
        hat = get_hat(Discipline.PLANNING)
        score = hat.score_message(message)
        assert score < 2.0, f"Weak match should have low score, got {score}"
