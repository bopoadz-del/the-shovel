"""
Test that grounding rules are enforced across all manifests.

The grounding doctrine:
- refuse_unaided_figures = true  (never fabricate numbers)
- cite_required = true            (every claim needs a source)
- standards_advisory = true       (flag when standards not met)
- grounding != "none"             (must require some form of evidence)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Package imports, not sys.path.insert(app/agents) -- see
# tests/test_no_implicit_relative_imports.py. The old form loaded these
# files a second time under top-level names, so a pydantic class from
# `models` failed isinstance against the same class from
# `app.agents.models`.

from app.agents.models import AgentManifest, GroundingPolicy

MANIFEST_DIR = Path(__file__).parent.parent / "app" / "agents" / "manifests"


def _all_manifests():
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        yield path.name, AgentManifest(**raw)


# -- Grounding Rules ------------------------------------------------------


class TestGroundingEnforced:
    """Grounding must be strict across all manifests."""

    @pytest.mark.parametrize("name,manifest", list(_all_manifests()), ids=lambda x: x[0] if isinstance(x, tuple) else "")
    def test_refuse_unaided_figures(self, name, manifest):
        """Every manifest must refuse unaided figures."""
        assert manifest.verification.refuse_unaided_figures, \
            f"'{name}' has refuse_unaided_figures=false"

    @pytest.mark.parametrize("name,manifest", list(_all_manifests()), ids=lambda x: x[0] if isinstance(x, tuple) else "")
    def test_cite_required(self, name, manifest):
        """Every manifest must require citations."""
        assert manifest.verification.cite_required, \
            f"'{name}' has cite_required=false"

    @pytest.mark.parametrize("name,manifest", list(_all_manifests()), ids=lambda x: x[0] if isinstance(x, tuple) else "")
    def test_standards_advisory(self, name, manifest):
        """Every manifest must enable standards advisory."""
        assert manifest.verification.standards_advisory, \
            f"'{name}' has standards_advisory=false"

    @pytest.mark.parametrize("name,manifest", list(_all_manifests()), ids=lambda x: x[0] if isinstance(x, tuple) else "")
    def test_grounding_not_none(self, name, manifest):
        """No manifest may have grounding=none."""
        assert manifest.verification.grounding != GroundingPolicy.NONE, \
            f"'{name}' has grounding=none"

    def test_base_grounding_is_documents_or_formulas(self):
        """Base manifest should use documents_or_formulas grounding."""
        from app.agents.catalog import get_base
        base = get_base()
        assert base.verification.grounding == GroundingPolicy.DOCUMENTS_OR_FORMULAS, \
            f"Base grounding is {base.verification.grounding}, expected documents_or_formulas"

    def test_all_hats_have_stricter_or_equal_grounding(self):
        """Hat grounding should match or exceed base strictness."""
        from app.agents.catalog import get_base, list_hats
        base = get_base()
        base_level = base.verification.grounding

        for hat in list_hats():
            assert hat.verification.grounding in {
                GroundingPolicy.DOCUMENTS_OR_FORMULAS,
                GroundingPolicy.DOCUMENTS_OR_LEARNING_ONLY,
            }, f"Hat '{hat.id}' has unexpected grounding: {hat.verification.grounding}"


# -- Failure Policy -------------------------------------------------------


class TestFailurePolicy:
    """Failure policies should be conservative (ask, don't guess)."""

    @pytest.mark.parametrize("name,manifest", list(_all_manifests()), ids=lambda x: x[0] if isinstance(x, tuple) else "")
    def test_on_formula_error_asks_clarifying(self, name, manifest):
        """Formula errors should ask clarifying questions, not guess."""
        from app.agents.models import MissingContextPolicy
        assert manifest.failure_policy.on_formula_error in {
            MissingContextPolicy.ASK_CLARIFYING,
            MissingContextPolicy.REFUSE,
        }, f"'{name}' on_formula_error too permissive: {manifest.failure_policy.on_formula_error}"

    @pytest.mark.parametrize("name,manifest", list(_all_manifests()), ids=lambda x: x[0] if isinstance(x, tuple) else "")
    def test_on_unverified_claim_flags(self, name, manifest):
        """Unverified claims should be flagged, not silently accepted."""
        assert manifest.failure_policy.on_unverified_claim == "flag_and_continue", \
            f"'{name}' on_unverified_claim should be 'flag_and_continue', got {manifest.failure_policy.on_unverified_claim}"


# -- Memory Safety --------------------------------------------------------


class TestMemorySafety:
    """Memory config must never persist sensitive data."""

    def test_base_never_persist_includes_secrets(self):
        from app.agents.catalog import get_base
        base = get_base()
        assert base.memory is not None, "Base should have memory config"
        assert base.memory.never_persist is not None, "Base should define never_persist"

        sensitive = ["secrets", "llm_raw_text"]
        for item in sensitive:
            assert item in base.memory.never_persist, \
                f"'{item}' should be in never_persist list"

    def test_base_never_persist_includes_cross_tenant(self):
        from app.agents.catalog import get_base
        base = get_base()
        assert "cross_tenant_data" in base.memory.never_persist, \
            "cross_tenant_data should never be persisted"

    def test_base_never_persist_includes_unverified(self):
        from app.agents.catalog import get_base
        base = get_base()
        assert "unverified_claims" in base.memory.never_persist, \
            "unverified_claims should never be persisted"

    def test_base_saves_intent_and_hats(self):
        from app.agents.catalog import get_base
        base = get_base()
        assert "user_intent" in base.memory.save_on_turn_end, \
            "user_intent should be saved"
        assert "activated_hats" in base.memory.save_on_turn_end, \
            "activated_hats should be saved"
