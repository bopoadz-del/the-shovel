"""
Test that every formula reference in every manifest resolves to a callable.

The honesty guard: any formula declared in a manifest must be callable,
or CI fails. This prevents "dormant" formulas that look wired but aren't.

Two namespaces:
  - app.lib.construction_formulas.<name>  -> construction_formulas.CALCULATORS[name]
  - app.core.construction_knowledge.<name> -> construction_knowledge.<name>
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# Package imports, not sys.path.insert(app/agents) -- see
# tests/test_no_implicit_relative_imports.py. The old form loaded these
# files a second time under top-level names, so a pydantic class from
# `models` failed isinstance against the same class from
# `app.agents.models`.

from app.agents.models import AgentManifest

MANIFEST_DIR = Path(__file__).parent.parent / "app" / "agents" / "manifests"


def _all_manifests():
    """Yield (path, manifest) for all manifests."""
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        yield path, AgentManifest(**raw)


# -- Binding Resolution (simulated) ---------------------------------------

# Simulated construction_formulas.CALCULATORS -- these are the formulas
# that exist in the real repo. We mock them here to test binding resolution.
MOCK_CALCULATORS = {
    # Materials
    "unit_weight_concrete": lambda: None,
    "modulus_of_elasticity_concrete": lambda: None,
    # Structural
    "beam_deflection_ss_udl": lambda: None,
    "beam_deflection_cantilever_udl": lambda: None,
    # Commercial
    "cost_buildup_concrete": lambda: None,
    "cost_buildup_rebar": lambda: None,
    "cost_buildup_formwork": lambda: None,
    # QAQC
    "concrete_mix_design_sg": lambda: None,
    "concrete_mix_slip_form": lambda: None,
    "formwork_striking_time": lambda: None,
    "concrete_maturity_strength": lambda: None,
    "concrete_thermal_cracking_check": lambda: None,
    "modulus_of_rupture": lambda: None,
    "grout_pressure_calc": lambda: None,
    "fineness_modulus": lambda: None,
    "shear_stress_check": lambda: None,
    # Procurement / shared
    "crane_planning": lambda: None,
}

# Namespace: construction_formulas_additions.ADDITIONAL_CALCULATORS
# (post-live-chat-test additions; mirrors formulas._get_construction_additional_calculators)
MOCK_ADDITIONAL_CALCULATORS = {
    "guardrail_top_rail_height": lambda: None,
    "calculate_interim_payment": lambda **_: None,
}

MOCK_KNOWLEDGE_FNS = {
    "calculate_evm": lambda: None,
    "calculate_payment": lambda: None,
    "score_risk": lambda: None,
    "evaluate_tender": lambda: None,
    "check_review_timeline": lambda: None,
    "next_ncr_status": lambda: None,
    "generate_doc_number": lambda: None,
    "enforce_critical_rules": lambda: None,
    "validate_design_status": lambda: None,
}


def _build_mock_registry():
    """Build a registry matching what formulas.py would build."""
    registry = {}
    for name, fn in MOCK_CALCULATORS.items():
        registry[f"app.lib.construction_formulas.{name}"] = fn
        registry[name] = fn
    for name, fn in MOCK_ADDITIONAL_CALCULATORS.items():
        registry[f"app.lib.construction_formulas_additions.{name}"] = fn
        registry[name] = fn
    for name, fn in MOCK_KNOWLEDGE_FNS.items():
        registry[f"app.core.construction_knowledge.{name}"] = fn
        registry[name] = fn
    return registry


@pytest.fixture(scope="module")
def registry():
    """The REAL binding registry -- the hand-kept mock below went stale the
    first time new hats landed (quantities/safety, 2026-08-15), which is the
    exact stale-twin failure mode test_hats_are_dormant warns about. The mock
    is retained for the shape-documentation tests only."""
    from app.agents.formulas import _build_registry
    return dict(_build_registry())


# -- Tests ----------------------------------------------------------------


class TestFormulaBindings:
    """Every formula.implemented_by must resolve to a callable."""

    def test_all_formula_bindings_resolve(self, registry):
        """Every implemented_by reference in every manifest must be in registry."""
        errors = []
        for path, manifest in _all_manifests():
            for formula in manifest.playbook.formulas:
                if formula.implemented_by:
                    if formula.implemented_by not in registry:
                        errors.append(
                            f"{path.name} :: formula '{formula.id}' -> "
                            f"'{formula.implemented_by}' not resolvable"
                        )

        if errors:
            pytest.fail("\n".join(errors))

    def test_no_duplicate_formula_ids_across_hats(self):
        """Formula IDs should be unique across all manifests (or intentionally shared)."""
        all_formula_ids = []
        for path, manifest in _all_manifests():
            for formula in manifest.playbook.formulas:
                all_formula_ids.append((path.name, formula.id))

        from collections import Counter
        counts = Counter(fid for _, fid in all_formula_ids)
        duplicates = {fid: c for fid, c in counts.items() if c > 1}

        allowed_shared = {"crane_planning", "modulus_of_elasticity_concrete", "guardrail_top_rail_height"}
        unexpected = {k: v for k, v in duplicates.items() if k not in allowed_shared}

        if unexpected:
            msg = "Unexpected duplicate formula IDs:\n"
            for fid, count in unexpected.items():
                locations = [f for p, f in all_formula_ids if f == fid]
                msg += f"  {fid}: appears {count} times\n"
            pytest.fail(msg)

    def test_shared_formulas_are_intentional(self):
        """crane_planning appears in both procurement and planning -- this is by design."""
        manifest_formulas = {}
        for path, manifest in _all_manifests():
            for formula in manifest.playbook.formulas:
                if formula.id not in manifest_formulas:
                    manifest_formulas[formula.id] = []
                manifest_formulas[formula.id].append(manifest.id)

        assert "crane_planning" in manifest_formulas, "crane_planning should exist"
        assert "fork.hat.procurement" in manifest_formulas["crane_planning"], \
            "crane_planning should be in procurement hat"

        assert "modulus_of_elasticity_concrete" in manifest_formulas, \
            "modulus_of_elasticity_concrete should exist"

    def test_all_formulas_have_implemented_by(self):
        """Every formula should declare implemented_by (no orphaned formulas)."""
        errors = []
        for path, manifest in _all_manifests():
            for formula in manifest.playbook.formulas:
                if not formula.implemented_by:
                    errors.append(
                        f"{path.name} :: formula '{formula.id}' has no implemented_by"
                    )
        if errors:
            pytest.fail("\n".join(errors))

    def test_formula_descriptions_are_meaningful(self):
        """Every formula description should be > 10 chars."""
        errors = []
        for path, manifest in _all_manifests():
            for formula in manifest.playbook.formulas:
                if not formula.description or len(formula.description) < 10:
                    errors.append(
                        f"{path.name} :: formula '{formula.id}' description too short"
                    )
        if errors:
            pytest.fail("\n".join(errors))

    def test_formula_domains_are_set(self):
        """Every formula should have a domain declared."""
        errors = []
        for path, manifest in _all_manifests():
            for formula in manifest.playbook.formulas:
                if not formula.domain:
                    errors.append(
                        f"{path.name} :: formula '{formula.id}' has no domain"
                    )
        if errors:
            pytest.fail("\n".join(errors))


class TestBindingRegistry:
    """The binding registry itself must be consistent."""

    def test_registry_has_all_namespaces(self, registry):
        """Registry contains both full paths and bare names."""
        assert "app.lib.construction_formulas.crane_planning" in registry
        assert "app.core.construction_knowledge.evaluate_tender" in registry
        assert "crane_planning" in registry
        assert "evaluate_tender" in registry

    def test_registry_callables_are_callable(self, registry):
        """Every registry entry must be callable."""
        for name, fn in registry.items():
            assert callable(fn), f"Registry entry '{name}' is not callable"

    def test_expected_formulas_in_registry(self, registry):
        """Key formulas expected by manifests must be in registry."""
        expected = {
            "unit_weight_concrete",
            "modulus_of_elasticity_concrete",
            "calculate_evm",
            "cost_buildup_concrete",
            "cost_buildup_rebar",
            "cost_buildup_formwork",
            "calculate_payment",
            "score_risk",
            "generate_doc_number",
            "enforce_critical_rules",
            "next_ncr_status",
            "concrete_mix_design_sg",
            "concrete_maturity_strength",
            "concrete_thermal_cracking_check",
            "modulus_of_rupture",
            "grout_pressure_calc",
            "fineness_modulus",
            "shear_stress_check",
            "validate_design_status",
            "check_review_timeline",
            "crane_planning",
            "evaluate_tender",
        }
        for formula_id in expected:
            assert formula_id in registry, f"Expected formula '{formula_id}' not in registry"
