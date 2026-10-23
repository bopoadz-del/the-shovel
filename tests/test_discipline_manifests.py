"""
Test discipline manifest loading, validation, and structural integrity.

Covers:
- All manifests load and validate against Pydantic models
- Unique manifest IDs
- Hat extends references exist
- Exactly one base manifest
- Required fields present
- Enum values valid
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Add app/agents to path
# Package imports, not sys.path.insert(app/agents) -- see
# tests/test_no_implicit_relative_imports.py. The old form loaded these
# files a second time under top-level names, so a pydantic class from
# `models` failed isinstance against the same class from
# `app.agents.models`.

from app.agents.models import (
    ActivationMode,
    AgentManifest,
    Discipline,
    GroundingPolicy,
    ManifestKind,
    TriggerType,
)

MANIFEST_DIR = Path(__file__).parent.parent / "app" / "agents" / "manifests"


def _all_manifest_paths():
    """Return all manifest JSON files."""
    return sorted(MANIFEST_DIR.glob("*.json"))


def _load_raw(path: Path):
    """Load raw JSON dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -- Basic Loading --------------------------------------------------------


class TestManifestLoading:
    """Every manifest file must load and validate."""

    @pytest.mark.parametrize("path", _all_manifest_paths(), ids=lambda p: p.name)
    def test_manifest_loads(self, path: Path):
        raw = _load_raw(path)
        manifest = AgentManifest(**raw)
        assert manifest.id
        assert manifest.name
        assert manifest.discipline

    def test_manifest_dir_exists(self):
        assert MANIFEST_DIR.exists(), f"Manifest dir not found: {MANIFEST_DIR}"

    def test_at_least_one_manifest(self):
        paths = list(_all_manifest_paths())
        assert len(paths) >= 2, f"Expected >=2 manifests, found {len(paths)}"


# -- Structural Validation ------------------------------------------------


class TestManifestStructure:
    """Structural rules: IDs, base/hat relationships, required fields."""

    @pytest.fixture(scope="class")
    def all_manifests(self) -> dict[str, AgentManifest]:
        """Load all manifests into a dict keyed by id."""
        catalog = {}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            m = AgentManifest(**raw)
            catalog[m.id] = m
        return catalog

    def test_unique_ids(self, all_manifests: dict):
        """No duplicate manifest IDs across files."""
        ids_seen = {}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            mid = raw["id"]
            assert mid not in ids_seen, f"Duplicate manifest id '{mid}' in {path.name} and {ids_seen[mid]}"
            ids_seen[mid] = path.name

    def test_exactly_one_base(self, all_manifests: dict):
        """Exactly one base manifest must exist."""
        bases = [m for m in all_manifests.values() if m.is_base]
        assert len(bases) == 1, f"Expected exactly 1 base, found {len(bases)}: {[b.id for b in bases]}"

    def test_base_is_fork_base(self, all_manifests: dict):
        """Base manifest must be fork.base."""
        base = [m for m in all_manifests.values() if m.is_base][0]
        assert base.id == "fork.base"

    def test_hats_extend_fork_base(self, all_manifests: dict):
        """All hats must extend fork.base (or a valid existing manifest)."""
        for mid, m in all_manifests.items():
            if m.is_hat:
                assert m.extends, f"Hat '{mid}' missing 'extends'"
                assert m.extends in all_manifests, f"Hat '{mid}' extends unknown '{m.extends}'"

    def test_hat_kinds_are_correct(self, all_manifests: dict):
        """Base kind=base, all others kind=hat."""
        for mid, m in all_manifests.items():
            if m.id == "fork.base":
                assert m.is_base, f"fork.base should be kind=base"
            else:
                assert m.is_hat, f"'{mid}' should be kind=hat"

    def test_all_hats_have_discipline(self, all_manifests: dict):
        """Every hat has a non-base discipline."""
        for mid, m in all_manifests.items():
            if m.is_hat:
                assert m.discipline != Discipline.BASE, f"Hat '{mid}' must have non-base discipline"

    def test_all_hats_have_system_prompt(self, all_manifests: dict):
        """Every hat has system prompt guidance."""
        for mid, m in all_manifests.items():
            if m.is_hat:
                assert m.system_prompt_guidance, f"Hat '{mid}' missing system_prompt_guidance"
                assert len(m.system_prompt_guidance) > 20, f"Hat '{mid}' system prompt too short"


# -- Enum Validation ------------------------------------------------------


class TestEnumValues:
    """All enum values in manifests must be valid."""

    def test_all_disciplines_valid(self):
        valid = {d.value for d in Discipline}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            d = raw.get("discipline")
            assert d in valid, f"Invalid discipline '{d}' in {path.name}. Valid: {valid}"

    def test_all_kinds_valid(self):
        valid = {k.value for k in ManifestKind}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            k = raw.get("kind")
            assert k in valid, f"Invalid kind '{k}' in {path.name}"

    def test_all_activation_modes_valid(self):
        valid = {m.value for m in ActivationMode}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            mode = raw.get("activation", {}).get("mode")
            if mode:
                assert mode in valid, f"Invalid activation mode '{mode}' in {path.name}"

    def test_all_trigger_types_valid(self):
        valid = {t.value for t in TriggerType}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            for trigger in raw.get("activation", {}).get("triggers", []):
                ttype = trigger.get("type")
                assert ttype in valid, f"Invalid trigger type '{ttype}' in {path.name}"

    def test_all_grounding_policies_valid(self):
        valid = {g.value for g in GroundingPolicy}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            g = raw.get("verification", {}).get("grounding")
            if g:
                assert g in valid, f"Invalid grounding '{g}' in {path.name}"

    def test_all_formula_tiers_valid(self):
        valid = {"foundational", "operational", "strategic"}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            for formula in raw.get("playbook", {}).get("formulas", []):
                tier = formula.get("tier")
                if tier:
                    assert tier in valid, f"Invalid tier '{tier}' in {path.name}"


# -- Content Validation ---------------------------------------------------


class TestManifestContent:
    """Content-specific checks for completeness."""

    @pytest.fixture(scope="class")
    def all_manifests(self) -> dict[str, AgentManifest]:
        catalog = {}
        for path in _all_manifest_paths():
            raw = _load_raw(path)
            m = AgentManifest(**raw)
            catalog[m.id] = m
        return catalog

    def test_base_has_memory_config(self, all_manifests: dict):
        base = all_manifests["fork.base"]
        assert base.memory is not None, "Base manifest should have memory config"
        assert base.memory.never_persist, "Base should define never_persist items"

    def test_all_hats_have_triggers(self, all_manifests: dict):
        for mid, m in all_manifests.items():
            if m.is_hat:
                assert len(m.activation.triggers) > 0, f"Hat '{mid}' has no triggers"

    def test_all_hats_have_formulas(self, all_manifests: dict):
        for mid, m in all_manifests.items():
            if m.is_hat:
                assert len(m.playbook.formulas) > 0, f"Hat '{mid}' has no formulas"

    def test_all_hats_have_context_sources(self, all_manifests: dict):
        for mid, m in all_manifests.items():
            if m.is_hat:
                assert len(m.context_sources) > 0, f"Hat '{mid}' has no context_sources"

    def test_all_hats_have_allowed_actions(self, all_manifests: dict):
        for mid, m in all_manifests.items():
            if m.is_hat:
                assert len(m.allowed_actions) > 0, f"Hat '{mid}' has no allowed_actions"

    def test_handoffs_have_valid_targets(self, all_manifests: dict):
        """Handoff targets must be valid discipline names."""
        valid_disciplines = {d.value for d in Discipline}
        for mid, m in all_manifests.items():
            if m.handoffs:
                for h in m.handoffs:
                    assert h.to in valid_disciplines, (
                        f"Hat '{mid}' handoffs to invalid discipline '{h.to}'. "
                        f"Valid: {valid_disciplines}"
                    )

    def test_expected_hats_exist(self, all_manifests: dict):
        """All 5 expected hats must be present."""
        expected = {"planning", "commercial", "contracts", "qaqc", "procurement"}
        found = {
            m.discipline.value for m in all_manifests.values()
            if m.is_hat
        }
        missing = expected - found
        assert not missing, f"Missing hats: {missing}. Found: {found}"

    def test_formula_ids_unique_within_manifest(self, all_manifests: dict):
        """No duplicate formula IDs within a single manifest."""
        for mid, m in all_manifests.items():
            ids = [f.id for f in m.playbook.formulas]
            assert len(ids) == len(set(ids)), f"Duplicate formula IDs in '{mid}': {ids}"

    def test_trigger_scores_in_valid_range(self, all_manifests: dict):
        """All trigger scores between 0 and 1."""
        for mid, m in all_manifests.items():
            for t in m.activation.triggers:
                assert 0 <= t.score <= 1, (
                    f"Trigger '{t.value}' score {t.score} out of range in '{mid}'"
                )

    def test_context_source_weights_in_valid_range(self, all_manifests: dict):
        """All context source weights between 0 and 1."""
        for mid, m in all_manifests.items():
            for cs in m.context_sources:
                assert 0 <= cs.weight <= 1, (
                    f"Context source weight {cs.weight} out of range in '{mid}'"
                )

    def test_expected_hats_count(self, all_manifests: dict):
        """Should have exactly 8 hats + 1 base = 9 manifests (quantities and safety added 2026-08-15; design added 2026-10-08, F-DRIVER Phase A)."""
        hats = [m for m in all_manifests.values() if m.is_hat]
        assert len(hats) == 8, f"Expected 8 hats, found {len(hats)}: {[h.id for h in hats]}"
