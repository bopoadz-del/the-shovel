"""
The Fork -- Agent Manifest Catalog

Loads and validates all manifests at import time.
Fails loudly on schema violation -- this is the honesty guard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agents.models import AgentManifest, Discipline, ManifestKind


_MANIFEST_DIR = Path(__file__).parent / "manifests"
_SCHEMA_PATH = Path(__file__).parent / "manifest.schema.json"

# Runtime cache
_catalog: Optional[Dict[str, AgentManifest]] = None


def _load_all_manifests() -> Dict[str, AgentManifest]:
    """Load every manifest in manifests/ and validate against schema."""
    catalog: Dict[str, AgentManifest] = {}

    manifest_dir = Path(os.environ.get("AGENT_MANIFEST_DIR", _MANIFEST_DIR))
    if not manifest_dir.exists():
        raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")

    for path in sorted(manifest_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)

        try:
            manifest = AgentManifest(**raw)
        except Exception as e:
            raise ValueError(f"Invalid manifest {path.name}: {e}") from e

        if manifest.id in catalog:
            raise ValueError(f"Duplicate manifest id '{manifest.id}' in {path.name}")

        catalog[manifest.id] = manifest

    # Validate hat extends references
    for mid, m in catalog.items():
        if m.is_hat and m.extends and m.extends not in catalog:
            raise ValueError(f"Hat '{mid}' extends unknown base '{m.extends}'")

    # Ensure exactly one base exists
    bases = [m for m in catalog.values() if m.is_base]
    if len(bases) != 1:
        raise ValueError(f"Expected exactly 1 base manifest, found {len(bases)}")

    return catalog


def get_agent_catalog() -> Dict[str, AgentManifest]:
    """Return all loaded manifests. Loads on first call."""
    global _catalog
    if _catalog is None:
        _catalog = _load_all_manifests()
    return _catalog


def get_base() -> AgentManifest:
    """Return the base manifest."""
    for m in get_agent_catalog().values():
        if m.is_base:
            return m
    raise RuntimeError("No base manifest found")


def get_hat(discipline: Discipline) -> Optional[AgentManifest]:
    """Return the hat manifest for a discipline."""
    for m in get_agent_catalog().values():
        if m.is_hat and m.discipline == discipline:
            return m
    return None


def get_hat_by_id(manifest_id: str) -> Optional[AgentManifest]:
    """Return a manifest by its id."""
    return get_agent_catalog().get(manifest_id)


def list_hats() -> List[AgentManifest]:
    """Return all hat manifests."""
    return [m for m in get_agent_catalog().values() if m.is_hat]


def resolve_hat_with_base(hat: AgentManifest) -> AgentManifest:
    """
    Merge a hat onto its base manifest.
    Hat values override base values where both exist.
    Hat allowed_actions are ADDED to base allowed_actions.
    Hat playbook formulas are ADDED to base playbook formulas.
    Hat context_sources are ADDED to base context_sources.
    Hat handoffs are ADDED to base handoffs.
    Hat triggers are ADDED to base triggers.
    """
    base = get_base()
    if hat.extends != base.id:
        base_candidate = get_agent_catalog().get(hat.extends)
        if base_candidate:
            base = base_candidate

    merged_actions = list(dict.fromkeys(base.allowed_actions + hat.allowed_actions))
    merged_formulas = list(base.playbook.formulas)
    existing_formula_ids = {f.id for f in merged_formulas}
    for f in hat.playbook.formulas:
        if f.id not in existing_formula_ids:
            merged_formulas.append(f)

    merged_sources = list(base.context_sources)
    existing_kinds = {s.kind for s in merged_sources}
    for s in hat.context_sources:
        if s.kind not in existing_kinds:
            merged_sources.append(s)

    merged_triggers = list(base.activation.triggers) + list(hat.activation.triggers)
    merged_handoffs = list(base.handoffs or []) + list(hat.handoffs or [])

    import copy
    merged = copy.deepcopy(base)

    merged.id = hat.id
    merged.name = hat.name
    merged.description = hat.description
    merged.kind = hat.kind
    merged.discipline = hat.discipline
    merged.extends = hat.extends
    if hat.system_prompt_guidance:
        merged.system_prompt_guidance = hat.system_prompt_guidance
    merged.allowed_actions = merged_actions
    merged.context_sources = merged_sources
    merged.activation.triggers = merged_triggers
    merged.activation.mode = hat.activation.mode
    merged.activation.default_enabled = hat.activation.default_enabled
    merged.handoffs = merged_handoffs
    if hat.memory:
        merged.memory = hat.memory
    merged.verification = hat.verification
    merged.failure_policy = hat.failure_policy
    merged.playbook.formulas = merged_formulas
    if hat.playbook.flows:
        merged.playbook.flows = list(hat.playbook.flows)

    return merged
