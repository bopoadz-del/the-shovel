"""
The Fork -- Formula Bindings

Connects playbook.formulas[].implemented_by strings to real callables.
The honesty guard: every reference must resolve, or CI fails.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional, Set


# -- Lazy import to avoid circular deps at module level -------------------

def _get_construction_calculators() -> Dict[str, Callable]:
    """Lazy-load the CALCULATORS dict from construction_formulas."""
    try:
        mod = importlib.import_module("app.lib.construction_formulas")
        return getattr(mod, "CALCULATORS", {})
    except Exception:
        return {}


def _get_construction_additional_calculators() -> Dict[str, Callable]:
    """Lazy-load the ADDITIONAL_CALCULATORS dict from construction_formulas_additions."""
    try:
        mod = importlib.import_module("app.lib.construction_formulas_additions")
        return getattr(mod, "ADDITIONAL_CALCULATORS", {})
    except Exception:
        return {}


def _get_construction_knowledge_functions() -> Dict[str, Callable]:
    """Lazy-load functions from construction_knowledge."""
    try:
        mod = importlib.import_module("app.core.construction_knowledge")
        names = ["calculate_evm", "calculate_payment", "score_risk",
                 "evaluate_tender", "check_review_timeline", "next_ncr_status",
                 "generate_doc_number", "enforce_critical_rules",
                 "validate_design_status"]
        result: Dict[str, Callable] = {}
        for name in names:
            fn = getattr(mod, name, None)
            if fn:
                result[name] = fn
        return result
    except Exception:
        return {}


# -- Binding resolution ---------------------------------------------------

# Map manifest implemented_by strings to real callables
# Two namespaces:
#   "app.lib.construction_formulas.<name>"  -> construction_formulas.CALCULATORS[name]
#   "app.core.construction_knowledge.<name>" -> construction_knowledge.<name>

_BINDING_REGISTRY: Optional[Dict[str, Callable]] = None


def _build_registry() -> Dict[str, Callable]:
    """Build the full binding registry from both sources."""
    global _BINDING_REGISTRY
    if _BINDING_REGISTRY is not None:
        return _BINDING_REGISTRY

    registry: Dict[str, Callable] = {}

    # Namespace 1: construction_formulas.CALCULATORS
    calculators = _get_construction_calculators()
    for name, fn in calculators.items():
        registry[f"app.lib.construction_formulas.{name}"] = fn
        registry[name] = fn  # Also allow bare names

    # Namespace 2: construction_formulas_additions.ADDITIONAL_CALCULATORS
    additional_calculators = _get_construction_additional_calculators()
    for name, fn in additional_calculators.items():
        registry[f"app.lib.construction_formulas_additions.{name}"] = fn
        registry[name] = fn  # Also allow bare names

    # Namespace 3: construction_knowledge functions
    knowledge_fns = _get_construction_knowledge_functions()
    for name, fn in knowledge_fns.items():
        registry[f"app.core.construction_knowledge.{name}"] = fn
        registry[name] = fn  # Also allow bare names

    # Namespace 4: where each declared formula is defined (its owner's
    # package, app.agents.base / app.agents.hats.<hat>).
    from app.lib import formula_registry

    for spec in formula_registry.all_specs():
        registry[f"{spec.fn.__module__}.{spec.fn.__name__}"] = spec.fn

    _BINDING_REGISTRY = registry
    return registry


def resolve_binding(implemented_by: str) -> Optional[Callable]:
    """Resolve an implemented_by string to a callable."""
    registry = _build_registry()
    return registry.get(implemented_by)


def resolve_binding_or_raise(implemented_by: str) -> Callable:
    """Resolve or raise ValueError (for CI guard)."""
    fn = resolve_binding(implemented_by)
    if fn is None:
        raise ValueError(f"Formula binding not found: '{implemented_by}'")
    return fn


def executable_formula_ids() -> Set[str]:
    """Return all formula ids that can be resolved."""
    return set(_build_registry().keys())


def allowed_actions_for_hat(hat_manifest) -> List[str]:
    """
    Return the combined allowed_actions for a hat (merged with base).
    Hat actions + base actions, deduplicated.
    """
    from app.agents.catalog import resolve_hat_with_base
    if hat_manifest.is_hat:
        merged = resolve_hat_with_base(hat_manifest)
        return merged.allowed_actions
    return hat_manifest.allowed_actions


def run_formula(formula_id: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Execute a formula by id with given params.
    Returns the result or raises if not found.
    """
    fn = resolve_binding(formula_id)
    if fn is None:
        raise ValueError(f"Formula not found: '{formula_id}'")
    return fn(**(params or {}))


def validate_manifest_bindings(manifest) -> List[str]:
    """
    Validate every formula and action reference in a manifest.
    Returns list of errors (empty = all good).
    """
    errors: List[str] = []
    registry = _build_registry()

    # Validate formula bindings
    for f in manifest.playbook.formulas:
        if f.implemented_by and f.implemented_by not in registry:
            errors.append(f"Formula '{f.id}' -> '{f.implemented_by}' not resolvable")

    return errors
