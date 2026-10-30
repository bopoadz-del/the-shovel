"""
The Fork -- Hat Activation Adapter

Thin layer that plugs into runtime.py to enable discipline-hat routing.
Env-flag gated: OFF = no-op (existing pilot unchanged), ON = hats active.

Usage in runtime.py (one-line integration):
    from app.agents.activation import HatActivationAdapter
    adapter = HatActivationAdapter()
    manifest = adapter.select_hat_for_message(user_message)
    if manifest:
        system_prompt = manifest.system_prompt_guidance
        allowed_actions = manifest.allowed_actions
        # ... use manifest to guide response
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from app.agents.catalog import (
    get_agent_catalog,
    get_base,
    list_hats,
    resolve_hat_with_base,
)
from app.agents.models import AgentManifest

# -- Always on (K-T0a) -----------------------------------------------------
# The hat router is part of the product, not an opt-in: FORK_HATS_ENABLED is
# removed. The declared-action gap is zero (tests/test_hats_are_dormant.py
# hard-asserts it), so nothing is left to gate.


def hats_enabled() -> bool:
    """The hat router is always on. Kept as a function for its callers."""
    return True

# Default score threshold for hat activation (configurable)
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("FORK_HAT_SCORE_THRESHOLD", "0.5"))

# Multi-hat mode: if True, can return multiple hats when scores are close
MULTI_HAT_ENABLED = os.environ.get("FORK_MULTI_HAT", "").strip() in ("1", "true", "True", "TRUE")

# Tolerance for considering two scores "close enough" to multi-hat
_MULTI_HAT_TOLERANCE = 0.15


class HatActivationAdapter:
    """
    Adapter that scores incoming messages against all discipline hats
    and returns the best-matching merged manifest (hat + base).

    The router is always on (K-T0a); a message below the score threshold
    still gets None / empty.
    """

    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        multi_hat: bool = MULTI_HAT_ENABLED,
    ) -> None:
        self.score_threshold = score_threshold
        self.multi_hat = multi_hat
        self._catalog: Optional[Dict[str, AgentManifest]] = None
        self._hat_cache: Optional[List[AgentManifest]] = None

    # -- Public API --------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return True if hat routing is active."""
        return hats_enabled()

    def select_hat_for_message(self, message: str) -> Optional[AgentManifest]:
        """
        Select the best hat for a user message and merge with base.
        Returns None if hats disabled or no hat matches above threshold.
        """
        if not hats_enabled():
            return None

        scores = self._score_all_hats(message)
        if not scores:
            return None

        best_hat, best_score = scores[0]
        if best_score < self.score_threshold:
            return None

        # Single hat mode: return best match
        if not self.multi_hat:
            return resolve_hat_with_base(best_hat)

        # Multi-hat mode: if second-best is within tolerance, merge both
        if len(scores) > 1:
            second_hat, second_score = scores[1]
            if second_score > self.score_threshold and (best_score - second_score) <= _MULTI_HAT_TOLERANCE:
                return self._merge_multi_hat(best_hat, second_hat)

        return resolve_hat_with_base(best_hat)

    def score_all_hats(self, message: str) -> Dict[str, float]:
        """
        Return discipline -> score mapping for all hats.
        Useful for debugging and UI display. Returns empty dict if disabled.
        """
        if not hats_enabled():
            return {}

        scores = self._score_all_hats(message)
        return {hat.discipline.value: score for hat, score in scores}

    def get_active_hat_names(self, message: str) -> List[str]:
        """Return list of hat names that would activate for this message."""
        if not hats_enabled():
            return []

        scores = self._score_all_hats(message)
        threshold = self.score_threshold
        return [
            hat.name for hat, score in scores
            if score >= threshold
        ]

    # -- Internal ----------------------------------------------------------

    def _score_all_hats(self, message: str) -> List[Tuple[AgentManifest, float]]:
        """Score all hats against message, return sorted (hat, score) desc."""
        hats = self._get_hats()
        scored: List[Tuple[AgentManifest, float]] = []

        for hat in hats:
            score = hat.score_message(message)
            if score > 0:
                scored.append((hat, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _get_hats(self) -> List[AgentManifest]:
        """Return cached list of hat manifests."""
        if self._hat_cache is None:
            self._hat_cache = list_hats()
        return self._hat_cache

    def _merge_multi_hat(self, primary: AgentManifest, secondary: AgentManifest) -> AgentManifest:
        """
        Merge two hats onto base. Primary hat takes precedence for
        system_prompt_guidance, but actions/formulas/sources accumulate.
        """
        base = get_base()

        # Start with primary hat merged onto base
        merged = resolve_hat_with_base(primary)

        # Accumulate secondary hat's actions
        for action in secondary.allowed_actions:
            if action not in merged.allowed_actions:
                merged.allowed_actions.append(action)

        # Accumulate secondary hat's formulas (avoid duplicates by id)
        existing_ids = {f.id for f in merged.playbook.formulas}
        for f in secondary.playbook.formulas:
            if f.id not in existing_ids:
                merged.playbook.formulas.append(f)

        # Accumulate secondary hat's triggers
        merged.activation.triggers = list(merged.activation.triggers) + list(secondary.activation.triggers)

        # Accumulate secondary hat's handoffs
        if secondary.handoffs:
            merged.handoffs = list(merged.handoffs or []) + list(secondary.handoffs)

        # Accumulate secondary hat's context sources
        existing_kinds = {s.kind for s in merged.context_sources}
        for s in secondary.context_sources:
            if s.kind not in existing_kinds:
                merged.context_sources.append(s)

        # Tag as multi-hat
        merged.name = f"{primary.name} + {secondary.name}"
        merged.id = f"{primary.id}__{secondary.id}"

        return merged


# -- Module-level convenience ---------------------------------------------

_default_adapter: Optional[HatActivationAdapter] = None


def get_adapter() -> HatActivationAdapter:
    """Return singleton adapter instance."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = HatActivationAdapter()
    return _default_adapter


def select_hat(message: str) -> Optional[AgentManifest]:
    """One-shot: select hat for message using default adapter."""
    return get_adapter().select_hat_for_message(message)


def quick_score(message: str) -> Dict[str, float]:
    """One-shot: score all hats for message using default adapter."""
    return get_adapter().score_all_hats(message)
