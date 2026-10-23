"""Cross-registry seam invariants — every routed action name must resolve.

The platform has four hand-maintained action registries in separate files:

  1. ``ACTION_PATTERNS``           app/blocks/smart_orchestrator.py  (router emits)
  2. ``SCOPED_DISPATCH_ALLOWLIST`` app/core/predefined_reasoning.py  (deliverable gate)
  3. ``GENERATIVE_INTENTS``        app/core/action_router.py         (agent gate)
  4. ``ACTION_HINTS``              app/core/action_router.py         (LLM hints)

They are loose strings, so Python cannot catch drift between them. The
2026-07-24 sweep found the orchestrator emitting ``bim_extractor`` while the
container implements ``bim_extract`` — routable, hinted, and permanently
undeliverable, with every per-file test green. These tests pin the seams:
any name present in a routing/gating registry MUST resolve to a real handler
(container action, procedure-engine action, or standalone block).
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_construction_kit


def _resolvable_names() -> set:
    """Every action name that a route can actually land on."""
    from app.containers.construction import ConstructionContainer
    from app.blocks._procedure_routing import PROCEDURE_ROUTING_ADDITIONS

    container_actions = set(ConstructionContainer().get_actions().keys())
    procedure_actions = {name for name, _ in PROCEDURE_ROUTING_ADDITIONS}
    # Actions served by standalone blocks rather than the container.
    block_actions = {"construction_advisor"}
    return container_actions | procedure_actions | block_actions


@requires_construction_kit
class TestActionSeams:
    def test_orchestrator_emits_resolve(self):
        from app.blocks.smart_orchestrator import ACTION_PATTERNS

        emitted = {name for name, _ in ACTION_PATTERNS}
        orphans = emitted - _resolvable_names()
        assert not orphans, (
            f"smart_orchestrator emits action names with no handler: {sorted(orphans)}. "
            "Rename the emit or implement the action — a route to nowhere only ever "
            "produces a chat hint."
        )

    def test_dispatch_allowlist_is_implemented(self):
        from app.core.predefined_reasoning import SCOPED_DISPATCH_ALLOWLIST
        from app.containers.construction import ConstructionContainer

        impl = set(ConstructionContainer().get_actions().keys())
        orphans = set(SCOPED_DISPATCH_ALLOWLIST) - impl
        assert not orphans, (
            f"SCOPED_DISPATCH_ALLOWLIST names unimplemented actions: {sorted(orphans)}"
        )

    def test_generative_intents_resolve(self):
        from app.core.action_router import GENERATIVE_INTENTS

        orphans = set(GENERATIVE_INTENTS) - _resolvable_names()
        assert not orphans, (
            f"GENERATIVE_INTENTS names with no handler: {sorted(orphans)}"
        )

    def test_action_hints_resolve(self):
        from app.core.action_router import ACTION_HINTS

        orphans = set(ACTION_HINTS) - _resolvable_names()
        assert not orphans, (
            f"ACTION_HINTS keys with no handler: {sorted(orphans)}"
        )
