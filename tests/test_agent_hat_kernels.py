"""Every agent carries its discipline kernels; the orchestrator carries all.

Operator directives 2026-08-15: "we created few kernels for construction --
if they dont exist, create kernels for each", "each agent should has his own
kernels", "the orchestrator should have it all". The kernel system already
existed (app/agents/manifests/ hat manifests + catalog + activation adapter)
but was UNWIRED -- runtime.py never referenced it, so the manifests' formula
bindings, grounding policy, and discipline guidance influenced nothing.

The wiring is per-agent and load-time: a ``hats:`` frontmatter key binds an
agent to manifest ids (or ``all``); each hat's system_prompt_guidance is
merged into the agent's system prompt under a Discipline Kernels section and
the manifest's formula bindings are validated at load, so a manifest that
points at a missing implementation fails at boot, not mid-conversation.

Two hats were missing and are now real: quantities (take-off / BOQ
production) and safety (HSE engineering checks).
"""
from __future__ import annotations

import pytest

from app.agents import catalog as hat_catalog
from app.agents.formulas import validate_manifest_bindings
from app.agents.runtime import load_agents


@pytest.fixture(scope="module")
def agents():
    return load_agents()


def test_the_catalog_carries_all_eight_discipline_hats():
    hats = {m.id for m in hat_catalog.list_hats()}
    assert hats == {
        "fork.hat.commercial", "fork.hat.contracts", "fork.hat.planning",
        "fork.hat.procurement", "fork.hat.qaqc",
        "fork.hat.quantities", "fork.hat.safety", "fork.hat.design",
    }


def test_every_manifest_binding_resolves_to_a_real_formula():
    for m in hat_catalog.get_agent_catalog().values():
        assert validate_manifest_bindings(m) == [], m.id


def test_discipline_agents_carry_their_own_kernels(agents):
    expected = {
        "quantity-surveyor": ["fork.hat.quantities", "fork.hat.commercial"],
        "safety-officer": ["fork.hat.safety"],
        "contracts-manager": ["fork.hat.contracts"],
        "construction-pm": ["fork.hat.planning"],
    }
    for name, hats in expected.items():
        agent = agents[name]
        assert agent.hats == hats, (name, agent.hats)
        assert "## Discipline kernels" in agent.system_prompt, name


def test_the_orchestrator_has_it_all(agents):
    orch = agents["smart-orchestrator"]
    assert set(orch.hats) == {m.id for m in hat_catalog.list_hats()}
    # every discipline's guidance is present in its prompt
    for m in hat_catalog.list_hats():
        assert m.name in orch.system_prompt, m.id


def test_the_project_assistant_has_it_all(agents):
    """Default UI agent was unhatted while FORK_HATS_ENABLED=1 is live."""
    pa = agents["project-assistant"]
    assert set(pa.hats) == {m.id for m in hat_catalog.list_hats()}
    assert "## Discipline kernels" in pa.system_prompt
    for m in hat_catalog.list_hats():
        assert m.name in pa.system_prompt, m.id


def test_unhatted_agents_are_untouched(agents):
    for name in ("validation", "document-analyst", "heavy-reasoning"):
        assert agents[name].hats == [], name
        assert "## Discipline kernels" not in agents[name].system_prompt


def test_an_unknown_hat_id_fails_loudly(tmp_path):
    from app.agents.runtime import _parse_agent_file
    bad = tmp_path / "bad-agent.md"
    bad.write_text(
        "---\nname: bad-agent\nhats:\n  - fork.hat.nonexistent\n---\nprompt body\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown hat"):
        _parse_agent_file(bad)


def test_the_orchestrator_contract_routes_novel_computations_to_self_coding(agents):
    """'The orchestrator should have it all' includes the code path for
    computations no calculator covers: delegate to the self-coding agent,
    which sandbox-runs generated Python."""
    prompt = agents["smart-orchestrator"].system_prompt
    assert "self-coding" in prompt
    assert agents["smart-orchestrator"].can_delegate is True
    assert "self-coding" in agents, "self-coding agent must exist to delegate to"
