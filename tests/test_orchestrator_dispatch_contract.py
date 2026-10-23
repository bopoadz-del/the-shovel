"""The traffic cop's contract must survive edits: dispatch, never fabricate.

Live-fire campaign phase 3, F19, second finding. After PR #343 taught the
agent to route computations past an empty router match, the live gate
produced the right number (24 m3) with a WRONG mechanism: the TIMING log
showed iter=0 calling only ``smart_orchestrator`` and iter=1 going straight
to the final answer -- ``formula_executor_v2`` was never invoked, yet the
reply said "Routed to: formula_executor_v2". The agent did the arithmetic
inline and fabricated the dispatch claim. Correct-but-unauditable is a
wrap-around green: on a computation hard enough to actually need the
calculator, the same behaviour returns a confident wrong number.

The contract now forbids claiming a dispatch that did not happen. Prose
rules are enforced live (the TIMING log must show the calculator call);
this fence pins the rules through the REAL config loader so a later edit
cannot silently drop them.
"""
from __future__ import annotations

import pathlib

import pytest

CONFIG = pathlib.Path("app/agents/configs/smart-orchestrator-agent.md")


@pytest.fixture(scope="module")
def agent():
    from app.agents.runtime import _parse_agent_file
    return _parse_agent_file(CONFIG)


def test_the_config_still_loads_through_the_real_parser(agent):
    assert agent.name == "smart-orchestrator"
    assert "formula_executor_v2" in agent.allowed_blocks
    assert "construction" in agent.allowed_blocks


def test_empty_router_match_routes_computations_to_a_calculator(agent):
    """PR #343: an empty match is not a routing decision -- computations go
    to the calculator the agent already holds, never the document fallback."""
    prompt = agent.system_prompt
    assert "matched_actions: []" in prompt
    assert "construction_calc" in prompt
    assert "intelligent_workflow" in prompt  # named as the dead end to avoid


def test_fabricated_dispatch_is_forbidden(agent):
    """The second F19 finding: 'Routed to:' may only name a block that was
    actually invoked, and Result may only carry what the tool returned."""
    prompt = agent.system_prompt
    assert "Never claim a dispatch you did not make" in prompt
    assert "ACTUALLY invoked" in prompt
    # the specific fabrication observed live is called out verbatim
    assert "fabrication" in prompt


def test_the_reasoning_token_floor_still_covers_this_agent(agent):
    """D1/D2: 1024 max_tokens guaranteed empty answers on K2 reasoning
    models. The configured value must stay at or above the provider floor."""
    assert agent.max_tokens >= 4096
