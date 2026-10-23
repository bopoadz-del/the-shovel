"""A percentage in the contract, times a figure in the contract, is money.

Live 589e637, SET4 M3, asked 10 times in the sheet's own words ("delay damages
in SAR per calendar day for a single Milestone"): 0/10 gave a SAR figure. Every
run found the clause and stopped there --

    "I can give you the contractual basis, but not a SAR figure - the Contract
     Data expresses delay damages as a percentage of the Contract Price, not
     as a fixed SAR amount."

-- while the same Contract Data states the Accepted Contract Amount. Both
numbers are the contract's own, so 0.015% x SAR 1,754,504,456.25 is arithmetic,
not invention, and the operator asked for it in SAR.

The cost-grounding gate is NOT the cause: fed that answer, it passes it
through untouched (checked locally, 23 Sep). The refusal is the agent's own,
so the rule belongs in the agent contract.
"""
import re
from pathlib import Path

import pytest

CONFIGS = Path("app/agents/configs")
AGENTS = ["project-assistant", "heavy-reasoning"]
RULE_HEAD = "A contractual percentage of a figure you have is money you can state"


def _config_text(name):
    return (CONFIGS / f"{name}.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_contract_carries_the_rule(agent):
    assert RULE_HEAD.lower() in _config_text(agent).lower(), f"{agent} lost the rule"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_is_a_hard_rule(agent):
    text = _config_text(agent)
    sections = []
    for m in re.finditer(r"^## Hard rules\s*$", text, re.MULTILINE):
        nxt = text.find("\n## ", m.end())
        sections.append(text[m.start():nxt if nxt != -1 else len(text)])
    assert any(RULE_HEAD in s for s in sections), (
        f"{agent}: the rule must sit in Hard rules, not read as advice")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_names_the_refusal_it_exists_to_stop(agent):
    text = _config_text(agent).lower()
    rule = text[text.index(RULE_HEAD.lower()):]
    rule = rule[:rule.index("\n- ")] if "\n- " in rule else rule
    # The live refusal sentence, so a reader of the config knows the shape.
    assert "expresses this as a percentage" in rule or "not a fixed amount" in rule
    # And the honest handling when the clause's own base is not available.
    assert "final account may differ" in rule or "base you do have" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_requires_the_base_to_be_named(agent):
    text = _config_text(agent).lower()
    rule = text[text.index(RULE_HEAD.lower()):]
    rule = rule[:rule.index("\n- ")] if "\n- " in rule else rule
    # The naming must be part of the COMPUTE instruction, not only of the
    # fallback sentence further down: a money figure whose base is unstated
    # cannot be checked by the reader.
    compute = rule[:rule.index("both numbers came from the contract")]
    assert "naming the base you used" in compute, compute[:200]
    assert "x the accepted contract amount" in compute, "the worked shape is the example"


def test_the_rule_does_not_licence_inventing_a_base():
    rule = _config_text("project-assistant")
    rule = rule[rule.index(RULE_HEAD):]
    rule = rule[:rule.index("\n- ")] if "\n- " in rule else rule
    assert "in the Contract Data you retrieved" in rule, (
        "the base must be a figure actually retrieved, never supplied by the model")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_loaded_prompt_carries_the_rule(agent):
    from app.agents import load_agents
    from app.agents.runtime import AGENT_REGISTRY

    load_agents()
    loaded = AGENT_REGISTRY.get(agent)
    assert loaded is not None
    assert RULE_HEAD.lower() in (loaded.system_prompt or "").lower()
