""""The base is not in the retrieved context" is a reason to look, not to stop.

SET4 M3 on live 8cd5f1e, after the percentage rule shipped (#698). The refusal
changed shape but not outcome -- 0/10 still gave a SAR figure:

    "I can give you the rate, but not a SAR figure - the contract expresses
     milestone delay damages as a percentage, and the base it applies to (the
     Contract Price) is not in the retrieved Contract Data. ... To convert
     0.015% into SAR per calendar day I need the Contract Price (or Accepted
     Contract Amount)."

The Accepted Contract Amount IS in the corpus -- it is the Contract Data entry
at Clause 1.1.1, and the agent has a search tool it did not use. One search
closes the gap; refusing while holding an unused tool does not.
"""
import re
from pathlib import Path

import pytest

CONFIGS = Path("app/agents/configs")
AGENTS = ["project-assistant", "heavy-reasoning"]
RULE_HEAD = "A contractual percentage of a figure you have is money you can state"


def _rule(agent):
    text = (CONFIGS / f"{agent}.md").read_text(encoding="utf-8")
    rest = text[text.index(RULE_HEAD):]
    return rest[:rest.index("\n- ")] if "\n- " in rest else rest


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_sends_the_agent_to_look_first(agent):
    rule = _rule(agent).lower()
    assert "search for it once" in rule
    assert "search_project_documents" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_search_is_bounded_to_one(agent):
    rule = _rule(agent)
    assert "SEARCH for it once" in rule, "an unbounded search loop is its own defect"


@pytest.mark.parametrize("agent", AGENTS)
def test_refusal_is_allowed_only_after_looking(agent):
    rule = _rule(agent).lower()
    assert "refuse only" in rule
    assert "comes back without it" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_quotes_the_live_refusal_it_replaces(agent):
    rule = _rule(agent).lower()
    assert "not in the retrieved" in rule, (
        "the config should name the sentence it is overruling")
    assert "reason to look" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_is_still_one_hard_rule(agent):
    text = (CONFIGS / f"{agent}.md").read_text(encoding="utf-8")
    sections = []
    for m in re.finditer(r"^## Hard rules\s*$", text, re.MULTILINE):
        nxt = text.find("\n## ", m.end())
        sections.append(text[m.start():nxt if nxt != -1 else len(text)])
    assert any(RULE_HEAD in s for s in sections)
    # Still one bullet, not a new section that could be read separately.
    assert _rule(agent).count("\n") <= 1, "the amendment must stay inside the bullet"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_loaded_prompt_carries_the_amendment(agent):
    from app.agents import load_agents
    from app.agents.runtime import AGENT_REGISTRY

    load_agents()
    assert "search for it once" in (AGENT_REGISTRY[agent].system_prompt or "").lower()
