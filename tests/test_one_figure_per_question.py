"""An answer states ONE figure for the quantity asked.

Measured on live 589e637, 23 Sep 2026, each question asked 20 times in a fresh
chat. The right figure was almost always there -- and so was a second one:

    R1   10/20   365 days, plus a stray 30 days from a "for completeness"
                 list of other Contract Data entries
    T11   3/20   40 days, plus a calendar-day conversion (46, 47, 48, 50)
    T12  12/20   27,806 MPa, plus the same property at another grade (29,725)
    T2    8/20   2.53 mm, plus span/360 and span/250 deflection limits
    T20   1/10   240 mm, plus 200 mm for a different support condition

A reader cannot tell which number is the answer, so the second one makes the
answer wrong. The rule lives in the agent contract; these tests keep it there
and keep it loaded.
"""
import re
from pathlib import Path

import pytest

CONFIGS = Path("app/agents/configs")
AGENTS = ["project-assistant", "heavy-reasoning"]
RULE_HEAD = "One figure per quantity asked"
# The named shapes the rule must keep naming: each one is a live failure above.
SHAPES = ["for completeness", "calendar-day conversion", "different grade",
          "support condition", "without stating its number"]


def _config_text(name):
    return (CONFIGS / f"{name}.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_agent_contract_carries_the_one_figure_rule(agent):
    text = _config_text(agent).lower()
    assert RULE_HEAD.lower() in text, f"{agent} lost the one-figure rule"
    for shape in SHAPES:
        assert shape in text, f"{agent} no longer names: {shape}"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_is_a_hard_rule_not_a_suggestion(agent):
    text = _config_text(agent)
    sections = []
    for m in re.finditer(r"^## Hard rules\s*$", text, re.MULTILINE):
        nxt = text.find("\n## ", m.end())
        sections.append(text[m.start():nxt if nxt != -1 else len(text)])
    assert sections, f"{agent} has no Hard rules section"
    assert any(RULE_HEAD in s for s in sections), (
        f"{agent}: the rule must live inside the Hard rules section, not in prose "
        f"elsewhere where it reads as advice")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_loaded_agent_prompt_contains_the_rule(agent):
    from app.agents import load_agents
    from app.agents.runtime import AGENT_REGISTRY

    load_agents()
    loaded = AGENT_REGISTRY.get(agent)
    assert loaded is not None, f"{agent} did not load"
    assert RULE_HEAD.lower() in (loaded.system_prompt or "").lower(), (
        f"{agent}'s system prompt reaches the model without the rule")


def test_the_rule_does_not_forbid_the_answers_own_working():
    # The rule must not read as "one number in the whole answer": a derivation
    # (3.2 x 3.2 x 0.75 x 18) and its result are the same quantity, shown once.
    text = _config_text("project-assistant")
    rule = text[text.index(RULE_HEAD):]
    rule = rule[:rule.index("\n-")] if "\n-" in rule else rule
    assert re.search(r"quantity asked", rule)
    assert "for the quantity" in rule.lower() or "quantity asked" in rule.lower()
    assert "no second scenario" in rule.lower()
