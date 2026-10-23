"""Two readings of the input must not become two answers.

Live 138044f, SET4 T11 asked 20 times after the one-figure rule shipped: the
right figure (40 days) was in every answer, and 13 of 20 also carried a second
duration --

    "If 12 m2/day was meant as the whole crew's output, the duration becomes
     400 days - worth confirming which basis your rate is"
    "apply your own efficiency factor (e.g. 80% efficiency -> 50 working days)"

Both are the model being helpful about a genuine ambiguity, and both leave the
operator holding two durations. Asking costs one line and leaves one number in
the answer.
"""
import re
from pathlib import Path

import pytest

CONFIGS = Path("app/agents/configs")
AGENTS = ["project-assistant", "heavy-reasoning"]
RULE_HEAD = "An ambiguous input is a question, not two answers"


def _text(agent):
    return (CONFIGS / f"{agent}.md").read_text(encoding="utf-8")


def _rule(agent):
    text = _text(agent)
    start = text.index(RULE_HEAD)
    rest = text[start:]
    return rest[:rest.index("\n- ")] if "\n- " in rest else rest


@pytest.mark.parametrize("agent", AGENTS)
def test_both_agents_carry_the_rule(agent):
    assert RULE_HEAD.lower() in _text(agent).lower()


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_is_a_hard_rule(agent):
    text = _text(agent)
    sections = []
    for m in re.finditer(r"^## Hard rules\s*$", text, re.MULTILINE):
        nxt = text.find("\n## ", m.end())
        sections.append(text[m.start():nxt if nxt != -1 else len(text)])
    assert any(RULE_HEAD in s for s in sections), f"{agent}: rule outside Hard rules"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_forbids_computing_the_other_reading(agent):
    rule = _rule(agent).lower()
    assert "without computing the other reading" in rule, (
        "the point is not to mention the ambiguity -- it is to leave the second "
        "number uncomputed")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_still_requires_the_operator_to_be_asked(agent):
    rule = _rule(agent).lower()
    assert "ask which was meant" in rule
    assert "one line" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_shows_the_live_failure_and_its_replacement(agent):
    rule = _rule(agent)
    assert "400 days" in rule, "the live wrong shape is the clearest teaching"
    assert "per mason" in rule, "and so is the one-number version that replaces it"


def test_it_does_not_contradict_the_one_figure_rule():
    # Both rules live in Hard rules and point the same way: one number for the
    # asked quantity. If this one ever licensed a second figure they would
    # conflict, and the model would follow whichever it read last.
    rule = _rule("project-assistant").lower()
    assert "two answers" in rule
    assert "one number" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_loaded_prompt_carries_the_rule(agent):
    from app.agents import load_agents
    from app.agents.runtime import AGENT_REGISTRY

    load_agents()
    assert RULE_HEAD.lower() in (AGENT_REGISTRY[agent].system_prompt or "").lower()
