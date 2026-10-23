"""In a multi-contract corpus, "the contract" has no referent.

Live 5719512, SET4 R8 measured both ways:

    "Which law governs the contract?"        -> "the laws of the Emirate of
                                                 Dubai and the federal laws of
                                                 the United Arab Emirates"  0/2
    "Which law governs contract <number>?"   -> "the law of the Kingdom"     2/2

The platform is right when the question names the contract and wrong when it
does not -- because Master Corpus holds many contracts and "the contract"
picks whichever one retrieval surfaced. Neither answer said which document it
came from, so the operator could not tell a Saudi contract's governing law
from a Dubai template's.

The rule does not invent a referent: it requires the answer to NAME the
contract it used, and to ask when the choice would change the answer.
"""
import re
from pathlib import Path

import pytest

CONFIGS = Path("app/agents/configs")
AGENTS = ["project-assistant", "heavy-reasoning"]
RULE_HEAD = "Name the contract you answered from"


def _text(agent):
    return (CONFIGS / f"{agent}.md").read_text(encoding="utf-8")


def _rule(agent):
    body = _text(agent)
    rest = body[body.index(RULE_HEAD):]
    return rest[:rest.index("\n- ")] if "\n- " in rest else rest


@pytest.mark.parametrize("agent", AGENTS)
def test_both_agents_carry_the_rule(agent):
    assert RULE_HEAD.lower() in _text(agent).lower()


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_is_a_hard_rule(agent):
    body = _text(agent)
    sections = []
    for m in re.finditer(r"^## Hard rules\s*$", body, re.MULTILINE):
        nxt = body.find("\n## ", m.end())
        sections.append(body[m.start():nxt if nxt != -1 else len(body)])
    assert any(RULE_HEAD in s for s in sections), f"{agent}: rule outside Hard rules"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_name_goes_in_the_first_line_not_a_footnote(agent):
    rule = _rule(agent).lower()
    assert "first line" in rule, (
        "a document named only in a citation at the end is not what the reader "
        "checks before acting on the figure")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_covers_the_other_bare_referents(agent):
    rule = _rule(agent).lower()
    for phrase in ("the contract", "the project", "the specification"):
        assert phrase in rule, f"{phrase} is the same ambiguity"


@pytest.mark.parametrize("agent", AGENTS)
def test_asking_is_required_when_the_choice_changes_the_answer(agent):
    rule = _rule(agent).lower()
    assert "ask which is meant" in rule
    assert "instead of choosing silently" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_carries_the_live_evidence(agent):
    rule = _rule(agent)
    assert "Dubai" in rule and "Kingdom" in rule, (
        "the two answers to the same question are the argument for the rule")


def test_the_rule_does_not_ask_the_model_to_guess_a_contract():
    rule = _rule("project-assistant").lower()
    # It must never read as "pick the most likely contract" -- the rule says
    # name what you used, or ask; it never authorises a choice.
    for guessy in ("pick the", "choose the", "assume the", "most likely", "best match"):
        assert guessy not in rule, f"the rule reads as an instruction to guess: {guessy}"
    assert "name the one you used" in rule


@pytest.mark.parametrize("agent", AGENTS)
def test_the_loaded_prompt_carries_the_rule(agent):
    from app.agents import load_agents
    from app.agents.runtime import AGENT_REGISTRY

    load_agents()
    assert RULE_HEAD.lower() in (AGENT_REGISTRY[agent].system_prompt or "").lower()
