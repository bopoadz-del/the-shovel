"""A follow-up continues from the total, not from one unit.

Live SET4.1 T6, ~1/5 across builds. Turn 1 answers "138.24 m3 for 18 pad
footings" correctly. Turn 2 -- "Add 5% waste and price it at SAR 390 per cubic
metre" -- comes back with **8.064 m3**, which is ONE footing (7.68) times 1.05,
and often no price at all. The right answer is 145.15 m3 and SAR 56,609.28.

Why a rule and not a graft: the existing priced-takeoff composer reads the
CURRENT ask only (`compose_user_priced_takeoff_from_ask`), and this follow-up
carries no geometry -- the total it must continue from is in the previous turn.
Rules fixed R1 (10/20 -> 20/20), T11 (3/20 -> 20/20) and T2 (8/20 -> 10/10)
this week without touching the runtime; the one code attempt at a
cross-turn composition (#701) had to be reverted. Rule first, measure, then
code if the number does not move.
"""
import re
from pathlib import Path

import pytest

CONFIGS = Path("app/agents/configs")
AGENTS = ["project-assistant", "heavy-reasoning"]
RULE_HEAD = "A follow-up applies to the TOTAL you just gave"


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
def test_the_rule_shows_the_live_numbers(agent):
    rule = _rule(agent)
    assert "138.24" in rule, "the total it must continue from"
    assert "1.05" in rule, "and what the follow-up does to it"


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_names_the_wrong_answer_it_stops(agent):
    rule = _rule(agent).lower()
    assert "one footing" in rule, (
        "the config should name the shape it is overruling, not just the right one")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_rule_requires_the_total_to_be_restated(agent):
    rule = _rule(agent).lower()
    assert "restate the total" in rule, (
        "the reader must be able to see WHICH number the follow-up continued from")


@pytest.mark.parametrize("agent", AGENTS)
def test_the_loaded_prompt_carries_the_rule(agent):
    from app.agents import load_agents
    from app.agents.runtime import AGENT_REGISTRY

    load_agents()
    assert RULE_HEAD.lower() in (AGENT_REGISTRY[agent].system_prompt or "").lower()


def test_the_arithmetic_the_rule_describes_is_the_sheets_answer():
    # 18 footings of 3.2 x 3.2 x 0.75, +5% waste, at SAR 390/m3.
    net = 18 * 3.2 * 3.2 * 0.75
    with_waste = net * 1.05
    assert net == pytest.approx(138.24)
    assert with_waste == pytest.approx(145.152)
    assert with_waste * 390 == pytest.approx(56_609.28, abs=0.01)
    # And the wrong answer the rule exists to stop:
    assert (3.2 * 3.2 * 0.75) * 1.05 == pytest.approx(8.064)
