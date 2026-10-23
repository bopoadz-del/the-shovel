"""An instruction that forbids tool use is work, not a capability question.

Live-fire find 2026-08-15 (F22). A costing request that opened with "this
turn is a costing analysis in text only: do not run any schedule, workbook,
or generator tool" was short-circuited in one second into the agent's tool
roster ("I'm the heavy-reasoning agent. Tools I can use: ...") -- the
capability classifier treats any co-occurrence of you/your with the word
"tool" as a question about the agent. The user was TELLING the agent not to
use tools; the platform answered a question nobody asked and the costing
never ran.

Prohibition phrasing now wins over the co-occurrence heuristic. Genuine
capability questions keep short-circuiting.
"""
from __future__ import annotations

from app.agents.runtime import _is_capability_request

LIVE_MESSAGE = (
    "This turn is a costing analysis in text only: do not run any schedule, "
    "workbook, or generator tool. Write a resource-based cost build-up for "
    "the measured finishes bill below. Derive labour cost from trade day "
    "rates with a stated daily output per tradesman and its basis. If a "
    "needed figure has no source in the knowledge base, write NO SOURCED "
    "RATE in that cell instead of inventing one."
)


def test_the_live_costing_instruction_is_not_a_capability_question():
    """RED before the fix: True -> tool-roster short-circuit."""
    assert _is_capability_request(LIVE_MESSAGE) is False


def test_other_prohibition_phrasings_are_work_not_questions():
    for msg in (
        "don't use any tool this turn, just write the text",
        "answer without tools please, prose only",
        "never invoke a tool for this, I want your written analysis",
        "run no tool; this is drafting only",
    ):
        assert _is_capability_request(msg) is False, msg


def test_genuine_capability_questions_still_short_circuit():
    for msg in (
        "what tools do you have?",
        "which tools can you use?",
        "tell me about your tools and capabilities",
        "are you able to parse a Primavera schedule?",
    ):
        assert _is_capability_request(msg) is True, msg
