"""BUILD 0b: the prompts ask for the answer contract instead of the padding.

The answer-exit contract (rules 1-4) removes Source lines, machinery words and
padding from an answer after the model writes them. The cause was upstream:
the agent configs and the driver's standing rules TOLD the model to write a
"Source:" line, to quote the block and action it ran, and to cite chunk
numbers. These tests read every prompt the model is given and fail on any
instruction of that kind, so a new config cannot bring the cause back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "app" / "agents" / "configs").glob("*.md"))
RULES = ROOT / "config" / "driver_standing_rules.md"

#: An instruction (not an example of what to avoid) that asks the model to put
#: a citation line, chunk numbers, or the tool / block it ran into the text.
_ASKS_FOR_MACHINERY = [
    re.compile(r"^\s*>?\s*\*?\s*Sources?:\s*[\[<]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*-\s*Sources?:\s*<", re.IGNORECASE | re.MULTILINE),
    re.compile(r"chunks?\s*\[?\s*N\b|chunks\s+\d+\s*,\s*\d+", re.IGNORECASE),
    re.compile(r"\b(?:quote|cite|name)\s+the\s+(?:source\s+)?block\b", re.IGNORECASE),
    re.compile(
        r"\bcite\s+the\s+`?\[source:[^\]]*\]`?\s+filename\s+in\s+the\s+answer",
        re.IGNORECASE,
    ),
    re.compile(r"give each figure its source\.", re.IGNORECASE),
]


def _machinery_asks(text: str) -> list[str]:
    return [m.group(0) for rx in _ASKS_FOR_MACHINERY for m in rx.finditer(text)]


def test_the_scanner_finds_an_invented_source_line_instruction():
    prompt = (
        "## Output\n"
        '> *"The slab is 200 mm thick."*\n'
        '> *Source: [filename from the header], chunks [N, M]."*\n'
        "- Always cite the source block for every number.\n"
    )
    found = _machinery_asks(prompt)
    assert any("Source:" in f for f in found)
    assert any("block" in f for f in found)


def test_the_scanner_passes_an_invented_prompt_that_asks_for_tags():
    prompt = (
        "## Output\n"
        "Open with the figure and its unit. Tag a document figure with "
        "`[source: <filename>, p. N]` on its sentence. Never write a Source line.\n"
        "- Result: <value with units>\n"
    )
    assert _machinery_asks(prompt) == []


@pytest.mark.parametrize("path", CONFIGS + [RULES], ids=lambda p: p.name)
def test_no_prompt_asks_for_source_lines_or_tool_names(path):
    asks = _machinery_asks(path.read_text(encoding="utf-8"))
    assert asks == [], f"{path.name} asks the model for machinery in the answer: {asks}"


def test_the_front_door_prompt_states_the_four_rules():
    text = (ROOT / "app" / "agents" / "configs" / "project-assistant.md").read_text(
        encoding="utf-8"
    )
    for rule in (
        "**Answer first.**",
        "**Length follows the question.**",
        "**No machinery in the text.**",
        "**Citations as tags, not lines.**",
    ):
        assert rule in text
    # The honesty rules ride with the form rules, unchanged.
    assert "a refusal stays a refusal" in text


def test_the_driver_rules_ask_for_the_answer_first():
    text = RULES.read_text(encoding="utf-8")
    assert "The first line is the answer" in text
    assert "never state how many documents are indexed" in text
