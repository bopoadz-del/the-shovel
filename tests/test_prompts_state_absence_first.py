"""Answer first holds for an absence too: no prompt hands the model an opener
that narrates the search.

Answer contract rule 1 says narration of what was searched never opens an
answer, but ``_narration_first`` keeps a narration lead that reports an
absence -- it is the refusal itself (rule 5), and rewriting a sentence after
the model wrote it is not the contract's job. The cause was upstream: the
front-door prompt GAVE the model the opener to copy, a quoted template that
starts "I searched the project documents for ...". These tests read every
prompt the model is given and fail on any quoted answer template whose first
words are the writer's own search, as the contract itself reads them, so a
new config cannot bring the cause back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.agents.answer_contract import _NARRATION_LEAD_RE

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "app" / "agents" / "configs").glob("*.md"))
RULES = ROOT / "config" / "driver_standing_rules.md"

#: A quoted piece of answer text in a prompt: straight or curly double quotes
#: (a template may wrap across lines), or a markdown blockquote line.
_QUOTED_RE = re.compile(r'"([^"]+)"|“([^”]+)”|^[ \t]*>[ \t]*(.+)$', re.MULTILINE)


def _narration_templates(text: str) -> list[str]:
    """Every quoted template in ``text`` that opens on the writer's search."""
    found = []
    for m in _QUOTED_RE.finditer(text):
        body = next(g for g in m.groups() if g is not None)
        lead = " ".join(body.split()).lstrip("*_\"“ ")
        if _NARRATION_LEAD_RE.match(lead):
            found.append(lead)
    return found


@pytest.mark.parametrize(
    "prompt",
    [
        "## When nothing is found\n"
        'Say so, e.g. **"I searched the drawings for <item> and did not\n'
        '   find it."**\n',
        "## Fallback\n"
        '   > "I looked through the schedule for [activity] but could not find it. '
        'Based on general practice: [answer]."\n',
    ],
    ids=["bold-quoted-wrapped", "blockquote"],
)
def test_the_scanner_finds_an_invented_narration_opener(prompt):
    assert len(_narration_templates(prompt)) == 1


@pytest.mark.parametrize(
    "prompt",
    [
        "## When nothing is found\n"
        'Say so, e.g. **"<Item> is not in the drawings I searched."** Any detail\n'
        "of the search may follow, never lead.\n",
        "## Fallback\n"
        '   > "Specific information on [activity] is not in the schedule I searched. '
        'Based on general practice: [answer]."\n',
    ],
    ids=["bold-quoted", "blockquote"],
)
def test_the_scanner_passes_an_invented_absence_first_template(prompt):
    assert _narration_templates(prompt) == []


@pytest.mark.parametrize("path", CONFIGS + [RULES], ids=lambda p: p.name)
def test_no_prompt_hands_the_model_a_search_narration_opener(path):
    found = _narration_templates(path.read_text(encoding="utf-8"))
    assert found == [], f"{path.name} gives the model a search-narration opener: {found}"


def test_the_front_door_answer_first_rule_covers_an_absence():
    text = (ROOT / "app" / "agents" / "configs" / "project-assistant.md").read_text(
        encoding="utf-8"
    )
    rule = next(ln for ln in text.splitlines() if "**Answer first.**" in ln)
    assert "absent" in rule
    assert "never lead" in rule
    # The template it gives opens on the missing thing, and the contract's own
    # narration reading does not see a search lead in it.
    example = re.search(r'\("([^"]+)"\)', rule).group(1)
    assert not _NARRATION_LEAD_RE.match(example)
    # Honesty is unchanged: the absence stays scoped to what was searched.
    assert "searched" in example
