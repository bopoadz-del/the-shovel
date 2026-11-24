"""A refusal never asks the user to upload a document the search already read.

The rate rule gave the model one refusal for every "no rate" case: "No rate on
file for that -- upload your priced BOQ / rate schedule ...". When the project
HAS a priced bill and the search read it and found no line for the item, the
model copied that template and then contradicted it in the next sentence (the
bill was scanned and has no such line). The cause is the prompt: an upload
request offered with no searched-and-absent alternative beside it.

These tests read every prompt the model is given. Any rule (a paragraph) that
hands the model a refusal asking the user to upload something must also hand it
the refusal for the case where the search read that kind of document and the
item is not in it -- naming what was searched, without an upload request. A
template that itself says the kind of document is missing from the project
needs no twin: there, asking for the document is the honest answer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "app" / "agents" / "configs").glob("*.md"))
RULES = ROOT / "config" / "driver_standing_rules.md"

_QUOTED_RE = re.compile(r'"([^"]+)"|“([^”]+)”')
_UPLOAD_RE = re.compile(r"\bupload\b", re.IGNORECASE)
_SEARCHED_RE = re.compile(r"\bI searched\b", re.IGNORECASE)
#: The template itself says the KIND of document is missing from the project
#: (not an item missing from it): asking for that document is then honest.
_DOC_ABSENT_RE = re.compile(
    r"\b(?:couldn't|could not|did not|didn't) find an? [^.]{0,60}\bin (?:this|the) project's documents\b",
    re.IGNORECASE,
)
#: A quote is a template to copy -- not a banned phrase listed as a warning --
#: when it is bold-wrapped or the words just before it tell the model to say it.
_SAY_RE = re.compile(r"\b(?:say|reply|refuse|tell the user)\b[^\"“”.]{0,20}$", re.IGNORECASE)


def _templates(paragraph: str) -> list[str]:
    found = []
    for m in _QUOTED_RE.finditer(paragraph):
        bold = paragraph[max(0, m.start() - 2) : m.start()] == "**"
        if bold or _SAY_RE.search(paragraph[max(0, m.start() - 60) : m.start()]):
            found.append(" ".join(next(g for g in m.groups() if g is not None).split()))
    return found


def _unguarded_upload_refusals(text: str) -> list[str]:
    """Upload-request templates in ``text`` whose rule has no searched-absent twin."""
    bad = []
    for paragraph in re.split(r"\n[ \t]*\n", text):
        quoted = _templates(paragraph)
        uploads = [q for q in quoted if _UPLOAD_RE.search(q) and not _DOC_ABSENT_RE.search(q)]
        if not uploads:
            continue
        twin = any(_SEARCHED_RE.search(q) and not _UPLOAD_RE.search(q) for q in quoted)
        if not twin:
            bad.extend(uploads)
    return bad


@pytest.mark.parametrize(
    "prompt",
    [
        "- If no price was retrieved, say exactly:\n"
        '  **"No price on file - upload your tender schedule."**\n',
        'When the drawing is missing reply "Please upload your drawing register first."\n'
        "\n"
        'A separate rule: say "The bolt grade is not in the drawings I searched."\n',
        "- If no line item matched, tell the user:\n"
        '  "I could not find that item. Upload your drawing register and ask again."\n',
    ],
    ids=["bare-upload", "twin-in-another-rule", "item-missing-asks-upload"],
)
def test_the_scanner_flags_an_invented_upload_refusal_with_no_searched_twin(prompt):
    assert len(_unguarded_upload_refusals(prompt)) == 1


@pytest.mark.parametrize(
    "prompt",
    [
        "- If the tender schedule was searched and lacks the item, say\n"
        '  **"<Item> has no price in the tender schedule I searched."** Only when no\n'
        '  schedule was retrieved, say **"No price on file - upload your tender schedule."**\n',
        'Missing drawing: when the register was read, say "<Code> is not in the drawing\n'
        'register I searched."; otherwise reply "Please upload your drawing register."\n',
        "### Phrases that are banned when chunks are present\n"
        '- "you would need to upload the actual"\n'
        '- Never write "please upload" when the block answers the question.\n',
        "- If the search returns no document of the kind asked about, tell the user:\n"
        '  "I couldn\'t find a schedule in this project\'s documents. Upload one and I\'ll read it."\n',
    ],
    ids=["bold-pair", "inline-pair", "banned-phrase-list", "document-kind-missing"],
)
def test_the_scanner_passes_an_invented_upload_refusal_with_its_searched_twin(prompt):
    assert _unguarded_upload_refusals(prompt) == []


@pytest.mark.parametrize("path", CONFIGS + [RULES], ids=lambda p: p.name)
def test_no_prompt_offers_an_upload_refusal_without_the_searched_case(path):
    found = _unguarded_upload_refusals(path.read_text(encoding="utf-8"))
    assert found == [], (
        f"{path.name} asks the user to upload with no refusal for a document "
        f"that was searched and lacks the item: {found}"
    )
