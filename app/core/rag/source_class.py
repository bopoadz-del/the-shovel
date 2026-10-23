"""What KIND of source an excerpt is, so an answer cannot mistake one for
another (owner's numbered item 2, 2026-09-01).

Two battery failures, one cause. Both are in
``gate_battery_13b2bf7_2026-08-31.md``:

* **Schedule register ask (Sev-1)** — asked what Schedule 10 of the contract contains, the
  platform answered that it "sets out any applicable Works Guarantees" and
  quoted **contract template** wording. The project's own Contract Data says
  ``Schedule 10: Not Used``. Nothing in the context said which excerpt was
  the contract and which was a standard form, so the model blended them.
* **Delay-rate ask (F-KB-1)** — asked for the Delay Damages, it reproduced
  a shipped ``docs/knowledge/`` FIDIC reference almost verbatim instead of
  the project's own figure at 8.8.1, which neighbouring asks prove is retrievable.
  The knowledge base answered a question the corpus could answer better.

The retriever has always KNOWN the difference: ``Chunk.layer`` is set on
every returned chunk ("own" / "general_knowledge" / "master_corpus"). It was
an internal isolation signal that never reached the answer layer. This
module turns it into a stated class, the marker carries it, and the citation
guard -- which has carried ``source_class`` on every evidence record since
#464 -- can finally read it.

THE CLASSES

``project_corpus``   this project's own record. The only class whose words
                     may be presented as what THIS contract says.
``knowledge_base``   curated cross-project reference (FIDIC administration
                     notes, CESMM, OSHA, rate books). Background, never the
                     contract.
``template``         a blank or standard form -- pro forma, specimen, model
                     form, or a chunk that is mostly placeholders. The
                     template-as-contract class. A template can sit in ANY layer, including the
                     project's own folder, so it is decided before layer.
``master_corpus``    the disclosed empty/thin fallback corpus (STEP 0b).

The owner named three classes. ``master_corpus`` is a fourth because the
alternatives are both wrong: calling it ``project_corpus`` licenses another
project's documents as this one's own words, and calling it
``knowledge_base`` would strip the attributions off every answer on the
fallback path, which is a working, disclosed feature. It is named for what
it is instead.
"""

from __future__ import annotations

import re
from typing import Any

#: Every class this module can return. Ordered as the header presents them.
SOURCE_CLASSES: tuple[str, ...] = (
    "project_corpus",
    "knowledge_base",
    "template",
    "master_corpus",
)

DEFAULT_CLASS = "project_corpus"

#: Visible labels at the glass (Sources panel + docx footer). The classifier
#: returns the machine class; this map is the only place the human wording
#: is defined so the UI cannot drift from the footer.
SOURCE_CLASS_LABELS: dict[str, str] = {
    "project_corpus": "this contract",
    "master_corpus": "master corpus",
    "knowledge_base": "knowledge base",
    "template": "template",
}

# ── template detection ────────────────────────────────────────────────────
#
# Conservative on purpose. A document that IS part of the contract must not
# be demoted to a template by the bare word "form" — "Schedule 8 form of
# Parent Company Guarantee" is a real upload name. PCG honesty (Contract
# Data 4.3.7 "not required" over that form's blank %) is a dedicated
# retriever/compose gate, not this classifier. Only standalone markers
# that mean "this is not filled in" count.
#
# The pattern has a TRAILING word boundary and deliberately no leading one.
# The trailing boundary is what protects a real document: "Blanket wayleave
# agreement" contains "blank" and must not be reclassified. A LEADING
# boundary would protect nothing on this list and would miss the shape
# engineering filenames actually take -- "BOQtemplate.xlsx",
# "DWGTEMPLATE-A.pdf" -- where the marker is glued to a discipline code.
_TEMPLATE_NAME_RE = re.compile(
    r"("
    r"template|templates"
    r"|pro[\s_-]?forma"
    r"|specimen"
    r"|blank"
    r"|model[\s_-]form"
    r"|standard[\s_-]form"
    r"|unexecuted"
    r")(?![A-Za-z])",
    re.IGNORECASE,
)

# Placeholder shapes. A filled contract does not carry these; an unfilled
# form carries several. Two DISTINCT kinds are required so that one stray
# rule of underscores in a scanned table cannot reclassify a real document.
_PLACEHOLDER_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[\s*(?:insert|name|address|date|amount|state|specify)\b[^\]]*\]",
               re.IGNORECASE),
    re.compile(r"\[\s*[••●■…]+\s*\]"),
    re.compile(r"<<[^>\n]{1,60}>>"),
    re.compile(r"_{6,}"),
    re.compile(r"(?<![A-Za-z0-9])[Xx]{4,}(?![A-Za-z0-9])"),
    re.compile(r"\[\s*(?:to\s+be\s+(?:completed|advised|inserted)|tbc|tba)\s*\]",
               re.IGNORECASE),
)
_MIN_PLACEHOLDER_KINDS = 2

#: Chunk.layer -> class, for anything that is not a template.
_LAYER_CLASS = {
    "own": "project_corpus",
    "general_knowledge": "knowledge_base",
    "master_corpus": "master_corpus",
}


def name_marks_template(source_name: str | None) -> bool:
    """True when the DOCUMENT NAME says it is a blank or standard form."""
    return bool(_TEMPLATE_NAME_RE.search(source_name or ""))


def placeholder_kinds(text: str | None) -> int:
    """How many DISTINCT placeholder shapes appear in this text."""
    body = text or ""
    return sum(1 for rx in _PLACEHOLDER_RES if rx.search(body))


def text_marks_template(text: str | None) -> bool:
    """True when the excerpt itself is mostly unfilled.

    Two distinct shapes, not two hits of one shape: a table of dotted leaders
    or a signature rule is one shape and is not evidence of anything.
    """
    return placeholder_kinds(text) >= _MIN_PLACEHOLDER_KINDS


def classify(
    layer: str | None,
    source_name: str | None = "",
    text: str | None = "",
) -> str:
    """The class of one excerpt.

    Template is decided FIRST and beats every layer. That ordering is the
    schedule-register defect:
    the template that answered "what does Schedule 10 contain" would have
    been ``project_corpus`` by layer, and quoting it as the contract is
    exactly the defect. A blank form in the project's own folder is still a
    blank form.
    """
    if name_marks_template(source_name) or text_marks_template(text):
        return "template"
    return _LAYER_CLASS.get((layer or "").strip(), DEFAULT_CLASS)


def classify_chunk(chunk: Any) -> str:
    """``classify`` for a ``Chunk``, tolerant of partially-built objects."""
    return classify(
        getattr(chunk, "layer", None),
        getattr(chunk, "source_name", "") or "",
        getattr(chunk, "text", "") or "",
    )


_SNAKE = re.compile(r"\b[a-z]+_[a-z_]+\b")
_KEY_VALUE = re.compile(r"\b\w+=\w+")


def _words(token: str) -> str:
    """``alpha_beta`` → "alpha beta". The machine id is not a label."""
    return " ".join(part for part in (token or "").replace("_", " ").split() if part)


def label_for(source_class: str | None) -> str:
    """Human wording for a class already decided by ``classify``.

    Does not re-tag. A class this map does not name yet is still shown,
    as words (underscores become spaces), never as the machine token.
    """
    key = (source_class or DEFAULT_CLASS).strip().lower()
    mapped = SOURCE_CLASS_LABELS.get(key)
    if mapped:
        return mapped
    return _words(key) or SOURCE_CLASS_LABELS[DEFAULT_CLASS]


def visible_label(source_class: str | None, provided: str | None = None) -> str:
    """The words a person reads. A provided label that is itself an id
    (snake_case or key=value) is not shown; the plain label is."""
    label = (provided or "").strip()
    if label and not _SNAKE.search(label) and not _KEY_VALUE.search(label):
        return label
    return label_for(source_class)
