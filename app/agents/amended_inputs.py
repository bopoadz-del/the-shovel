"""A follow-up that only changes figures the user already stated.

A follow-up such as "<label> is <new figure>, not <old figure>." or "And
<label> was <new figure>." asks nothing new: it changes one input of the
question asked earlier in the thread. The thread still holds the earlier answer, worked on the old figure,
so the model answered the follow-up from that answer -- it re-printed the old
table with one line touched, ran no tool, and in one case said the changed
input "does not enter" the result.

This module reads the figures a user message states under its own words
(``<label> <figure>``), and decides whether the latest message is such an
amendment: every figure it states sits under a label an earlier message of the
same thread already stated, at least one of them with a different value, and
it asks no question of its own. The note it returns names the question being
worked, each changed input (old and new), says every other stated figure
stands, and says the earlier answer is out of date.

Nothing here knows a calculation, a label or a figure: the labels are the
user's own words, read from the thread.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# A figure: digits with optional thousands commas and decimals, then an
# optional percent sign or word. A digit glued to a letter (m3, A1) is not one.
_FIGURE_RE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\w|[.,]\d)\s*(%|percent\b)?",
    re.IGNORECASE,
)
# A clause ends at sentence punctuation, a bracket, a semicolon or colon, a
# line break, or a comma that is not inside a number.
_CLAUSE_END_RE = re.compile(r"[.;:!?()\[\]\n]|,(?!\d{3})")
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
# Words that join a label to its figure, or open a follow-up, and are not
# part of the label: "and retention is", "advance was", "not 5".
_EDGE_WORDS = frozenset({
    "a", "also", "an", "and", "are", "at", "be", "but", "by", "equals",
    "for", "is", "it", "not", "now", "of", "or", "should", "so", "that",
    "the", "then", "to", "was", "were", "will", "would",
})
_LABEL_WORDS = 4


@dataclass(frozen=True)
class StatedFigure:
    label: tuple[str, ...]
    value: float
    percent: bool
    text: str


@dataclass(frozen=True)
class Amendment:
    label: str
    old: str
    new: str


def _label_before(text: str, start: int, end: int) -> tuple[str, ...]:
    """The words between ``start`` (the end of the figure before) and ``end``,
    inside the clause that holds ``end``."""
    for m in _CLAUSE_END_RE.finditer(text, start, end):
        start = m.end()
    words = [w.lower() for w in _WORD_RE.findall(text[start:end])][-_LABEL_WORDS:]
    while words and words[0] in _EDGE_WORDS:
        words.pop(0)
    while words and words[-1] in _EDGE_WORDS:
        words.pop()
    return tuple(words)


def stated_figures(text: str) -> list[StatedFigure]:
    """Every figure in ``text`` with the words its own clause puts before it.

    A figure with no words of its own before it ("not 5", "(800,000)") has no
    label and is not returned; nor is the second of two figures with only a
    joining word between them ("20 by 15")."""
    out: list[StatedFigure] = []
    previous_end = 0
    for m in _FIGURE_RE.finditer(text or ""):
        label = _label_before(text, previous_end, m.start())
        previous_end = m.end()
        if not label:
            continue
        value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        out.append(StatedFigure(label, value, bool(m.group(3)), m.group(0).strip()))
    return out


def _same_label(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    sa, sb = set(a), set(b)
    return sa <= sb or sb <= sa


def _same_value(a: StatedFigure, b: StatedFigure) -> bool:
    return a.value == b.value and a.percent == b.percent


def _asks_its_own_question(text: str) -> bool:
    return "?" in (text or "")


def _changes(message: str, stated: list[StatedFigure]) -> Optional[list[tuple[StatedFigure, StatedFigure]]]:
    """The (earlier, new) pairs ``message`` changes, or None when it is not an
    amendment of ``stated``."""
    figures = stated_figures(message)
    if not figures or _asks_its_own_question(message):
        return None
    pairs: list[tuple[StatedFigure, StatedFigure]] = []
    for new in figures:
        earlier = [s for s in stated if _same_label(s.label, new.label)]
        if not earlier:
            return None
        if not _same_value(earlier[-1], new):
            pairs.append((earlier[-1], new))
    return pairs or None


def amendments(message: str, earlier_user_messages: Iterable[str]) -> tuple[str, list[Amendment]]:
    """The question ``message`` amends and every change made to it since.

    The question is the latest earlier message that is not itself an
    amendment. Changes made by amendments after it are carried, so a second
    follow-up keeps the first one's change; each change runs from the figure
    the thread held before it to the latest one. Returns ``("", [])`` when
    ``message`` is not an amendment."""
    earlier = [m for m in earlier_user_messages if (m or "").strip()]
    base_index = None
    for i in range(len(earlier) - 1, -1, -1):
        prior_stated = [f for m in earlier[:i] for f in stated_figures(m)]
        if _changes(earlier[i], prior_stated) is None:
            base_index = i
            break
    if base_index is None:
        return "", []
    stated = stated_figures(earlier[base_index])
    changed: dict[tuple[str, ...], Amendment] = {}

    def carry(pairs: list[tuple[StatedFigure, StatedFigure]]) -> None:
        for old, new in pairs:
            changed[old.label] = Amendment(" ".join(old.label), old.text, new.text)
            stated.append(StatedFigure(old.label, new.value, new.percent, new.text))

    for follow_up in earlier[base_index + 1:]:
        carry(_changes(follow_up, stated) or [])
    pairs = _changes(message, stated)
    if pairs is None:
        return "", []
    carry(pairs)
    return earlier[base_index].strip(), list(changed.values())


def amendment_note(message: str, earlier_user_messages: Iterable[str]) -> Optional[str]:
    """A system note for a turn that only changes stated figures, else None."""
    question, changed = amendments(message, earlier_user_messages)
    if not changed:
        return None
    lines = "; ".join(f"{a.label}: {a.old} -> {a.new}" for a in changed)
    return (
        "The user's latest message changes figures they stated earlier and "
        "asks nothing new. The question is still the earlier one:\n"
        f"\"{question}\"\n"
        f"Changed inputs: {lines}. Every other figure the user stated stands. "
        "Any answer earlier in this thread was worked on the old figures and is "
        "out of date: work the question again from the full set of inputs, "
        "running the calculation again as you did for the earlier answer. Do "
        "not copy an earlier result or table, and do not say a changed input "
        "has no effect unless the working shows it."
    )
