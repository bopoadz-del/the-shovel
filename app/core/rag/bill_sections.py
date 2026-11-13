"""A priced bill's section map, read from the bill's own text (S-A7).

A multi-section bill of quantities prints the same item words (pipe, steel,
concrete ...) in several of its bills. Ranking the item's rows across the
whole document hands the model rows from whichever bill wins lexically. This
module derives the document's sections from its chunks so the bill-rows
action can keep only the section an ask names, or label every row with the
section it belongs to.

Everything is read from the document itself, in ``chunk_index`` order:

* a section heading is a line that starts ``Bill No. <n>`` (also ``Bill
  Number <n>`` / ``Bill # <n>``); its title is the rest of that line, or the
  next line when the heading line holds only the number;
* a line that also says the section is "carried to" a summary / collection is
  the section's closing total row, not a heading, and closes the section;
* a chunk that names two or more different section numbers in heading form
  is an index (the general summary lists every bill), not a section start.

Nothing here names a project, a document, a bill number or a bill title.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: A heading line: "Bill No. 4", "BILL NO 4 - EARTHWORKS", "Bill Number 12A".
_HEADING_RE = re.compile(
    r"^\s*bill\s*(?:no\.?|number|#)\s*(\d{1,3}[a-z]?)\b[\s.:\-–—/]*(.*)$",
    re.IGNORECASE,
)
#: The closing total row of a section: "carried to (the) (general) summary".
_CARRIED_RE = re.compile(
    r"\bcarried\s+(?:forward\s+)?to\b.*\b(?:summary|collection)\b", re.IGNORECASE,
)
#: A section named inside an ask / plan item: "bill no. 4", "bill 4".
_NAMED_NUMBER_RE = re.compile(
    r"\bbill\s*(?:no\.?|number|#)?\s*(\d{1,3}[a-z]?)\b", re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z]+")
_TITLE_MAX = 80
_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "at", "by",
    "with", "or", "no", "bill", "section",
})


@dataclass(frozen=True)
class BillSection:
    """One section of one document: number, title and its chunk span.

    ``start``/``end`` are inclusive ``chunk_index`` bounds. A chunk that
    closes one section and opens the next lies inside both spans.
    """

    number: str
    title: str
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"Bill No. {self.number}" + (f" - {self.title}" if self.title else "")


def _title_after(rest: str, following: Sequence[str]) -> str:
    """The section title: the rest of the heading line, else the next line."""
    for candidate in (rest, *following[:1]):
        text = (candidate or "").strip(" \t.:-–—/")
        if text and any(ch.isalpha() for ch in text) and not _HEADING_RE.match(text):
            return text[:_TITLE_MAX]
    return ""


def _headings(text: str) -> tuple[list[tuple[str, str]], list[str], bool]:
    """``([(number, title) opened], [numbers closed], opens_at_top)``.

    ``opens_at_top`` is True when the chunk's first line is the heading it
    opens, i.e. the chunk prints nothing of the section before it.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    opened: list[tuple[str, str]] = []
    closed: list[str] = []
    opens_at_top = False
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        number = match.group(1).upper()
        if _CARRIED_RE.search(line):
            closed.append(number)
            continue
        if any(number == n for n, _ in opened):
            continue
        opens_at_top = opens_at_top or (i == 0 and not opened)
        opened.append((number, _title_after(match.group(2), lines[i + 1:])))
    return opened, closed, opens_at_top


def section_map(chunks: Iterable[tuple[int, str]]) -> list[BillSection]:
    """The sections of one document from its ``(chunk_index, text)`` chunks.

    Returns ``[]`` when the document prints no section heading. That is the
    signal for "behave as before".
    """
    ordered = sorted(((int(i), t or "") for i, t in chunks), key=lambda p: p[0])
    if not ordered:
        return []
    last_index = ordered[-1][0]
    sections: list[BillSection] = []
    current: tuple[str, str, int] | None = None  # (number, title, start)

    def close(end: int) -> None:
        nonlocal current
        if current is not None:
            number, title, start = current
            sections.append(BillSection(number, title, start, max(start, end)))
            current = None

    for index, text in ordered:
        opened, closed, opens_at_top = _headings(text)
        if current is not None and current[0] in closed:
            close(index)
        if len(opened) > 1:
            # An index page (the general summary) lists every section.
            continue
        if not opened:
            continue
        number, title = opened[0]
        if current is not None and current[0] == number:
            continue  # the section's own running page header
        # A chunk that prints the previous section's last rows above the
        # new heading lies inside both spans; one that opens with the
        # heading belongs to the new section only.
        close(index - 1 if opens_at_top else index)
        current = (number, title, index)
    close(last_index)
    return [s for s in sections if s.start <= s.end]


def words(text: str) -> set[str]:
    """Content words of ``text``: lowercased, 3+ letters, plural folded."""
    out = set()
    for word in _WORD_RE.findall((text or "").lower()):
        if len(word) < 3 or word in _STOPWORDS:
            continue
        out.add(word[:-1] if word.endswith("s") and len(word) > 3 else word)
    return out


def _distinctive_title_words(sections: Sequence[BillSection]) -> dict[str, set[str]]:
    """Each section's title words minus the words most titles share.

    A word printed in half or more of the document's titles ("works",
    "items") does not tell one section from another, so naming it names none.
    """
    by_number: dict[str, set[str]] = {}
    for section in sections:
        by_number.setdefault(section.number, set()).update(words(section.title))
    if len(by_number) < 2:
        return by_number
    counts: dict[str, int] = {}
    for title_words in by_number.values():
        for word in title_words:
            counts[word] = counts.get(word, 0) + 1
    common = {w for w, n in counts.items() if n * 2 >= len(by_number)}
    return {num: ws - common for num, ws in by_number.items()}


def named_sections(text: str, sections: Sequence[BillSection]) -> set[str]:
    """Section numbers of ``sections`` that ``text`` names.

    By number ("Bill No. 4", "bill 4"), or by title when every distinctive
    word of the section's title appears in ``text``. Numbers the map does not
    hold name nothing.
    """
    if not sections or not (text or "").strip():
        return set()
    known = {s.number for s in sections}
    named = {
        m.group(1).upper() for m in _NAMED_NUMBER_RE.finditer(text)
    } & known
    asked = words(text)
    for number, title_words in _distinctive_title_words(sections).items():
        if title_words and title_words <= asked:
            named.add(number)
    return named


def sections_of(index: int | None, sections: Sequence[BillSection]) -> list[BillSection]:
    """The sections whose span holds chunk ``index`` (``[]`` when unknown)."""
    if index is None:
        return []
    return [s for s in sections if s.start <= index <= s.end]


def section_label(index: int | None, sections: Sequence[BillSection]) -> str:
    """``"[Bill No. 4 - EARTHWORKS]"`` for the chunk at ``index``, or ``""``."""
    owners = sections_of(index, sections)
    if not owners:
        return ""
    return "[" + " | ".join(dict.fromkeys(s.label for s in owners)) + "]"


def in_named_span(
    index: int | None, sections: Sequence[BillSection], numbers: Iterable[str],
) -> bool:
    """True when chunk ``index`` lies inside a section numbered in ``numbers``."""
    wanted = set(numbers)
    return any(s.number in wanted for s in sections_of(index, sections))


#: The label of a row whose bill prints no section heading.
UNKNOWN_SECTION = "[section unknown]"
#: A compressed row keeps the end of its text: the figures and the
#: description printed just above them.
_ROW_MAX = 240
#: A row's identity for deduplication: its last tokens (unit, qty, rate,
#: amount and the description words printed just before them).
_ROW_KEY_TOKENS = 8
#: Column and aggregate words of a bill ("total quantity", "rate", "amount").
#: They say what to compute about an item, not which item it is, so a row is
#: never matched on them.
_COLUMN_WORDS = frozenset({
    "total", "quantity", "qty", "rate", "amount", "sum", "price", "priced",
    "cost", "value", "item", "boq", "row", "unit", "many", "much", "what",
    "all", "every", "listed",
})


def _covers(word: str, row_words: set[str]) -> bool:
    """``word`` is in ``row_words``, or one is a 4+ letter prefix of the other
    ("pipe" / "pipework")."""
    if word in row_words:
        return True
    return any(
        min(len(word), len(other)) >= 4
        and (other.startswith(word) or word.startswith(other))
        for other in row_words
    )


def _row_owner(
    owners: list[BillSection], segment: str, current: BillSection | None,
) -> BillSection | None:
    """The section a row belongs to inside a chunk spanning ``owners``.

    A chunk that closes one section and opens the next prints the old
    section's rows above the new heading: the row switches to a section
    whose heading it prints, else stays with the current one.
    """
    for line in segment.splitlines():
        match = _HEADING_RE.match(line)
        if not match or _CARRIED_RE.search(line):
            continue
        number = match.group(1).upper()
        for section in owners:
            if section.number == number:
                current = section
    return current


def _labelled_rows(chunks, sections, numbers, row_segments):
    """``[(label, compact row text)]`` of every priced row, in bill order."""
    rows: list[tuple[str, str]] = []
    for index, text in sorted(chunks, key=lambda p: p[0]):
        owners = sorted(sections_of(index, sections), key=lambda s: s.start)
        if numbers and not any(s.number in numbers for s in owners):
            continue
        current = owners[0] if owners else None
        for segment in row_segments(text):
            current = _row_owner(owners, segment, current)
            if numbers and (current is None or current.number not in numbers):
                continue
            label = f"[{current.label}]" if current is not None else UNKNOWN_SECTION
            rows.append((label, " ".join(segment.split())[-_ROW_MAX:]))
    return rows


def item_rows(
    chunks: Iterable[tuple[int, str]],
    sections: Sequence[BillSection],
    item: str,
    *,
    row_segments,
    numbers: Iterable[str] = (),
) -> list[str]:
    """Every priced row of one bill that is ``item``, labelled by section.

    ``chunks`` are the bill's ``(chunk_index, text)``; ``row_segments(text)``
    splits a chunk into its priced rows. The result is complete, not ranked:
    every row whose text prints at least half of the item's words (the words
    the bill's rows print at all; column words such as "total" or "rate" do
    not count), deduplicated on the row's figures and last description
    words, each led by its section label, or
    :data:`UNKNOWN_SECTION` when the bill prints no heading. With
    ``numbers`` only the rows of those sections are kept.
    """
    wanted = {str(n) for n in numbers}
    rows = _labelled_rows(list(chunks), sections, wanted, row_segments)
    row_words = [words(text) for _label, text in rows]
    asked = {
        w for w in words(item) - _COLUMN_WORDS
        if any(_covers(w, ws) for ws in row_words)
    }
    if not asked:
        return []
    need = (len(asked) + 1) // 2
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for (label, text), ws in zip(rows, row_words):
        if sum(_covers(w, ws) for w in asked) < need:
            continue
        # Overlapping chunks print the same row with a different lead-in:
        # the row is its figures and the description words just before them.
        key = (label, " ".join(text.lower().split()[-_ROW_KEY_TOKENS:]))
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{label} {text}")
    return out
