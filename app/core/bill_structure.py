"""A bill's section structure, read from its rows' heading paths (one module, two users).

A scanned bill converted into rows keeps, for every row, a flat pile of the
lines printed above it: the page header (with the page label whose numbers
change per page), the first page's header carried onto every row, SOMETIMES
the section's title (a section prints it on some of its pages only), the
wrapped lines of the item preamble, and last the row's own sub-heading.
Matching an ask's section words against that pile keeps the rows of the
pages that print the title and silently drops the rest.

This module turns the pile into columns, the same way for both users:

* the reader (``bill_aggregate``) annotates rows stored before the converter
  wrote the columns, so the answers are right on the copies stored today;
* the converter writes the columns into the derived sheet, so a new copy
  carries them and the reader reads them directly.

The rules, all read from the document itself:

(a) RUNNING HEADERS. A path line that repeats, digits masked, on more than
    half of the document's rows is a running header, not a heading.
(b) SECTION KEY. Among the running headers that print digits, the one with
    the most distinct readings is the page label. Its numeric fields that
    vary across the document, counted from the right, are the section key
    (all but the last) and the page (the last). The first page's label,
    carried unchanged onto every row, is not a row's own label. A row whose
    label is unreadable inherits the key of the nearest labelled row before
    it in stored order; that inheritance is counted.
(c) TITLE PROPAGATION. A capitalised heading line (not a running header)
    seen on ANY row of a section key is that section's title, for EVERY row
    of the key.
(d) CONTINUATION. A continued heading ("X (cont'd)") is its original "X".
(e) PREAMBLE. The lines between the title and the row's last level are the
    wrapped item preamble: ONE element, not heading levels.

Nothing here names a document, a section, a label word or a row.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

#: A line repeating on more than this share of rows is a running header.
RUNNING_SHARE = 0.5
#: A title line has at least this many letters, nearly all capitals.
_TITLE_MIN_LETTERS = 6
_TITLE_UPPER_SHARE = 0.85
#: An OCR reading of the label: its masked form this close to the label's.
_LABEL_LIKENESS = 0.9
#: Lines shorter than this are compared exactly, never by likeness.
_LIKENESS_MIN_CHARS = 12
#: At most this many title lines name one section.
_TITLE_LINES = 3

#: The derived sheet's section columns: (``SheetRow`` attribute, header).
COLUMNS = (
    ("section_key", "Section Key"),
    ("section_title", "Section Title"),
    ("sub_heading", "Sub-heading"),
    ("preamble", "Item Preamble"),
)

_DIGITS = re.compile(r"\d+")
#: A trailing continuation mark: "(cont'd)", "[cont'd]", "contd.", "continued",
#: with OCR variants of the apostrophe and of the closing bracket.
_CONTINUED = re.compile(
    r"[\s\-–—,:;]*[(\[{|]?\s*\bcont(?:inued|['’`´]?\s*[dq]|d)\.?\s*[)\]}|]?\s*$",
    re.IGNORECASE,
)


def mask(line: str) -> str:
    """``line`` with every run of digits replaced by ``#``."""
    return _DIGITS.sub("#", line)


def base_heading(line: str) -> str:
    """A continued heading's original: ``"X (cont'd)"`` -> ``"X"`` (rule d)."""
    text = (line or "").strip()
    previous = None
    while previous != text:
        previous = text
        text = _CONTINUED.sub("", text).strip()
    return text


def _is_title(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < _TITLE_MIN_LETTERS:
        return False
    return sum(c.isupper() for c in letters) >= _TITLE_UPPER_SHARE * len(letters)


@dataclass(frozen=True)
class RowStructure:
    """One row's columns. ``key_inherited`` is True when its label was unreadable."""

    section_key: tuple[str, ...] | None
    section_title: str
    sub_heading: str
    preamble: str
    page: str
    key_inherited: bool


@dataclass(frozen=True)
class BillStructure:
    """The columns of every row, in the order given, and how they were read."""

    rows: tuple[RowStructure, ...]
    running: frozenset[str]
    running_words: str
    label: str | None
    labelled: int
    inherited: int
    unkeyed: int

    def section_counts(self) -> dict[tuple[str, ...] | None, int]:
        return dict(Counter(r.section_key for r in self.rows))


def _label_mask(paths: Sequence[Sequence[str]], running: set[str]) -> str | None:
    variants: dict[str, set[str]] = defaultdict(set)
    # A page label is printed above a row, never as the row's own last level.
    own_level = {mask(path[-1]) for path in paths if path}
    for path in paths:
        for line in path:
            m = mask(line)
            if m in running and m != line and m not in own_level:
                variants[m].add(line)
    varied = {m: v for m, v in variants.items() if len(v) > 1}
    if not varied:
        return None
    return max(sorted(varied), key=lambda m: len(varied[m]))


def _varying_positions(readings: Sequence[str]) -> list[int]:
    """Right-counted positions (1 = last) of the numeric fields that vary."""
    fields = [_DIGITS.findall(r) for r in readings]
    width = max((len(f) for f in fields), default=0)
    out = []
    for pos in range(1, width + 1):
        seen = {f[-pos] for f in fields if len(f) >= pos}
        if len(seen) > 1:
            out.append(pos)
    return sorted(out, reverse=True)


def structure(paths: Sequence[Sequence[str]]) -> BillStructure:
    """The section columns of a bill's rows from their heading paths.

    ``paths`` are the rows' heading paths (outer to inner) in stored order.
    """
    paths = [tuple(p.strip() for p in path if p and p.strip()) for path in paths]
    n = len(paths)
    masked_seen: Counter[str] = Counter()
    exact_seen: Counter[str] = Counter()
    for path in paths:
        masked_seen.update(set(map(mask, path)))
        exact_seen.update(set(path))
    running = {m for m, c in masked_seen.items() if c > RUNNING_SHARE * n}
    carried = {line for line, c in exact_seen.items() if c > RUNNING_SHARE * n}
    label = _label_mask(paths, running)

    # (b) a row's own label: the last line of the label's form (OCR readings
    # included) that is not the first page's label carried onto every row.
    likeness: dict[str, bool] = {}

    def reads_label(line: str) -> bool:
        m = mask(line)
        if m not in likeness:
            likeness[m] = bool(label) and (m == label or SequenceMatcher(
                None, m.casefold(), label.casefold()).ratio() >= _LABEL_LIKENESS)
        return likeness[m]

    own: list[str | None] = []
    for path in paths:
        # A first-page row prints the carried label once more as its own.
        mine = [ln for ln in path if reads_label(ln)
                and (ln not in carried or path.count(ln) > 1)]
        own.append(mine[-1] if mine else None)
    positions = _varying_positions([o for o in own if o])
    key_positions, page_position = positions[:-1], (positions[-1] if positions else None)

    keys: list[tuple[str, ...] | None] = []
    pages: list[str] = []
    inherited_flags: list[bool] = []
    current: tuple[str, ...] | None = None
    labelled = inherited = 0
    for reading in own:
        fields = _DIGITS.findall(reading) if reading else []
        page = fields[-page_position] if page_position and len(fields) >= page_position else ""
        key = tuple(fields[-p] for p in key_positions if len(fields) >= p) if fields else ()
        if fields and len(key) == len(key_positions):
            current = key
            labelled += 1
            inherited_flags.append(False)
        else:
            inherited += current is not None
            inherited_flags.append(current is not None)
        keys.append(current)
        pages.append(page)

    # (a) an OCR reading of a running header is a running header too.
    near_running: dict[str, bool] = {}

    def is_running(line: str) -> bool:
        m = mask(line)
        if m not in near_running:
            near_running[m] = m in running or (len(m) >= _LIKENESS_MIN_CHARS and any(
                SequenceMatcher(None, m.casefold(), r.casefold()).ratio() >= _LABEL_LIKENESS
                for r in running if len(r) >= _LIKENESS_MIN_CHARS))
        return near_running[m]

    # (c)+(d) title lines per key; (e) preamble and sub-heading per row. The
    # row's last level is its sub-heading; capitalised lines above it are
    # title lines; the rest above it is the wrapped preamble.
    holders: dict[tuple[str, ...] | None, Counter[str]] = defaultdict(Counter)
    split: list[tuple[list[str], str]] = []
    for path, key in zip(paths, keys):
        own_lines = [base_heading(ln) for ln in path if not is_running(ln)]
        own_lines = [ln for ln in own_lines if ln]
        above, sub = own_lines[:-1], (own_lines[-1] if own_lines else "")
        for head in dict.fromkeys(ln for ln in above if _is_title(ln)):
            holders[key][head] += 1
        split.append((above, sub))
    # A section's title belongs to one section: the key whose rows print it
    # most (a boundary page also prints the next section's title). A line
    # printed above rows of most sections, none of which holds most of its
    # rows, names none of them.
    spread = Counter(head for counter in holders.values() for head in counter)
    printed = Counter()
    for counter in holders.values():
        printed.update(counter)
    owner: dict[str, tuple[str, ...] | None] = {}
    for key, counter in holders.items():
        for head, count in counter.items():
            if head not in owner or count > holders[owner[head]][head]:
                owner[head] = key
    shared = {
        h for h, c in spread.items()
        if len(holders) > 1 and c * 2 > len(holders)
        and holders[owner[h]][h] * 2 <= printed[h]
    }
    titles: dict[tuple[str, ...] | None, str] = {}
    for key, counter in holders.items():
        ranked = sorted(
            (h for h in counter if h not in shared and owner[h] == key),
            key=lambda h: (-counter[h], h),
        )
        titles[key] = " / ".join(ranked[:_TITLE_LINES])

    rows = []
    for (above, sub), key, page, inh in zip(split, keys, pages, inherited_flags):
        rows.append(RowStructure(
            section_key=key,
            section_title=titles.get(key, ""),
            sub_heading=sub,
            preamble=" ".join(ln for ln in above if not _is_title(ln)),
            page=page,
            key_inherited=inh,
        ))
    running_lines = [ln for ln in carried if not reads_label(ln)]
    return BillStructure(
        rows=tuple(rows),
        running=frozenset(running),
        running_words=" ".join(sorted(running_lines)),
        label=label,
        labelled=labelled,
        inherited=inherited,
        unkeyed=sum(k is None for k in keys),
    )


def key_text(key: tuple[str, ...] | None) -> str:
    """A section key as a sheet cell: ``("12",)`` -> ``"12"``; none -> ``""``."""
    return ".".join(key) if key else ""


def log_line(found: BillStructure) -> str:
    """How a bill's sections were read: counts only, never a heading's text."""
    return (
        f"bill_structure rows={len(found.rows)} sections={len(found.section_counts())} "
        f"labelled={found.labelled} inherited={found.inherited} unkeyed={found.unkeyed} "
        f"empty_title={sum(not r.section_title for r in found.rows)}"
    )
