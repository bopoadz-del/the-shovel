"""Noise chunks the index refuses (RQ-1).

Three kinds of chunk text carry nothing a question can be answered from,
yet each is embedded and FTS-indexed like content and competes for the
retrieval budget:

* an EMPTY ROW -- a spreadsheet row whose cells are all blank or zero
  (a template's formula rows evaluate to ``0 | 0 | 0 ...``);
* a HEADER REPEAT -- a label-only row (no figure in any cell) that the
  same document already stated, e.g. a column header re-printed above
  every bill or every sheet;
* a PLATFORM GUARD sentence -- prose the platform itself once wrote into
  the chunk store (the retired BOQ total guard, now a document attribute).

Every rule reads the text's own structure (cells, figures, repetition) or
the platform's own fixed lead-ins. Nothing here names a document, a sheet
or a project.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

# A table row's cell separator as the spreadsheet extractor writes it
# (``" | "``) plus the row chunker's ``"; "`` field separator.
_CELL_SPLIT = re.compile(r"\s*[|;\t]\s*")
_LETTER = re.compile(r"[^\W\d_]")
_DIGIT = re.compile(r"\d")
# A token that states no quantity: a dash, or a number equal to zero
# (``0``, ``0.00``, ``-0``, ``0%``, ``(0)``, ``0,000.00``). A cell is zero
# when every whitespace-separated token in it is (the word chunker re-flows
# a run of rows onto one line, so ``0 0`` is two zero cells).
_ZERO_TOKEN = re.compile(r"^[\-–—(]*(?:0+(?:[.,]0*)*)?%?\)?$")


def _is_zero_cell(cell: str) -> bool:
    return all(_ZERO_TOKEN.match(tok) for tok in cell.split())


def _guard_prefixes() -> tuple[str, ...]:
    """The fixed lead-ins of sentences the platform wrote as chunk text.

    Read from the module that owns each retired writer, so the table and
    the purge plan cannot drift.
    """
    from app.core.ingest_reconcile import LEGACY_TOTAL_GUARD_PREFIX

    return (LEGACY_TOTAL_GUARD_PREFIX,)


def cells(row: str) -> list[str]:
    """The non-blank cells of one row (``a | b | c`` or ``a; b; c``)."""
    return [c for c in (p.strip() for p in _CELL_SPLIT.split(row or "")) if c]


def is_empty_row(row: str) -> bool:
    """True when every cell of ``row`` is blank or zero and none has a word.

    ``"0 | 0 | 0"`` and ``"- | 0.00 | 0"`` are empty rows; ``"12 | 4.5 | 54"``
    (a figure) and ``"Excavation | 0"`` (a description) are not.
    """
    parts = cells(row)
    if not parts:
        return True
    return all(_is_zero_cell(c) and not _LETTER.search(c) for c in parts)


def is_label_row(row: str) -> bool:
    """A table row of two or more cells that states no figure (a header shape)."""
    parts = cells(row)
    return len(parts) >= 2 and not any(_DIGIT.search(c) for c in parts)


def row_key(row: str) -> str:
    return " | ".join(c.casefold() for c in cells(row))


def is_guard_sentence(text: str) -> bool:
    """True when ``text`` starts with a platform guard sentence's lead-in."""
    head = (text or "").lstrip()
    return any(head.startswith(p) for p in _guard_prefixes())


def is_empty_chunk(text: str) -> bool:
    """True when every line of ``text`` is an empty row (no word, no non-zero figure)."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return all(is_empty_row(ln) for ln in lines)


def refusal_reason(text: str) -> str | None:
    """Why the index refuses chunk ``text``, or ``None`` to keep it."""
    if is_guard_sentence(text):
        return "guard_sentence"
    if is_empty_chunk(text):
        return "empty_row"
    return None


def clean_table_text(text: str) -> tuple[str, dict[str, int]]:
    """Drop empty rows and repeated label rows from row-wise table text.

    ``text`` is one row per line (the spreadsheet extractor's shape). The
    first occurrence of a label row is kept -- it is the column header the
    rows under it are read against -- and every later identical label row
    is dropped. Returns ``(text, {"empty_row": n, "header_repeat": m})``.
    """
    seen_labels: set[str] = set()
    kept: list[str] = []
    counts = {"empty_row": 0, "header_repeat": 0}
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        if is_empty_row(line):
            counts["empty_row"] += 1
            continue
        if is_label_row(line):
            key = row_key(line)
            if key in seen_labels:
                counts["header_repeat"] += 1
                continue
            seen_labels.add(key)
        kept.append(line)
    return "\n".join(kept), counts


def refuse_chunks(
    chunks: Iterable[str], pages: list | None = None,
) -> tuple[list[str], list | None, dict[str, int]]:
    """Drop refused chunks, keeping ``pages`` aligned when it is given.

    Returns ``(chunks, pages, {reason: count})``.
    """
    chunks = list(chunks)
    aligned = pages is not None and len(pages) == len(chunks)
    kept: list[str] = []
    kept_pages: list = []
    counts: dict[str, int] = {}
    for i, text in enumerate(chunks):
        reason = refusal_reason(text)
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
            continue
        kept.append(text)
        if aligned:
            kept_pages.append(pages[i])
    return kept, (kept_pages if aligned else pages), counts
