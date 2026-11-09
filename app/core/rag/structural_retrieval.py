"""Structural retrieval over a parsed spreadsheet (item 2).

The structural action is "all rows of sheet S where a column matches" — the
primitive the WBS document tool already uses (``XLSXDocument.sheets``), here
exposed as a pure helper so it is unit-testable with two BOQ layouts (a
row-per-item sheet and a sectioned sheet with header rows) without a live
index. It is lexical and knows nothing about a specific item, column or
project.
"""
from __future__ import annotations

from collections.abc import Sequence


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _contains_sequence(hay: list[str], needle: list[str]) -> bool:
    """True when ``needle`` appears as a contiguous run of words in ``hay``.

    Callers guard the empty-needle case, so this never sees an empty ``needle``.
    """
    n = len(needle)
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def rows_matching(
    rows: Sequence[Sequence[str]],
    term: str,
) -> list[tuple[str, ...]]:
    """Rows where any cell holds ``term`` (case-insensitive, whole words).

    Works for both a row-per-item sheet and a sectioned sheet whose header
    rows are interspersed: the match is on a row's own cells, not its position,
    and a section-title row has no item term so it is not matched.
    """
    needle = _normalize(term).split()
    if not needle:
        return []
    out: list[tuple[str, ...]] = []
    for row in rows:
        cells = tuple(str(c) for c in row)
        hay = _normalize(" ".join(cells)).split()
        if _contains_sequence(hay, needle):
            out.append(cells)
    return out


def render_rows(rows: Sequence[Sequence[str]], header: Sequence[str] = ()) -> str:
    """Rows as aligned TSV text (columns tab-separated) for the LLM.

    A header row, when supplied, is rendered first. Empty leading cells are
    preserved so columns stay aligned.
    """
    lines: list[str] = []
    if header:
        lines.append("\t".join(str(c) for c in header))
    for row in rows:
        lines.append("\t".join(str(c) for c in row))
    return "\n".join(lines)
