"""A table drawn from a computed row set prints every row of that set, or says
which it left out.

A computed tool that counts and sums rows (``bill_rows_aggregate``) hands the
turn the exact rows behind its total: their labels and the figure each one
carries (``tool_registry.Statement.rows`` / ``row_figures``). The model then
prints those rows as a table. Nothing checks that table against the row set
it was drawn from: ``answer_row_totals`` reads only the answer's own text, so
when the table silently drops a row of the set while the answer states the
set's total, the text alone cannot tell a dropped row from a wrong total.
Measured on the live bill band (21 Nov, build f4181ebf): the tool counted 13
rows on every one of six identical asks, and one answer printed 12 of them,
summing to one less than the total it stated, and then explained the gap in
prose with a row it did not print.

The rule, decided from the turn's own computed result and the answer's own
table: a table is a rendering of a row set when every data row it prints
names exactly one distinct row of the set (by the set's own row label) and
the figure it prints for that row is the figure the set carries for it. A
row names a set row when one of its cells is that row's label, or, when no
cell is a whole label, when every word cell it prints is a piece of that
one row's label and of no other: a bill whose
reference sits inside the description cell gives each row its description
as its label, and a table prints that label split across cells (the band in
one, the reference in another). When
such a table leaves rows of the set out, and the answer states the set's
total, the rows it left out are named under the table with the figures the
set carries for them, so the printed rows and the named rows together are the
stated total. The table itself is not rewritten and nothing is guessed: a
such a table leaves rows of the set out, and the answer states the set's
total, the rows it left out are named under the table with the figures the
set carries for them, so the printed rows and the named rows together are the
stated total. The table itself is not rewritten and nothing is guessed: a
table with a row the set does not hold, a row printed twice, a figure that
is not the set's, or a set that states no per-row figures, is left alone.

Nothing here names a project, a document, a bill, an item, a reference or a
figure: the row labels and figures come from the tool's result on the turn,
and the table from the answer.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.agents.answer_row_totals import (
    _TOTAL_CELL_RE,
    _agrees,
    _cells,
    _number,
    _table_blocks,
)

_LOG = logging.getLogger(__name__)

#: The page a row label carries after its reference ("K 133 (p.12)").
_LABEL_PAGE_RE = re.compile(r"\s*\(\s*p\.[^)]*\)\s*$")

#: A figure printed in the answer's prose.
_PROSE_FIGURE_RE = re.compile(r"(?<![\d.,])\d[\d,]*(?:\.\d+)?")

#: A table has at least this many rows of the set before it is read as a
#: rendering of that set.
_MIN_MATCHED = 2


@dataclass(frozen=True)
class RowSet:
    """One computed total and the rows behind it."""

    total: float
    unit: str
    labels: tuple[str, ...]
    figures: tuple[float, ...]


def _key(text: str) -> str:
    """A row's name compared without spacing, case or its page note."""
    return re.sub(r"\s+", "", _LABEL_PAGE_RE.sub("", text or "")).lower()


def row_sets(statements: Sequence[Any]) -> list[RowSet]:
    """The summed row sets the turn's computed statements state, each once.

    Only a sum whose every row carries a figure and a distinct name is a set
    a table can be checked against.
    """
    out: list[RowSet] = []
    seen: set[tuple] = set()
    for st in statements:
        if getattr(st, "basis", "") != "sum" or getattr(st, "figure", None) is None:
            continue
        labels = tuple(getattr(st, "rows", ()) or ())
        figures = tuple(getattr(st, "row_figures", ()) or ())
        if len(labels) < _MIN_MATCHED or len(figures) != len(labels):
            continue
        if any(f is None for f in figures):
            continue
        if len({_key(label) for label in labels}) != len(labels):
            continue
        row_set = RowSet(float(st.figure), str(getattr(st, "unit", "") or ""),
                         labels, tuple(float(f) for f in figures))
        identity = (row_set.total, row_set.unit, tuple(_key(x) for x in labels))
        if identity in seen:
            continue
        seen.add(identity)
        out.append(row_set)
    return out


def _row_match(row: list[str], keys: dict[str, int]) -> list[int]:
    """The set rows a table row names: one index per cell that is a row label,
    or, when no cell is a whole label, the one row whose label holds every
    word cell the table row prints (``_piece_match``)."""
    whole = [keys[k] for k in (_key(cell) for cell in row) if k in keys]
    return whole or _piece_match(row, keys)


#: Typographic variants a table prints for the characters a label holds.
_FOLD = str.maketrans({"–": "-", "—": "-", "−": "-", "×": "x"})

#: A cell that is only a figure: it is checked against the set's figures,
#: never read as part of a row's name.
_BARE_FIGURE_RE = re.compile(r"[\d,]*\.?\d+")


def _fold(key: str) -> str:
    return key.translate(_FOLD)


def _piece_match(row: list[str], keys: dict[str, int]) -> list[int]:
    """The one set row whose label holds, as pieces, every word cell of ``row``.

    A word cell is a cell that is not a bare figure (a figure is checked
    against the set's figures, not read as a name). A row whose word cells
    fit no label, or more than one (words the rows of a set share, or a row
    that prints figures only), names nothing.
    """
    pieces = [_fold(_key(cell)) for cell in row]
    pieces = [p for p in pieces if p and not _BARE_FIGURE_RE.fullmatch(p)]
    hits = [index for label, index in keys.items()
            if all(p in _fold(label) for p in pieces)]
    return hits if len(hits) == 1 else []


def _figure_column(rows: list[list[str]], indices: list[int], row_set: RowSet) -> int | None:
    """The column whose every cell is the figure the set carries for that row."""
    width = min(len(r) for r in rows)
    for col in range(width - 1, -1, -1):
        values = [_number(r[col]) for r in rows]
        if all(v is not None and _agrees(row_set.figures[i], v)
               for v, i in zip(values, indices)):
            return col
    return None


def _states_total(lines: list[str], table: set[int], total: float) -> bool:
    """True when the answer's prose (outside the table) prints ``total``."""
    for index, line in enumerate(lines):
        if index in table:
            continue
        for match in _PROSE_FIGURE_RE.finditer(line):
            try:
                value = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            if _agrees(total, value):
                return True
    return False


@dataclass
class _Omission:
    after_line: int
    row_set: RowSet
    printed: int
    missing: list[int]
    sample: str


def _block_omission(block: list[int], cells: list[list[str] | None],
                    lines: list[str], row_set: RowSet) -> _Omission | None:
    keys = {_key(label): i for i, label in enumerate(row_set.labels)}
    data: list[list[str]] = []
    indices: list[int] = []
    for position, line_index in enumerate(block):
        row = cells[line_index] or []
        first = next(((c or "").strip() for c in row if (c or "").strip()), "")
        if _TOTAL_CELL_RE.match(first):
            continue
        if not any((c or "").strip().strip("-: ") for c in row):
            continue  # a markdown separator row
        named = _row_match(row, keys)
        if not named:
            if position == 0:
                continue  # the header
            return None  # a row the set does not hold: not a rendering of it
        if len(set(named)) != 1:
            return None
        data.append(row)
        indices.append(named[0])
    if len(indices) < _MIN_MATCHED or len(set(indices)) != len(indices):
        return None
    missing = [i for i in range(len(row_set.labels)) if i not in set(indices)]
    if not missing:
        return None
    col = _figure_column(data, indices, row_set)
    if col is None:
        return None
    if not _states_total(lines, set(block), row_set.total):
        return None
    return _Omission(block[-1], row_set, len(indices), missing, data[0][col])


def _figure_text(value: float, sample: str) -> str:
    """``value`` at its own precision (a figure is never rounded to fit),
    grouped with thousands separators when the table's figures are."""
    grouped = "," in (sample or "")
    body = f"{value:.6f}".rstrip("0").rstrip(".")
    whole, _, decimals = body.partition(".")
    head = f"{int(whole):,}" if grouped else whole
    return f"{head}.{decimals}" if decimals else head


def _note(omission: _Omission) -> str:
    row_set = omission.row_set
    unit = f" {row_set.unit}" if row_set.unit else ""
    total = _figure_text(row_set.total, omission.sample)
    count = len(row_set.labels)
    left = ", ".join(
        f"{row_set.labels[i]}: {_figure_text(row_set.figures[i], omission.sample)}{unit}"
        for i in omission.missing
    )
    rows = "row" if len(omission.missing) == 1 else "rows"
    return (
        f"The {total}{unit} total counts {count} rows; the table above prints "
        f"{omission.printed} of them. Not printed ({len(omission.missing)} {rows}, "
        f"from the same computed row set): {left}."
    )


def _name_dropped(raw: str, statements: Sequence[Any]) -> str:
    sets = row_sets(statements)
    lines = raw.split("\n")
    cells = [_cells(line) for line in lines]
    omissions: list[_Omission] = []
    for block in _table_blocks(cells):
        for row_set in sets:
            found = _block_omission(block, cells, lines, row_set)
            if found is not None:
                omissions.append(found)
                break
    if not omissions:
        return raw
    for omission in sorted(omissions, key=lambda o: o.after_line, reverse=True):
        lines[omission.after_line + 1:omission.after_line + 1] = ["", _note(omission)]
    _LOG.info("counted_rows_named tables=%d rows=%d", len(omissions),
              sum(len(o.missing) for o in omissions))
    return "\n".join(lines)


def name_dropped_rows(text: str, statements: Sequence[Any]) -> str:
    """Name the rows of a computed row set that the answer's table leaves out.

    ``statements`` are the turn's computed statements. An answer whose tables
    print every row of the set they render, or that renders no set, is
    returned byte for byte. Never raises: a check that can break an answer is
    a check that gets switched off.
    """
    raw = text or ""
    try:
        return _name_dropped(raw, statements or ())
    except Exception:  # noqa: BLE001 - a check must never break a turn
        _LOG.warning("counted-row check failed; answer passed through", exc_info=True)
        return raw
