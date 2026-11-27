"""Rows from a model's read of bill pages: the LLM says which figure is which, code checks.

The derived bill sheet (``bill_derived``) is written by
``bill_table_geometry.write_bill_workbook`` from ``BillRow`` objects. Where
the page's word geometry cannot place a figure, a model reads the page and
returns, per page, its lines as JSON::

    {"type": "item", "ref": "A", "description": "...", "unit": "m3",
     "quantity": 1277, "rate": 148, "amount": 188996,
     "figures": {"<a money column the page prints>": "<its value>"}}

This module turns those lines into ``BillRow`` objects for the same writer.
It decides nothing about the figures; it only states facts about them:

* a ``heading`` line is not a row; a run of consecutive heading lines is the
  heading path of the rows that follow it, under the label of the bill the
  read was for (the caller passes it, taken from the document's own summary);
* every other line is a row and keeps the model's line type in the
  ``Line Type`` column, so a collection or a carried line stays in the sheet
  and the sheet's own arithmetic (``bill_aggregate``) can confirm it;
* any further money column the page prints (``figures``) is kept as its own
  column under the label the page gives it; nothing is forced into rate or
  amount;
* each row carries its check: ``""`` when accepted, otherwise what failed
  (a figure that is not a number, or quantity x rate != amount). A figure
  that fails is kept as read and flagged; it is never repaired;
* a page whose read is missing or is not a list of lines is NOT READ, never
  "a page with no rows";
* a key of the read that is not a page number (decimal digits only) is not dropped
  in silence: it is listed in ``keys_not_pages`` so the caller can say so.

No word of any line is read: types come from the model, checks from cells.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core import bill_table_geometry as g

HEADING = "heading"
LINE_TYPE_LABEL = "Line Type"
REF_LABEL = "Item"
CHECK_NOT_A_NUMBER = "figure not a number"

#: The model's figure keys, mapped to the sheet's canonical fields.
_FIGURES = (("quantity", "quantity"), ("rate", "rate"), ("amount", "total"))


@dataclass
class PageReadRows:
    """Rows of a model read, the pages that gave none, keys that are not pages, flagged rows."""

    rows: list[g.BillRow] = field(default_factory=list)
    pages_not_read: list[int] = field(default_factory=list)
    keys_not_pages: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[g.BillRow]:
        return [r for r in self.rows if r.check]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _figure(value: Any) -> tuple[str, bool]:
    """The cell text for one figure and whether it reads as a number.

    A string is kept exactly as the model gave it (its printed places carry the
    bill's own rounding). A number is written without a spurious ``.0``.
    """
    if value is None or value == "":
        return "", True
    if isinstance(value, bool):
        return str(value), False
    if isinstance(value, int):
        return str(value), True
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value), False
        return (str(int(value)) if value.is_integer() else repr(value)), True
    text = _text(value)
    return text, g.parse_number(text) is not None


def rows_from_page_reads(
    pages: Mapping[Any, Any], bill_label: str = "",
) -> PageReadRows:
    """``BillRow`` objects from a model's per-page line reads, pages in order."""
    out = PageReadRows()
    keyed: list[tuple[int, Any]] = []
    for key, lines in pages.items():
        name = _text(key)
        if name.isdecimal():
            keyed.append((int(name), lines))
        else:
            out.keys_not_pages.append(name)
    keyed.sort(key=lambda kv: kv[0])
    top = (bill_label.strip(),) if bill_label.strip() else ()
    headings: list[str] = []
    after_row = True
    for page, lines in keyed:
        if (not isinstance(lines, list) or not lines
                or not all(isinstance(line, Mapping) for line in lines)):
            out.pages_not_read.append(page)
            continue
        for line in lines:
            kind = _text(line.get("type")).lower()
            if kind == HEADING:
                if after_row:
                    headings = []
                    after_row = False
                label = _text(line.get("description"))
                if label:
                    headings.append(label)
                continue
            after_row = True
            out.rows.append(_row(line, page, top + tuple(headings), kind))
    return out


def _row(line: Mapping[str, Any], page: int, path: tuple[str, ...], kind: str) -> g.BillRow:
    fields = {"description": _text(line.get("description")),
              "unit": _text(line.get("unit"))}
    bad = False
    for key, name in _FIGURES:
        text, ok = _figure(line.get(key))
        fields[name] = text
        bad = bad or not ok
    extra: dict[str, str] = {}
    ref = _text(line.get("ref"))
    if ref:
        extra[REF_LABEL] = ref
    extra[LINE_TYPE_LABEL] = kind
    printed = line.get("figures")
    if isinstance(printed, Mapping):
        for label, value in printed.items():
            name = _text(label)
            if not name:
                continue
            text, ok = _figure(value)
            extra[name] = text
            bad = bad or not ok
    check = CHECK_NOT_A_NUMBER if bad else g.check_row(fields)
    return g.BillRow(
        section=path[-1] if path else "",
        source_page=page,
        fields=fields,
        extra=extra,
        check=check,
        path=path,
    )


def rows_of_model_reads(
    read_page: Callable[[Any], Any], bill_label: str = "",
) -> Callable[[Sequence[Any]], tuple[list[g.BillRow], list[int]]]:
    """A ``bill_derived.RowsOfReads`` from a per-page model reader.

    ``read_page`` gets one ``bill_classify.PageRead`` (0-based ``index``) and
    returns the model's lines for that page (or anything else, which is NOT
    READ). The rows and the 1-based pages not read are what
    :func:`rows_from_page_reads` gives for those pages."""

    def rows_of(page_reads: Sequence[Any]) -> tuple[list[g.BillRow], list[int]]:
        pages = {str(r.index + 1): read_page(r) for r in page_reads}
        got = rows_from_page_reads(pages, bill_label)
        return got.rows, got.pages_not_read

    return rows_of
