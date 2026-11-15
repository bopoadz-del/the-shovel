"""Counts-only geometry diagnostic of a bill plan's word boxes (P-B3).

A plan can report "0 unreconciled" because no row carries all three figures,
and it can carry one page's header columns across many later pages on trust.
This module measures, from the same word boxes the plan read, what the page
geometry itself holds, so a column fix is chosen from counts, not from the
plan's own verdict:

1. ``fields`` - of the plan's rows, how many hold a number in the quantity,
   rate and amount cells, in all three, and how many of those reconcile
   (``bill_table_geometry.reconciles``).
2. ``headers`` - per header line, columns resolved to each canonical field
   and columns left unmapped; of the unmapped, how many hold mostly numbers
   in the cells below them (down to the next header line).
3. ``bands`` - with no header at all: lines that carry numbers in at least
   three vertical bands (a band is a right-edge cluster of numbers that at
   least :data:`MIN_BAND_LINES` lines of the page share), and on how many
   of those the rightmost of three such numbers equals the product of the
   two before it within the plan's own rounding tolerance.
4. ``labels`` - heading-label lines per page (min, median, max), by reason,
   and how many hold a number in a figure-column cell.
5. ``header_left_spread_px`` - per canonical field, the spread of that
   column's left edge over every header line that resolved it.

Every output is a count, a pixel measure or a field / reason code. No word,
label or row text leaves this module. Nothing here names a bill, a page or a
document.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

from app.core import bill_table_geometry as g

#: A numeric right-edge cluster is a vertical band when at least this many
#: lines of the page put a number in it.
MIN_BAND_LINES = 3
#: An unmapped column "holds mostly numbers" when more than this share of
#: its non-empty cells below the header parse as numbers, over at least
#: :data:`MIN_COLUMN_CELLS` cells.
MOSTLY_NUMERIC = 0.5
MIN_COLUMN_CELLS = 3
#: Canonical fields whose cells are figures, as the plan names them.
FIGURE_FIELDS = ("quantity", "rate", "total")


def _numbers(text: str) -> list[float]:
    return [v for v in (g.parse_number(t) for t in (text or "").split()) if v is not None]


def field_counts(rows: Iterable[g.BillRow]) -> dict[str, int]:
    """Item 1: rows with a number in each figure cell, all three, reconciled."""
    out = {"rows": 0, **{f: 0 for f in FIGURE_FIELDS}, "all_three": 0, "reconciled": 0}
    for row in rows:
        out["rows"] += 1
        has = {f: bool(_numbers(row.fields.get(f, ""))) for f in FIGURE_FIELDS}
        for f in FIGURE_FIELDS:
            out[f] += has[f]
        if all(has.values()):
            out["all_three"] += 1
            out["reconciled"] += row.check == ""
    return out


def _pages(words: Iterable[g.Word]) -> dict[int, list[list[g.Word]]]:
    by_page: dict[int, list[g.Word]] = {}
    for w in words:
        by_page.setdefault(w.page, []).append(w)
    return {p: g.group_lines(ws) for p, ws in sorted(by_page.items())}


def header_counts(pages: Mapping[int, list[list[g.Word]]], block) -> dict[str, Any]:
    """Items 2 and 5: per header line, columns by field and unmapped; the
    unmapped columns holding mostly numbers below; left-edge spread per field."""
    headers: list[dict[str, Any]] = []
    lefts: dict[str, list[float]] = {}
    current: list[g._Column] | None = None
    cells: list[list[int]] = []  # per column of the current header: [non-empty, numeric]

    def close() -> None:
        if current is None:
            return
        unmapped = [i for i, c in enumerate(current) if c.field is None]
        headers[-1]["unmapped_mostly_numeric"] = sum(
            1 for i in unmapped
            if cells[i][0] >= MIN_COLUMN_CELLS and cells[i][1] > MOSTLY_NUMERIC * cells[i][0]
        )

    for page in pages:
        for line in pages[page]:
            columns, _reason = g.header_test(line, block)
            if columns is not None:
                close()
                current = columns
                cells = [[0, 0] for _ in columns]
                by_field: dict[str, int] = {}
                for c in columns:
                    if c.field:
                        by_field[c.field] = by_field.get(c.field, 0) + 1
                        lefts.setdefault(c.field, []).append(c.left)
                headers.append({
                    "page": page,
                    "columns": len(columns),
                    "by_field": dict(sorted(by_field.items())),
                    "unmapped": sum(1 for c in columns if c.field is None),
                })
                continue
            if current is None:
                continue
            buckets: list[list[str]] = [[] for _ in current]
            for w in line:
                buckets[g._assign(w, current)].append(w.text)
            for i, bucket in enumerate(buckets):
                if bucket:
                    cells[i][0] += 1
                    cells[i][1] += bool(_numbers(" ".join(bucket)))
    close()
    totals: dict[str, int] = {}
    for h in headers:
        for f, n in h["by_field"].items():
            totals[f] = totals.get(f, 0) + n
    return {
        "header_lines": len(headers),
        "columns_by_field": dict(sorted(totals.items())),
        "columns_unmapped": sum(h["unmapped"] for h in headers),
        "unmapped_mostly_numeric": sum(h["unmapped_mostly_numeric"] for h in headers),
        "per_header": headers,
        "header_left_spread_px": {
            f: {"headers": len(xs), "spread": round(max(xs) - min(xs), 1)}
            for f, xs in sorted(lefts.items())
        },
    }


def _bands(numeric: Sequence[tuple[int, g.Word]], tol: float) -> list[int]:
    """Band index per numeric token (sorted by right edge, a new band where
    the gap to the previous right edge exceeds ``tol``)."""
    order = sorted(range(len(numeric)), key=lambda i: numeric[i][1].right)
    band = [0] * len(numeric)
    current, prev = -1, None
    for i in order:
        r = numeric[i][1].right
        if prev is None or r - prev > tol:
            current += 1
        band[i] = current
        prev = r
    return band


def band_counts(pages: Mapping[int, list[list[g.Word]]]) -> dict[str, int]:
    """Item 3: header-free three-band numeric lines and product lines."""
    out = {"lines": 0, "lines_three_bands": 0, "lines_product": 0,
           "lines_product_nontrivial": 0, "pages_with_product_line": 0}
    for lines in pages.values():
        out["lines"] += len(lines)
        heights = [w.height for ln in lines for w in ln if w.height > 0]
        tol = statistics.median(heights) if heights else 1.0
        numeric = [(li, w) for li, ln in enumerate(lines) for w in ln
                   if g.parse_number(w.text) is not None]
        band = _bands(numeric, tol)
        support: dict[int, set[int]] = {}
        for (li, _w), b in zip(numeric, band):
            support.setdefault(b, set()).add(li)
        vertical = {b for b, ls in support.items() if len(ls) >= MIN_BAND_LINES}
        per_line: dict[int, dict[int, g.Word]] = {}
        for (li, w), b in zip(numeric, band):
            if b in vertical:
                per_line.setdefault(li, {}).setdefault(b, w)  # one token per band
        page_has = False
        for toks in per_line.values():
            if len(toks) < 3:
                continue
            out["lines_three_bands"] += 1
            vals = [g.parse_number(w.text) for w in sorted(toks.values(), key=lambda w: w.left)]
            hits = [(a, b, c) for a, b, c in combinations(vals, 3) if g.reconciles(a, b, c)]
            if hits:
                out["lines_product"] += 1
                page_has = True
            if any(c != 0 and a not in (0, 1) and b not in (0, 1) for a, b, c in hits):
                out["lines_product_nontrivial"] += 1
        out["pages_with_product_line"] += page_has
    return out


def label_counts(events: Sequence[g.LabelEvent], pages: Iterable[int]) -> dict[str, Any]:
    """Item 4: label lines per page (min, median, max), by reason, and the
    labels holding a number in a figure-column cell."""
    per_page = {p: 0 for p in pages}
    reasons: dict[str, int] = {}
    for e in events:
        per_page[e.page] = per_page.get(e.page, 0) + e.lines
        reasons[e.reason] = reasons.get(e.reason, 0) + e.lines
    counts = list(per_page.values()) or [0]
    return {
        "label_lines": sum(counts),
        "per_page_min": min(counts),
        "per_page_median": statistics.median(counts),
        "per_page_max": max(counts),
        "by_reason": dict(sorted(reasons.items())),
        "with_figure_number": sum(e.lines for e in events if e.figure_number),
    }


def geometry_diag(words: Sequence[g.Word], pages: Iterable[int], block=None) -> dict[str, Any]:
    """All five counts over one page range's word boxes (counts only)."""
    block = block or g._resolver()
    events: list[g.LabelEvent] = []
    rows = g.extract_bill_rows(words, block, on_label=events.append)
    lines = _pages(words)
    headers = header_counts(lines, block)
    return {
        "fields": field_counts(rows),
        "headers": {k: v for k, v in headers.items() if k != "header_left_spread_px"},
        "bands": band_counts(lines),
        "labels": label_counts(events, pages),
        "header_left_spread_px": headers["header_left_spread_px"],
    }
