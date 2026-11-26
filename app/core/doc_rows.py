"""Table rows held as facts: the row shape, its readers, reconciliation, query.

A table-like document (a priced bill, a spreadsheet, a rate list) is read
into rows once, at ingest. Each row keeps the locator a citation needs: the
document, the table, the heading path above it, its page (a PDF) or sheet
row (a spreadsheet), and the ordinal of the retrievable chunk made from it.

The readers are the ones that already exist; nothing here reads a cell a
second way:

* spreadsheets: :func:`app.core.doc_index.spreadsheet_rows` (the same reader
  that makes the row chunks, so chunk ``i`` and row ``i`` are one row);
* PDF bills: :func:`app.core.bill_table_geometry.extract_bill_rows`.

Each table gets one of four reconciliation states (:func:`reconcile_table`),
from the figures the table itself prints. :func:`rows_query` answers filter,
lookup, sum, count and compare over the rows with the arithmetic in code,
and every answer names the rows it used.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.bill_table_geometry import BillRow, check_row, parse_number

#: Every check the table makes possible agrees, and there is at least one:
#: a row stating quantity, rate and amount, or a printed total the amounts
#: sum to (within the rounding of the printed figures).
RECONCILED = "RECONCILED"
#: Rows were read, but no row states quantity, rate and amount together and
#: no printed total was given: nothing in the table can be checked.
UNCHECKED = "UNCHECKED"
#: At least one check the table makes possible disagrees.
FAILED = "FAILED"
#: The reader returned no row for the table.
NOT_READ = "NOT READ"

STATES = (RECONCILED, UNCHECKED, FAILED, NOT_READ)

#: The figure fields a row may carry, as the column resolver names them.
FIGURE_FIELDS = ("quantity", "rate", "total")


@dataclass(frozen=True)
class DocRow:
    """One table row with its locator."""

    document_id: str
    table_no: int
    row_no: int
    path: tuple[str, ...] = ()
    fields: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, str] = field(default_factory=dict)
    page: int | None = None
    sheet: str = ""
    sheet_row: int | None = None
    chunk_ordinal: int | None = None
    check: str = ""

    def figure(self, name: str) -> float | None:
        """The row's ``name`` figure as a number, ``None`` when it prints none."""
        return parse_number(self.fields.get(name, ""))

    def locator(self) -> dict[str, Any]:
        """What a citation of this row names."""
        loc: dict[str, Any] = {"document_id": self.document_id,
                               "table_no": self.table_no, "row_no": self.row_no}
        if self.path:
            loc["path"] = list(self.path)
        if self.page is not None:
            loc["page"] = self.page
        if self.sheet:
            loc["sheet"] = self.sheet
        if self.sheet_row is not None:
            loc["sheet_row"] = self.sheet_row
        if self.chunk_ordinal is not None:
            loc["chunk_ordinal"] = self.chunk_ordinal
        return loc


@dataclass(frozen=True)
class PrintedTotal:
    """A total line a table prints: the amount it states and where it stands.

    ``after_row`` is the ``row_no`` of the last measured row above the line
    (0 when none is above it), so the rows the line closes are known.
    """

    table_no: int
    after_row: int
    value: float
    sheet_row: int | None = None


def _spreadsheet_walk(rows: Sequence[Any]) -> Iterable[tuple[int, int, int, Any]]:
    """``(table_no, row_no, ordinal, row)`` over spreadsheet rows, total lines
    included.

    One table per sheet, numbered in the order the sheets are read. ``row_no``
    counts the measured rows of its own table; ``ordinal`` counts measured
    rows across the whole list from 0, so it is the chunk ordinal: the row
    chunks are made from the measured rows of the same list in order. A total
    line carries the ``row_no`` of the last measured row above it in its
    table (0 when none is).
    """
    tables: dict[str, int] = {}
    in_table: dict[int, int] = {}
    ordinal = 0
    for r in rows:
        table_no = tables.setdefault(r.sheet, len(tables) + 1)
        if not getattr(r, "total_line", False):
            in_table[table_no] = in_table.get(table_no, 0) + 1
            ordinal += 1
        yield table_no, in_table.get(table_no, 0), ordinal - 1, r


def rows_from_spreadsheet(document_id: str, rows: Sequence[Any]) -> list[DocRow]:
    """``DocRow`` per measured :class:`app.core.doc_index.SpreadsheetRow`."""
    out: list[DocRow] = []
    for table_no, row_no, ordinal, r in _spreadsheet_walk(rows):
        if getattr(r, "total_line", False):
            continue
        fields = dict(r.fields)
        out.append(DocRow(
            document_id=document_id,
            table_no=table_no,
            row_no=row_no,
            path=tuple(r.path),
            fields=fields,
            extra=dict(r.others),
            sheet=r.sheet,
            sheet_row=r.row_no,
            chunk_ordinal=ordinal,
            check=check_row(fields),
        ))
    return out


def spreadsheet_printed_totals(rows: Sequence[Any]) -> list[PrintedTotal]:
    """The total lines a spreadsheet prints, each with the amount it states."""
    out: list[PrintedTotal] = []
    for table_no, row_no, _, r in _spreadsheet_walk(rows):
        if not getattr(r, "total_line", False):
            continue
        value = parse_number(r.fields.get("total", ""))
        if value is not None:
            out.append(PrintedTotal(table_no, row_no, value, r.row_no))
    return out


def rows_from_bill(document_id: str, rows: Sequence[BillRow], table_no: int = 1) -> list[DocRow]:
    """``DocRow`` per :class:`BillRow` of one bill read, with its page."""
    return [
        DocRow(
            document_id=document_id,
            table_no=table_no,
            row_no=i + 1,
            path=tuple(r.path) or ((r.section,) if r.section else ()),
            fields=dict(r.fields),
            extra=dict(r.extra),
            page=r.source_page,
            check=r.check,
        )
        for i, r in enumerate(rows)
    ]


@dataclass(frozen=True)
class TableCheck:
    """The reconciliation of one table: its state and the counts behind it."""

    state: str
    rows: int
    checked: int
    failed: int
    total_printed: float | None = None
    total_of_rows: float | None = None


def reconcile_table(rows: Sequence[DocRow], printed_total: float | None = None) -> TableCheck:
    """The state of one table from the figures it prints.

    A row is checkable when it states quantity, rate and amount; it fails when
    they disagree (:func:`check_row`). When ``printed_total`` is given, the
    amounts of the rows must sum to it within the rounding the printed
    figures allow (:func:`_rounding_allowance`), never a share of the total:
    a total short by one row's amount is FAILED. A printed total that is not
    a finite number is never agreed with.
    """
    if not rows:
        return TableCheck(NOT_READ, 0, 0, 0, printed_total)
    checked = failed = 0
    for r in rows:
        if all(r.figure(f) is not None for f in FIGURE_FIELDS):
            checked += 1
            failed += 1 if check_row(dict(r.fields)) else 0
    amounts = [a for a in (r.figure("total") for r in rows) if a is not None]
    total_of_rows = sum(amounts) if amounts else None
    total_ok: bool | None = None
    if printed_total is not None:
        # A non-finite printed total is never within a finite allowance.
        total_ok = (total_of_rows is not None
                    and abs(total_of_rows - printed_total) <= _rounding_allowance(
                        [r.fields.get("total", "") for r in rows if r.figure("total") is not None],
                        printed_total))
    if failed or total_ok is False:
        state = FAILED
    elif checked or total_ok:
        state = RECONCILED
    else:
        state = UNCHECKED
    return TableCheck(state, len(rows), checked, failed, printed_total, total_of_rows)


def reconcile_tables(
    rows: Iterable[DocRow], printed_totals: Mapping[int, float] | None = None,
) -> dict[int, TableCheck]:
    """:func:`reconcile_table` for every ``table_no`` in ``rows``."""
    by_table: dict[int, list[DocRow]] = {}
    for r in rows:
        by_table.setdefault(r.table_no, []).append(r)
    totals = printed_totals or {}
    out = {t: reconcile_table(rs, totals.get(t)) for t, rs in by_table.items()}
    for t, total in totals.items():
        out.setdefault(t, reconcile_table([], total))
    return out


def _half_unit(text: str) -> float:
    """Half a unit of the last digit ``text`` prints (``"1,200.50"`` -> 0.005,
    ``"1,200"`` -> 0.5)."""
    digits = re.sub(r"[^0-9.]", "", text or "")
    _, dot, frac = digits.rpartition(".")
    return 0.5 * 10.0 ** -(len(frac) if dot and frac else 0)


def _rounding_allowance(amounts: Sequence[str], printed_total: float) -> float:
    """How far a sum of printed ``amounts`` may sit from ``printed_total``
    by rounding alone: half a unit of each figure's own last printed digit."""
    total_half = 0.5 if float(printed_total).is_integer() else 0.005
    return sum(_half_unit(a) for a in amounts) + total_half


@dataclass(frozen=True)
class TotalLineCheck:
    """One printed total line against the rows it closes.

    A total line states either the rows since the previous total line (a
    subtotal) or every row above it (a running total: carried forward,
    brought forward). ``agrees`` is True when the printed amount matches
    either sum, False when it matches neither, None when no row above it
    prints an amount.
    """

    line: PrintedTotal
    since_previous: float | None
    running: float | None
    agrees: bool | None


def _amount_sum(rows: Iterable[DocRow]) -> tuple[float, list[str]] | None:
    """The sum of the amounts ``rows`` print and the printed texts behind it;
    None when no row prints an amount."""
    priced = [r for r in rows if r.figure("total") is not None]
    if not priced:
        return None
    return sum(r.figure("total") for r in priced), [r.fields.get("total", "") for r in priced]


def _sum_agrees(summed: tuple[float, list[str]], printed: float) -> bool:
    """A sum agrees with a printed total within the rounding its printed
    figures allow (:func:`_rounding_allowance`), never a share of the total."""
    value, texts = summed
    return abs(value - printed) <= _rounding_allowance(texts, printed)


def check_total_lines(rows: Sequence[DocRow], lines: Sequence[PrintedTotal]) -> list[TotalLineCheck]:
    """:class:`TotalLineCheck` for every printed total line, in table order."""
    out: list[TotalLineCheck] = []
    previous: dict[int, int] = {}
    for line in sorted(lines, key=lambda x: (x.table_no, x.after_row)):
        mine = [r for r in rows if r.table_no == line.table_no and r.row_no <= line.after_row]
        start = previous.get(line.table_no, 0)
        since = _amount_sum(r for r in mine if r.row_no > start)
        running = _amount_sum(mine)
        sums = [v for v in (since, running) if v is not None]
        agrees = any(_sum_agrees(v, line.value) for v in sums) if sums else None
        out.append(TotalLineCheck(line, since[0] if since else None,
                                  running[0] if running else None, agrees))
        previous[line.table_no] = line.after_row
    return out


def reconcile_tables_with_lines(
    rows: Sequence[DocRow], lines: Sequence[PrintedTotal],
) -> dict[int, TableCheck]:
    """Each table's state from its rows and the total lines it prints.

    A total line that agrees with neither the rows it closes nor every row
    above it makes the table FAILED; one that agrees makes a table with no
    failing row RECONCILED; a table with no total line keeps the state its
    rows give.
    """
    out = reconcile_tables(rows)
    checks = check_total_lines(rows, lines)
    for t in {c.line.table_no for c in checks}:
        mine = [c for c in checks if c.line.table_no == t]
        base = out.get(t) or reconcile_table([])
        if any(c.agrees is False for c in mine) or base.state == FAILED:
            state = FAILED
        elif any(c.agrees for c in mine):
            state = RECONCILED
        else:
            state = base.state
        out[t] = TableCheck(state, base.rows, base.checked, base.failed,
                            mine[-1].line.value, base.total_of_rows)
    return out


# ── the bill's own skeleton: page labels, listing pages, printed totals ─────
#
# A long priced document prints a label on every page (its running footer
# names the page), and its summary and collection pages list other pages by
# that label with the amount carried from each. Read those pages first and
# the document says which pages make each part and what each part totals;
# rows are then assigned by page, not by a heading guessed from their text.

#: A page's own label sits on a line whose other words recur on at least this
#: share of the pages (the running footer or header).
FURNITURE_SHARE = 0.5


def _is_label(token: str) -> bool:
    """A page-label shape: segments joined by ``/``, the last a page number,
    at least one letter (so a date or a fraction is not a label)."""
    parts = token.split("/")
    return (len(parts) >= 2 and parts[-1].isdigit() and all(p.isalnum() for p in parts)
            and any(c.isalpha() for c in token))


def _line_figure(tokens: Sequence[str]) -> tuple[float, str] | None:
    """The last figure on a line and the text it is printed as, None when
    the line prints none."""
    for t in reversed(tokens):
        if any(c.isdigit() for c in t):
            v = parse_number(t)
            if v is not None:
                return v, t
    return None


def _line_amount(tokens: Sequence[str]) -> float | None:
    """The last figure on a line, None when it prints none."""
    fig = _line_figure(tokens)
    return fig[0] if fig else None


def page_labels(pages: Sequence[Sequence[Sequence[str]]],
                furniture_share: float = FURNITURE_SHARE) -> dict[int, str]:
    """``{page number (1-based): the label the page prints for itself}``.

    ``pages`` is each page's lines, each line its tokens in reading order. A
    page's own label is the one label on a line whose other words recur on at
    least ``furniture_share`` of the pages (the running footer or header);
    a label inside the page body (an index, a cross-reference) is not. A
    label two pages claim is dropped from both. No pages claim nothing.
    """
    seen: dict[str, int] = {}
    for lines in pages:
        for word in {t for line in lines for t in line}:
            seen[word] = seen.get(word, 0) + 1
    claims: dict[int, str] = {}
    for no, lines in enumerate(pages, 1):
        best: tuple[float, str] | None = None
        for line in lines:
            labels = [t for t in line if _is_label(t)]
            words = [t for t in line if not _is_label(t)]
            if len(labels) != 1 or not words:
                continue
            share = sum(seen[w] for w in words) / len(words) / len(pages)
            if share >= furniture_share and (best is None or share > best[0]):
                best = (share, labels[0])
        if best:
            claims[no] = best[1]
    counts: dict[str, int] = {}
    for label in claims.values():
        counts[label] = counts.get(label, 0) + 1
    return {no: label for no, label in claims.items() if counts[label] == 1}


def page_listings(pages: Sequence[Sequence[Sequence[str]]],
                  labels: Mapping[int, str]) -> dict[int, list[tuple[str, float | None]]]:
    """``{page: [(label of a listed page, amount printed for it or None)]}``.

    A listing line names exactly one other page by its label; the amount is
    the last figure after that label (None when the line prints none, e.g. a
    page carried as included). A page with fewer than two such lines lists
    nothing.
    """
    page_of = {label: no for no, label in labels.items()}
    out: dict[int, list[tuple[str, float | None]]] = {}
    for no, lines in enumerate(pages, 1):
        entries: list[tuple[str, float | None]] = []
        for line in lines:
            hits = [i for i, t in enumerate(line) if t in page_of and page_of[t] != no]
            if len(hits) == 1:
                entries.append((line[hits[0]], _line_amount(line[hits[0] + 1:])))
        if len(entries) >= 2:
            out[no] = entries
    return out


@dataclass(frozen=True)
class SkeletonPart:
    """One part of the document as its summary page lists it."""

    label: str
    summary_page: int
    pages: tuple[int, ...]
    printed_total: float | None
    listed: tuple[tuple[int, float | None], ...]


def bill_skeleton(pages: Sequence[Sequence[Sequence[str]]],
                  furniture_share: float = FURNITURE_SHARE) -> tuple[int | None, list[SkeletonPart]]:
    """The summary page and the parts it lists.

    The summary is the listing page no other listing names, the one that
    lists the most pages. Each part is a page it lists: when that page itself
    lists pages (a collection page), the part is every page walked under it, and
    ``listed`` is each page under it that lists nothing, with the amount its
    collection carries for it.
    """
    labels = page_labels(pages, furniture_share)
    listings = page_listings(pages, labels)
    page_of = {label: no for no, label in labels.items()}
    named = {page_of[t] for entries in listings.values() for t, _ in entries}
    roots = [p for p in listings if p not in named]
    if not roots:
        return None, []
    root = max(roots, key=lambda p: (len(listings[p]), p))

    def leaves(no: int, amount: float | None, path: frozenset[int],
               walked: set[int]) -> list[tuple[int, float | None]]:
        walked.add(no)
        if no not in listings or no in path:
            return [(no, amount)]
        out: list[tuple[int, float | None]] = []
        for t, a in listings[no]:
            out += leaves(page_of[t], a, path | {no}, walked)
        return out

    parts = []
    for label, printed in listings[root]:
        top = page_of[label]
        under: set[int] = set()
        listed = leaves(top, printed, frozenset({root}), under)
        parts.append(SkeletonPart(label, top, tuple(sorted(under)), printed, tuple(listed)))
    return root, parts


def summary_totals(pages: Sequence[Sequence[Sequence[str]]],
                   furniture_share: float = FURNITURE_SHARE) -> list[float]:
    """The total lines the summary page prints after its first entry, in order.

    Each part it lists adds its amount; a later line with a figure and no
    label is a total when it equals the amounts since the previous total or
    every amount above it (a sub-total, then a grand total), and otherwise
    an addition of its own (a contingency) that later totals include. The
    last one is the document's grand total.
    """
    root, _ = bill_skeleton(pages, furniture_share)
    if root is None:
        return []
    labels = page_labels(pages, furniture_share)
    listed = {label for label, _ in page_listings(pages, labels)[root]}
    totals: list[float] = []
    since: list[str] = []
    running: list[str] = []
    count = 0
    for line in pages[root - 1]:
        hits = [t for t in line if t in listed]
        if len(hits) == 1:
            fig = _line_figure(line[line.index(hits[0]) + 1:])
        elif count and not any(_is_label(t) for t in line):
            fig = _line_figure(line)
            if fig is not None and any(_texts_agree(texts, fig[0]) for texts in (since, running)):
                totals.append(fig[0])
                since = []
                continue
        else:
            continue
        if fig is not None:
            since.append(fig[1])
            running.append(fig[1])
        count += 1
    return totals


def _texts_agree(texts: Sequence[str], printed: float) -> bool:
    """The figures printed as ``texts`` sum to ``printed`` within the
    rounding they print (:func:`_sum_agrees`)."""
    values = [parse_number(t) for t in texts]
    return _sum_agrees((sum(v for v in values if v is not None), list(texts)), printed)


def page_total_lines(rows: Sequence[DocRow]) -> set[int]:
    """Indexes of rows that are a printed total, not an item.

    Within one page, a row printing an amount with no quantity and no rate,
    whose amount equals the items since the previous such line, is the
    page's carried total, whatever words it prints.
    """
    out: set[int] = set()
    run: list[str] = []
    page = None
    for i, r in enumerate(rows):
        if i == 0 or r.page != page:
            run, page = [], r.page
        amount = r.figure("total")
        if amount is None:
            continue
        if (run and r.figure("quantity") is None and r.figure("rate") is None
                and _texts_agree(run, amount)):
            out.add(i)
            run = []
            continue
        run.append(r.fields.get("total", ""))
    return out


@dataclass(frozen=True)
class PartCheck:
    """One part's item rows against the totals its own summary pages print."""

    part: SkeletonPart
    state: str
    rows: int
    total_of_rows: float | None
    listed_sum: float | None
    pages_agree: int
    pages_disagree: int
    pages_unpriced: int


def reconcile_skeleton(parts: Sequence[SkeletonPart], rows: Sequence[DocRow]) -> list[PartCheck]:
    """Each part's item rows (page totals left out) against its printed total.

    RECONCILED when the items sum to the printed total; FAILED when they do
    not; UNCHECKED when the summary prints no amount for the part; NOT READ
    when no item row with an amount lies on the part's pages.
    ``pages_agree`` / ``pages_disagree`` count the listed pages whose items
    match / miss the amount the collection carries for them;
    ``pages_unpriced`` the listed pages it carries no amount for.
    """
    totals = page_total_lines(rows)
    per_page: dict[int | None, list[str]] = {}
    for i, r in enumerate(rows):
        if i not in totals and r.figure("total") is not None:
            per_page.setdefault(r.page, []).append(r.fields.get("total", ""))
    out = []
    for part in parts:
        amounts = [a for p in part.pages for a in per_page.get(p, [])]
        total = sum(parse_number(a) or 0.0 for a in amounts) if amounts else None
        printed = [a for _, a in part.listed if a is not None]
        agree = disagree = unpriced = 0
        for p, a in part.listed:
            if a is None:
                unpriced += 1
            elif per_page.get(p) and _texts_agree(per_page[p], a):
                agree += 1
            else:
                disagree += 1
        if total is None:
            state = NOT_READ
        elif part.printed_total is None:
            state = UNCHECKED
        elif _texts_agree(amounts, part.printed_total):
            state = RECONCILED
        else:
            state = FAILED
        out.append(PartCheck(part, state, len(amounts), total, sum(printed) if printed else None,
                             agree, disagree, unpriced))
    return out


def lines_of_words(words: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """A page's word boxes (``text``, ``left``, ``top``, ``height``) as lines
    of tokens in reading order: a word starts a new line when its top is more
    than 0.6 of its height from the line's first word."""
    out: list[list[Mapping[str, Any]]] = []
    top: float | None = None
    for w in sorted(words, key=lambda w: (w["top"], w["left"])):
        if top is None or abs(w["top"] - top) > w["height"] * 0.6:
            out.append([w])
            top = w["top"]
        else:
            out[-1].append(w)
    return [[str(w["text"]) for w in sorted(line, key=lambda w: w["left"])] for line in out]


# ── rows_query ──────────────────────────────────────────────────────────────

OPERATIONS = ("filter", "lookup", "sum", "count", "compare")


def _text_of(row: DocRow, column: str) -> str:
    if column == "path":
        return " > ".join(row.path)
    if column in row.fields:
        return row.fields[column]
    return row.extra.get(column, "")


#: A word or a number; ``.``, ``/`` and ``-`` between letters or digits stay
#: inside it, so an item reference like ``1.10`` is one token.
_TOKEN_RE = re.compile(r"[^\W_]+(?:[./-][^\W_]+)*")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).casefold())


def _matches(row: DocRow, where: Mapping[str, str]) -> bool:
    """Every ``column -> text`` in ``where`` appears in the row's cell as
    whole tokens in order, case-insensitively: ``Bill 1`` matches ``Bill 1 >
    Earthworks`` but not ``Bill 10``. An empty ``where`` keeps every row; a
    value with no token matches nothing."""
    for k, v in where.items():
        want, have = _tokens(v), _tokens(_text_of(row, k))
        n = len(want)
        if not n or not any(have[i:i + n] == want for i in range(len(have) - n + 1)):
            return False
    return True


def rows_query(
    rows: Sequence[DocRow],
    operation: str,
    where: Mapping[str, str] | None = None,
    column: str = "total",
    other: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Answer one question over ``rows`` with the arithmetic in code.

    * ``filter``: the rows matching ``where``;
    * ``lookup``: the one row matching ``where`` (an error when none or more);
    * ``sum``: the sum of ``column`` over the matching rows;
    * ``count``: how many rows match;
    * ``compare``: ``column`` summed over ``where`` and over ``other``, and
      the difference (first minus second); an error when either side is not
      given.

    Every result carries ``rows``: the locators of the rows it used. A row
    whose ``column`` prints no number is named under ``skipped``, never
    counted as zero.
    """
    if operation not in OPERATIONS:
        return {"error": f"unknown operation {operation!r}", "operations": list(OPERATIONS)}
    picked = [r for r in rows if _matches(r, where or {})]

    def summed(rs: Sequence[DocRow]) -> dict[str, Any]:
        used = [r for r in rs if r.figure(column) is not None]
        skipped = [r.locator() for r in rs if r.figure(column) is None]
        return {"value": sum(r.figure(column) for r in used) if used else None,
                "column": column, "rows": [r.locator() for r in used],
                "skipped": skipped}

    if operation == "filter":
        return {"rows": [r.locator() | {"fields": dict(r.fields)} for r in picked],
                "count": len(picked)}
    if operation == "count":
        return {"value": len(picked), "rows": [r.locator() for r in picked]}
    if operation == "lookup":
        if len(picked) != 1:
            return {"error": f"{len(picked)} rows match; lookup needs exactly one",
                    "rows": [r.locator() for r in picked]}
        r = picked[0]
        return {"row": r.locator() | {"fields": dict(r.fields), "extra": dict(r.extra)},
                "rows": [r.locator()]}
    if operation == "sum":
        return summed(picked)
    missing = [name for name, side in (("where", where), ("other", other)) if not side]
    if missing:
        return {"error": f"compare needs both sides; missing: {', '.join(missing)}", "rows": []}
    second = [r for r in rows if _matches(r, other or {})]
    a, b = summed(picked), summed(second)
    diff = None if a["value"] is None or b["value"] is None else a["value"] - b["value"]
    return {"first": a, "second": b, "difference": diff,
            "rows": a["rows"] + b["rows"]}
