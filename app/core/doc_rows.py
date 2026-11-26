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
from the total the document prints for it. This table is the whole rule; the
code and ``tests/test_doc_rows.py::test_the_state_table`` follow it:

==========================================  ==========  =======================
the table as read                           state       charged to
==========================================  ==========  =======================
no printed total (none, or blank text)      UNCHECKED   nothing to check
printed total the checker cannot parse      NOT READ    the checker, never the
(no one figure in its text), or no rows                 document
rows' amounts add up to the printed total   RECONCILED  -
(to the document's own rounding)
rows' amounts do not add up to it           FAILED      the table as read
==========================================  ==========  =======================

:func:`rows_query` answers
filter, lookup, sum, count and compare over the rows with the arithmetic in
code, compares typed cell values, and every answer names the rows it used.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.bill_table_geometry import BillRow, _resolver, check_row, parse_number

#: The amounts of the rows read add up to the total the document prints, to
#: the document's own rounding. Nothing else is RECONCILED.
RECONCILED = "RECONCILED"
#: Rows were read but the document prints no total for the table: whatever
#: its rows say, the table as read cannot be checked against the document.
UNCHECKED = "UNCHECKED"
#: A printed total exists and the amounts read do not add up to it.
FAILED = "FAILED"
#: The checker could not read the table: no row came back, or the printed
#: total's text holds no one figure it can parse. The checker's limit, never
#: charged to the document.
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


def rows_from_spreadsheet(document_id: str, rows: Sequence[Any]) -> list[DocRow]:
    """``DocRow`` per :class:`app.core.doc_index.SpreadsheetRow`.

    One table per sheet, numbered in the order the sheets are read. The
    chunk ordinal is the row's position in ``rows``: the row chunks are made
    from the same list in the same order.
    """
    tables: dict[str, int] = {}
    in_table: dict[int, int] = {}
    out: list[DocRow] = []
    for i, r in enumerate(rows):
        table_no = tables.setdefault(r.sheet, len(tables) + 1)
        in_table[table_no] = in_table.get(table_no, 0) + 1
        fields = dict(r.fields)
        out.append(DocRow(
            document_id=document_id,
            table_no=table_no,
            row_no=in_table[table_no],
            path=tuple(r.path),
            fields=fields,
            extra=dict(r.others),
            sheet=r.sheet,
            sheet_row=r.row_no,
            chunk_ordinal=i,
            check=check_row(fields),
        ))
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
    """The reconciliation of one table: its state and the counts behind it.

    ``checked``/``failed`` count the rows stating quantity, rate and amount
    and those whose arithmetic disagrees; they are reported, the state does
    not rest on them.
    """

    state: str
    rows: int
    checked: int
    failed: int
    total_printed: float | None = None
    total_of_rows: float | None = None
    #: A printed total was given as text but no one number could be read from
    #: it: the state is NOT READ, the checker's limit, not the document's.
    total_unread: bool = False


def reconcile_table(
    rows: Sequence[DocRow], printed_total: float | str | None = None,
) -> TableCheck:
    """The state of one table against the total the document prints for it.

    * no rows: NOT READ;
    * no printed total (``None`` or blank text): UNCHECKED;
    * a printed text with no one number the checker can parse from it (no
      figure, or more than one): NOT READ with ``total_unread``; the checker
      failed, not the document, so the table is not charged FAILED;
    * RECONCILED only when the amounts of the rows add up to the printed
      total within half a unit of the last digit the document prints
      (:func:`_rounding_allowance`); otherwise FAILED. One row's amount
      missing from the read is therefore FAILED, whatever the row count.

    ``printed_total`` given as the printed text is read for its one number,
    whatever currency or mark prints beside it (``"AED 6,000"``,
    ``"6,000/-"``), and held to that number's own last digit (``"6,000"`` to
    0.5, ``"6,000.00"`` to 0.005); given as a number,
    to the finest digit the table prints, in its amounts or in the number.
    A printed total that is not a finite number is never agreed with.
    """
    unread = False
    if isinstance(printed_total, str):
        printed_total = _printed_figure(printed_total)
        unread = printed_total == ""
        printed_total = printed_total or None
    if not rows or unread:
        return TableCheck(NOT_READ, len(rows), 0, 0, _as_number(printed_total), total_unread=unread)
    checked = failed = 0
    for r in rows:
        if all(r.figure(f) is not None for f in FIGURE_FIELDS):
            checked += 1
            failed += 1 if check_row(dict(r.fields)) else 0
    with_amount = [r for r in rows if r.figure("total") is not None]
    total_of_rows = sum(r.figure("total") for r in with_amount) if with_amount else None
    total = _as_number(printed_total)
    if printed_total is None:
        state = UNCHECKED
    elif (total is not None and total_of_rows is not None and math.isfinite(total)
          and abs(total_of_rows - total) <= _rounding_allowance(
              [r.fields.get("total", "") for r in with_amount], printed_total)):
        state = RECONCILED
    else:
        state = FAILED
    return TableCheck(state, len(rows), checked, failed, total, total_of_rows, unread)


#: One printed figure: digits with comma grouping and an optional decimal part.
_FIGURE_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _printed_figure(text: str) -> str | None:
    """The one figure a printed total's text holds, as printed.

    ``None`` when the text is blank (no total); the text itself when the
    number reader takes it whole (``"1 200.00"``); else the single figure
    among currency or other marks (``"AED 6,000"`` -> ``"6,000"``,
    ``"6000."`` -> ``"6000"``); ``""`` when the text holds no figure or more
    than one, so which is the total cannot be told."""
    if not text.strip():
        return None
    if parse_number(text) is not None:
        return text
    found = _FIGURE_RE.findall(text)
    return found[0] if len(found) == 1 and parse_number(found[0]) is not None else ""


def _as_number(value: float | str | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return parse_number(value)
    return float(value)


def reconcile_tables(
    rows: Iterable[DocRow], printed_totals: Mapping[int, float | str] | None = None,
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


def _decimals(text: str) -> int:
    """How many digits ``text`` prints after its decimal point."""
    digits = re.sub(r"[^0-9.]", "", text or "")
    _, dot, frac = digits.rpartition(".")
    return len(frac) if dot and frac else 0


def _rounding_allowance(amounts: Sequence[str], printed_total: float | str) -> float:
    """How far the sum of the amounts read may sit from the printed total:
    half a unit of the last digit the document prints for the total. As
    text, the total's own digits say it; as a number, the finest digit the
    amounts or the number print. It does not grow with the row count."""
    if isinstance(printed_total, str):
        return _half_unit(printed_total)
    number = float(printed_total)
    places = [_decimals(a) for a in amounts]
    if not number.is_integer():
        places.append(_decimals(repr(number)))
    return 0.5 * 10.0 ** -max(places, default=0)


# ── rows_query ──────────────────────────────────────────────────────────────

OPERATIONS = ("filter", "lookup", "sum", "count", "compare")
#: How a ``where`` value meets a cell. ``exact`` (the default) compares the
#: typed value; ``contains`` is a partial text match, used only when asked
#: for by name and named in the result.
MATCHES = ("exact", "contains")


def _cells(row: DocRow, column: str) -> list[str]:
    """The cell(s) ``column`` names: the path is the whole path and each of
    its headings; any other column is its one cell (absent: none)."""
    if column == "path":
        return [" > ".join(row.path), *row.path] if row.path else []
    if column in row.fields:
        return [row.fields[column]]
    if column in row.extra:
        return [row.extra[column]]
    return []


def _norm(text: Any) -> str:
    """Case and spacing removed: ``"BILL No. 1"`` and ``"bill no.1"`` meet."""
    return re.sub(r"\s+", "", str(text)).casefold()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    return parse_number(str(value))


def _cell_matches(column: str, cell: str, value: Any, match: str) -> bool:
    if match == "exact" and column in FIGURE_FIELDS:
        want, have = _number(value), parse_number(cell)
        return want is not None and have is not None and want == have
    want = _norm(value)
    if not want:
        return False
    return want in _norm(cell) if match == "contains" else want == _norm(cell)


def _matches(row: DocRow, where: Mapping[str, Any], match: str = "exact") -> bool:
    """Every ``column -> value`` in ``where`` meets one of the row's cells for
    that column (:func:`_cells`). Exact: a figure column (quantity, rate,
    total) compares numbers, so ``200`` is not ``1,200`` and ``1200`` is
    ``1,200.00``; any other column compares the whole cell with case and
    spacing removed, so ``Bill 1`` is not ``Bill 10``. ``contains`` is a
    partial text match on any column. An empty ``where`` keeps every row; a
    value that is empty, or not a number on a figure column, matches nothing."""
    return all(any(_cell_matches(k, c, v, match) for c in _cells(row, k))
               for k, v in where.items())


def rows_query(
    rows: Sequence[DocRow],
    operation: str,
    where: Mapping[str, str] | None = None,
    column: str = "total",
    other: Mapping[str, str] | None = None,
    match: str = "exact",
) -> dict[str, Any]:
    """Answer one question over ``rows`` with the arithmetic in code.

    * ``filter``: the rows matching ``where``;
    * ``lookup``: the one row matching ``where`` (an error when none or more);
    * ``sum``: the sum of ``column`` over the matching rows;
    * ``count``: how many rows match;
    * ``compare``: ``column`` summed over ``where`` and over ``other``, and
      the difference (first minus second); an error when either side is not
      given.

    ``where`` and ``other`` are matched as :func:`_matches` says; ``match``
    is ``exact`` unless ``contains`` is asked for by name, and every result
    that is not an error says which was used.

    Every result carries ``rows``: the locators of the rows it used. A row
    whose ``column`` prints no number is named under ``skipped``, never
    counted as zero.
    """
    if operation not in OPERATIONS:
        return {"error": f"unknown operation {operation!r}", "operations": list(OPERATIONS)}
    if match not in MATCHES:
        return {"error": f"unknown match {match!r}", "matches": list(MATCHES), "rows": []}
    named = [*(where or {}), *(other or {}), column]
    columns: dict[str, str] = {}
    for name in named:
        resolved = _resolve_column(rows, name)
        if isinstance(resolved, list):
            return {"error": f"column {name!r} names more than one column",
                    "candidates": resolved, "rows": []}
        if resolved is None:
            return {"error": f"unknown column {name!r}",
                    "columns": sorted(_known_columns(rows)), "rows": []}
        columns[name] = resolved
    for side in (where, other):
        targets = [columns[k] for k in side or {}]
        if len(set(targets)) < len(targets):
            twice = sorted(k for k in side or {} if targets.count(columns[k]) > 1)
            return {"error": f"{', '.join(map(repr, twice))} name the same column",
                    "rows": []}
    out = _rows_query(rows, operation, _renamed(where, columns), columns[column],
                      _renamed(other, columns), match)
    if "error" not in out:
        out["match"] = match
        changed = {k: v for k, v in columns.items() if k != v}
        if changed:
            out["columns"] = changed
    return out


#: The fields the row readers resolve a column to (``DocRow.fields`` keys).
READ_FIELDS = ("description", "unit", *FIGURE_FIELDS)


def _known_columns(rows: Sequence[DocRow]) -> set[str]:
    """Every column a query may name over ``rows``: the path, the fields the
    readers resolve, and each column the rows carry under its own header."""
    known = {"path", *READ_FIELDS}
    for r in rows:
        known.update(r.fields)
        known.update(r.extra)
    return known


def _resolve_column(rows: Sequence[DocRow], name: str) -> str | list[str] | None:
    """The column ``name`` means: itself when the rows know it; else the
    field its header text resolves to through the readers' own alias table
    (``"Amount"`` -> ``total``, ``"Qty"`` -> ``quantity``); a list of the
    candidates when the text resolves to more than one field and no alias
    matches it whole; ``None`` when it names no column the rows know."""
    known = _known_columns(rows)
    if name in known:
        return name
    block = _resolver()
    hits = [f for f in block._resolve_columns([name]) if f in known]
    if len(hits) > 1:
        whole = block._normalize_col(name)
        hits = [f for f in hits if whole in block._COL_MAP.get(f, ())] or hits
    if len(hits) > 1:
        return sorted(hits)
    return hits[0] if hits else None


def _renamed(where: Mapping[str, Any] | None, columns: Mapping[str, str]) -> dict[str, Any] | None:
    if where is None:
        return None
    return {columns[k]: v for k, v in where.items()}


def _rows_query(rows: Sequence[DocRow], operation: str, where: Mapping[str, str] | None,
                column: str, other: Mapping[str, str] | None, match: str) -> dict[str, Any]:
    picked = [r for r in rows if _matches(r, where or {}, match)]

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
    second = [r for r in rows if _matches(r, other or {}, match)]
    a, b = summed(picked), summed(second)
    diff = None if a["value"] is None or b["value"] is None else a["value"] - b["value"]
    return {"first": a, "second": b, "difference": diff,
            "rows": a["rows"] + b["rows"]}


# ── the doc_rows table ──────────────────────────────────────────────────────

_table_ready_for: str | None = None


def _ensure_table() -> None:
    """Create ``doc_rows`` on a fresh SQLite DB (Alembic 0041 owns PostgreSQL)."""
    global _table_ready_for
    from app.core.db import engine, get_database_url
    from app.core.models import DocRowRecord

    url = get_database_url()
    if _table_ready_for != url:
        DocRowRecord.__table__.create(bind=engine, checkfirst=True)
        _table_ready_for = url


def store_document_rows(
    project_id: str, doc_id: str, rows: Sequence[DocRow], *, keep_existing: bool = False,
) -> int:
    """Write one document's rows; returns how many were written.

    The document's rows are its reader's whole output, so a re-index replaces
    them. With ``keep_existing`` (the additive backfill) a document that
    already has rows is left as it is and nothing is written.
    """
    import json
    import uuid
    from datetime import datetime, timezone

    from app.core.db import SessionLocal
    from app.core.models import DocRowRecord

    _ensure_table()
    now = datetime.now(timezone.utc).isoformat()
    with SessionLocal() as session:
        mine = session.query(DocRowRecord).filter(
            DocRowRecord.project_id == project_id, DocRowRecord.doc_id == doc_id)
        if keep_existing and mine.first() is not None:
            return 0
        mine.delete(synchronize_session=False)
        session.add_all(
            DocRowRecord(
                id=str(uuid.uuid4()), project_id=project_id, doc_id=doc_id,
                table_no=r.table_no, row_no=r.row_no,
                path=json.dumps(list(r.path)), fields=json.dumps(dict(r.fields)),
                extra=json.dumps(dict(r.extra)), page=r.page, sheet=r.sheet or None,
                sheet_row=r.sheet_row, chunk_index=r.chunk_ordinal,
                row_check=r.check, created_at=now,
            )
            for r in rows
        )
        session.commit()
    return len(rows)


def load_rows(project_id: str, doc_id: str | None = None) -> list[DocRow]:
    """The stored rows of a project (or one of its documents), in table order."""
    import json

    from app.core.db import SessionLocal
    from app.core.models import DocRowRecord

    _ensure_table()
    with SessionLocal() as session:
        q = session.query(DocRowRecord).filter(DocRowRecord.project_id == project_id)
        if doc_id is not None:
            q = q.filter(DocRowRecord.doc_id == doc_id)
        recs = q.order_by(DocRowRecord.doc_id, DocRowRecord.table_no, DocRowRecord.row_no).all()
        return [
            DocRow(
                document_id=rec.doc_id, table_no=rec.table_no, row_no=rec.row_no,
                path=tuple(json.loads(rec.path)), fields=json.loads(rec.fields),
                extra=json.loads(rec.extra), page=rec.page, sheet=rec.sheet or "",
                sheet_row=rec.sheet_row, chunk_ordinal=rec.chunk_index, check=rec.row_check,
            )
            for rec in recs
        ]
