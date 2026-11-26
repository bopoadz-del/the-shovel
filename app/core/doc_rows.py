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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.bill_table_geometry import BillRow, check_row, parse_number, reconciles

#: Every quantity x rate a table states agrees with its amount, and at least
#: one row states all three (and a printed total, when given, agrees too).
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


def rows_from_spreadsheet(document_id: str, rows: Sequence[Any]) -> list[DocRow]:
    """``DocRow`` per :class:`app.core.doc_index.SpreadsheetRow`.

    One table per sheet, numbered in the order the sheets are read. The
    chunk ordinal is the row's position in ``rows``: the row chunks are made
    from the same list in the same order.
    """
    tables: dict[str, int] = {}
    out: list[DocRow] = []
    for i, r in enumerate(rows):
        table_no = tables.setdefault(r.sheet, len(tables) + 1)
        fields = dict(r.fields)
        out.append(DocRow(
            document_id=document_id,
            table_no=table_no,
            row_no=i + 1,
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
    amounts of the rows must sum to it within the bill rounding tolerance.
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
        total_ok = total_of_rows is not None and reconciles(1.0, total_of_rows, printed_total)
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


# ── rows_query ──────────────────────────────────────────────────────────────

OPERATIONS = ("filter", "lookup", "sum", "count", "compare")


def _text_of(row: DocRow, column: str) -> str:
    if column == "path":
        return " > ".join(row.path)
    if column in row.fields:
        return row.fields[column]
    return row.extra.get(column, "")


def _matches(row: DocRow, where: Mapping[str, str]) -> bool:
    """Every ``column -> text`` in ``where`` is contained in the row's cell,
    case-insensitively. An empty ``where`` keeps every row."""
    return all(str(v).casefold() in _text_of(row, k).casefold() for k, v in where.items())


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
      the difference (first minus second).

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
    second = [r for r in rows if _matches(r, other or {})]
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
