"""Count and sum bill rows deterministically, over the derived bill sheet (P-B3).

A question that asks how many bill rows match a description, or for the total
quantity or amount of those rows, is answered here and not by the model adding
whichever row chunks retrieval happened to return. Retrieval gives the top K
chunks, so a sum built from them is short by every matching row that was not
retrieved.

The source is the derived spreadsheet ``bill_derived.derive_bill_sheet``
writes: one sheet per heading path, a banner row holding that path (outer to
inner, joined by ``bill_table_geometry._PATH_SEP``), the canonical columns, any
columns the page printed that map to no canonical field, then ``Check`` and
``Source Page``. Every row of that sheet is read; none is sampled.

An item the page break splits is one item. The extractor starts a new item at
a line that prints a reference, but each page starts fresh, so an item whose
description begins at the foot of one page and whose figures (or the rest of
its description) are printed at the head of the next page comes out as two
rows. :func:`join_page_continuations` puts them back together: a row on the
next page with no reference, directly following the previous page's last row
of the same heading path, where one of the two rows carries no figure, is the
rest of that item.

Matching is by words, not by a phrase: every term must appear as a word in
the row's description (and every section term in its heading path). A number
in the terms is a SIZE (:func:`sizes_in`), matched against the sizes the row
prints: a size with a unit is read to metres on both sides, so every spelling
of it with a unit selects the same rows; a bare number is itself, never read
at another magnitude (``1.8`` does not select ``18``, ``180`` or ``1800``).
Nothing here names a bill, a section, a row or a question.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core import bill_table_geometry as g

_LOG = logging.getLogger(__name__)

MEASURE_QUANTITY = "quantity"
MEASURE_AMOUNT = "amount"
MEASURES = (MEASURE_QUANTITY, MEASURE_AMOUNT)

_CANONICAL = {label: name for name, label in g._FIELD_LABELS}
_CHECK_LABEL = "Check"
_PAGE_LABEL = "Source Page"
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass
class SheetRow:
    """One item of the derived sheet, with every page it is printed on."""

    path: tuple[str, ...]
    ref: str
    description: str
    unit: str
    quantity: str
    rate: str
    amount: str
    pages: list[int] = field(default_factory=list)

    def figure(self, measure: str) -> float | None:
        text = self.quantity if measure == MEASURE_QUANTITY else self.amount
        return g.parse_number(text)

    def has_figure(self) -> bool:
        return any(g.parse_number(t) is not None for t in (self.quantity, self.rate, self.amount))


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


_PAGE_RE = re.compile(r"\d+(?:\.0+)?")


def _page(value: Any) -> int | None:
    """The page number a ``Source Page`` cell holds; ``None`` when it holds none.

    A blank or non-numeric cell is a row whose page the sheet did not record,
    not an error, so it is told apart by its text rather than by catching.
    """
    text = "" if value is None else str(value).strip()
    if not _PAGE_RE.fullmatch(text):
        return None
    return int(float(text))


def read_sheet_rows(xlsx_path: str) -> list[SheetRow]:
    """Every row of every sheet of a derived bill workbook, in sheet order.

    ``xlsx_path`` is a stored document: ``bill_derived`` writes the sheet
    through the byte store, which encrypts it when a data key is set, so the
    workbook is read from its plaintext (``file_crypto.open_plaintext``).
    """
    from app.core import file_crypto

    with file_crypto.open_plaintext(xlsx_path) as readable:
        return _read_workbook_rows(readable)


def _read_workbook_rows(xlsx_path: str) -> list[SheetRow]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    out: list[SheetRow] = []
    try:
        for ws in wb.worksheets:
            path: tuple[str, ...] = ()
            header: list[str] | None = None
            for raw in ws.iter_rows(values_only=True):
                cells = [_cell(v) for v in raw]
                if header is None:
                    if "Description" in cells:
                        header = cells
                    elif any(cells):
                        banner = next(c for c in cells if c)
                        path = tuple(p.strip() for p in banner.split(g._PATH_SEP.strip()) if p.strip())
                    continue
                if not any(cells):
                    continue
                named = dict(zip(header, cells))
                # The extra columns sit between the canonical columns and Check,
                # in the order the pages printed them; the reference is the first.
                extras = [h for h in header if h and h not in _CANONICAL
                          and h not in (_CHECK_LABEL, _PAGE_LABEL)]
                page = _page(named.get(_PAGE_LABEL))
                out.append(SheetRow(
                    path=path,
                    ref=named.get(extras[0], "") if extras else "",
                    description=named.get("Description", ""),
                    unit=named.get("Unit", ""),
                    quantity=named.get("Quantity", ""),
                    rate=named.get("Rate", ""),
                    amount=named.get("Amount", ""),
                    pages=[page] if page is not None else [],
                ))
    finally:
        wb.close()
    return out


def join_page_continuations(rows: Sequence[SheetRow]) -> list[SheetRow]:
    """Rejoin each item the page break split into two rows (see module doc)."""
    out: list[SheetRow] = []
    for row in rows:
        prev = out[-1] if out else None
        if (
            prev is not None
            and not row.ref
            and row.path == prev.path
            and row.pages and prev.pages
            and row.pages[0] == prev.pages[-1] + 1
            and not (row.has_figure() and prev.has_figure())
        ):
            prev.description = " ".join(t for t in (prev.description, row.description) if t)
            for attr in ("unit", "quantity", "rate", "amount"):
                if not getattr(prev, attr):
                    setattr(prev, attr, getattr(row, attr))
            prev.pages.append(row.pages[0])
            continue
        out.append(SheetRow(**{**row.__dict__, "pages": list(row.pages)}))
    return out


#: Length units a size is printed in, and their factor to metres (the base
#: unit). Units, not sizes: no size of any bill is listed here.
_LENGTH_UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                 "مم": 0.001, "ملم": 0.001, "سم": 0.01, "م": 1.0, "كم": 1000.0}
#: Arabic-Indic and Eastern Arabic-Indic digits, and the Arabic decimal and
#: thousands separators, read as the Latin ones.
_DIGITS = str.maketrans({**{chr(0x0660 + i): str(i) for i in range(10)},
                         **{chr(0x06F0 + i): str(i) for i in range(10)},
                         "٫": ".", "٬": ","})
_NUM = r"\d+(?:[.,]\d+)*"
_UNIT = "(?:" + "|".join(sorted(map(re.escape, _LENGTH_UNITS), key=len, reverse=True)) + r")(?![^\W_])"
_MEMBER = rf"{_NUM}(?:\s*{_UNIT})?"
_SIZE_SEP = r"\s*(?:x|×|\*|by|في)\s*"
#: A size: one number, or numbers joined by ``x`` (a dimension), each with
#: its unit when printed. Not a number glued to a letter (a code: ``DN150``).
_SIZE_RE = re.compile(rf"(?<![^\W_])(?<!\.){_MEMBER}(?:{_SIZE_SEP}{_MEMBER})*", re.IGNORECASE)
_MEMBER_RE = re.compile(rf"({_NUM})\s*({_UNIT})?", re.IGNORECASE)


def _number(text: str) -> float:
    """``1.8`` and ``1,8`` -> 1.8; ``1,800`` and ``1,800.5`` -> thousands."""
    seps = re.findall(r"[.,]", text)
    if not seps:
        return float(text)
    last = seps[-1]
    head, _, tail = text.rpartition(last)
    if all(s == last for s in seps) and (len(seps) > 1 or (last == "," and len(tail) == 3)):
        # One mark repeated, or a lone comma before three digits: thousands.
        return float(text.replace(last, ""))
    # The last mark is the decimal point; any other mark groups thousands.
    return float(re.sub(r"[.,]", "", head) + "." + tail)


@dataclass(frozen=True)
class Size:
    """A size as printed (``text``), its members as the plain numbers printed
    (``printed``) and, when a unit is printed on any member, each member's
    length in metres (``metres``; ``None`` for a size with no unit)."""

    text: str
    printed: tuple[float, ...]
    metres: tuple[float, ...] | None

    def same(self, other: Size) -> bool:
        """Two sizes with a unit are one size when their lengths in metres
        are equal (``1.8 x 1.8 m`` is ``1800 x 1800 mm``). A size with no
        unit is the plain number it prints: it is the same as another size
        only when the printed numbers are equal, never at another magnitude
        (a bare ``1.8`` is not ``18``, ``180`` or ``1800``)."""
        if self.metres is not None and other.metres is not None:
            return self.metres == other.metres
        return self.printed == other.printed


def sizes_in(text: str) -> list[Size]:
    """Every size printed in ``text``.

    A member with no unit takes the unit printed on the next member, else on
    the one before (``1.8 x 1.8 m`` is metres throughout); a size with no unit
    on any member borrows none: it keeps only its printed numbers."""
    out: list[Size] = []
    for m in _SIZE_RE.finditer((text or "").translate(_DIGITS)):
        members = [(_number(n), (u or "").casefold()) for n, u in _MEMBER_RE.findall(m.group(0))]
        units = [u for _, u in members]
        metres: tuple[float, ...] | None = None
        if any(units):
            filled = []
            for i, u in enumerate(units):
                after = next((x for x in units[i:] if x), "")
                filled.append(_LENGTH_UNITS[u or after or next(x for x in reversed(units[:i]) if x)])
            metres = tuple(round(v * f, 9) for (v, _), f in zip(members, filled))
        out.append(Size(" ".join(m.group(0).split()), tuple(v for v, _ in members), metres))
    return out


def _without_sizes(text: str) -> str:
    return _SIZE_RE.sub(" ", (text or "").translate(_DIGITS))


def _words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RE.findall(text or "")}


def _terms(terms: Iterable[str] | str | None) -> set[str]:
    if isinstance(terms, str):
        return _words(terms)
    return set().union(*(_words(t) for t in terms)) if terms else set()


def matches(row: SheetRow, terms: set[str], section_terms: set[str], *,
            sizes: Sequence[Size] = ()) -> bool:
    """Every term is a word of the description, every section term a word of
    the heading path, and every size is printed (as a size, in any spelling
    of it, :meth:`Size.same`) in the heading path or the description."""
    if terms and not terms <= _words(row.description):
        return False
    if sizes:
        printed = sizes_in(" ".join(row.path)) + sizes_in(row.description)
        if not all(any(want.same(s) for s in printed) for want in sizes):
            return False
    return not section_terms or section_terms <= _words(" ".join(row.path))


def aggregate(
    rows: Sequence[SheetRow],
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None = None,
    measure: str = MEASURE_QUANTITY,
) -> dict[str, Any]:
    """Count and sum the items matching ``terms`` (and ``section``).

    Quantities are summed per unit: adding cubic metres to metres is not a
    total, so a mixed-unit match returns one sum per unit and no single sum.
    A matching item without the figure is counted, listed, and named in
    ``without_figure``; it adds nothing to a sum.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    # A size in the call's terms (a number, with its unit when printed, or a
    # dimension ``a x b``) is matched as a size, never as loose digit words:
    # ``1.8`` read as words is ``1`` and ``8`` and selects any row printing
    # both anywhere. A bare number is itself (:meth:`Size.same`).
    texts = [terms] if isinstance(terms, str) else list(terms or [])
    want, where = _terms([_without_sizes(t) for t in texts]), _terms(section)
    asked = [s for text in texts for s in sizes_in(text)]
    if not want and not where and not asked:
        raise ValueError("give description terms or a section to match")
    items = join_page_continuations(rows)
    used, without = [], []
    sums: dict[str, float] = {}
    for row in items:
        if not matches(row, want, where, sizes=asked):
            continue
        value = row.figure(measure)
        entry = {
            "ref": row.ref,
            "pages": row.pages,
            "section": list(row.path),
            "description": row.description,
            "unit": row.unit,
            measure: value,
        }
        used.append(entry)
        if value is None:
            without.append({"ref": row.ref, "pages": row.pages, "description": row.description})
            continue
        key = row.unit if measure == MEASURE_QUANTITY else ""
        sums[key] = sums.get(key, 0.0) + value
    single = len(sums) <= 1
    return {
        "measure": measure,
        # The sizes are named with the words: the statement says what the
        # rows were selected by, and a size is part of that.
        "terms": sorted(want) + [s.text for s in asked],
        "section_terms": sorted(where),
        "sizes": [s.text for s in asked],
        "rows_scanned": len(items),
        "count": len(used),
        "sum": (next(iter(sums.values()), 0.0) if single else None),
        "unit": (next(iter(sums), "") if single and measure == MEASURE_QUANTITY else None),
        "sum_by_unit": ({u: s for u, s in sums.items()} if measure == MEASURE_QUANTITY else None),
        "without_figure": without,
        "rows": used,
    }


NO_DERIVED_SHEET = "no_derived_bill_sheet"
NO_READABLE_BYTES = "derived_sheet_has_no_readable_bytes"
UNREADABLE_SHEET = "derived_sheet_is_not_a_readable_workbook"
#: Every reason a bill (or the whole call) did not return; the tool declares
#: these, and they are the only error words of it a user is shown.
REASON_CODES = (NO_DERIVED_SHEET, NO_READABLE_BYTES, UNREADABLE_SHEET)


def _unreadable_sheet_errors() -> tuple[type[BaseException], ...]:
    """What reading a stored sheet raises when its bytes are not a workbook
    this key can open: not a zip (or a cut one), not an xlsx inside, or
    encrypted under another key. That is the bill's error, not the turn's."""
    import zipfile

    from openpyxl.utils.exceptions import InvalidFileException

    from app.core import file_crypto

    return (zipfile.BadZipFile, InvalidFileException, file_crypto.DecryptionError)


def derived_bill_documents(project_id: str, document_id: str | None = None) -> list[dict[str, Any]]:
    """The visible derived bill sheets of the project; with ``document_id``,
    only the one derived from (or being) that document."""
    from app.core import bill_derived, projects

    out = []
    for doc in projects.list_documents(project_id):
        if doc.get("retrieval_visible") is False:
            continue
        meta = projects.coerce_document_metadata(doc.get("metadata"))
        if meta.get("derivation") != bill_derived.DERIVATION:
            continue
        if document_id and document_id not in (doc.get("id"), meta.get("derived_from")):
            continue
        out.append(doc)
    return out


def aggregate_project_bills(
    project_id: str,
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None = None,
    measure: str = MEASURE_QUANTITY,
    document_id: str | None = None,
) -> dict[str, Any]:
    """:func:`aggregate` over every row of each derived bill sheet in scope,
    one result per bill (two bills are never added together)."""
    from app.core import projects

    docs = derived_bill_documents(project_id, document_id)
    if not docs:
        out = {"status": "error", "reason": NO_DERIVED_SHEET}
        _LOG.info("%s", selection_log_line(out, terms=terms, section=section, measure=measure))
        return out
    bills = []
    for doc in docs:
        path = projects.materialize_document_path(doc)
        meta = projects.coerce_document_metadata(doc.get("metadata"))
        head = {"document_id": doc.get("id"), "filename": doc.get("original_name"),
                "derived_from": meta.get("derived_from")}
        if not path:
            bills.append({**head, "status": "error", "reason": NO_READABLE_BYTES})
            continue
        try:
            rows = read_sheet_rows(path)
        except _unreadable_sheet_errors() as exc:
            _LOG.warning("derived bill sheet %s is not readable: %s", doc.get("id"), type(exc).__name__)
            bills.append({**head, "status": "error", "reason": UNREADABLE_SHEET})
            continue
        result = aggregate(rows, terms=terms, section=section, measure=measure)
        bills.append({**head, "status": "success", **result})
    out = {"status": "success", "bills": bills}
    _LOG.info("%s", selection_log_line(out, terms=terms, section=section, measure=measure))
    return out


def _digest(items: Any) -> str:
    """A short, stable fingerprint of ``items``: equal inputs give equal
    digests, so two calls can be compared from the log without the log
    holding any of the words or rows themselves."""
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _set_field(words: set[str]) -> str:
    return f"{len(words)}:{_digest(sorted(words)) if words else '-'}"


def selection_log_line(
    result: dict[str, Any],
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None,
    measure: str,
) -> str:
    """One line naming which term set and which rows a call selected.

    The term sets are logged as size and digest of the casefolded word set
    (the matcher's own view of them, so word order and case do not change
    it); each bill as its status, rows scanned, rows counted and a digest of
    the selected rows (reference, pages and heading path). Two asks that
    selected different rows show different row digests; whether they also
    show different term digests says whether the term choice was the cause.
    No description text and no figure is written.
    """
    parts = [
        "bill_rows_aggregate",
        f"measure={measure}",
        f"terms={_set_field(_terms(terms))}",
        f"section={_set_field(_terms(section))}",
        f"status={result.get('status')}",
    ]
    if result.get("reason"):
        parts.append(f"reason={result['reason']}")
    bills = result.get("bills") or []
    parts.append(f"bills={len(bills)}")
    for bill in bills:
        if bill.get("status") != "success":
            parts.append(f"[{bill.get('document_id')} {bill.get('status')} reason={bill.get('reason')}]")
            continue
        selected = [[r.get("ref"), r.get("pages"), r.get("section")] for r in bill.get("rows") or []]
        parts.append(
            f"[{bill.get('document_id')} success scanned={bill.get('rows_scanned')} "
            f"count={bill.get('count')} without_figure={len(bill.get('without_figure') or [])} "
            f"rows={_digest(selected) if selected else '-'}]"
        )
    return " ".join(parts)
