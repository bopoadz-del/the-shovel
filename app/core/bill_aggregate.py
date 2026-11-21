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
the row's description (and every section term in its heading path). Nothing
here names a bill, a section, a row or a question.

When the caller passes the ask itself (``ask=``), the scope is read off the
ask rather than off the model's per-turn word set (see :class:`AskScope`): a
size and a depth band are compared as numbers with the cells of the row's own
line, a bill named in the ask admits only the rows whose own path names it,
and the ask's other words scope by the cell the document prints them in. The
same ask therefore selects the same rows on every trial.
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


def _words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RE.findall(text or "")}


def _terms(terms: Iterable[str] | str | None) -> set[str]:
    if isinstance(terms, str):
        return _words(terms)
    return set().union(*(_words(t) for t in terms)) if terms else set()


def matches(row: SheetRow, terms: set[str], section_terms: set[str]) -> bool:
    if terms and not terms <= _words(row.description):
        return False
    return not section_terms or section_terms <= _words(" ".join(row.path))


# ── the scope an ask states ──────────────────────────────────────────────────
#
# A count or a total over bill rows is scoped by what the ask says, not by the
# word set the model happens to pass on one turn: the same ask must select the
# same rows on every trial. Three phrases in it carry a scope, and each is read
# as a NUMBER or as a structural cell of the row's own line, never as a token:
#
# * a size -- "W x H m" -> the two dimensions, compared numerically with
#   the dimensions the row's own description states. The word splitting that
#   turns "W.5m" into "W" and "5m" cannot satisfy a size, and neither can an
#   item code that happens to carry one of its digits.
# * a band -- "LO to HI m", "LO-HI m" -> the two ends of a depth band,
#   compared numerically with the band cell of the row ITSELF (never with a
#   neighbour's, a header's or the previous row's label).
# * a bill -- "Bill No. <n>" -> a row is admitted only when its own path names
#   that bill: a bill heading, or the document's page label.
#
# The ask's remaining words scope by the cell the document prints them in: a
# word the document prints in heading paths (a network, a section, a sheet
# head) scopes structurally, a word it prints in item descriptions scopes the
# item, and a word it prints nowhere scopes nothing -- it cannot select a row
# and must not empty the answer.

_DIM = r"\d+(?:\.\d+)?"
#: A size: two dimensions multiplied, each with an optional unit.
_SIZE_RE = re.compile(
    rf"({_DIM})\s*(mm|cm|m)?\s*[x\u00d7X]\s*({_DIM})\s*(mm|cm|m)?(?!\w)", re.IGNORECASE,
)
#: A depth band: two ends joined by "to", a hyphen or a dash, with a unit.
_BAND_RE = re.compile(
    rf"({_DIM})\s*(mm|cm|m)?\s*(?:to|[\-\u2013\u2014])\s*({_DIM})\s*(mm|cm|m)\b",
    re.IGNORECASE,
)
#: A bill named in an ask: "Bill No. <n>", "bill <n>", "Bill Number <n><a>".
_ASK_BILL_RE = re.compile(r"\bbill\s*(?:no\.?|number|#)?\s*(\d{1,3}[a-z]?)\b", re.IGNORECASE)
#: A bill named by the document's own page label: "Page D/13/2", "D/13/10".
_PAGE_LABEL_RE = re.compile(r"\b([A-Za-z]{1,4})\s*/\s*(\d{1,3}[a-z]?)\s*/\s*\d{1,4}\b")

_UNIT_M = {"mm": 0.001, "cm": 0.01, "m": 1.0}
#: A dimension printed with no unit at all is metres unless it is too large to
#: be one: a bill prints a pipe in millimetres as a bare 4-digit number.
_UNITLESS_MM_FROM = 100.0
_SCOPE_STOP = frozenset({
    "and", "the", "for", "how", "many", "much", "what", "which", "all", "any",
    "are", "was", "were", "its", "with", "from", "that", "this", "total", "totals",
    "unit", "rate", "rated", "billed", "bill", "bills", "number", "count", "counted",
    "value", "amount", "quantity", "price", "priced", "depth", "depths", "together",
    "across", "does", "did", "into", "per", "each", "row", "rows", "item", "items",
    "give", "list", "show", "tell", "please", "there", "their", "over", "under",
})


@dataclass(frozen=True)
class AskScope:
    """The scope an ask states, read as numbers and structural cells.

    ``size``/``band`` are in metres; ``bills`` are the bill numbers the ask
    names; ``level``/``item``/``either`` are the ask's content words graded by
    where the document prints them (a heading path, an item description, or
    both). Empty dimensions do not scope.
    """

    size: tuple[float, float] | None = None
    band: tuple[float, float] | None = None
    bills: frozenset[str] = frozenset()
    level: frozenset[str] = frozenset()
    item: frozenset[str] = frozenset()
    either: frozenset[str] = frozenset()

    def states(self) -> bool:
        return bool(self.size or self.band or self.bills or self.level or self.item or self.either)

    def states_a_dimension(self) -> bool:
        """Whether the ask states the scope a word set cannot express: a size,
        a depth band or a bill. Those are read as numbers and as structural
        cells, so they -- not a per-turn token set -- govern the row set."""
        return bool(self.size or self.band or self.bills)

    def admits(self, row: SheetRow) -> bool:
        """Whether the row is inside every scope the ask states."""
        if self.size is not None and _row_size(row) != self.size:
            return False
        if self.band is not None and _row_band(row) != self.band:
            return False
        if self.bills and not (row_bills(row) & self.bills):
            return False
        if self.level and not self.level <= _row_path_words(row):
            return False
        if self.item and not self.item <= _words(row.description):
            return False
        if self.either and not (self.either & (_words(row.description) | _row_path_words(row))):
            return False
        return True


def _fold(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _scope_words(text: str) -> set[str]:
    """The ask's content words, folded, with the words a bill question uses to
    ask rather than to name an item left out (they name no row anyway)."""
    out = set()
    for word in _WORD_RE.findall((text or "").casefold()):
        if len(word) < 3 or word.isdigit() or word in _SCOPE_STOP:
            continue
        out.add(_fold(word))
    return out


def _row_path_words(row: SheetRow) -> set[str]:
    return {_fold(w) for w in _words(" ".join(row.path))}


def _desc_words(row: SheetRow) -> set[str]:
    return {_fold(w) for w in _words(row.description)}


def _dimensions(match: re.Match[str]) -> tuple[float, float]:
    """The two ends of a dimension expression, in metres.

    A unit written on either end applies to both; a size written with no unit
    at all is metres unless both ends are too large for that, in which case it
    is millimetres (a bill prints a chamber in millimetres as bare numbers)."""
    first, unit_a, second, unit_b = match.groups()
    values = (float(first), float(second))
    unit = (unit_a or unit_b or "").lower()
    if unit:
        scale = _UNIT_M.get(unit, 1.0)
        return (round(values[0] * scale, 6), round(values[1] * scale, 6))
    if min(values) >= _UNITLESS_MM_FROM:
        return (round(values[0] / 1000.0, 6), round(values[1] / 1000.0, 6))
    return values


def _row_size(row: SheetRow) -> tuple[float, float] | None:
    """The size the row's own description states, in metres, or ``None``."""
    found = _SIZE_RE.search(row.description or "")
    return _dimensions(found) if found else None


def _row_band(row: SheetRow) -> tuple[float, float] | None:
    """The depth band of the row's OWN band cell, in metres, or ``None``.

    The band is the row's own description when it states one, else the deepest
    heading-path level of that row that is a band. A neighbour's label, the
    sheet's header and the previous row's band are never used."""
    found = _BAND_RE.search(row.description or "")
    if found:
        return _dimensions(found)
    for level in reversed(row.path):
        found = _BAND_RE.search(level or "")
        if found:
            return _dimensions(found)
    return None


def row_bills(row: SheetRow) -> set[str]:
    """The bill numbers the row's own path names.

    A bill heading (``Bill No. <n> - ...``) or the document's page label
    (``D/13/2``: part, bill, page) printed in one of the row's own heading
    levels. A path that names no bill names none.
    """
    found: set[str] = set()
    for level in row.path:
        for match in _ASK_BILL_RE.finditer(level or ""):
            found.add(match.group(1).upper())
        for match in _PAGE_LABEL_RE.finditer(level or ""):
            found.add(match.group(2).upper())
    return found


def ask_scope(ask: str | None, items: Sequence[SheetRow]) -> AskScope | None:
    """The scope ``ask`` states over ``items``; ``None`` when it states none.

    The ask's words are graded by the cell the document prints them in, so a
    word that names a heading level scopes structurally and a word that names
    an item scopes the description. A word the document prints nowhere is left
    out: it cannot select a row, and requiring it would empty the answer.
    """
    text = (ask or "").strip()
    if not items:
        return None
    if not text:
        return None
    size = _SIZE_RE.search(text)
    band = _BAND_RE.search(text)
    bills = frozenset(m.group(1).upper() for m in _ASK_BILL_RE.finditer(text))
    in_paths: set[str] = set()
    in_desc: set[str] = set()
    for row in items:
        in_paths |= _row_path_words(row)
        in_desc |= _desc_words(row)
    asked = _scope_words(text)
    scope = AskScope(
        size=_dimensions(size) if size else None,
        band=_dimensions(band) if band else None,
        bills=bills,
        level=frozenset(asked & in_paths - in_desc),
        item=frozenset(asked & in_desc - in_paths),
        either=frozenset(asked & in_paths & in_desc),
    )
    return scope if scope.states() else None


def aggregate(
    rows: Sequence[SheetRow],
    *,
    terms: Iterable[str] | str | None,
    section: Iterable[str] | str | None = None,
    measure: str = MEASURE_QUANTITY,
    ask: str | None = None,
) -> dict[str, Any]:
    """Count and sum the items matching ``terms`` (and ``section``).

    Quantities are summed per unit: adding cubic metres to metres is not a
    total, so a mixed-unit match returns one sum per unit and no single sum.
    A matching item without the figure is counted, listed, and named in
    ``without_figure``; it adds nothing to a sum.

    With ``ask``, a size, a depth band or a bill the ask states governs the
    row set (:func:`ask_scope`): those are read as numbers and as the row's own
    structural cells, so the same ask selects the same rows whatever ``terms``
    the model passed on the turn. The model's terms are then reported but not
    applied -- an unparsed token set never defines a scope the ask states. An
    ask that states none of the three leaves ``terms`` and ``section`` to
    select, exactly as before, so a question asked in several parts still gets
    one result per part.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    want, where = _terms(terms), _terms(section)
    items = join_page_continuations(rows)
    scope = ask_scope(ask, items)
    if scope is not None and not scope.states_a_dimension() and (want or where):
        # The ask names no size, band or bill: the model's terms select, and an
        # ask word only selects when the model passed none.
        scope = None
    if scope is None and not want and not where:
        raise ValueError("give description terms or a section to match")
    used, without = [], []
    sums: dict[str, float] = {}
    for row in items:
        admitted = scope.admits(row) if scope is not None else matches(row, want, where)
        if not admitted:
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
        "terms": sorted(want),
        "section_terms": sorted(where),
        "scope": ({
            "size": list(scope.size) if scope.size else None,
            "band": list(scope.band) if scope.band else None,
            "bills": sorted(scope.bills),
            "level": sorted(scope.level),
            "item": sorted(scope.item),
            "either": sorted(scope.either),
        } if scope is not None else None),
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
    ask: str | None = None,
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
        result = aggregate(rows, terms=terms, section=section, measure=measure, ask=ask)
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
