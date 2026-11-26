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

A dimension the ask asks to total (``the total billed length``, ``how many
metres of pipe``) admits only rows whose own Unit cell measures that dimension
and that carry the figure: a row measured in counts, a section's carry total
and a row with no figure are not lengths, so they never enter a length total.
Each side is read through the same vocabulary -- the ask's word and the row's
unit cell are the same table -- so neither side is a list of spellings.
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
            # A line printing only an amount is a page total: it closes its
            # page, so it is never the first half of an item the break split.
            and not _amount_only(prev)
        ):
            prev.description = " ".join(t for t in (prev.description, row.description) if t)
            for attr in ("unit", "quantity", "rate", "amount"):
                if not getattr(prev, attr):
                    setattr(prev, attr, getattr(row, attr))
            prev.pages.append(row.pages[0])
            continue
        out.append(SheetRow(**{**row.__dict__, "pages": list(row.pages)}))
    return out


ROW_ITEM = "item"
ROW_COLLECTION = "collection"
ROW_REPEAT = "repeat"
ROW_KINDS = (ROW_ITEM, ROW_COLLECTION, ROW_REPEAT)


def _amount_only(row: SheetRow) -> bool:
    """A line that prints an amount and neither a quantity nor a rate: the
    shape of a page or section total, read off the cells, not the words."""
    return (g.parse_number(row.amount) is not None
            and g.parse_number(row.quantity) is None
            and g.parse_number(row.rate) is None)


def _last_place(text: str) -> float:
    """One unit in the last decimal place the amount cell prints: the
    rounding the document itself states for that figure (``1,234.50`` ->
    0.01, ``1,234`` -> 1)."""
    t = (text or "").strip().strip("()-").replace(",", "").replace(" ", "")
    if "." not in t:
        return 1.0
    return 10.0 ** -len(t.split(".", 1)[1])


def _same_figure(a: float, b: float, tol: float) -> bool:
    """``a`` and ``b`` agree within ``tol`` (the binary float error of the
    subtraction itself is not a printed difference)."""
    return abs(a - b) <= tol * (1 + 1e-9)


def row_kinds(items: Sequence[SheetRow]) -> list[str]:
    """The kind of each row, decided by arithmetic, in the order of ``items``.

    Rows are walked page by page (stable, so the sheet order is kept inside a
    page). A line that prints only an amount is

    * a ``collection`` when that amount equals the sum of the lines it follows
      on its page since the last collection (at least one line), within one
      unit of the amount cell's own last printed place;
    * a ``repeat`` when it equals a collection already read on an earlier page
      (a figure brought forward, or a page total listed on a collection page).
      A repeat is covered by the next collection of its page, as an item is.

    Every other row, and every row with no recorded page, is an ``item``. A
    bill's total is the sum of its items; a collection or a repeat restates
    figures that are already in that sum.
    """
    return _classify(items)[0]


def _classify(items: Sequence[SheetRow]) -> tuple[list[str], set[int]]:
    """:func:`row_kinds`, plus the indices of the collections whose covered
    lines all print only an amount: a line that adds up other total lines (the
    page totals a collection page lists), i.e. the total the document prints
    for the pages it lists. It is read off the cells, so an item the read got
    wrong does not hide the total it must be checked against."""
    kinds = [ROW_ITEM] * len(items)
    summaries: set[int] = set()
    order = sorted((i for i, r in enumerate(items) if r.pages), key=lambda i: items[i].pages[-1])
    earlier: list[tuple[float, float]] = []   # (amount, tolerance) of collections on earlier pages
    page: int | None = None
    page_collections: list[tuple[float, float]] = []
    covered, lines, totals = 0.0, 0, 0
    for i in order:
        row = items[i]
        if row.pages[-1] != page:
            earlier.extend(page_collections)
            page, page_collections, covered, lines, totals = row.pages[-1], [], 0.0, 0, 0
        amount = g.parse_number(row.amount)
        if amount is None:
            continue
        tol = _last_place(row.amount)
        if _amount_only(row):
            if lines and _same_figure(amount, covered, tol):
                kinds[i] = ROW_COLLECTION
                if totals == lines:
                    summaries.add(i)
                page_collections.append((amount, tol))
                covered, lines, totals = 0.0, 0, 0
                continue
            if any(_same_figure(amount, a, max(t, tol)) for a, t in earlier):
                kinds[i] = ROW_REPEAT
            totals += 1
        covered += amount
        lines += 1
    return kinds, summaries


STATE_RECONCILED = "RECONCILED"
STATE_FAILED = "FAILED"
STATE_UNCHECKED = "UNCHECKED"


def bill_checks(items: Sequence[SheetRow], bills: Iterable[str]) -> list[dict[str, Any]]:
    """The arithmetic check of each bill in ``bills`` over its own rows.

    ``items`` are the rows after :func:`join_page_continuations`. A row is in a
    bill when its own path names the bill (:func:`row_bills`). Per bill:

    * ``rows_summed`` -- the bill's item rows that carry an amount, and
      ``items_sum`` their sum; ``rows_without_amount`` -- its item rows that
      carry none (they add nothing; they are reported, not guessed);
    * ``printed_total`` -- the bill's own total line, found by arithmetic: a
      collection whose covered lines all print only an amount (the page
      totals a collection page lists), the last such line of the bill by
      page. No word of the line is read;
    * ``state`` -- ``UNCHECKED`` when the bill prints no such line,
      ``RECONCILED`` when ``items_sum`` equals it within one unit of the
      printed total's own last place, ``FAILED`` otherwise.

    The sum of a bill that does not reconcile is never stated as the bill's
    total: the state goes with it.
    """
    kinds, summaries = _classify(items)
    out: list[dict[str, Any]] = []
    for bill in sorted({_bill_number(b) for b in bills}):
        mine = [i for i, row in enumerate(items) if bill in row_bills(row)]
        amounts = [g.parse_number(items[i].amount) for i in mine if kinds[i] == ROW_ITEM]
        summed = [a for a in amounts if a is not None]
        total = sum(summed)
        lines = sorted((i for i in mine if i in summaries), key=lambda i: items[i].pages[-1])
        check: dict[str, Any] = {
            "bill": bill,
            "rows_summed": len(summed),
            "rows_without_amount": len(amounts) - len(summed),
            "items_sum": total,
            "printed_total": None,
            "printed_page": None,
            "state": STATE_UNCHECKED,
        }
        if lines:
            line = items[lines[-1]]
            printed = g.parse_number(line.amount)
            check["printed_total"] = printed
            check["printed_page"] = line.pages[-1]
            check["state"] = (STATE_RECONCILED if _same_figure(total, printed, _last_place(line.amount))
                              else STATE_FAILED)
        out.append(check)
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
# * a dimension -- "the total billed length" -> the measurement the ask wants
#   totalled. A row is admitted only when its own Unit cell measures that
#   dimension and the row carries the figure; one vocabulary reads the ask's
#   noun and the row's unit cell the same way, so a count ("No", "Nr"), a
#   section's carry total ("sum") and a row with no figure cannot satisfy it.
#
# The ask's remaining words scope by the cell the document prints them in: a
# word the document prints in heading paths (a network, a section, a sheet
# head) scopes structurally, a word it prints in item descriptions scopes the
# item, and a word it prints nowhere scopes nothing -- it cannot select a row
# and must not empty the answer.

#: How a bill separates the groups of a long number and where it puts the
#: decimal mark: a tender prints both conventions, so a number is read the same
#: whichever it is written in. Groups of three are joined by a comma, a space,
#: a no-break space or a thin space; a comma with one or two digits behind it
#: is the decimal mark (``2,5`` is two and a half, ``1,500`` is fifteen hundred).
_GROUP = "[,\u00a0\u2009 ]"
_GROUPED_SEP_RE = re.compile(rf"(?<=\d){_GROUP}(?=\d{{3}}(?!\d))")
_DIM = rf"\d{{1,3}}(?:{_GROUP}\d{{3}})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
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
#: A bill number may be grouped like any other number (``Bill No. 1,313``), so
#: the groups are part of the number and never a second bill.
_ASK_BILL_RE = re.compile(
    r"\bbill\s*(?:no\.?|number|#)?\s*(\d{1,3}(?:,\d{3})+[a-z]?|\d+[a-z]?)\b", re.IGNORECASE,
)
#: A bill named by the document's own page label: "Page D/13/2", "D/13/10".
_PAGE_LABEL_RE = re.compile(r"\b([A-Za-z]{1,4})\s*/\s*(\d{1,3}[a-z]?)\s*/\s*\d{1,4}\b")

_UNIT_M = {"mm": 0.001, "cm": 0.01, "m": 1.0}
#: A dimension printed with no unit at all is metres unless it is too large to
#: be one: a bill prints a pipe in millimetres as a bare 4-digit number.
_UNITLESS_MM_FROM = 100.0

#: The dimensions a quantity is measured in. One vocabulary reads BOTH sides of
#: a dimensional test: the unit a document prints in a row's own Unit cell and
#: the noun an ask uses for the figure it wants totalled. Nothing here names a
#: document, a row, a bill or a question -- these are the spellings a bill and
#: an ask use for a measurement.
LENGTH = "length"
AREA = "area"
VOLUME = "volume"
MASS = "mass"
COUNT = "count"
TIME = "time"
TOTAL = "total"

#: Unit spellings a document prints in its Unit cell, normalised to letters and
#: digits: ``m?``, ``m>`` and ``M`` are one unit, ``| sum`` is ``sum``. Only the
#: singular is listed; a plural printed with a trailing ``s`` is read through
#: the same entry by :func:`_read`, so ``Mtrs`` and ``Mtr`` are one unit.
_UNIT_DIMENSION = {
    "m": LENGTH, "mm": LENGTH, "cm": LENGTH, "km": LENGTH, "mtr": LENGTH,
    "metre": LENGTH, "meter": LENGTH,
    "in": LENGTH, "inch": LENGTH, "inches": LENGTH, "ft": LENGTH, "feet": LENGTH,
    "m2": AREA, "cm2": AREA, "mm2": AREA, "ft2": AREA, "sqm": AREA, "sqft": AREA,
    "sft": AREA,
    "m3": VOLUME, "cm3": VOLUME, "cum": VOLUME, "litre": VOLUME, "liter": VOLUME,
    "kg": MASS, "g": MASS, "t": MASS, "ton": MASS, "tonne": MASS,
    "tonnage": MASS,
    "no": COUNT, "nr": COUNT, "ea": COUNT, "each": COUNT,
    "pc": COUNT, "piece": COUNT,
    "day": TIME, "hr": TIME, "hour": TIME,
    "sum": TOTAL, "total": TOTAL, "subtotal": TOTAL, "carry": TOTAL, "carried": TOTAL,
}

#: The nouns an ask uses for a dimension it wants totalled, singular only (the
#: same :func:`_read` covers a plural). An ask can also name the measurement by
#: its unit ("how many metres of pipe"); that reading is :func:`_ask_unit_measure`
#: and is deliberately narrower, because a unit spelling in prose is usually the
#: size of a thing ("at 2.5 to 3 m depth") rather than a total of it.
_ASK_DIMENSION_NOUNS = {
    "length": LENGTH,
    "area": AREA,
    "volume": VOLUME,
    "weight": MASS, "mass": MASS, "tonnage": MASS,
}
_DIMENSION_NOUN_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_ASK_DIMENSION_NOUNS, key=len, reverse=True)) + r")s?\b",
    re.IGNORECASE,
)
#: The dimensions an ask can total and a row's own Unit cell can state. A count
#: and a carry total are what a row measures *instead* of a dimension, so an ask
#: printing one of those spellings ("each of the pipes") states no dimension:
#: those rows are counted, not measured, and a count total is not a length.
_MEASURED_DIMENSIONS = frozenset({LENGTH, AREA, VOLUME, MASS})
#: The qualifiers that make a metre a square or a cubic one: "square metres of
#: formwork" totals an area and "cubic metres of concrete" a volume, so the word
#: in front of the unit, not the unit alone, names the dimension.
_UNIT_QUALIFIERS = {"square": AREA, "sq": AREA, "cubic": VOLUME, "cu": VOLUME}
#: The words that put a dimension noun in a measuring position: it asks for a
#: total ("the total billed length"), or it heads a genitive ("the length of
#: the pipes"). The same noun elsewhere ("the surface water area") names a
#: place or an item and scopes nothing by dimension. The cue must sit within
#: the run of words before the noun, never across a sentence or a punctuation.
_TOTAL_CUE_RE = re.compile(
    r"\b(?:total|totalled|totaled|combined|overall|whole|sum|summed|aggregate)\b",
    re.IGNORECASE,
)
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
    names; ``dimension`` is the measurement the ask asks to total (see
    :func:`_ask_dimension`); ``level``/``item``/``either`` are the ask's
    content words graded by where the document prints them (a heading path,
    an item description, or both). Empty dimensions do not scope.
    """

    size: tuple[float, float] | None = None
    band: tuple[float, float] | None = None
    bills: frozenset[str] = frozenset()
    dimension: str | None = None
    level: frozenset[str] = frozenset()
    item: frozenset[str] = frozenset()
    either: frozenset[str] = frozenset()

    def states(self) -> bool:
        return bool(self.size or self.band or self.bills or self.dimension
                    or self.level or self.item or self.either)

    def states_a_dimension(self) -> bool:
        """Whether the ask states the scope a word set cannot express: a size,
        a depth band, a bill or the dimension it asks to total. Those are read
        as numbers and as structural cells, so they -- not a per-turn token
        set -- govern the row set."""
        return bool(self.size or self.band or self.bills or self.dimension)

    def admits(self, row: SheetRow, measure: str = MEASURE_QUANTITY) -> bool:
        """Whether the row is inside every scope the ask states."""
        if self.size is not None and _row_size(row) != self.size:
            return False
        if self.band is not None and _row_band(row) != self.band:
            return False
        if self.bills and not (row_bills(row) & self.bills):
            return False
        if self.dimension is not None and not _measures(row, self.dimension, measure):
            return False
        if self.level and not self.level <= _row_path_words(row):
            return False
        # The scope's words are folded when they are read from the ask, so the
        # row's words are folded the same way here: a row that prints an item
        # in the plural binds a scope word written in either number. Grading
        # one side folded and the other unfolded empties the item and either
        # scopes on every ask whose noun the document prints with an "s".
        if self.item and not self.item <= _desc_words(row):
            return False
        if self.either and not (self.either & (_desc_words(row) | _row_path_words(row))):
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


def _number(text: str) -> float:
    """One number as the document prints it.

    Groups of three are joined (``1,500``, ``1 500``); a comma left with one or
    two digits behind it is the decimal mark (``2,5``)."""
    plain = _GROUPED_SEP_RE.sub("", (text or "").strip())
    return float(plain.replace(",", "."))


def _dimensions(match: re.Match[str]) -> tuple[float, float]:
    """The two ends of a dimension expression, in metres, low end first.

    A unit written on either end applies to both; a size written with no unit
    at all is metres unless both ends are too large for that, in which case it
    is millimetres (a bill prints a chamber in millimetres as bare numbers).
    A dimension is an extent, not a print order: ``2.0 x 1.5`` and
    ``1.5 x 2.0`` are one size, and a band written high-to-low is the band."""
    first, unit_a, second, unit_b = match.groups()
    values = (_number(first), _number(second))
    unit = (unit_a or unit_b or "").lower()
    if unit:
        scale = _UNIT_M.get(unit, 1.0)
        ends = (values[0] * scale, values[1] * scale)
    elif min(values) >= _UNITLESS_MM_FROM:
        ends = (values[0] / 1000.0, values[1] / 1000.0)
    else:
        ends = values
    return (round(min(ends), 6), round(max(ends), 6))


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


def _read(table: dict[str, str], token: str) -> str | None:
    """One word read through one vocabulary, both sides read the same way.

    A word printed in the plural is the word printed in the singular: a bill
    writes ``Nos`` where its neighbour writes ``No``, an ask writes ``lengths``
    where another writes ``length``. The rule is the vocabulary's, not any one
    spelling's, so a word the table does not list in the plural still reads.
    """
    if token and token not in table and token.endswith("s"):
        token = token[:-1]
    return table.get(token or "")


def _unit_dimension(unit: str) -> str | None:
    """The dimension a row's own Unit cell measures, or ``None``.

    The cell is normalised the way the document prints it -- case, spacing,
    punctuation and superscripts belong to the print, not to the unit -- and
    read through the one vocabulary above, the same way the ask's noun is.
    A blank cell, a count (``No``, ``Nr``), a section's carry total (``sum``)
    and any spelling the vocabulary does not know measure nothing, so they
    cannot satisfy a stated dimension.
    """
    text = (unit or "").casefold().replace("\u00b2", "2").replace("\u00b3", "3")
    return _read(_UNIT_DIMENSION, re.sub(r"[^a-z0-9]", "", text))


def _measuring_nouns(text: str) -> list[tuple[str, str]]:
    """Every dimension noun in a measuring position, in the ask's own order.

    A noun is in a measuring position behind a total word, as "the total
    billed length of ...", or before "of", as "the length of the pipes"; a
    place name ("the surface water area") is not. Each pair is the word as the
    ask's content words are folded and the dimension it names, so the caller
    can keep a word read as a measurement out of the word scopes: it named a
    measurement, not a word to look for in a row.
    """
    text = text or ""
    found: list[tuple[str, str]] = []
    for match in _DIMENSION_NOUN_RE.finditer(text):
        before = text[max(0, match.start() - 40):match.start()]
        after = text[match.end():match.end() + 4]
        if not (_TOTAL_CUE_RE.search(before) or after.lstrip().casefold().startswith("of")):
            continue
        dimension = _read(_ASK_DIMENSION_NOUNS, match.group(0).casefold())
        if dimension is not None:
            found.append((_fold(match.group(0).casefold()), dimension))
    return found


def _ask_unit_measure(text: str) -> tuple[frozenset[str], str] | None:
    """A measurement the ask names by its unit, in a measuring position.

    "how many metres of pipe" and "the total tonnes of steel" name the
    dimension with a unit instead of with the noun :func:`_measuring_nouns`
    reads. The unit is the row's own vocabulary read the same way -- one table,
    both sides -- and it names a measurement only where the ask totals the
    thing the unit measures: the unit directly heads a genitive and is not
    attached to a number. "at 2.5 to 3 m depth" states the size of a thing,
    not a total of it, and "in Bill No. 13" is a preposition the unit table
    happens to spell, so neither names a dimension.

    The first half of the pair is every word the ask printed to name the
    measurement -- the unit and, when it decided the dimension, the qualifier in
    front of it ("square metres") -- folded as content words are, so the caller
    keeps them out of the word scopes: they named a measurement, not a row.
    """
    words = [(match.group(0), _fold(match.group(0).casefold()))
             for match in _WORD_RE.finditer(text or "")]
    for index, (word, folded) in enumerate(words):
        dimension = _read(_UNIT_DIMENSION, re.sub(r"[^a-z0-9]", "", word.casefold()))
        if dimension not in _MEASURED_DIMENSIONS:
            continue
        if index + 1 >= len(words) or words[index + 1][0].casefold() != "of":
            continue
        before = words[index - 1][0].casefold() if index else ""
        if before[:1].isdigit() or before in {"x", "to", "and"}:
            continue
        qualifier = _UNIT_QUALIFIERS.get(before)
        if qualifier is None:
            return frozenset({folded}), dimension
        return frozenset({folded, _fold(before)}), qualifier
    return None


def _ask_dimension(text: str) -> str | None:
    """The dimension the ask asks to total, or ``None`` (the first it states).

    The noun the ask totals is read first; an ask that names the measurement by
    its unit ("how many metres of pipe") is read by :func:`_ask_unit_measure`,
    which never reads a unit attached to a number -- "at 2.5 to 3 m depth"
    states a band, not a length total.
    """
    found = _measuring_nouns(text)
    if found:
        return found[0][1]
    unit = _ask_unit_measure(text)
    return unit[1] if unit else None


def _measures(row: SheetRow, dimension: str, measure: str) -> bool:
    """Whether the row itself measures ``dimension`` with the figure
    ``measure`` asks for.

    Both halves are the row's own: its Unit cell states the dimension (read
    through the same vocabulary the ask's noun is read through), and the row
    carries the figure the total is built from. A row whose unit is a count or
    a carry total, or that carries no figure, is not a member of a total of
    that dimension."""
    if _unit_dimension(row.unit) != dimension:
        return False
    return row.figure(measure) is not None


def _bill_number(text: str) -> str:
    """A bill number as the ask or the heading prints it, its groups joined:
    ``1,313`` and ``1313`` are one bill."""
    return (text or "").replace(",", "").replace(" ", "").upper()


def row_bills(row: SheetRow) -> set[str]:
    """The bill numbers the row's own path names.

    A bill heading (``Bill No. <n> - ...``) or the document's page label
    (``D/13/2``: part, bill, page) printed in one of the row's own heading
    levels. A path that names no bill names none.
    """
    found: set[str] = set()
    for level in row.path:
        for match in _ASK_BILL_RE.finditer(level or ""):
            found.add(_bill_number(match.group(1)))
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
    bills = frozenset(_bill_number(m.group(1)) for m in _ASK_BILL_RE.finditer(text))
    in_paths: set[str] = set()
    in_desc: set[str] = set()
    for row in items:
        in_paths |= _row_path_words(row)
        in_desc |= _desc_words(row)
    asked = _scope_words(text)
    measured = _measuring_nouns(text)
    # A word read as a measurement is not also a content word: the row set is
    # bound by the row's own unit and figure (``dimension``), so requiring the
    # word in the row as well would narrow a dimension total to the rows that
    # happen to print the noun -- and empty it on a document that prints none.
    # The same holds for the unit an ask names its measurement by ("metres").
    nouns = {word for word, _dimension in measured}
    named_unit = _ask_unit_measure(text) if not measured else None
    if named_unit is not None:
        nouns |= named_unit[0]
    scope = AskScope(
        size=_dimensions(size) if size else None,
        band=_dimensions(band) if band else None,
        bills=bills,
        dimension=_ask_dimension(text),
        level=frozenset((asked & in_paths) - in_desc - nouns),
        item=frozenset((asked & in_desc) - in_paths - nouns),
        either=frozenset((asked & in_paths & in_desc) - nouns),
    )
    return scope if scope.states() else None


def fullest_sections(scope: AskScope, admitted: Sequence[SheetRow]) -> list[SheetRow]:
    """The admitted rows whose heading section is not named by fewer of the
    ask's words than another admitted section.

    The ask's content words are graded over the whole document, so a word that
    one stray description prints anywhere is read as optional rather than as a
    heading the row must sit under, and a sibling section that shares only a
    generic word is admitted beside the section the ask names. The rows that
    are already admitted settle it among themselves: each heading section
    (the rows that share one heading path) is read as the ask's words its path
    and its admitted rows print. A section whose words are a strict subset of
    another admitted section's words is a sibling the ask did not name, and it
    is left out. Two sections that each print a word the other does not are
    both named (an ask that lists two items), so both stay. A section is read
    as a whole, so a word one of its rows happens to print never splits it,
    and a row outside the admitted set never moves a section in or out.
    """
    words = scope.level | scope.item | scope.either
    named: dict[tuple[str, ...], set[str]] = {}
    for row in admitted:
        key = tuple(row.path)
        named.setdefault(key, set(words & _row_path_words(row)))
        named[key] |= words & _desc_words(row)
    keep = {
        key for key, have in named.items()
        if not any(have < other for other in named.values())
    }
    return [row for row in admitted if tuple(row.path) in keep]


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

    With ``ask``, a size, a depth band, a bill or a dimension the ask totals
    governs the row set (:func:`ask_scope`): those are read as numbers and as
    the row's own structural cells, so the same ask selects the same rows
    whatever ``terms`` the model passed on the turn. A dimension admits only
    rows whose own Unit cell measures it and that carry the figure. The
    model's terms are then reported but not applied -- an unparsed token set
    never defines a scope the ask states. An ask that states none of the four
    leaves ``terms`` and ``section`` to select, exactly as before, so a
    question asked in several parts still gets one result per part.

    Collection and repeat lines are never summed (:func:`row_kinds`); they are
    listed in ``not_items``. Each bill the ask names gets its check in
    ``bill_checks`` (:func:`bill_checks`): rows summed, rows without an amount,
    the bill's printed total and the state, so a sum is never handed back as
    a bill's total without saying whether it reconciles.
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
    joined = items
    kinds = row_kinds(items)
    not_items = [
        {"ref": row.ref, "pages": row.pages, "section": list(row.path),
         "description": row.description, "kind": kind, "amount": g.parse_number(row.amount)}
        for row, kind in zip(items, kinds) if kind != ROW_ITEM
    ]
    items = [row for row, kind in zip(items, kinds) if kind == ROW_ITEM]
    if scope is not None:
        selected = fullest_sections(scope, [row for row in items if scope.admits(row, measure)])
    else:
        selected = [row for row in items if matches(row, want, where)]
    for row in selected:
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
            "dimension": scope.dimension,
            "level": sorted(scope.level),
            "item": sorted(scope.item),
            "either": sorted(scope.either),
        } if scope is not None else None),
        "rows_scanned": len(items) + len(not_items),
        "not_items": not_items,
        "bill_checks": bill_checks(joined, scope.bills if scope is not None else ()),
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
