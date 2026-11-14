"""Scanned bill page -> table rows, from OCR word geometry (P-B3).

A scanned priced bill read as plain OCR text loses its table: on many pages
the engine emits the ref codes, then the units, then the quantities, then the
rates as separate vertical runs, so a row's description and its figures are
never on one line and no text-level row assembler can rejoin them.

This module works from the word BOXES instead of the word order. Each word
carries its page, left, top, width and height. Words are grouped into lines by
their vertical centre, the page's own column-header line is found by the BOQ
processor's column aliasing (the single source of truth for what a
description / quantity / unit / rate / amount column is called), and every
word below it is assigned to the header cell whose span it falls under. The
reading order the OCR engine chose is therefore irrelevant: the same boxes in
row-major or column-major order give the same rows.

Nothing here names a bill, a section, an item or a page. The columns are read
from each page's header words; a page with no header line continues with the
columns of the last page that had one (a section split across pages).

Output is a list of :class:`BillRow` and, through :func:`write_bill_workbook`,
an ``.xlsx`` with one sheet per bill section whose header row resolves through
the same aliasing, so ``doc_index._spreadsheet_row_chunks`` reads it as one
chunk per measured row.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

# Output column order for the derived sheet. The labels resolve through
# BOQProcessorBlock._resolve_columns (asserted by the tests), so the existing
# spreadsheet row chunker maps them without a second alias table.
_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("description", "Description"),
    ("unit", "Unit"),
    ("quantity", "Quantity"),
    ("rate", "Rate"),
    ("total", "Amount"),
)
_FIGURE_FIELDS = ("quantity", "rate", "total")

# Relative tolerance for qty x rate == amount; rounding in printed bills.
_CHECK_REL_TOL = 0.005
_CHECK_ABS_TOL = 0.01

# A label is printed "heavier" than another when its words are this much
# taller (a larger or bolder heading face).
_HEAVIER_RATIO = 1.2
# Path separator in the sheet banner (the row chunker's inherited heading).
_PATH_SEP = " > "

_NUMBER_RE = re.compile(r"^-?\(?\d[\d,]*(?:\.\d+)?\)?$")


@dataclass(frozen=True)
class Word:
    """One OCR word box. Coordinates are in the page image's pixel space."""

    text: str
    page: int
    left: float
    top: float
    width: float
    height: float

    @property
    def cx(self) -> float:
        return self.left + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.top + self.height / 2.0

    @property
    def right(self) -> float:
        return self.left + self.width


@dataclass
class _Column:
    label: str            # header text as printed on the page
    field: str | None     # resolved alias field, None for an unmapped column
    left: float
    right: float


@dataclass(frozen=True)
class _Label:
    """A heading line: its text, left edge and word height (its print style)."""

    text: str
    left: float
    height: float


def _label_of(words: Sequence[Word]) -> _Label:
    heights = [w.height for w in words if w.height > 0]
    return _Label(
        text=" ".join(w.text for w in words),
        left=min(w.left for w in words),
        height=statistics.median(heights) if heights else 0.0,
    )


def _outer(a: _Label, b: _Label, indent: float) -> bool:
    """True when ``a`` is printed as an outer level relative to ``b``: further
    left by more than ``indent``, or in a heavier (taller) face."""
    return a.left < b.left - indent or a.height > b.height * _HEAVIER_RATIO


@dataclass
class _HeadingPath:
    """The nested labels above a measured row, outer to inner.

    Labels that follow each other with no measured row between them nest. A
    label after measured rows is a sibling of the innermost level and replaces
    it. A label printed further left or heavier than the innermost level
    replaces from its own level down: every level not printed outer to it is
    dropped. No label text is consulted, only the word boxes.
    """

    levels: list[_Label] = field(default_factory=list)
    after_rows: bool = False

    def push(self, label: _Label, indent: float) -> None:
        if self.levels and _outer(label, self.levels[-1], indent):
            keep = 0
            while keep < len(self.levels) and _outer(self.levels[keep], label, indent):
                keep += 1
            del self.levels[keep:]
        elif self.levels and self.after_rows:
            self.levels.pop()
        self.levels.append(label)
        self.after_rows = False

    def texts(self) -> tuple[str, ...]:
        return tuple(lv.text for lv in self.levels)


@dataclass
class BillRow:
    section: str
    source_page: int
    fields: dict[str, str] = field(default_factory=dict)  # alias field -> text
    extra: dict[str, str] = field(default_factory=dict)   # unmapped header label -> text
    check: str = ""
    path: tuple[str, ...] = ()  # heading labels above the row, outer to inner


def words_from_tesseract_data(data: dict, page: int) -> list[Word]:
    """Convert ``pytesseract.image_to_data(..., output_type=DICT)`` to words."""
    out: list[Word] = []
    texts = data.get("text", [])
    for i, raw in enumerate(texts):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            conf = float(data.get("conf", [0] * len(texts))[i])
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0:
            continue
        out.append(
            Word(
                text=text,
                page=page,
                left=float(data["left"][i]),
                top=float(data["top"][i]),
                width=float(data["width"][i]),
                height=float(data["height"][i]),
            )
        )
    return out


def group_lines(words: Sequence[Word]) -> list[list[Word]]:
    """Group one page's words into lines by vertical centre, top to bottom.

    Independent of input order: words are sorted by centre y first, and a word
    joins the current line when its centre lies within half the median word
    height of the line's running centre.
    """
    heights = [w.height for w in words if w.height > 0]
    tol = (statistics.median(heights) if heights else 1.0) * 0.5
    lines: list[list[Word]] = []
    centre = 0.0
    for w in sorted(words, key=lambda w: (w.cy, w.left)):
        if lines and abs(w.cy - centre) <= tol:
            lines[-1].append(w)
            centre = sum(x.cy for x in lines[-1]) / len(lines[-1])
        else:
            lines.append([w])
            centre = w.cy
    return [sorted(line, key=lambda w: w.left) for line in lines]


def _cells_by_gap(line: Sequence[Word]) -> list[list[Word]]:
    """Split a line into cells where the horizontal gap exceeds one word height."""
    if not line:
        return []
    heights = [w.height for w in line if w.height > 0]
    gap = statistics.median(heights) if heights else 1.0
    cells: list[list[Word]] = [[line[0]]]
    for prev, cur in pairwise(line):
        if cur.left - prev.right > gap:
            cells.append([cur])
        else:
            cells[-1].append(cur)
    return cells


def _resolver():
    from app.blocks.boq_processor import BOQProcessorBlock

    return BOQProcessorBlock()


def detect_header(line: Sequence[Word], block=None) -> list[_Column] | None:
    """Columns of ``line`` if it is a bill column-header line, else ``None``.

    A header must resolve a description column and at least one figure column
    through the BOQ processor's aliasing.
    """
    block = block or _resolver()
    cells = _cells_by_gap(line)
    labels = [" ".join(w.text for w in c) for c in cells]
    if len(labels) < 3:
        return None
    resolved = block._resolve_columns(labels)
    if "description" not in resolved:
        return None
    if not any(f in resolved for f in _FIGURE_FIELDS):
        return None
    field_of: dict[int, str] = {}
    for fld, label in resolved.items():
        field_of.setdefault(labels.index(label), fld)
    return [
        _Column(
            label=labels[i],
            field=field_of.get(i),
            left=min(w.left for w in c),
            right=max(w.right for w in c),
        )
        for i, c in enumerate(cells)
    ]


def _assign(word: Word, columns: Sequence[_Column]) -> int:
    """Index of the column a word belongs to.

    Boundaries sit at the midpoint of the gap between consecutive header
    cells; a word belongs to the column whose band holds its centre.
    """
    for i in range(len(columns) - 1):
        boundary = (columns[i].right + columns[i + 1].left) / 2.0
        if word.cx < boundary:
            return i
    return len(columns) - 1


def parse_number(text: str) -> float | None:
    t = (text or "").strip().replace(" ", "")
    if not t or not _NUMBER_RE.match(t):
        return None
    neg = t.startswith("-") or (t.startswith("(") and t.endswith(")"))
    # _NUMBER_RE admits only digits, commas and one decimal point here.
    v = float(t.strip("-()").replace(",", ""))
    return -v if neg else v


def check_row(fields: dict[str, str]) -> str:
    """``""`` when qty x rate agrees with amount (or cannot be checked), else
    ``"qty x rate != amount"``. The row is kept either way."""
    q = parse_number(fields.get("quantity", ""))
    r = parse_number(fields.get("rate", ""))
    a = parse_number(fields.get("total", ""))
    if q is None or r is None or a is None:
        return ""
    expected = q * r
    if abs(expected - a) <= max(_CHECK_ABS_TOL, abs(a) * _CHECK_REL_TOL):
        return ""
    return "qty x rate != amount"


def _ref_columns(columns: Sequence[_Column]) -> set[int]:
    """Indexes of unmapped columns printed left of the description column.

    On a bill these carry the item reference; a line that prints one starts a
    new item. A page whose header has no such column gives an empty set.
    """
    desc = next((i for i, c in enumerate(columns) if c.field == "description"), None)
    if desc is None:
        return set()
    return {i for i, c in enumerate(columns[:desc]) if c.field is None}


@dataclass
class _Line:
    text: str
    fields: dict[str, str]
    extra: dict[str, str]
    ref: str
    top: float
    bottom: float
    label: _Label


def _bucket_line(line: Sequence[Word], columns: Sequence[_Column], refs: set[int]) -> _Line:
    buckets: list[list[str]] = [[] for _ in columns]
    for w in line:
        buckets[_assign(w, columns)].append(w.text)
    fields: dict[str, str] = {}
    extra: dict[str, str] = {}
    for col, bucket in zip(columns, buckets):
        if col.field:
            fields[col.field] = " ".join(bucket)
        else:
            extra[col.label] = " ".join(bucket)
    ref = " ".join(" ".join(buckets[i]) for i in sorted(refs) if buckets[i])
    return _Line(
        text=" ".join(w.text for w in line),
        fields=fields,
        extra=extra,
        ref=ref,
        top=min(w.top for w in line),
        bottom=max(w.top + w.height for w in line),
        label=_label_of(line),
    )


def _merge_wrapped(pending: Sequence[_Line], last: _Line) -> tuple[dict[str, str], dict[str, str]]:
    """One row from an item whose description wraps over ``pending`` lines and
    whose figures print on ``last``: descriptions join top to bottom; a cell the
    figure line leaves empty (a unit printed on the opening line) keeps the
    first non-empty text of the chain, as every unmapped cell does."""
    chain = [*pending, last]
    fields = {
        f: last.fields.get(f) or next((ln.fields[f] for ln in chain if ln.fields.get(f)), "")
        for f in last.fields
    }
    fields["description"] = " ".join(
        d for d in (ln.fields.get("description", "") for ln in chain) if d
    )
    extra = {label: next((ln.extra[label] for ln in chain if ln.extra.get(label)), "") for label in last.extra}
    return fields, extra


def extract_bill_rows(words: Iterable[Word], block=None) -> list[BillRow]:
    """Rows of every bill table found in ``words`` (any number of pages).

    A line with no figure cell and some letters is a heading label. Labels
    nest into a heading path (see :class:`_HeadingPath`): consecutive labels
    go outer to inner, a label after measured rows replaces the innermost
    level, and a label printed further left or heavier replaces from its own
    level down. Every measured row carries the path; ``section`` is its
    innermost label.

    A description may wrap over several printed lines with the figures on the
    last one. The evidence for that is geometric, never textual: the first
    line prints an item reference (an unmapped column left of the description
    column) and a description but no figure; each following line prints only
    description text; the closing line prints the figures and no reference of
    its own (its description cell may be empty: figures printed on a line of
    their own under the text); and no blank line separates any of them
    (vertical gap at most one median word height). Such a chain is one row. A chain that is not closed
    that way is a section label, as before. A page whose header has no
    reference column has no such evidence and keeps the line-per-row rule.
    """
    block = block or _resolver()
    by_page: dict[int, list[Word]] = {}
    for w in words:
        by_page.setdefault(w.page, []).append(w)

    rows: list[BillRow] = []
    columns: list[_Column] | None = None
    refs: set[int] = set()
    path = _HeadingPath()
    indent = 1.0
    pending: list[_Line] = []

    def label(lab: _Label) -> None:
        path.push(lab, indent)

    def emit(page: int, fields: dict[str, str], extra: dict[str, str]) -> None:
        texts = path.texts()
        row = BillRow(
            section=texts[-1] if texts else "",
            source_page=page,
            fields=fields,
            extra=extra,
            path=texts,
        )
        row.check = check_row(fields)
        rows.append(row)
        path.after_rows = True

    def flush() -> None:
        if pending:
            first = pending[0].label
            label(_Label(" ".join(ln.text for ln in pending), first.left, first.height))
            pending.clear()

    for page in sorted(by_page):
        heights = [w.height for w in by_page[page] if w.height > 0]
        max_gap = statistics.median(heights) if heights else 1.0
        indent = max_gap
        for line in group_lines(by_page[page]):
            header = detect_header(line, block)
            if header is not None:
                flush()
                columns = header
                refs = _ref_columns(columns)
                continue
            if columns is None:
                if re.search(r"[A-Za-z]", " ".join(w.text for w in line)):
                    label(_label_of(line))
                continue
            cur = _bucket_line(line, columns, refs)
            has_figure = any(cur.fields.get(f) for f in _FIGURE_FIELDS)
            has_desc = bool(cur.fields.get("description"))
            other_text = any(
                t for lbl, t in cur.extra.items()
                if lbl not in {columns[i].label for i in refs}
            ) or cur.fields.get("unit")
            adjacent = bool(pending) and cur.top - pending[-1].bottom <= max_gap
            if not has_figure and has_desc:
                if cur.ref:
                    flush()
                    pending.append(cur)
                    continue
                if adjacent and not other_text:
                    pending.append(cur)
                    continue
            # The closing line may print figures only (its description cell
            # empty): the open chain is the evidence the figures belong to it.
            if has_figure and adjacent and not cur.ref:
                fields, extra = _merge_wrapped(pending, cur)
                pending.clear()
                emit(page, fields, extra)
                continue
            flush()
            # Same rule as the spreadsheet row chunker: a measured row has a
            # description AND a figure. Anything else with letters (a bill or
            # section label, even one that prints a number such as "Bill 3")
            # is a heading label of the rows that follow.
            if not has_figure or not has_desc:
                if re.search(r"[A-Za-z]", cur.text):
                    label(cur.label)
                continue
            emit(page, cur.fields, cur.extra)
        flush()
    return rows


def _sheet_title(name: str, used: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", " ", name or "Bill").strip()[:28] or "Bill"
    title, n = base, 2
    while title.lower() in used:
        title = f"{base[:25]} {n}"
        n += 1
    used.add(title.lower())
    return title


def write_bill_workbook(rows: Sequence[BillRow], path: str) -> dict[str, int]:
    """Write one sheet per heading path; returns ``{sheet title: row count}``.

    Each sheet carries the row's heading path, outer to inner, as a banner row
    above the header (the row chunker inherits it as the bill heading, so each
    row chunk reads the path, then the row); the sheet is titled by the
    innermost label. Then the canonical columns,
    any unmapped columns the page printed (item, ref ...), ``Check`` and
    ``Source Page``.
    """
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    order: list[tuple[str, ...]] = []
    grouped: dict[tuple[str, ...], list[BillRow]] = {}
    for r in rows:
        key = r.path or ((r.section,) if r.section else ())
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(r)

    used: set[str] = set()
    counts: dict[str, int] = {}
    for key in order:
        section_rows = grouped[key]
        extras: list[str] = []
        for r in section_rows:
            for label in r.extra:
                if label not in extras:
                    extras.append(label)
        title = _sheet_title(key[-1] if key else "", used)
        ws = wb.create_sheet(title)
        if key:
            ws.append([_PATH_SEP.join(key)])
        header = [lbl for f, lbl in _FIELD_LABELS] + extras + ["Check", "Source Page"]
        ws.append(header)
        for r in section_rows:
            ws.append(
                [r.fields.get(f, "") for f, _ in _FIELD_LABELS]
                + [r.extra.get(e, "") for e in extras]
                + [r.check, r.source_page]
            )
        counts[title] = len(section_rows)
    if not order:
        wb.create_sheet("Bill").append([lbl for _, lbl in _FIELD_LABELS])
    wb.save(path)
    return counts


def ocr_page_words(image, page: int, lang: str = "eng") -> list[Word]:
    """Word boxes for one rendered page image via tesseract."""
    import pytesseract

    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    return words_from_tesseract_data(data, page)
