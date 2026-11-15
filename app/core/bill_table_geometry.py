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
columns of the last page that had one (a section split across pages). A page
whose own lines prove its quantity, rate and total columns by arithmetic (the
rightmost of three aligned figures is the product of the two before it) is
read under those inferred columns instead, when they out-witness the header
(see :func:`infer_page_columns`).

Output is a list of :class:`BillRow` and, through :func:`write_bill_workbook`,
an ``.xlsx`` with one sheet per bill section whose header row resolves through
the same aliasing, so ``doc_index._spreadsheet_row_chunks`` reads it as one
chunk per measured row.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

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


#: Why a line is not a column-header line (the order :func:`header_test`
#: checks them in). Codes only: a diagnostic counts them, it never prints labels.
HEADER_FEW_CELLS = "fewer_than_3_cells"
HEADER_NO_DESCRIPTION = "no_description_column"
HEADER_NO_FIGURE = "no_figure_column"
HEADER_REJECTIONS = (HEADER_FEW_CELLS, HEADER_NO_DESCRIPTION, HEADER_NO_FIGURE)


def detect_header(line: Sequence[Word], block=None) -> list[_Column] | None:
    """Columns of ``line`` if it is a bill column-header line, else ``None``.

    A header must resolve a description column and at least one figure column
    through the BOQ processor's aliasing.
    """
    return header_test(line, block)[0]


def header_test(
    line: Sequence[Word], block=None,
) -> tuple[list[_Column] | None, str | None]:
    """``(columns, None)`` for a header line, else ``(None, reason)`` with
    ``reason`` one of :data:`HEADER_REJECTIONS`. The one test both
    :func:`detect_header` and the classifier's diagnostic counts run."""
    block = block or _resolver()
    cells = _cells_by_gap(line)
    labels = [" ".join(w.text for w in c) for c in cells]
    if len(labels) < 3:
        return None, HEADER_FEW_CELLS
    resolved = block._resolve_columns(labels)
    if "description" not in resolved:
        return None, HEADER_NO_DESCRIPTION
    if not any(f in resolved for f in _FIGURE_FIELDS):
        return None, HEADER_NO_FIGURE
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
    ], None


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
    return "" if reconciles(q, r, a) else "qty x rate != amount"


def reconciles(quantity: float, rate: float, amount: float) -> bool:
    """``quantity x rate == amount`` within the printed-bill rounding tolerance."""
    return abs(quantity * rate - amount) <= max(_CHECK_ABS_TOL, abs(amount) * _CHECK_REL_TOL)


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


#: Why a line became a heading label (codes for :func:`extract_bill_rows`'s
#: ``on_label`` observer; a diagnostic counts them, it never prints labels).
LABEL_BEFORE_HEADER = "before_header"
LABEL_UNCLOSED_CHAIN = "unclosed_chain"
LABEL_NO_FIGURE = "no_figure"
LABEL_NO_DESCRIPTION = "no_description"


@dataclass(frozen=True)
class LabelEvent:
    """One heading label as :func:`extract_bill_rows` made it: the page, why,
    how many printed lines it joins, and whether a figure-column cell of the
    line held a number (a measured row read as a label)."""

    page: int
    reason: str
    lines: int
    figure_number: bool


def has_letter(text: str) -> bool:
    """The label test's letter rule, the one place it lives: ``text`` holds a
    letter of any script (the Unicode letter categories ``L*``, which is what
    ``str.isalpha`` tests). Digits, punctuation and symbols are not letters, so
    a figure-only or punctuation-only line is never a label."""
    return any(ch.isalpha() for ch in text)


def _has_figure_number(fields: dict[str, str]) -> bool:
    return any(
        parse_number(tok) is not None
        for f in _FIGURE_FIELDS for tok in (fields.get(f) or "").split()
    )



# --- columns inferred from the rows (the reconciliation test) -------------
#
# A scanned bill may print its rate and amount columns under header words the
# aliasing does not resolve, or print no header on the page at all. The page
# geometry still holds the columns: on a priced line the rightmost of three
# aligned figures is the product of the two before it. The rule below reads
# that arithmetic from each page's own word boxes; it names no word, column
# position, page or bill.

#: A numeric right-edge cluster is a vertical band when at least this many
#: lines of the page put a number in it.
MIN_BAND_LINES = 3
#: A page earns inferred columns only from at least this many witness lines
#: agreeing on one (quantity, rate, total) band assignment...
MIN_WITNESS_LINES = 3
#: ...and only when that assignment has more than this many times the
#: witness lines of the best competing assignment (a clear majority).
WITNESS_MAJORITY = 2
#: The unit is a single word of at most this many characters printed as its
#: own cell between the description and the quantity band.
UNIT_MAX_CHARS = 6
#: Column label of words on an inferred page that fall in no figure band and
#: right of the description (kept, never dropped).
UNPLACED_LABEL = "Unplaced"

#: Where a page's columns came from (codes for :class:`ColumnStats`).
SOURCE_HEADER = "header"
SOURCE_INFERRED = "inferred"
SOURCE_NONE = "none"


def numeric_bands(lines: Sequence[Sequence[Word]]) -> dict[int, dict[int, Word]]:
    """Per line index, ``{band: word}`` for the numbers of one page that sit
    in a vertical band (one word per band per line, the leftmost).

    Numbers are clustered by right edge: sorted by right edge, a new cluster
    starts where the gap to the previous right edge exceeds the page's median
    word height. A cluster is a band when at least :data:`MIN_BAND_LINES`
    lines put a number in it. Band ids grow left to right.
    """
    heights = [w.height for ln in lines for w in ln if w.height > 0]
    tol = statistics.median(heights) if heights else 1.0
    numeric = [(li, w) for li, ln in enumerate(lines) for w in ln if parse_number(w.text) is not None]
    order = sorted(range(len(numeric)), key=lambda i: numeric[i][1].right)
    cluster = [0] * len(numeric)
    current, prev = -1, None
    for i in order:
        r = numeric[i][1].right
        if prev is None or r - prev > tol:
            current += 1
        cluster[i] = current
        prev = r
    support: dict[int, set[int]] = {}
    for (li, _w), c in zip(numeric, cluster):
        support.setdefault(c, set()).add(li)
    vertical = {c for c, ls in support.items() if len(ls) >= MIN_BAND_LINES}
    per_line: dict[int, dict[int, Word]] = {}
    for (li, w), c in sorted(zip(numeric, cluster), key=lambda t: t[0][1].left):
        if c in vertical:
            per_line.setdefault(li, {}).setdefault(c, w)
    return per_line


def witness_triples(tokens: dict[int, Word]) -> list[tuple[int, int, int]]:
    """The band triples ``(q, r, t)``, left to right, for which this line's
    rightmost figure of the three is the product of the two before it
    (:func:`reconciles`). A factor of 0 or 1, or a total of 0, proves nothing
    and is not a witness."""
    bands = sorted(tokens)
    out: list[tuple[int, int, int]] = []
    for i, bq in enumerate(bands):
        for j in range(i + 1, len(bands)):
            br = bands[j]
            for bt in bands[j + 1:]:
                q, r, t = (parse_number(tokens[b].text) for b in (bq, br, bt))
                if q in (0, 1) or r in (0, 1) or t == 0:
                    continue
                if reconciles(q, r, t):
                    out.append((bq, br, bt))
    return out


@dataclass(frozen=True)
class InferredColumns:
    """A page's (quantity, rate, total) bands, as proved by its witness lines."""

    bands: tuple[int, int, int]
    witnesses: int        # lines that witness this assignment
    quantity_left: float  # left edge of the quantity band on this page
    band_of: dict[tuple[int, int], int]  # (line index, id(word)) -> band, figure bands only


def infer_page_columns(lines: Sequence[Sequence[Word]]) -> tuple[InferredColumns | None, int]:
    """``(columns, witness_lines)`` for one page, from its own word boxes.

    ``witness_lines`` counts lines that witness any assignment. Columns are
    ``None`` unless at least :data:`MIN_WITNESS_LINES` lines witness one
    assignment and it has more than :data:`WITNESS_MAJORITY` times the witness
    lines of every competing assignment. Nothing is carried from another page.
    """
    per_line = numeric_bands(lines)
    votes: dict[tuple[int, int, int], int] = {}
    witness_lines = 0
    for toks in per_line.values():
        triples = witness_triples(toks) if len(toks) >= 3 else []
        witness_lines += bool(triples)
        for t in triples:
            votes[t] = votes.get(t, 0) + 1
    if not votes:
        return None, witness_lines
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    best, n = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if n < MIN_WITNESS_LINES or n <= WITNESS_MAJORITY * runner_up:
        return None, witness_lines
    figure = set(best)
    band_of = {(li, id(w)): b for li, toks in per_line.items() for b, w in toks.items() if b in figure}
    q_left = min(w.left for toks in per_line.values() for b, w in toks.items() if b == best[0])
    return InferredColumns(best, n, q_left, band_of), witness_lines


def _bucket_inferred(li: int, line: Sequence[Word], cols: InferredColumns) -> _Line:
    """One printed line under inferred columns: figures from the bands; the
    description is the text left of the quantity band; the unit is a short
    single-word cell printed last before the quantity band, when present."""
    names = dict(zip(cols.bands, _FIGURE_FIELDS))
    figures: dict[str, list[str]] = {f: [] for f in _FIGURE_FIELDS}
    left: list[Word] = []
    unplaced: list[str] = []
    for w in line:
        band = cols.band_of.get((li, id(w)))
        if band is not None:
            figures[names[band]].append(w.text)
        elif w.cx < cols.quantity_left:
            left.append(w)
        else:
            unplaced.append(w.text)
    cells = _cells_by_gap(left)
    unit = ""
    if len(cells) > 1 and len(cells[-1]) == 1:
        tok = cells[-1][0].text
        if len(tok) <= UNIT_MAX_CHARS and parse_number(tok) is None:
            unit = tok
            cells = cells[:-1]
    fields = {"description": " ".join(w.text for c in cells for w in c), "unit": unit}
    fields.update({f: " ".join(v) for f, v in figures.items()})
    return _Line(
        text=" ".join(w.text for w in line),
        fields=fields,
        extra={UNPLACED_LABEL: " ".join(unplaced)} if unplaced else {},
        ref="",
        top=min(w.top for w in line),
        bottom=max(w.top + w.height for w in line),
        label=_label_of(line),
    )


def _header_reconciling(lines: Sequence[Sequence[Word]], columns: Sequence[_Column]) -> int:
    """Lines of a page on which ``columns`` read all three figures and they
    reconcile with no factor of 0 or 1 (the header's own witness count)."""
    n = 0
    for line in lines:
        f = _bucket_line(line, columns, set()).fields
        q, r, t = (parse_number(f.get(k, "")) for k in _FIGURE_FIELDS)
        if q is None or r is None or t is None or q in (0, 1) or r in (0, 1) or t == 0:
            continue
        n += reconciles(q, r, t)
    return n


def _header_conflicts(
    lines: Sequence[Sequence[Word]], columns: Sequence[_Column], cols: InferredColumns,
) -> int:
    """Figure fields the header resolved whose column is not where most words
    of that field's inferred band fall (or that the band never reaches)."""
    names = dict(zip(cols.bands, _FIGURE_FIELDS))
    under: dict[str, dict[int, int]] = {}
    for li, line in enumerate(lines):
        for w in line:
            band = cols.band_of.get((li, id(w)))
            if band is not None:
                hits = under.setdefault(names[band], {})
                idx = _assign(w, columns)
                hits[idx] = hits.get(idx, 0) + 1
    conflicts = 0
    for idx, col in enumerate(columns):
        if col.field in _FIGURE_FIELDS:
            hits = under.get(col.field, {})
            if not hits or max(hits, key=hits.__getitem__) != idx:
                conflicts += 1
    return conflicts


@dataclass
class ColumnStats:
    """How each page's columns were chosen, as counts (no text)."""

    pages: dict[str, int] = field(default_factory=lambda: dict.fromkeys(
        (SOURCE_HEADER, SOURCE_INFERRED, SOURCE_NONE), 0))
    witness_lines: int = 0
    header_conflicts: int = 0
    labels_reread_as_rows: int = 0

    def as_dict(self, rows: Sequence[BillRow]) -> dict[str, Any]:
        """The counts, with the rows' priced / not-reconciling split and the
        number of section paths."""
        priced = [r for r in rows
                  if all(parse_number(r.fields.get(f, "")) is not None for f in _FIGURE_FIELDS)]
        return {
            "pages_by_source": dict(self.pages),
            "witness_lines": self.witness_lines,
            "rows_priced_reconciled": sum(1 for r in priced if not r.check),
            "rows_figures_not_reconciling": sum(1 for r in priced if r.check),
            "priced_rows_empty_description": sum(
                1 for r in priced if not (r.fields.get("description") or "").strip()),
            "header_conflicts": self.header_conflicts,
            "labels_reread_as_rows": self.labels_reread_as_rows,
            "section_paths": len({tuple(r.path) or (r.section,) for r in rows}),
        }


def _header_label(line: Sequence[Word], columns: Sequence[_Column] | None, refs: set[int]) -> bool:
    """True when the header path would not read ``line`` as a measured row
    (no header in effect, or no figure or no description under its columns).

    With no header in effect the header path labels only a line that holds a
    letter (:data:`LABEL_BEFORE_HEADER`); a letter-free line was dropped, not
    labelled, so it is not a label re-read as a row."""
    if columns is None:
        return has_letter(" ".join(w.text for w in line))
    f = _bucket_line(line, columns, refs).fields
    return not (any(f.get(k) for k in _FIGURE_FIELDS) and f.get("description"))


def extract_bill_rows(
    words: Iterable[Word],
    block=None,
    on_label: Callable[[LabelEvent], None] | None = None,
    stats: ColumnStats | None = None,
) -> list[BillRow]:
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

    A page whose own word boxes prove its figure columns by the
    reconciliation test (:func:`infer_page_columns`) is read under those
    inferred columns when they have more witness lines than the header in
    effect on the page reconciles; otherwise the header path above applies.

    ``on_label`` (optional) observes every label as a :class:`LabelEvent`;
    ``stats`` (optional) collects the column-source counts. Neither changes
    the rows.
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

    def label(lab: _Label, page: int, reason: str, lines: int = 1, figure_number: bool = False) -> None:
        path.push(lab, indent)
        if on_label is not None:
            on_label(LabelEvent(page, reason, lines, figure_number))

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

    def flush(page: int) -> None:
        if pending:
            first = pending[0].label
            label(_Label(" ".join(ln.text for ln in pending), first.left, first.height),
                  page, LABEL_UNCLOSED_CHAIN, len(pending))
            pending.clear()

    stats = stats if stats is not None else ColumnStats()
    for page in sorted(by_page):
        heights = [w.height for w in by_page[page] if w.height > 0]
        max_gap = statistics.median(heights) if heights else 1.0
        indent = max_gap
        page_lines = group_lines(by_page[page])
        headers = [detect_header(line, block) for line in page_lines]
        inferred, witness_lines = infer_page_columns(page_lines)
        stats.witness_lines += witness_lines
        # The header in effect on this page: its own first header line, else
        # the one carried from an earlier page.
        page_header = next((h for h in headers if h is not None), columns)
        use_inferred = False
        if inferred is not None:
            header_witnesses = 0
            if page_header is not None:
                stats.header_conflicts += _header_conflicts(page_lines, page_header, inferred)
                header_witnesses = _header_reconciling(page_lines, page_header)
            use_inferred = inferred.witnesses > header_witnesses
        source = (SOURCE_INFERRED if use_inferred
                  else SOURCE_HEADER if page_header is not None else SOURCE_NONE)
        stats.pages[source] += 1
        for li, line in enumerate(page_lines):
            header = headers[li]
            if header is not None:
                flush(page)
                columns = header
                refs = _ref_columns(columns)
                continue
            if use_inferred:
                cur = _bucket_inferred(li, line, inferred)
                line_refs: set[int] = set()
            elif columns is None:
                if has_letter(" ".join(w.text for w in line)):
                    label(_label_of(line), page, LABEL_BEFORE_HEADER)
                continue
            else:
                cur = _bucket_line(line, columns, refs)
                line_refs = refs
            has_figure = any(cur.fields.get(f) for f in _FIGURE_FIELDS)
            has_desc = bool(cur.fields.get("description"))
            ref_labels = {columns[i].label for i in line_refs} if columns else set()
            other_text = any(
                t for lbl, t in cur.extra.items() if lbl not in ref_labels
            ) or cur.fields.get("unit")
            adjacent = bool(pending) and cur.top - pending[-1].bottom <= max_gap
            if not has_figure and has_desc:
                if cur.ref:
                    flush(page)
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
            flush(page)
            # Same rule as the spreadsheet row chunker: a measured row has a
            # description AND a figure. Anything else with letters (a bill or
            # section label, even one that prints a number such as "Bill 3")
            # is a heading label of the rows that follow.
            if not has_figure or not has_desc:
                if has_letter(cur.text):
                    label(cur.label, page,
                          LABEL_NO_FIGURE if not has_figure else LABEL_NO_DESCRIPTION,
                          figure_number=_has_figure_number(cur.fields))
                continue
            if use_inferred and _header_label(line, columns, refs):
                stats.labels_reread_as_rows += 1
            emit(page, cur.fields, cur.extra)
        flush(page)
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


# Tesseract page segmentation for a bill page: one uniform block of text.
# The automatic layout modes (3, 4) and the sparse modes (11, 12) drop an
# isolated one-character word, and a bill is full of them: a quantity of 1 to
# 9, a unit of "m" or "t". Reading order is irrelevant here (rows and columns
# come from the word boxes), so the block mode costs nothing and keeps them.
_BILL_PAGE_PSM = 6


def ocr_page_words(image, page: int, lang: str = "eng") -> list[Word]:
    """Word boxes for one rendered page image via tesseract."""
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--psm {_BILL_PAGE_PSM}",
        output_type=pytesseract.Output.DICT,
    )
    return words_from_tesseract_data(data, page)
