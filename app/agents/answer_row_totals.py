"""A stated total is the sum of the rows the answer itself prints.

An answer that lists bill rows and then states a total is making an
arithmetic claim about its own list. Nothing else checks that claim: the
derivation check (``runtime._annotate_derivation_mismatches``) reads only
explicit ``a x b = c`` chains, so a table whose Total cell is a figure the
model composed is invisible to it. Measured on the live bill band
(20 Nov, build 15e50367): a Total of 3,928 m over eight rows that sum to
4,225 m, a Total of 2,468 m over four rows that sum to 2,906 m, and a Total
of 168 No whose last entry is a "remaining depth bands" remainder rather
than a row.

Four rules, all decided from the answer's own text and its own figures:

(1) A total is the sum of the rows the answer prints, or the answer says it
    is incomplete. A total that disagrees with its own column is replaced by
    the column's sum - in a Total cell, in a printed ``a + b + c = N`` chain,
    and in every prose statement of that figure that carries the same unit,
    so the answer cannot keep the wrong number in its first line - and one
    plain line says what the total now covers and what it leaves out.
(2) A row printed without a figure (``-``, ``n/a``, an empty cell) is named
    as a row the answer did not read. It adds nothing to the sum.
(3) An entry that stands for rows rather than being one - "the remaining
    rows", "others", "balance", "not listed" - is a remainder bucket, not
    a row. Whether its figure is already in the total is decided from the
    bucket's own words: a bucket that names a reference the answer also
    prints on a row it counted describes rows already counted, so it is a
    truncated view and its figure is not counted twice (rule 3a); a bucket
    that names no such reference stands for rows the answer did not
    itemise, and it accounts for a total above the printed rows only when
    the answer itself accounts for the difference - the bucket states the
    figure those rows carry and printed + that figure is the total, or the
    bucket says it makes up the balance of the total (rule 3b). A bucket
    that only names further rows, with no figure and no claim to the
    balance, leaves the difference supported by nothing the answer prints.
(4) The same rule reads a row list the answer prints as prose rather than
    as a table ("The rows read: 1.5-2.0 m 18; 2.0-2.5 m 26; ..."): the
    figures those rows state are summed and checked against the total the
    answer states. A row printed without a quantity leaves its contribution
    unknown, and then nothing is rewritten.

A total that already equals its column is returned byte for byte: the check
may only rewrite an answer that disagrees with itself. Where the printed
evidence cannot settle a total - a row the answer prints without a figure,
whose contribution is unknown - the answer is not rewritten: a check that
guesses is worse than a check that stays silent. A total the answer
credits to a document ("[source: ...]", "as printed", "carried to") is that
document's figure, not a sum of the rows above it, and is left alone - as is
a prose total the answer itself discloses as partial ("the remaining rows
make up the balance to N").

Nothing here names a project, a document, a bill, an item, a bore size, a
band label or a figure. A column's unit is read from the answer's own header
cell, and the words that mark a remainder bucket are general quantifier
vocabulary, not any document's wording.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

_LOG = logging.getLogger(__name__)

#: A cell that states a total of the rows above it: "Total", "Sub-total",
#: "Grand total", "Sum", "Total quantity". Case and surrounding space ignored.
_TOTAL_CELL_RE = re.compile(r"(?i)^\s*(?:grand\s+|sub\s*[- ]?)?(?:total|totals|sum)\b")

#: An entry that stands for rows the answer does not print. General
#: quantifier vocabulary: a remainder, not an item.
_REMAINDER_RE = re.compile(
    r"(?i)\b(?:remaining|remainder|others?|further|balance|rest|"
    r"not\s+listed|unlisted|left\s+over|and\s+more|etc)\b",
)

#: An item reference a row prints: "K 133", "K137.8", "A-12". A remainder
#: bucket that repeats a reference the answer already prints on a row it
#: counted describes that row, so its figure is a restatement, not an addition.
_REFERENCE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])([A-Za-z]{1,3})\s?(\d[\d./\-]*\d|\d)(?![A-Za-z0-9])",
)

#: A remainder bucket that says it makes up the balance of the stated total
#: names the residual itself ("make up the balance to 1,000 m"), so the total
#: is defined as the printed rows plus that bucket and cannot disagree with
#: them. A bucket that merely mentions further rows makes no such claim.
_BALANCE_RE = re.compile(
    r"(?i)\b(?:balance|rest)\b(?=[^.;]{0,40}?(?:\d|total\b))",
)

#: A total row the answer credits to a document is that document's figure.
_ATTRIBUTED_RE = re.compile(
    r"(?i)(?:\[source\s*:|per\s+the\s+(?:bill|sheet|document)|as\s+printed|"
    r"carried\s+(?:to|forward))",
)

#: A cell holding one figure (633, 4,858, 772,260.00) or a sum of figures
#: ("1 + 1 + 1 + 1"). Anything else - a dimension ("400 mm"), a range, a
#: label - is not a figure.
_FIGURE_CELL_RE = re.compile(
    r"^\s*-?\d[\d,]*(?:\.\d+)?(?:\s*\+\s*-?\d[\d,]*(?:\.\d+)?)*\s*$",
)

#: A column's unit as the header prints it: "Length (m)", "Qty (No)".
_HEADER_UNIT_RE = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9/ ]{0,10})\s*\)\s*$")

#: A sentence that states a total OF ROWS: "the sum of the 15 priced rows",
#: "summed across all bores", "adds up to", "altogether". A figure stated in
#: such a sentence is an arithmetic claim about the rows around it.
_TOTAL_CLAIM_RE = re.compile(
    r"(?i)\bsums?\b|\bsummed\b|\btotal\s+of\b|\baltogether\b|\badds?\s+up\s+to\b",
)

#: A chain of figures the answer prints as its own row list:
#: ``18 + 26 + 64 + 22 = 130 No``. An operand may be an ellipsis (``…``,
#: ``...``) where the answer itself admits the list continues.
_CHAIN_RE = re.compile(
    r"(?<![\d,.])"
    r"(?P<terms>-?\d[\d,]*(?:\.\d+)?"
    r"(?:\s*\+\s*(?:-?\d[\d,]*(?:\.\d+)?|…|\.{2,3})){2,})"
    r"\s*=\s*(?P<total>-?\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*(?P<unit>[A-Za-z]{1,8})(?![A-Za-z]))?"
)
_ELLIPSIS_RE = re.compile(r"…|\.{2,3}")

#: Two figures agree within this share of the larger, or a hundredth - the
#: tolerance the derivation check uses, so a rounded total still agrees.
_TOLERANCE = 0.005
_ABS_TOLERANCE = 0.01

#: How many prose statements of a wrong figure are replaced, per line.
_MAX_REPLACEMENTS = 8

#: A table cell's separator: the answer prints its rows tab- or pipe-delimited.
_TAB = "\t"


@dataclass
class Correction:
    """One column of one table whose stated total disagrees with its rows."""

    line_index: int
    col: int
    stated_text: str
    stated: float
    computed: float
    unit: str
    rows: int
    without_figure: list[str] = field(default_factory=list)
    remainders: list[str] = field(default_factory=list)
    restated: list[str] = field(default_factory=list)
    #: False when the statement is prose ("the sum of the rows is 168 No")
    #: rather than a cell of the table.
    in_cell: bool = True


@dataclass
class ChainFinding:
    """One printed sum chain and what it states.

    ``computed`` is None when the chain is itself incomplete (an ellipsis
    among its terms): the answer then says its own row list is incomplete
    rather than stating a total its terms do not carry.
    """

    line_index: int
    stated_text: str
    stated: float
    computed: float | None
    unit: str
    terms: int


def _cells(line: str) -> list[str] | None:
    """The cells of a table line, or None when ``line`` is not one.

    A table line is tab-separated (at least one tab) or pipe-separated (at
    least two pipes), which is how an answer prints a list of rows.
    """
    if _TAB in line:
        return line.split(_TAB)
    if line.count("|") >= 2:
        body = line.strip().removeprefix("|").removesuffix("|")
        return body.split("|")
    return None


def _table_blocks(cells: list[list[str] | None]) -> list[list[int]]:
    """The runs of consecutive table lines: one block is one table."""
    blocks: list[list[int]] = []
    previous = -2
    for index in [i for i, row in enumerate(cells) if row is not None]:
        joined = index == previous + 1 and bool(blocks)
        blocks.append([*blocks.pop(), index] if joined else [index])
        previous = index
    return blocks


def _number(cell: str) -> float | None:
    """The figure a cell states, or None.

    One figure, or the sum of a printed sum expression ("1 + 1 + 1 + 1" is
    four). A cell holding words, a dimension or a range states no figure.
    """
    text = (cell or "").strip()
    parts = text.split("+") if _FIGURE_CELL_RE.match(text) else []
    if not parts:
        return None
    total = 0.0
    for part in parts:
        try:
            total += float(part.strip().replace(",", ""))
        except ValueError:  # pragma: no cover - _FIGURE_CELL_RE accepted it
            _LOG.warning("figure cell %r did not parse as a number", text)
            return None
    return total


def _format_like(value: float, sample: str) -> str:
    """``value`` printed the way ``sample`` prints a figure: the same
    thousands separators and the same number of decimals."""
    text = (sample or "").strip()
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    grouped = "," in text
    if decimals:
        return f"{value:,.{decimals}f}" if grouped else f"{value:.{decimals}f}"
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):,}" if grouped else str(round(value))
    body = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{float(body):,}" if grouped else body


def _agrees(stated: float, computed: float) -> bool:
    return abs(stated - computed) <= max(abs(stated), abs(computed)) * _TOLERANCE + _ABS_TOLERANCE


def _row_label(row: list[str], figure_col: int) -> str:
    """What a row is called: its own words, not its figure column.

    A reference cell alone ("A", "-") names nothing a reader can act on, so
    the row is named by the longest cell that states no figure - the
    description the answer prints for it.
    """
    words = [
        (cell or "").strip() for index, cell in enumerate(row)
        if index != figure_col and _number(cell) is None and (cell or "").strip()
    ]
    label = max(words, key=len) if words else ((row[0] if row else "") or "").strip()
    # "plus the remaining rows", "and further rows": the connective is the
    # sentence's, not the row's name.
    return re.sub(r"(?i)^(?:plus|and|with|\+)\s+", "", label or "row with no reference")


def _row_words(row: list[str], figure_col: int) -> str:
    """The row's own words: every cell but the figure it states."""
    return " ".join(
        (cell or "").strip() for index, cell in enumerate(row) if index != figure_col
    )


def _references(text: str) -> set[str]:
    """The item references a line prints, normalised so "K 133" equals "K133"."""
    return {
        f"{match.group(1).lower()}{match.group(2).lower()}"
        for match in _REFERENCE_RE.finditer(text or "")
    }


@dataclass
class _RowRead:
    """What a table's data rows contribute to one column's total.

    ``kept`` are the rows the answer printed with a figure; ``without`` are
    rows it printed with none; ``restated`` are remainder buckets whose own
    words name a reference already counted (a truncated view of the rows
    above); ``unread`` are remainder buckets that name no such row - rows the
    answer itself says it did not itemise.
    """

    kept: list[float] = field(default_factory=list)
    without: list[str] = field(default_factory=list)
    restated: list[str] = field(default_factory=list)
    unread: list[str] = field(default_factory=list)
    #: The figures the unread buckets state for themselves, where they state
    #: one; a bucket that states no figure adds nothing to this list.
    unread_figures: list[float] = field(default_factory=list)
    #: True when an unread bucket says in its own words that it makes up the
    #: balance of the stated total.
    balance: bool = False

    @property
    def printed(self) -> float:
        return sum(self.kept)


def _read_rows(rows: list[list[str]], figure_col: int, unit: str = "") -> _RowRead:
    """Sort a table's data rows into printed rows and remainder buckets.

    A bucket's own words decide what it is. It repeats a reference the answer
    already printed on a row it counted -> it describes rows already in the
    total (a truncated view), so it is named as such and adds nothing. It
    names no such row -> it stands for rows the answer did not itemise, and
    whatever figure it states for those rows is recorded so the caller can
    check the total against it. ``unit`` is the column's own unit, so a bucket
    that writes its figure into its wording ("remaining rows 6 No") states it
    as plainly as one that puts it in the figure column.
    """
    read = _RowRead()
    counted: set[str] = set()
    buckets: list[tuple[list[str], str]] = []
    for row in rows:
        words = _row_words(row, figure_col)
        wording = " ".join(cell or "" for cell in row)
        if _REMAINDER_RE.search(wording):
            buckets.append((row, words))
            continue
        value = _number(row[figure_col] if figure_col < len(row) else "")
        if value is None:
            read.without.append(_row_label(row, figure_col))
            continue
        read.kept.append(value)
        counted |= _references(words)
    for row, words in buckets:
        label = _row_label(row, figure_col)
        if _references(words) & counted:
            read.restated.append(label)
            continue
        read.unread.append(label)
        wording = " ".join(cell or "" for cell in row)
        figure = _number(row[figure_col] if figure_col < len(row) else "")
        if figure is None and unit:
            figure = _bucket_figure(wording, unit)
        if figure is not None:
            read.unread_figures.append(figure)
        if _BALANCE_RE.search(wording):
            read.balance = True
    return read


def _disclosed_total(stated: float, read: _RowRead) -> bool:
    """True when the answer's own text accounts for its total.

    A total equal to the printed rows is already the sum of the rows above.
    A total ABOVE the printed rows is the answer disclosing rows it left out
    only when the answer itself accounts for the difference: it states the
    figures those rows carry (printed + bucket figures agrees with the
    total), or it says the bucket makes up the balance of that total. A row
    printed without a figure leaves its contribution unknown, so the printed
    evidence cannot settle such a total and nothing is rewritten.

    A bucket that merely names further rows without stating any figure makes
    no claim about the difference: a total above the printed rows is then
    supported by nothing the answer prints, and is treated as a disagreement.
    A bucket already shown to restate counted rows never excuses a total.
    """
    if _agrees(stated, read.printed):
        return True
    if stated < read.printed:
        return False
    if read.balance:
        return True
    if not read.unread:
        return False
    if read.without:
        return True
    return bool(read.unread_figures) and _agrees(
        stated, read.printed + sum(read.unread_figures)
    )


def _header_unit(cell: str) -> str:
    """The unit a column header states in its own words ("Length (m)" -> "m")."""
    match = _HEADER_UNIT_RE.search((cell or "").strip())
    return match.group(1).strip() if match else ""


def _claim_figure(line: str, unit: str) -> tuple[str, float] | None:
    """The figure ``line`` states with ``unit``, as (text, value), or None."""
    if not unit:
        return None
    match = re.search(
        rf"(?<![\d,.])(\d[\d,]*(?:\.\d+)?)\s*{re.escape(unit)}(?![A-Za-z0-9])",
        line, re.IGNORECASE,
    )
    if match is None:
        return None
    value = _number(match.group(1))
    return None if value is None else (match.group(1), value)


def _column_correction(
    line_index: int, col: int, total_cells: list[str], data_rows: list[list[str]],
    header_cells: list[str], value_col: int | None = None,
) -> Correction | None:
    """One column's correction, or None when its total agrees or is not a sum.

    A column is a candidate only when its Total cell is a figure (so a
    description or reference column is never summed) and the row credits no
    document. Rule (2) records a row printed without a figure; rule (3) sorts
    the remainder buckets, and a total that already agrees with what the
    answer printed - the rows alone, or the rows plus the buckets that name
    no counted row - is left alone.

    ``value_col`` is the column the rows state their figures in when it is
    not the total's own column: a total row prints a blank cell before its
    figure (``Total \\t\\t 4,858``), so its figure can sit one column right of
    the column it totals.
    """
    stated_text = (total_cells[col] if col < len(total_cells) else "").strip()
    stated = _number(stated_text)
    if stated is None:
        return None
    if any(_ATTRIBUTED_RE.search(c or "") for c in total_cells):
        return None
    figures_in = col if value_col is None else value_col
    unit = _header_unit(header_cells[figures_in] if figures_in < len(header_cells) else "")
    read = _read_rows(data_rows, figures_in, unit)
    if not read.kept:
        return None
    if _disclosed_total(stated, read):
        return None
    return Correction(
        line_index=line_index, col=col, stated_text=stated_text, stated=stated,
        computed=read.printed, unit=unit, rows=len(read.kept),
        without_figure=read.without, remainders=read.unread, restated=read.restated,
    )


def _block_corrections(block: list[int], cells: list[list[str] | None]) -> list[Correction]:
    """Every column of one table block whose Total row disagrees with its rows."""
    rows = [cells[i] or [] for i in block]
    total_row = next(
        (i for i, row in enumerate(rows)
         if any(_TOTAL_CELL_RE.match(c or "") for c in row)),
        None,
    )
    if total_row is None or total_row == 0:
        return []
    above = [row for row in rows[:total_row] if any((c or "").strip() for c in row)]
    if not above:
        return []
    # The block's first line is a header when it states no figure in a column
    # the Total row totals; otherwise it is the first data row.
    total_cols = [c for c, cell in enumerate(rows[total_row]) if _number(cell) is not None]
    first = above[0]
    is_header = bool(total_cols) and all(
        _number(first[c] if c < len(first) else "") is None for c in total_cols
    )
    data = above[1:] if is_header else above
    if not data:
        return []
    header = first if is_header else []
    numeric_cols = [
        c for c in range(max((len(row) for row in data), default=0))
        if any(_number(row[c]) is not None for row in data if c < len(row))
    ]
    out: list[Correction] = []
    for col in range(len(rows[total_row])):
        value_col = col
        if _number(rows[total_row][col]) is not None and col not in numeric_cols:
            # The total's figure sits in a column the rows leave blank: a
            # table with one figure column names that column.
            if len(numeric_cols) != 1:
                continue
            value_col = numeric_cols[0]
        found = _column_correction(
            block[total_row], col, rows[total_row], data, header, value_col=value_col,
        )
        if found is not None:
            out.append(found)
    return out


def _prose_correction(
    block: list[int], cells: list[list[str] | None], lines: list[str],
) -> Correction | None:
    """A table with no Total row whose rows the answer says sum to a figure.

    "The sum of the 15 priced rows is 168 No" over a table that prints those
    rows is the same arithmetic claim as a Total cell, so it is checked the
    same way. Only a sentence that states a total OF its rows is read
    (``_TOTAL_CLAIM_RE``); a figure the answer credits to a document as that
    document's figure ("the bill's carried-to-collection total is ...")
    states no sum of these rows and is left alone.

    A prose claim is weaker evidence than a cell: an answer that says in its
    own words which rows it did not print ("the remaining ten rows make up
    the balance to 6,292 m") has already disclosed that its figure is the
    band's, not the printed rows', so its figure is left alone. A total cell
    that disagrees with the cells above it is a defect either way and is not
    excused here.
    """
    outside = [i for i in range(len(lines)) if i not in block]
    disclosed = any(cells[i] is None and _REMAINDER_RE.search(lines[i]) for i in outside)
    if disclosed:
        return None
    rows = [cells[i] or [] for i in block]
    data = [row for row in rows if any((c or "").strip() for c in row)]
    if not data or any(any(_TOTAL_CELL_RE.match(c or "") for c in row) for row in data):
        return None
    header = data[0]
    figure_cols = [
        c for c in range(max((len(row) for row in data), default=0))
        if sum(_number(row[c] if c < len(row) else "") is not None for row in data) > 1
    ]
    if len(figure_cols) != 1:
        return None
    col = figure_cols[0]
    unit = _header_unit(header[col] if col < len(header) else "")
    for index in outside:
        line = lines[index]
        stated = _claim_figure(line, unit) if _TOTAL_CLAIM_RE.search(line) else None
        if stated is None:
            continue
        stated_text, stated_value = stated
        read = _read_rows(data[1:], col)
        if not read.kept or _disclosed_total(stated_value, read):
            continue
        return Correction(
            line_index=index, col=col, stated_text=stated_text, stated=stated_value,
            computed=read.printed, unit=unit, rows=len(read.kept),
            without_figure=read.without, remainders=read.unread,
            restated=read.restated, in_cell=False,
        )
    return None


#: A total the answer credits to a document by its own words. A bare
#: ``[source: ...]`` citation is how the answer cites the rows it read, so it
#: is not, on its own, a claim that the figure is the document's.
_ATTRIBUTED_PROSE_RE = re.compile(
    r"(?i)(?:per\s+the\s+(?:bill|sheet|document)|as\s+printed|"
    r"carried\s+(?:to|forward))",
)

#: A citation the answer appends to a sentence: "[source: ...]".
_CITATION_RE = re.compile(r"\[[^\[\]]*\]")

#: The quantity a prose row states at the end of its segment: "... 2.0-2.5 m 26".
_ROW_QUANTITY_RE = re.compile(r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)\s*$")

#: The token directly before that figure: a unit ("m") or a reference ("K").
_LEADING_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,3})\s*$")

#: The figure a claim states with its unit: "26 No", "4,858 m".
_CLAIM_UNIT_RE = re.compile(r"(?<![\d,.])(\d[\d,]*(?:\.\d+)?)\s+([A-Za-z]{1,3})(?![A-Za-z0-9])")

#: Words that follow a figure without being its unit.
_CLAIM_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "by", "for", "and",
    "or", "is", "as", "it", "so", "we", "per", "each", "all",
})


def _segment_label(segment: str) -> str:
    """What a prose segment is called: its own words, without the connective."""
    text = re.sub(
        r"(?i)^(?:plus|and|with|\+)\s+", "", _strip_citations(segment).strip(),
    )
    return text.rstrip(" .,;")[:80] or "row with no reference"


def _strip_citations(text: str) -> str:
    """``text`` without the bracketed citations the answer appends."""
    return _CITATION_RE.sub("", text or "")


def _row_quantity(segment: str) -> float | None:
    """The quantity a prose row segment states, or None when it states none.

    The segment has to end with the figure, so a citation, a page reference or
    the sentence's own words after the figure mean the row states no quantity.
    The token before the figure is either a unit the answer prints in lower
    case ("m") or nothing; an upper-case prefix is a reference ("R 12"), not
    a row's quantity.
    """
    text = _strip_citations(segment).strip().rstrip(".,;")
    match = _ROW_QUANTITY_RE.search(text)
    if match is None:
        return None
    token = _LEADING_TOKEN_RE.search(text[: match.start()].rstrip())
    if token is not None and token.group(1) != token.group(1).lower():
        return None
    return _number(match.group(1))


def _row_segments(line: str) -> list[str]:
    """The segments of a prose row list.

    Rows are separated by semicolons. A remainder clause the answer appends
    after a dash ("18-18.5 m 1 - plus the remaining rows ...") is its own
    segment, so the quantity printed before it stays with its row.
    """
    text = re.sub(r"(?i)\s*(?:\u2014|\u2013|-)\s*plus\s+", "; plus ", line or "")
    text = re.sub(
        r"(?i)[;,]?\s*\bplus\s+(?=(?:the\s+|a\s+)?(?:remaining|further|rest|others|balance)\b)",
        "; plus ", text,
    )
    return [part for part in text.split(";") if part.strip()]


def _claim_unit(line: str) -> tuple[str, str, float] | None:
    """The figure and unit a claim line states first, as (text, unit, value)."""
    for match in _CLAIM_UNIT_RE.finditer(_strip_citations(line)):
        unit = match.group(2)
        if unit.lower() in _CLAIM_STOPWORDS:
            continue
        value = _number(match.group(1))
        if value is None:
            continue
        return match.group(1), unit, value
    return None


def _bucket_figure(text: str, unit: str) -> float | None:
    """The figure a bucket states for its own rows, when it carries the
    total's own unit - so a dimension inside a bucket's wording is not read as
    a quantity of the rows it stands for."""
    claim = _claim_unit(text)
    if claim is None or claim[1].lower() != (unit or "").lower():
        return None
    return claim[2]


def _is_row_list(line: str) -> bool:
    """True when a line prints a list of rows, most of them with a quantity.

    A row list is at least three segments, at least two of which state their
    own quantity, and at least two thirds of which do - so a sentence that
    merely happens to contain semicolons is not read as a list of rows, while
    a list with one row the answer could not read still is.
    """
    segments = _row_segments(line)
    quantified = sum(1 for segment in segments if _row_quantity(segment) is not None)
    return quantified >= 2 and quantified * 3 >= len(segments) * 2


def _prose_row_list_correction(
    cells: list[list[str] | None], lines: list[str],
    in_table: set[int] = frozenset(),
) -> Correction | None:
    """A row list the answer prints as prose, checked against its own total.

    Some answers print their rows as a `;`-separated sentence rather than a
    table ("The rows read: 1.5-2.0 m 18; 2.0-2.5 m 26; ..."). The
    arithmetic claim is the same one a Total cell makes, so it is read the
    same way: the stated total is the sum of the figures the answer prints,
    and it may only exceed them by figures the answer itself states for rows
    it did not itemise. A row a segment prints without a quantity leaves that
    row's contribution unknown, and then nothing is rewritten.
    """
    for index in [i for i, row in enumerate(cells) if row is None and i not in in_table]:
        line = lines[index]
        if _ATTRIBUTED_PROSE_RE.search(line) or not _TOTAL_CLAIM_RE.search(line):
            continue
        claim = _claim_unit(line)
        if claim is None:
            continue
        stated_text, unit, stated = claim
        listing_index = next(
            (
                i for i in range(len(lines))
                if i != index and i not in in_table and _is_row_list(lines[i])
            ),
            None,
        )
        if listing_index is None:
            continue
        read = _RowRead()
        for segment in _row_segments(lines[listing_index]):
            quantity = _row_quantity(segment)
            if _REMAINDER_RE.search(_strip_citations(segment)):
                read.unread.append(_segment_label(segment))
                figure = quantity if quantity is not None else _bucket_figure(segment, unit)
                if figure is not None:
                    read.unread_figures.append(figure)
                if _BALANCE_RE.search(segment):
                    read.balance = True
                continue
            if quantity is None:
                read.without.append(_segment_label(segment))
                continue
            read.kept.append(quantity)
        for other in range(len(lines)):
            if other in (index, listing_index) or other in in_table:
                continue
            if cells[other] is not None:
                continue
            text = lines[other]
            if not _REMAINDER_RE.search(text):
                continue
            read.unread.append(_segment_label(text))
            if _BALANCE_RE.search(text):
                read.balance = True
                continue
            figure = _bucket_figure(text, unit)
            if figure is not None:
                read.unread_figures.append(figure)
        if len(read.kept) < 2 or _disclosed_total(stated, read):
            continue
        return Correction(
            line_index=index, col=0, stated_text=stated_text, stated=stated,
            computed=read.printed, unit=unit, rows=len(read.kept),
            without_figure=read.without, remainders=read.unread, restated=[],
            in_cell=False,
        )
    return None


def _chain_findings(cells: list[list[str] | None], lines: list[str]) -> list[ChainFinding]:
    """Every sum chain the answer prints in prose, checked against its result.

    A line stating ``a + b + c = N`` makes the same arithmetic claim a Total
    cell makes, so it is checked the same way. A chain that prints an
    ellipsis states that its own list is incomplete: its result cannot be the
    complete total, and the answer says so without a figure being guessed.
    """
    out: list[ChainFinding] = []
    for index in [i for i, row in enumerate(cells) if row is None]:
        for match in _CHAIN_RE.finditer(lines[index]):
            raw_terms = match.group("terms")
            if _ELLIPSIS_RE.search(raw_terms):
                printed = [
                    value for value in (_number(part) for part in raw_terms.split("+"))
                    if value is not None
                ]
                out.append(ChainFinding(
                    line_index=index, stated_text=match.group("total"),
                    stated=_number(match.group("total")) or 0.0,
                    computed=None, unit=(match.group("unit") or "").strip(),
                    terms=len(printed),
                ))
                continue
            terms = [
                value for value in (_number(part) for part in raw_terms.split("+"))
                if value is not None
            ]
            total = _number(match.group("total"))
            if total is None or len(terms) < 3:
                continue
            computed = sum(terms)
            if _agrees(total, computed):
                continue
            out.append(ChainFinding(
                line_index=index, stated_text=match.group("total"), stated=total,
                computed=computed, unit=(match.group("unit") or "").strip(),
                terms=len(terms),
            ))
    return out


def _pipe_row(line: str, row: list[str]) -> str:
    """A rewritten pipe-delimited row, keeping the pipes the line printed."""
    stripped = line.strip()
    lead = "|" if stripped.startswith("|") else ""
    tail = "|" if stripped.endswith("|") else ""
    return lead + "|".join(row) + tail


def _apply_cell(
    lines: list[str], cells: list[list[str] | None], line_index: int, col: int, text: str,
) -> None:
    """Write a rewritten figure back into its cell, keeping every separator.

    The cell keeps its own padding, so a pipe table keeps its shape and a tab
    table keeps its columns.
    """
    row = cells[line_index]
    while len(row) <= col:
        row.append("")
    original = row[col] or ""
    lead = original[: len(original) - len(original.lstrip())]
    tail = original[len(original.rstrip()):]
    row[col] = f"{lead}{text}{tail}"
    tabbed = _TAB in lines[line_index]
    lines[line_index] = _TAB.join(row) if tabbed else _pipe_row(lines[line_index], row)


def _replaced_stated(
    lines: list[str], skip_line: int, stated: str, computed: str, unit: str,
) -> list[str]:
    """``lines`` with every statement of a wrong figure replaced.

    Only a figure stated WITH the column's own unit is replaced ("168 No",
    "2,468 m"), so a page number, an item code or a rate that repeats the
    digits is left alone. ``skip_line`` is the total's own line, whose cell or
    chain has already been rewritten.
    """
    pattern = re.compile(
        rf"(?<![\d,.]){re.escape(stated)}\s*{re.escape(unit)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    ) if unit else None
    return [
        line if pattern is None or offset == skip_line
        else pattern.subn(f"{computed} {unit}", line, count=_MAX_REPLACEMENTS)[0]
        for offset, line in enumerate(lines)
    ]


def _summary(correction: Correction, computed_text: str) -> str:
    """One plain line: what the total covers, and what it leaves out."""
    amounts = " ".join(t for t in (computed_text, correction.unit) if t)
    rows_word = "row" if correction.rows == 1 else "rows"
    parts = [
        (
            f"The rows printed above sum to {amounts} over {correction.rows} "
            f"{rows_word}, so rows this answer did not print are not part of "
            "that figure and the total is incomplete."
        )
    ]
    if correction.without_figure:
        parts.append(
            "Printed without a readable figure, so not in the total: "
            + ", ".join(correction.without_figure) + "."
        )
    if correction.remainders:
        named = ", ".join(correction.remainders)
        verb = "stands" if len(correction.remainders) == 1 else "stand"
        parts.append(
            f"{named} {verb} for rows this answer does not itemise; "
            "not a row of the bill, so not part of the total, and the total "
            "is therefore incomplete."
        )
    if correction.restated:
        named = ", ".join(correction.restated)
        verb = "restates" if len(correction.restated) == 1 else "restate"
        parts.append(
            f"{named} {verb} rows the answer has already counted - the same "
            "reference is printed above - so it is not a further row and is "
            "not added a second time."
        )
    return " ".join(parts)


def _chain_summary(finding: ChainFinding, computed_text: str) -> str:
    """One plain line for a printed sum chain."""
    unit = f" {finding.unit}" if finding.unit else ""
    stated = _format_like(finding.stated, finding.stated_text)
    if finding.computed is None:
        return (
            "The answer's own row list is incomplete (it prints an ellipsis over "
            f"{finding.terms} terms), so the {stated}{unit} it states at the end of "
            "that chain is not the sum of the rows it printed."
        )
    return (
        f"The {finding.terms} terms printed in the answer's own chain sum to "
        f"{computed_text}{unit}, not the {stated}{unit} it stated, so that total is "
        "corrected to the sum of the terms printed."
    )


def _reconcile(raw: str) -> str:
    lines = raw.split("\n")
    cells: list[list[str] | None] = [_cells(line) for line in lines]
    blocks = _table_blocks(cells)
    corrections = [c for block in blocks for c in _block_corrections(block, cells)]
    corrections.extend(
        prose for prose in (_prose_correction(b, cells, lines) for b in blocks)
        if prose is not None
    )
    listed = _prose_row_list_correction(
        cells, lines, {i for block in blocks for i in block},
    )
    if listed is not None:
        corrections.append(listed)
    chains = _chain_findings(cells, lines)
    if not corrections and not chains:
        return raw
    summaries: list[str] = []
    for correction in corrections:
        computed_text = _format_like(correction.computed, correction.stated_text)
        stated_text = _format_like(correction.stated, correction.stated_text)
        if correction.in_cell:
            _apply_cell(lines, cells, correction.line_index, correction.col, computed_text)
        skip_line = correction.line_index if correction.in_cell else -1
        lines[:] = _replaced_stated(
            lines, skip_line, stated_text, computed_text, correction.unit,
        )
        summaries.append(_summary(correction, computed_text))
    for chain in chains:
        stated = _format_like(chain.stated, chain.stated_text)
        if chain.computed is None:
            summaries.append(_chain_summary(chain, ""))
            continue
        computed_text = _format_like(chain.computed, chain.stated_text)
        line = lines[chain.line_index]
        swapped = False
        for match in _CHAIN_RE.finditer(line):
            if match.group("total") != chain.stated_text or swapped:
                continue
            start, end = match.span("total")
            line = line[:start] + computed_text + line[end:]
            swapped = True
        lines[chain.line_index] = line
        lines[:] = _replaced_stated(lines, chain.line_index, stated, computed_text, chain.unit)
        summaries.append(_chain_summary(chain, computed_text))
    _LOG.info(
        "row_total_reconciled columns=%d chains=%d", len(corrections), len(chains),
    )
    return "\n".join(
        lines + ["", "**Row total checked against the rows printed above:**"]
        + [f"- {s}" for s in summaries]
    )


def reconcile(text: str) -> str:
    """Check every stated row total against the rows the answer prints.

    An answer with no table and no printed sum chain, or whose totals already
    equal their rows, is returned byte for byte. Never raises: a check that
    can break an answer is a check that gets switched off.
    """
    raw = text or ""
    try:
        return _reconcile(raw)
    except Exception:  # noqa: BLE001 - a check must never break a turn
        _LOG.warning("row-total reconciliation failed; answer passed through", exc_info=True)
        return raw
