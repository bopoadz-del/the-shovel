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

Three rules, all decided from the answer's own text and its own figures:

(1) A total is the sum of the rows the answer prints, or the answer says it
    is incomplete. A total that disagrees with its own column is replaced by
    the column's sum - in a Total cell, in a printed ``a + b + c = N`` chain,
    and in every prose statement of that figure that carries the same unit,
    so the answer cannot keep the wrong number in its first line - and one
    plain line says what the total now covers and what it leaves out.
(2) A row printed without a figure (``-``, ``n/a``, an empty cell) is named
    as a row the answer did not read. It adds nothing to the sum.
(3) An entry that stands for rows the answer does not print - "the
    remaining rows", "others", "balance", "not listed" - is a remainder
    bucket, not a row. Its figure never enters a total; it is named in the
    appended line as rows that were not itemised.

A total that already equals its column is returned byte for byte: the check
may only rewrite an answer that disagrees with itself. A total the answer
credits to a document ("[source: ...]", "as printed", "carried to") is that
document's figure, not a sum of the rows above it, and is left alone.

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


def _number(cell: str) -> float | None:
    """The figure a cell states, or None.

    One figure, or the sum of a printed sum expression ("1 + 1 + 1 + 1" is
    four). A cell holding words, a dimension or a range states no figure.
    """
    text = (cell or "").strip()
    if not _FIGURE_CELL_RE.match(text):
        return None
    total = 0.0
    for part in text.split("+"):
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
    description the bill prints for it.
    """
    words = [
        (cell or "").strip() for index, cell in enumerate(row)
        if index != figure_col and _number(cell) is None and (cell or "").strip()
    ]
    if not words:
        return (row[0] if row else "").strip() or "row with no reference"
    label = max(words, key=len)
    # "plus the remaining rows", "and further rows": the connective is the
    # sentence's, not the row's name.
    return re.sub(r"(?i)^(?:plus|and|with|\+)\s+", "", label)


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
    document. Rule (3) removes a remainder bucket from the sum before the
    comparison; rule (2) records a row printed without a figure.

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
    kept: list[float] = []
    without: list[str] = []
    remainders: list[str] = []
    for row in data_rows:
        label = _row_label(row, figures_in)
        row_text = " ".join((c or "").strip() for c in row)
        value = _number(row[figures_in] if figures_in < len(row) else "")
        if _REMAINDER_RE.search(row_text):
            remainders.append(label)
            continue
        if value is None:
            without.append(label)
            continue
        kept.append(value)
    if not kept:
        return None
    computed = sum(kept)
    if _agrees(stated, computed):
        return None
    unit = _header_unit(header_cells[figures_in] if figures_in < len(header_cells) else "")
    return Correction(
        line_index=line_index, col=col, stated_text=stated_text, stated=stated,
        computed=computed, unit=unit, rows=len(kept),
        without_figure=without, remainders=remainders,
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
    if any(cells[i] is None and _REMAINDER_RE.search(lines[i]) for i in range(len(lines))
           if i not in block):
        return None
    rows = [cells[i] or [] for i in block]
    if not rows:
        return None
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
    if not unit:
        return None
    for index, line in enumerate(lines):
        if index in block:
            continue
        if not _TOTAL_CLAIM_RE.search(line):
            continue
        stated = _claim_figure(line, unit)
        if stated is None:
            continue
        stated_text, stated_value = stated
        kept: list[float] = []
        without: list[str] = []
        remainders: list[str] = []
        for row in data[1:]:
            label = _row_label(row, col)
            row_text = " ".join((c or "").strip() for c in row)
            value = _number(row[col] if col < len(row) else "")
            if _REMAINDER_RE.search(row_text):
                remainders.append(label)
                continue
            if value is None:
                without.append(label)
                continue
            kept.append(value)
        if not kept or _agrees(stated_value, sum(kept)):
            continue
        return Correction(
            line_index=index, col=col, stated_text=stated_text, stated=stated_value,
            computed=sum(kept), unit=unit, rows=len(kept),
            without_figure=without, remainders=remainders, in_cell=False,
        )
    return None


def _chain_findings(lines: list[str], cells: list[list[str] | None]) -> list[ChainFinding]:
    """Every sum chain the answer prints, checked against its own result.

    A line stating ``a + b + c = N`` makes the same arithmetic claim a Total
    cell makes, so it is checked the same way. A chain that prints an
    ellipsis states that its own list is incomplete: its result cannot be the
    complete total, and the answer says so without a figure being guessed.
    """
    out: list[ChainFinding] = []
    for index, line in enumerate(lines):
        if cells[index] is not None:
            continue
        for match in _CHAIN_RE.finditer(line):
            total = _number(match.group("total"))
            if total is None:
                continue
            raw_terms = match.group("terms")
            if _ELLIPSIS_RE.search(raw_terms):
                printed = [
                    value for value in (_number(part) for part in raw_terms.split("+"))
                    if value is not None
                ]
                out.append(ChainFinding(
                    line_index=index, stated_text=match.group("total"), stated=total,
                    computed=None, unit=(match.group("unit") or "").strip(),
                    terms=len(printed),
                ))
                continue
            terms = [
                value for value in (_number(part) for part in raw_terms.split("+"))
                if value is not None
            ]
            if len(terms) < 3:
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


def _apply_cell(
    lines: list[str], cells: list[list[str] | None], line_index: int, col: int, text: str,
) -> None:
    """Write a rewritten figure back into its cell, keeping every separator."""
    row = cells[line_index]
    if row is None:  # pragma: no cover - only a table line is ever rewritten
        return
    while len(row) <= col:
        row.append("")
    # Keep the cell's own padding so a pipe table keeps its shape.
    original = row[col] or ""
    lead = original[: len(original) - len(original.lstrip())]
    tail = original[len(original.rstrip()):]
    row[col] = f"{lead}{text}{tail}"
    if _TAB in lines[line_index]:
        lines[line_index] = _TAB.join(row)
        return
    lead = "|" if lines[line_index].strip().startswith("|") else ""
    tail = "|" if lines[line_index].strip().endswith("|") else ""
    lines[line_index] = lead + "|".join(row) + tail


def _replace_stated(
    lines: list[str], skip_line: int, stated: str, computed: str, unit: str,
) -> int:
    """Replace this answer's prose statements of a wrong figure, in place.

    Only a figure stated WITH the column's own unit is replaced ("168 No",
    "2,468 m"), so a page number, an item code or a rate that repeats the
    digits is left alone. ``skip_line`` is the total's own line, whose cell or
    chain has already been rewritten.
    """
    if not unit:
        return 0
    pattern = re.compile(
        rf"(?<![\d,.]){re.escape(stated)}\s*{re.escape(unit)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    replaced = 0
    for offset, line in enumerate(lines):
        if offset == skip_line:
            continue
        new_line, count = pattern.subn(f"{computed} {unit}", line, count=_MAX_REPLACEMENTS)
        if count:
            lines[offset] = new_line
            replaced += count
    return replaced


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
    corrections: list[Correction] = []
    block_start: int | None = None
    for index in range(len(lines) + 1):
        row = cells[index] if index < len(lines) else None
        if row is not None:
            if block_start is None:
                block_start = index
            continue
        if block_start is None:
            continue
        block = list(range(block_start, index))
        corrections.extend(_block_corrections(block, cells))
        prose = _prose_correction(block, cells, lines)
        if prose is not None:
            corrections.append(prose)
        block_start = None
    chains = _chain_findings(lines, cells)
    if not corrections and not chains:
        return raw
    summaries: list[str] = []
    for correction in corrections:
        computed_text = _format_like(correction.computed, correction.stated_text)
        stated_text = _format_like(correction.stated, correction.stated_text)
        skip_line = correction.line_index
        if correction.in_cell:
            _apply_cell(lines, cells, correction.line_index, correction.col, computed_text)
        else:
            skip_line = -1
        _replace_stated(lines, skip_line, stated_text, computed_text, correction.unit)
        summaries.append(_summary(correction, computed_text))
    for chain in chains:
        stated = _format_like(chain.stated, chain.stated_text)
        if chain.computed is None:
            summaries.append(_chain_summary(chain, ""))
            continue
        computed_text = _format_like(chain.computed, chain.stated_text)
        line = lines[chain.line_index]
        replaced = False
        for match in _CHAIN_RE.finditer(line):
            if match.group("total") != chain.stated_text or replaced:
                continue
            start, end = match.span("total")
            line = line[:start] + computed_text + line[end:]
            replaced = True
        lines[chain.line_index] = line
        _replace_stated(lines, chain.line_index, stated, computed_text, chain.unit)
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
    if not (text or "").strip():
        return text
    try:
        return _reconcile(text)
    except Exception:  # noqa: BLE001 - a check must never break a turn
        _LOG.warning("row-total reconciliation failed; answer passed through", exc_info=True)
        return text
