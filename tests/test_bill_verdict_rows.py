"""P-B3: the bill verdict judges the rows the plan holds, whatever the column
source of their page; the plan counts priced rows with an empty description;
and the label test's letter rule accepts a letter of any script.

Every input is a synthetic page laid out here from invented word boxes; no
document, bill, page or column position is named in product code. Each
decision is exercised with at least two inputs.
"""

from __future__ import annotations

import pytest

from app.core import bill_classify as bc
from app.core import bill_table_geometry as g

H = 20.0
CW = 9.0
LINE = 34.0


def _cell(text: str, x: float, y: float, page: int, right: bool) -> list[g.Word]:
    toks = text.split()
    width = sum(CW * len(t) for t in toks) + CW * (len(toks) - 1)
    cur = x - width if right else x
    out = []
    for tok in toks:
        w = CW * len(tok)
        out.append(g.Word(tok, page, cur, y, w, H))
        cur += w + CW
    return out


def _layout(columns, lines, page: int, header=None) -> list[g.Word]:
    """``columns`` = [(x, right_aligned)]; a str line is a label at the left."""
    words: list[g.Word] = []
    y = 100.0
    if header:
        for (x, right), label in zip(columns, header):
            words += _cell(label, x, y, page, right)
        y += LINE
    for line in lines:
        if isinstance(line, str):
            words += _cell(line, columns[0][0], y, page, False)
        else:
            for (x, right), text in zip(columns, line):
                if text:
                    words += _cell(text, x, y, page, right)
        y += LINE
    return words


def _read(words: list[g.Word], index: int) -> bc.PageRead:
    return bc.PageRead(index, bc.SOURCE_TEXT_LAYER, tuple(words))


def _priced(rows):
    return [r for r in rows
            if all(g.parse_number(r.fields.get(f, "")) is not None for f in ("quantity", "rate", "total"))]


COLS_1 = [(40, False), (520, False), (700, True), (850, True), (1020, True)]
PRICED_1 = [
    ["Supply pipe 150 dia", "m", "120", "45.50", "5,460.00"],
    ["Gate valve 150 dia", "Nr", "4", "1,200.00", "4,800.00"],
    ["Thrust block", "Nr", "6", "300.00", "1,800.00"],
    ["Bedding sand", "m3", "35", "62.00", "2,170.00"],
]
HEADER_1 = ["Description", "Unit", "Qty", "Rate", "Amount"]
COLS_2 = [(60, False), (410, False), (560, True), (720, True), (900, True)]
PRICED_2 = [
    ["Excavate trench", "m3", "200", "10.00", "2,000.00"],
    ["Trim and compact", "m2", "300", "2.50", "750.00"],
    ["Backfill selected", "m3", "180", "14.25", "2,565.00"],
]
HEADER_2 = ["Description", "Unit", "Quantity", "Rate", "Amount"]


# --- 1. the verdict sees pages read under inferred columns ------------------

@pytest.mark.parametrize("cols,lines", [(COLS_1, PRICED_1), (COLS_2, PRICED_2)])
def test_verdict_counts_priced_rows_of_inferred_pages(cols, lines):
    reads = [_read(_layout(cols, lines, page=0), 0), _read(_layout(cols, lines, page=1), 1)]
    v = bc.verdict_of_reads(reads)
    # No header line anywhere: the page-by-page header count is still 0 ...
    assert v.header_pages == 0
    # ... and the rows the plan reads under inferred columns are judged.
    assert v.inferred_pages == 2
    assert v.priced_rows == v.reconciled_rows == 2 * len(lines)
    assert (v.is_bill, v.reason) == (True, bc.BILL_PRICED)
    # The header-only reading (no rows given) still says "no header".
    words = [w for r in reads for w in r.words]
    assert bc.bill_verdict(words).reason == bc.NOT_BILL_NO_HEADER


@pytest.mark.parametrize("cols,lines,header,other_cols,other_lines", [
    (COLS_1, PRICED_1, HEADER_1, COLS_2, PRICED_2),
    (COLS_2, PRICED_2, HEADER_2, COLS_1, PRICED_1),
])
def test_verdict_priced_rows_equal_the_plans_priced_rows(cols, lines, header, other_cols, other_lines):
    """A header page, a page laid out on other columns (read under inferred
    columns: the carried header does not fit it) and a page that continues
    the header: the verdict's priced count is the plan's own extraction's."""
    pages = [
        _layout(cols, lines, page=0, header=header),
        _layout(other_cols, other_lines, page=1),
        _layout(cols, lines[:2], page=2),  # too few witnesses: the header carries
    ]
    reads = [_read(p, i) for i, p in enumerate(pages)]
    stats = g.ColumnStats()
    plan_rows = g.extract_bill_rows([w for p in pages for w in p], stats=stats)
    assert stats.pages == {"header": 2, "inferred": 1, "none": 0}
    v = bc.verdict_of_reads(reads)
    assert v.priced_rows == len(_priced(plan_rows)) == len(lines) + len(other_lines) + 2
    assert v.header_pages == 1 and v.inferred_pages == 1
    assert v.reason == bc.BILL_PRICED


@pytest.mark.parametrize("cols,lines,header,gap", [
    (COLS_1, PRICED_1, HEADER_1, 7),
    (COLS_2, PRICED_2, HEADER_2, 3),
])
def test_a_header_never_carries_across_a_gap_between_sampled_pages(cols, lines, header, gap):
    """Adjacent pages carry the header, as in the plan; a sampled page after
    a gap does not."""
    first = _layout(cols, lines, page=0, header=header)
    follow = lines[:2]  # too few witnesses to infer: only a carried header reads it
    adjacent = bc.verdict_of_reads([_read(first, 0), _read(_layout(cols, follow, page=1), 1)])
    apart = bc.verdict_of_reads([_read(first, 0), _read(_layout(cols, follow, page=gap), gap)])
    assert adjacent.priced_rows == len(lines) + len(follow)
    assert apart.priced_rows == len(lines)


@pytest.mark.parametrize("cols,lines", [
    (COLS_1, ["Preliminaries", "General items", "Site establishment"]),
    (COLS_2, [["Sundries", "item", "7", "3.00", "99.00"],
              ["Extras", "item", "5", "4.00", "31.00"],
              ["Other", "item", "9", "6.00", "17.00"]]),
])
def test_no_header_and_no_inference_is_still_not_a_bill(cols, lines):
    v = bc.verdict_of_reads([_read(_layout(cols, lines, page=0), 0)])
    assert v.inferred_pages == 0 and v.header_pages == 0
    assert (v.is_bill, v.reason) == (False, bc.NOT_BILL_NO_HEADER)


# --- 2. the plan counts priced rows with an empty description ---------------

def _row(desc: str, q: str = "2", r: str = "3.00", t: str = "6.00") -> g.BillRow:
    return g.BillRow(section="S", source_page=0,
                     fields={"description": desc, "quantity": q, "rate": r, "total": t},
                     extra={}, path=["S"])


@pytest.mark.parametrize("rows,empty", [
    ([_row("Pipe"), _row("Valve"), _row("Bend")], 0),
    ([_row(""), _row("  "), _row("Valve"), _row("", r="", t="")], 2),  # the last is unpriced
])
def test_plan_counts_priced_rows_with_an_empty_description(rows, empty):
    out = g.ColumnStats().as_dict(rows)
    assert out["priced_rows_empty_description"] == empty


# --- 3. the label test accepts a letter of any script -----------------------

@pytest.mark.parametrize("text,want", [
    ("Earthworks", True),
    ("أعمال الحفر", True),       # Arabic
    ("Земляные работы", True),   # Cyrillic
    ("土方工程", True),            # CJK
    ("1,200.00", False),
    ("— 12 / 3 —", False),
    ("", False),
])
def test_has_letter_is_the_unicode_letter_property(text, want):
    assert g.has_letter(text) is want


@pytest.mark.parametrize("cols,lines,heading", [
    (COLS_1, PRICED_1, "أعمال الحفر"),
    (COLS_2, PRICED_2, "Земляные работы"),
])
def test_heading_in_another_script_becomes_the_section_label(cols, lines, heading):
    """Under inferred columns a no-figure line with letters is a label."""
    rows = g.extract_bill_rows(_layout(cols, [heading, *lines], page=0))
    assert len(rows) == len(lines)
    assert all(tuple(r.path) == (heading,) and r.section == heading for r in rows)


@pytest.mark.parametrize("cols,lines,header,heading", [
    (COLS_1, PRICED_1, HEADER_1, "أعمال الحفر"),
    (COLS_2, PRICED_2, HEADER_2, "Земляные работы"),
])
def test_heading_in_another_script_before_the_header_is_a_label(cols, lines, header, heading):
    """With no header yet in effect, a line in another script is a label too."""
    words = _layout(cols, [heading], page=0)
    words += _layout(cols, lines, page=1, header=header)
    events: list[g.LabelEvent] = []
    rows = g.extract_bill_rows(words, on_label=events.append)
    assert [e.reason for e in events] == [g.LABEL_BEFORE_HEADER]
    assert all(r.path[0] == heading for r in rows)


@pytest.mark.parametrize("cols,lines,stray", [
    (COLS_1, PRICED_1, "— — —"),
    (COLS_2, PRICED_2, "12 / 7"),
])
def test_a_letter_free_line_is_still_not_a_label(cols, lines, stray):
    rows = g.extract_bill_rows(_layout(cols, [stray, *lines], page=0))
    assert len(rows) == len(lines)
    assert all(tuple(r.path) == () for r in rows)
