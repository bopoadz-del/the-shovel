"""P-B3: a page's quantity, rate and total columns inferred from its own rows
by the reconciliation test (the rightmost of three aligned figures is the
product of the two before it).

Every input is a synthetic page laid out here from invented word boxes; no
document, bill, page or column position is named in product code. Each
decision is exercised with at least two inputs.
"""

from __future__ import annotations

import pytest

from app.core import bill_table_geometry as g
from scripts import bill_plan_once as bpo

H = 20.0   # word height in px
CW = 9.0   # px per character
LINE = 34.0


def _cell(text: str, x: float, y: float, page: int, right: bool) -> list[g.Word]:
    """Words of one cell; a right-aligned cell ends at ``x``."""
    toks = text.split()
    width = sum(CW * len(t) for t in toks) + CW * (len(toks) - 1)
    cur = x - width if right else x
    out = []
    for tok in toks:
        w = CW * len(tok)
        out.append(g.Word(tok, page, cur, y, w, H))
        cur += w + CW
    return out


def _layout(columns, lines, page=1, header=None):
    """``columns`` = [(x, right_aligned)], one per cell; ``header`` (optional)
    is one label per column, printed above; a str line is a label at the
    left margin."""
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


# description (left), unit (left), quantity / rate / total right-aligned
COLS_1 = [(40, False), (520, False), (700, True), (850, True), (1020, True)]
PRICED_1 = [
    ["Supply pipe 150 dia", "m", "120", "45.50", "5,460.00"],
    ["Gate valve 150 dia", "Nr", "4", "1,200.00", "4,800.00"],
    ["Thrust block", "Nr", "6", "300.00", "1,800.00"],
    ["Bedding sand", "m3", "35", "62.00", "2,170.00"],
]
COLS_2 = [(60, False), (410, False), (560, True), (720, True), (900, True)]
PRICED_2 = [
    ["Excavate trench", "m3", "200", "10.00", "2,000.00"],
    ["Trim and compact", "m2", "300", "2.50", "750.00"],
    ["Backfill selected", "m3", "180", "14.25", "2,565.00"],
]


def _rows(words, stats=None):
    return g.extract_bill_rows(words, stats=stats)


def _priced(rows):
    return [r for r in rows if r.fields.get("rate") and r.fields.get("total")]


# --- 1. a page with witnesses and no header is read under inferred columns ---

@pytest.mark.parametrize("cols,lines", [(COLS_1, PRICED_1), (COLS_2, PRICED_2)])
def test_header_free_page_with_witnesses_is_priced(cols, lines):
    stats = g.ColumnStats()
    rows = _rows(_layout(cols, lines), stats)
    assert len(rows) == len(lines)
    for row, want in zip(rows, lines):
        assert row.fields["description"] == want[0]
        assert row.fields["unit"] == want[1]
        assert (row.fields["quantity"], row.fields["rate"], row.fields["total"]) == tuple(want[2:])
        assert row.check == ""
    out = stats.as_dict(rows)
    assert out["pages_by_source"] == {"header": 0, "inferred": 1, "none": 0}
    assert out["witness_lines"] == len(lines)
    assert out["rows_priced_reconciled"] == len(lines)
    assert out["rows_figures_not_reconciling"] == 0
    assert out["labels_reread_as_rows"] == len(lines)  # no header: all were labels


@pytest.mark.parametrize("cols,lines", [(COLS_1, PRICED_1), (COLS_2, PRICED_2)])
def test_inference_is_independent_of_word_order(cols, lines):
    words = _layout(cols, lines)
    a = [r.fields for r in _rows(words)]
    b = [r.fields for r in _rows(list(reversed(words)))]
    assert a == b and len(a) == len(lines)


# --- 2. too few witnesses: the page keeps today's behaviour ---------------

@pytest.mark.parametrize("cols,lines", [(COLS_1, PRICED_1[:2]), (COLS_2, PRICED_2[:2])])
def test_page_with_too_few_witnesses_is_not_inferred(cols, lines):
    # The third line keeps the bands vertical but does not reconcile.
    bad = [*lines, ["Sundries", "item", "7", "3.00", "99.00"]]
    words = _layout(cols, bad)
    columns, witnesses = g.infer_page_columns(g.group_lines(words))
    assert columns is None and witnesses == 2
    stats = g.ColumnStats()
    assert _rows(words, stats) == []  # no header, no proof: labels, as today
    assert stats.pages == {"header": 0, "inferred": 0, "none": 1}


@pytest.mark.parametrize("cols,lines", [(COLS_1, PRICED_1[:2]), (COLS_2, PRICED_2[:2])])
def test_too_few_witnesses_under_a_partial_header_stays_unpriced(cols, lines):
    header = ["Description", "Unit", "Qty", "", ""]
    bad = [*lines, ["Sundries", "item", "7", "3.00", "99.00"]]
    stats = g.ColumnStats()
    rows = _rows(_layout(cols, bad, header=header), stats)
    assert rows and not _priced(rows)
    assert stats.pages == {"header": 1, "inferred": 0, "none": 0}


def test_no_carry_of_inferred_columns_to_another_page():
    words = _layout(COLS_1, PRICED_1, page=1) + _layout(COLS_1, PRICED_1[:2], page=2)
    stats = g.ColumnStats()
    rows = _rows(words, stats)
    assert {r.source_page for r in rows} == {1}
    assert stats.pages == {"header": 0, "inferred": 1, "none": 1}


# --- 3. a header that disagrees with the bands ------------------------------

@pytest.mark.parametrize("cols,lines,header", [
    # rate and amount labels printed over each other's figures
    (COLS_1, PRICED_1, ["Description", "Unit", "Qty", "Amount", "Rate"]),
    (COLS_2, PRICED_2, ["Description", "Unit", "Quantity", "Amount", "Rate"]),
])
def test_disagreeing_header_is_counted_and_out_witnessed(cols, lines, header):
    stats = g.ColumnStats()
    rows = _rows(_layout(cols, lines, header=header), stats)
    assert stats.header_conflicts == 2
    assert stats.pages["inferred"] == 1
    assert all(r.check == "" for r in rows) and len(_priced(rows)) == len(lines)
    assert [r.fields["total"] for r in rows] == [ln[4] for ln in lines]


@pytest.mark.parametrize("cols,lines,header", [
    (COLS_1, PRICED_1, ["Description", "Unit", "Qty", "Rate", "Amount"]),
    (COLS_2, PRICED_2, ["Description", "Unit", "Quantity", "Rate", "Amount"]),
])
def test_agreeing_header_keeps_the_header_path(cols, lines, header):
    stats = g.ColumnStats()
    rows = _rows(_layout(cols, lines, header=header), stats)
    assert stats.header_conflicts == 0
    assert stats.pages == {"header": 1, "inferred": 0, "none": 0}
    assert stats.labels_reread_as_rows == 0
    assert len(_priced(rows)) == len(lines)


# --- 4. a factor of 0 or 1 is not a witness ---------------------------------

def _toks(*values):
    return {i: g.Word(v, 1, 100.0 * i, 0, CW * len(v), H) for i, v in enumerate(values)}


@pytest.mark.parametrize("values,want", [
    (("1", "500.00", "500.00"), []),       # quantity of 1
    (("12", "1", "12.00"), []),            # rate of 1
    (("0", "45.00", "0.00"), []),          # zero
    (("12", "5.00", "60.00"), [(0, 1, 2)]),
    (("3", "7.00", "21.00", "9"), [(0, 1, 2)]),
])
def test_witness_triples(values, want):
    assert g.witness_triples(_toks(*values)) == want


@pytest.mark.parametrize("factor_one", [
    [["Lump sum a", "item", "1", "500.00", "500.00"],
     ["Lump sum b", "item", "1", "750.00", "750.00"],
     ["Lump sum c", "item", "1", "125.00", "125.00"]],
    [["Prov sum a", "sum", "300", "1", "300.00"],
     ["Prov sum b", "sum", "450", "1", "450.00"],
     ["Prov sum c", "sum", "600", "1", "600.00"]],
])
def test_lines_with_a_factor_of_one_earn_no_columns(factor_one):
    columns, witnesses = g.infer_page_columns(g.group_lines(_layout(COLS_1, factor_one)))
    assert columns is None and witnesses == 0


# --- 5. a non-reconciling row on an inferred page is not priced -------------

@pytest.mark.parametrize("cols,lines,bad", [
    (COLS_1, PRICED_1, ["Manhole cover", "Nr", "3", "410.00", "1,300.00"]),
    (COLS_2, PRICED_2, ["Disposal off site", "m3", "50", "8.00", "500.00"]),
])
def test_non_reconciling_row_keeps_its_flag_and_is_not_priced(cols, lines, bad):
    stats = g.ColumnStats()
    rows = _rows(_layout(cols, [*lines, bad]), stats)
    flagged = [r for r in rows if r.check]
    assert [r.fields["description"] for r in flagged] == [bad[0]]
    out = stats.as_dict(rows)
    assert out["rows_priced_reconciled"] == len(lines)
    assert out["rows_figures_not_reconciling"] == 1


# --- the majority rule --------------------------------------------------------

COLS_4 = [*COLS_1, (1180, True)]  # a fourth figure band


@pytest.mark.parametrize("dup_lines,inferred", [
    (4, False),  # every line also reconciles into the fourth band: a tie
    (1, True),   # one line does: the first assignment has a clear majority
])
def test_assignment_needs_a_clear_majority(dup_lines, inferred):
    lines = [[*ln, ln[4] if i < dup_lines else "9"] for i, ln in enumerate(PRICED_1)]
    columns, _ = g.infer_page_columns(g.group_lines(_layout(COLS_4, lines)))
    assert (columns is not None) is inferred
    if inferred:
        assert columns.witnesses == len(PRICED_1)


# --- unit cell ------------------------------------------------------------------

@pytest.mark.parametrize("unit,want", [("m2", "m2"), ("", "")])
def test_unit_is_the_short_cell_before_the_quantity(unit, want):
    lines = [[ln[0], unit, *ln[2:]] for ln in PRICED_1]
    rows = _rows(_layout(COLS_1, lines))
    assert {r.fields["unit"] for r in rows} == {want}
    assert [r.fields["description"] for r in rows] == [ln[0] for ln in PRICED_1]


# --- plan summary carries the counts through the redaction ------------------------

@pytest.mark.parametrize("cols,lines", [(COLS_1, PRICED_1), (COLS_2, PRICED_2)])
def test_column_counts_pass_the_public_summary(cols, lines):
    stats = g.ColumnStats()
    rows = _rows(_layout(cols, lines), stats)
    counts = stats.as_dict(rows)
    assert "columns" in bpo.PLAN_FIELDS
    assert bpo.public_summary({"columns": counts})["columns"] == counts
