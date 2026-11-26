"""A page that prints its own column-header line naming figure columns, but
whose description label did not read, places those figure columns itself
instead of reading under a header carried from an earlier table.

Every input is a synthetic page laid out here from invented word boxes; no
document, bill, page or column position is named in product code. Each
decision is exercised with at least two inputs and has a case where it must
not fire.
"""

from __future__ import annotations

import pytest

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


def _line(cells, page: int, y: float) -> list[g.Word]:
    """``cells`` = [(text, x, right_aligned)]."""
    out: list[g.Word] = []
    for text, x, right in cells:
        out += _cell(text, x, y, page, right)
    return out


# The earlier table: a full header (description + quantity resolve) whose two
# right-hand figure labels do not resolve, printed at its own positions.
CARRIED = [("Item", 40, False), ("Description", 120, False), ("Unit", 520, False),
           ("Qty", 700, True), ("Charge", 790, True), ("Charges", 1180, True)]

# This page's table: quantity / rate / total printed right-aligned here.
Q_X, R_X, T_X = 700, 900, 1060


def _two_pages(own_header, rows):
    """Page 1 prints the carried header and nothing else; page 2 prints
    ``own_header`` (cells) then ``rows`` (description, unit, q, r, t)."""
    words = _line(CARRIED, 1, 100.0)
    y = 100.0
    words += _line(own_header, 2, y)
    for desc, unit, q, r, t in rows:
        y += LINE
        words += _line([(desc, 120, False), (unit, 520, False),
                        (q, Q_X, True), (r, R_X, True), (t, T_X, True)], 2, y)
    return words


# Fewer than MIN_WITNESS_LINES rows: no column inference, the header path only.
ROWS_A = [("Supply pump unit", "Nr", "2", "4,500", "9,000.00"),
          ("Lay cable duct", "m", "30", "15", "450.00")]
ROWS_B = [("Fix bracket", "Nr", "7", "120", "840.00"),
          ("Paint railing", "m2", "40", "25", "1,000.00")]

OWN_RATE_AMOUNT = [("an", 120, False), ("Code", 400, False),
                   ("Unit Rate", R_X, True), ("Amount", T_X, True)]
OWN_QTY_TOTAL = [("xx", 120, False), ("Quantity", Q_X, True), ("Total", T_X, True)]


@pytest.mark.parametrize("own,rows,want", [
    (OWN_RATE_AMOUNT, ROWS_A, [("4,500", "9,000.00"), ("15", "450.00")]),
    (OWN_QTY_TOTAL, ROWS_B, [(None, "840.00"), (None, "1,000.00")]),
])
def test_a_partial_own_header_places_its_figure_columns(own, rows, want):
    out = g.extract_bill_rows(_two_pages(own, rows))
    priced = [r for r in out if r.fields.get("total")]
    assert [r.fields["total"] for r in priced] == [t for _r, t in want]
    for r, (rate, _t) in zip(priced, want):
        if rate is not None:
            assert r.fields["rate"] == rate
    assert [r.fields["description"] for r in priced] == [d for d, *_ in rows]


@pytest.mark.parametrize("rows", [ROWS_A, ROWS_B])
def test_one_figure_label_is_not_a_header_and_the_carried_one_stays(rows):
    one = [("an", 120, False), ("Code", 400, False), ("Amount", T_X, True)]
    out = g.extract_bill_rows(_two_pages(one, rows))
    assert not any(r.fields.get("total") for r in out)


@pytest.mark.parametrize("cells,want", [
    (OWN_RATE_AMOUNT, ["rate", "total"]),
    (OWN_QTY_TOTAL, ["quantity", "total"]),
    ([("an", 120, False), ("Code", 400, False), ("Amount", T_X, True)], []),
    ([("Item", 40, False), ("Description", 120, False), ("Rate", R_X, True),
      ("Amount", T_X, True)], []),           # a full header: not partial
    ([("Unit Rate", R_X, True), ("Amount", T_X, True)], []),  # two cells only
    ([("ab", 120, False), ("cd", 400, False), ("ef", T_X, True)], []),
])
def test_own_figure_columns(cells, want):
    cols = g.own_figure_columns(_line(cells, 1, 100.0))
    assert [c.field for c in cols] == want


def _col(label, field, left, right):
    return g._Column(label=label, field=field, left=left, right=right)


@pytest.mark.parametrize("own,want", [
    # an own column replaces the carried columns it overlaps; others stay
    ([_col("Rate", "rate", 850, 900), _col("Amount", "total", 1000, 1060)],
     [("D", "description"), ("Q", "quantity"), ("Rate", "rate"), ("Amount", "total")]),
    # an own field the carried header already maps elsewhere replaces it
    ([_col("Qty", "quantity", 300, 340), _col("Amount", "total", 1000, 1060)],
     [("D", "description"), ("Qty", "quantity"), ("X", None), ("Amount", "total")]),
])
def test_complete_header(own, want):
    carried = [_col("D", "description", 100, 200), _col("Q", "quantity", 600, 640),
               _col("X", None, 860, 960), _col("Y", None, 1050, 1150)]
    out = g.complete_header(carried, own)
    assert [(c.label, c.field) for c in out] == want
    assert [c.left for c in out] == sorted(c.left for c in out)


# The rate column read on this page spans 850..900 (or 760..810); the carried
# column X sits left or right of it, touching it or one unit clear of it.
@pytest.mark.parametrize("x_span, own_left, expected", [
    ((700, 850), 850, ["D", "Rate", "Amount"]),         # X right edge on Rate left
    ((700, 850), 851, ["D", "X", "Rate", "Amount"]),    # one unit clear: X stays
    ((610, 760), 760, ["D", "Rate", "Amount"]),
    ((610, 760), 761, ["D", "X", "Rate", "Amount"]),
    ((900, 980), 850, ["D", "Rate", "Amount"]),         # X left edge on Rate right
    ((901, 980), 850, ["D", "Rate", "X", "Amount"]),
    ((810, 890), 760, ["D", "Rate", "Amount"]),
    ((811, 890), 760, ["D", "Rate", "X", "Amount"]),
])
def test_a_column_touching_an_own_column_edge_gives_way(x_span, own_left, expected):
    carried = [_col("D", "description", 100, 200), _col("X", None, *x_span)]
    out = g.complete_header(carried, [_col("Rate", "rate", own_left, own_left + 50),
                                      _col("Amount", "total", 1100, 1160)])
    assert [c.label for c in out] == expected


# Three reconciling rows: the page proves its own columns, and a header that
# reads them reconciles exactly as many lines (a tie).
TIE_A = [("Supply pump unit", "Nr", "2", "4,500", "9,000.00"),
         ("Lay cable duct", "m", "30", "15", "450.00"),
         ("Fit valve", "Nr", "4", "300", "1,200.00")]
TIE_B = [("Fix bracket", "Nr", "7", "120", "840.00"),
         ("Paint railing", "m2", "40", "25", "1,000.00"),
         ("Seal joint", "m", "12", "50", "600.00")]


@pytest.mark.parametrize("rows", [TIE_A, TIE_B])
def test_on_a_tie_the_page_proof_beats_a_completed_header(rows):
    stats = g.ColumnStats()
    g.extract_bill_rows(_two_pages(OWN_RATE_AMOUNT, rows), stats=stats)
    assert stats.pages == {"header": 1, "inferred": 1, "none": 0}


@pytest.mark.parametrize("rows", [TIE_A, TIE_B])
def test_on_a_tie_a_full_own_header_still_wins(rows):
    full = [("Description", 120, False), ("Unit", 520, False), ("Qty", Q_X, True),
            ("Rate", R_X, True), ("Amount", T_X, True)]
    stats = g.ColumnStats()
    out = g.extract_bill_rows(_two_pages(full, rows), stats=stats)
    assert stats.pages == {"header": 2, "inferred": 0, "none": 0}
    assert [r.fields["total"] for r in out if r.fields.get("total")] == [t for *_x, t in rows]
