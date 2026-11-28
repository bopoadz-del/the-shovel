"""One bill aggregate call states both measures over the rows it selected.

A row of a bill carries a quantity and an amount; a question that wants both
used to cost two calls, one per measure. The result now carries both figures
of every row and totals the measure not asked over the same rows
(``other_measure``), and the tool's statement states that total too, so a
figure the model prints from it is backed by the call. Every row and bill
here is invented in the test; each decision is driven by at least two inputs
with different outcomes, with a negative where a rule must not fire.
"""

from __future__ import annotations

import pytest

from app.agents.base.tools import bill_rows_statement
from app.core import bill_aggregate as agg


def _r(ref, desc, unit, qty, amount, page=1, path=("Roads", "Kerbs")):
    return agg.SheetRow(path=tuple(path), ref=ref, description=desc, unit=unit,
                        quantity=qty, rate="", amount=amount, pages=[page])


KERBS = [
    _r("K1", "Precast concrete kerb straight", "m", "120", "3600"),
    _r("K2", "Precast concrete kerb curved", "m", "30", "1050"),
    _r("K3", "Precast concrete kerb dropped", "m", "12", ""),
]
CHAMBERS = [
    _r("C1", "Inspection chamber depth 1.5m", "nr", "4", "8000", path=("Drainage", "Chambers")),
    _r("C2", "Inspection chamber cover slab", "m2", "6.5", "1300", path=("Drainage", "Chambers")),
    _r("C3", "Inspection chamber step irons", "nr", "", "250", path=("Drainage", "Chambers")),
]


# -- measure_totals: the one place a bill result's sums are made -------------

@pytest.mark.parametrize("rows,measure,want_sum,want_unit,want_by_unit,without", [
    (KERBS, "quantity", 162.0, "m", {"m": 162.0}, []),
    (KERBS, "amount", 4650.0, None, None, ["K3"]),
    (CHAMBERS, "quantity", None, None, {"nr": 4.0, "m2": 6.5}, ["C3"]),
    (CHAMBERS, "amount", 9550.0, None, None, []),
])
def test_measure_totals(rows, measure, want_sum, want_unit, want_by_unit, without):
    out = agg.measure_totals(rows, measure)
    assert out["sum"] == want_sum
    assert out["unit"] == want_unit
    assert out["sum_by_unit"] == want_by_unit
    assert [w["ref"] for w in out["without_figure"]] == without


@pytest.mark.parametrize("measure", ["rate", "", "Quantity "])
def test_measure_totals_refuses_a_measure_it_does_not_know(measure):
    with pytest.raises(ValueError):
        agg.measure_totals(KERBS, measure)


# -- other_measure_totals: no list repeated, no 0 read off nothing ------------

@pytest.mark.parametrize("rows,measure,want_sum,want_unit,missing", [
    # some rows carry the figure: the total is theirs, the rest are counted
    (KERBS, "amount", 4650.0, None, 1),
    (CHAMBERS, "quantity", None, None, 1),
    (CHAMBERS[:2], "amount", 9300.0, None, 0),
    (KERBS[:2], "quantity", 150.0, "m", 0),
    # no row carries the figure: there is no total to state
    ([KERBS[2]], "amount", None, None, 1),
    ([CHAMBERS[2]], "quantity", None, None, 1),
    # nothing selected: an empty total, the same as the asked measure's
    ([], "amount", 0.0, None, 0),
    ([], "quantity", 0.0, "", 0),
])
def test_other_measure_totals(rows, measure, want_sum, want_unit, missing):
    out = agg.other_measure_totals(rows, measure)
    assert out["measure"] == measure
    assert out["sum"] == want_sum
    assert out["unit"] == want_unit
    assert out["rows_without_figure"] == missing
    assert "without_figure" not in out


# -- aggregate: both figures per row, the other measure over the same rows ----

@pytest.mark.parametrize("rows,terms,measure,other,other_sum,other_by_unit", [
    (KERBS, "kerb", "quantity", "amount", 4650.0, None),
    (KERBS, "kerb", "amount", "quantity", 162.0, {"m": 162.0}),
    (CHAMBERS, "chamber", "amount", "quantity", None, {"nr": 4.0, "m2": 6.5}),
    (CHAMBERS, "chamber", "quantity", "amount", 9550.0, None),
])
def test_aggregate_totals_the_other_measure_over_the_same_rows(
        rows, terms, measure, other, other_sum, other_by_unit):
    out = agg.aggregate(rows, terms=terms, measure=measure)
    assert out["measure"] == measure
    om = out["other_measure"]
    assert om["measure"] == other
    assert om["sum"] == other_sum
    assert om["sum_by_unit"] == other_by_unit
    # every listed row carries both of its own figures
    for row, src in zip(out["rows"], rows):
        assert row["quantity"] == agg.SheetRow.figure(src, "quantity")
        assert row["amount"] == agg.SheetRow.figure(src, "amount")
    # the asked measure is unchanged by the second total
    assert out["sum"] == agg.measure_totals(rows, measure)["sum"]
    # the other total is ahead of the rows, so a cut that keeps the head keeps it
    keys = list(out)
    assert keys.index("other_measure") < keys.index("rows")


@pytest.mark.parametrize("terms,want_refs,other_sum", [
    ("curved", ["K2"], 1050.0),
    ("straight", ["K1"], 3600.0),
])
def test_aggregate_other_measure_follows_the_selection(terms, want_refs, other_sum):
    """The other total is over the selected rows only, never the whole sheet."""
    out = agg.aggregate(KERBS + CHAMBERS, terms=terms, measure="quantity")
    assert [r["ref"] for r in out["rows"]] == want_refs
    assert out["other_measure"]["sum"] == other_sum


# -- bill_rows_statement: the other total is stated where a row carries it ----

def _bill(rows, terms, measure):
    return {"status": "success", "bills": [
        {"status": "success", "filename": "bill.xlsx",
         **agg.aggregate(rows, terms=terms, measure=measure)}]}


@pytest.mark.parametrize("rows,terms,measure,other,want", [
    (KERBS, "kerb", "quantity", "amount", [(4650.0, "")]),
    (KERBS, "kerb", "amount", "quantity", [(162.0, "m")]),
    (CHAMBERS, "chamber", "amount", "quantity", [(4.0, "nr"), (6.5, "m2")]),
])
def test_bill_rows_statement_states_the_other_measure(rows, terms, measure, other, want):
    sts = bill_rows_statement(_bill(rows, terms, measure))
    asked = [s for s in sts if s.basis == "sum" and s.what == measure]
    others = [s for s in sts if s.basis == "sum" and s.what == other]
    assert asked, sts
    assert [(s.figure, s.unit) for s in others] == want
    for s in others:
        # each stated total is the sum of the row figures behind it
        assert sum(s.row_figures) == pytest.approx(s.figure)
        assert len(s.rows) == len(s.row_figures) > 0


@pytest.mark.parametrize("rows,terms,measure", [
    ([KERBS[2]], "dropped", "quantity"),     # no amount on the one row
    ([CHAMBERS[2]], "irons", "amount"),      # no quantity on the one row
])
def test_bill_rows_statement_states_no_other_total_off_nothing(rows, terms, measure):
    sts = bill_rows_statement(_bill(rows, terms, measure))
    other = next(m for m in agg.MEASURES if m != measure)
    assert [s for s in sts if s.what == other] == []
    assert [s for s in sts if s.basis == "count"], sts


@pytest.mark.parametrize("bad", [None, "amount", {"measure": "quantity", "sum": 5.0}, {"sum": 5.0}])
def test_bill_rows_statement_ignores_an_other_measure_it_cannot_read(bad):
    """A missing, malformed or same-measure ``other_measure`` adds nothing:
    the statement is what it was for the asked measure alone."""
    body = _bill(KERBS[:2], "kerb", "quantity")
    body["bills"][0]["other_measure"] = bad
    sts = bill_rows_statement(body)
    assert [(s.basis, s.what, s.figure) for s in sts] == [
        ("count", "quantity", 2.0), ("sum", "quantity", 150.0)]
