"""Row kind by arithmetic: a bill total is the sum of its item rows.

Every input is built here. Each decision is shown on two different inputs,
and every rule has a case where it must not fire.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg


def _r(page, desc, amount="", qty="", rate="", unit="", ref="", path=("Bill A",)):
    return agg.SheetRow(path=tuple(path), ref=ref, description=desc, unit=unit,
                        quantity=qty, rate=rate, amount=amount,
                        pages=[page] if page is not None else [])


def _kinds(rows):
    return agg.row_kinds(rows)


# --- collection: an amount-only line equal to the lines above it on its page --

@pytest.mark.parametrize("a, b, total", [
    ("100.00", "250.50", "350.50"),
    ("1,200", "3,400", "4,600"),
])
def test_amount_only_line_equal_to_the_page_lines_is_a_collection(a, b, total):
    rows = [_r(5, "x", a, "1", a), _r(5, "y", b, "2", "1"), _r(5, "page total", total)]
    assert _kinds(rows) == ["item", "item", "collection"]


@pytest.mark.parametrize("total", ["350.48", "351.00"])
def test_amount_only_line_that_does_not_add_up_stays_an_item(total):
    rows = [_r(5, "x", "100.00", "1", "100"), _r(5, "y", "250.50"), _r(5, "lump", total)]
    assert _kinds(rows)[-1] == "item"


@pytest.mark.parametrize("qty, rate", [("1", ""), ("", "350.50"), ("2", "175.25")])
def test_a_line_with_a_quantity_or_rate_is_never_a_collection(qty, rate):
    rows = [_r(5, "x", "100.00"), _r(5, "y", "250.50"), _r(5, "z", "350.50", qty, rate)]
    assert _kinds(rows)[-1] == "item"


@pytest.mark.parametrize("amount", ["500.00", "7", "0", "1", "0.01"])
def test_first_amount_line_of_a_page_is_an_item(amount):
    """No line above it: nothing to collect, so it is an item, never a total of nothing."""
    assert _kinds([_r(3, "only", amount)]) == ["item"]


@pytest.mark.parametrize("cell, total, kind", [
    ("100.00", "100.01", "collection"),   # one unit of the cell's last place: within
    ("100.00", "100.02", "item"),         # two units: outside
    ("100", "101", "collection"),
    ("100", "102", "item"),
])
def test_tolerance_is_one_unit_of_the_amount_cells_own_last_place(cell, total, kind):
    rows = [_r(1, "x", cell, "1", cell), _r(1, "t", total)]
    assert _kinds(rows)[-1] == kind


def test_lines_on_another_page_are_not_collected():
    rows = [_r(1, "x", "40", "1", "40"), _r(2, "y", "60", "1", "60"), _r(2, "t", "100")]
    assert _kinds(rows) == ["item", "item", "item"]


def test_a_collection_resets_the_running_sum():
    rows = [_r(1, "a", "10", "1", "10"), _r(1, "s1", "10"),
            _r(1, "b", "5", "1", "5"), _r(1, "s2", "5"), _r(1, "lump", "15")]
    assert _kinds(rows) == ["item", "collection", "item", "collection", "item"]


# --- repeat: a figure brought forward or listed on a collection page ---------

@pytest.mark.parametrize("p1, p2", [("300.00", "700.00"), ("12", "88")])
def test_brought_forward_and_collection_page(p1, p2):
    tot = f"{float(p1.replace(',', '')) + float(p2.replace(',', '')):.2f}"
    rows = [
        _r(1, "a", p1, "1", p1), _r(1, "carried", p1),
        _r(2, "brought", p1), _r(2, "b", p2, "1", p2), _r(2, "carried", tot),
        _r(3, "page 1", p1), _r(3, "page 2 net", p2, "1", p2), _r(3, "summary", tot),
    ]
    kinds = _kinds(rows)
    assert kinds[:5] == ["item", "collection", "repeat", "item", "collection"]
    assert kinds[5] == "repeat"
    assert kinds[7] == "collection"


def test_a_lump_equal_to_a_later_page_total_is_not_a_repeat():
    """A repeat restates a collection read on an EARLIER page only."""
    rows = [_r(1, "lump", "90"), _r(2, "a", "90", "1", "90"), _r(2, "t", "90")]
    assert _kinds(rows) == ["item", "item", "collection"]


def test_rows_with_no_page_stay_items():
    rows = [_r(None, "a", "10", "1", "10"), _r(None, "t", "10")]
    assert _kinds(rows) == ["item", "item"]


def test_empty_sheet_has_no_kinds():
    assert _kinds([]) == []


# --- the tool sums items only and names what it left out ---------------------

@pytest.mark.parametrize("a, b", [("100.00", "250.00"), ("7", "13")])
def test_aggregate_amount_sums_items_only(a, b):
    total = str(float(a) + float(b))
    rows = [_r(4, "pipe x", a, "1", a, unit="m"), _r(4, "pipe y", b, "1", b, unit="m"),
            _r(4, "pipe page total", total)]
    out = agg.aggregate(rows, terms=["pipe"], measure=agg.MEASURE_AMOUNT)
    assert out["sum"] == pytest.approx(float(a) + float(b))
    assert out["count"] == 2
    assert out["rows_scanned"] == 3
    assert [(n["kind"], n["amount"]) for n in out["not_items"]] == [("collection", float(total))]


def test_aggregate_without_totals_reports_no_left_out_rows():
    rows = [_r(4, "pipe x", "5", "1", "5"), _r(4, "pipe y", "6", "1", "6")]
    out = agg.aggregate(rows, terms=["pipe"], measure=agg.MEASURE_AMOUNT)
    assert out["sum"] == pytest.approx(11.0)
    assert out["not_items"] == []
    assert out["rows_scanned"] == 2
