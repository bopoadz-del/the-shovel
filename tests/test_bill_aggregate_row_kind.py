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


@pytest.mark.parametrize("a, b", [("40", "60"), ("1,250.00", "750.00")])
def test_lines_on_another_page_are_not_collected(a, b):
    tot = f"{float(a.replace(',', '')) + float(b.replace(',', '')):.2f}"
    rows = [_r(1, "x", a, "1", a), _r(2, "y", b, "1", b), _r(2, "t", tot)]
    assert _kinds(rows) == ["item", "item", "item"]


@pytest.mark.parametrize("a, b, both", [("10", "5", "15"), ("200.00", "35.50", "235.50")])
def test_a_collection_resets_the_running_sum(a, b, both):
    rows = [_r(1, "a", a, "1", a), _r(1, "s1", a),
            _r(1, "b", b, "1", b), _r(1, "s2", b), _r(1, "lump", both)]
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


@pytest.mark.parametrize("fig", ["90", "4,321.75"])
def test_a_lump_equal_to_a_later_page_total_is_not_a_repeat(fig):
    """A repeat restates a collection read on an EARLIER page only."""
    rows = [_r(1, "lump", fig), _r(2, "a", fig, "1", fig), _r(2, "t", fig)]
    assert _kinds(rows) == ["item", "item", "collection"]


@pytest.mark.parametrize("fig", ["10", "880.25"])
def test_rows_with_no_page_stay_items(fig):
    rows = [_r(None, "a", fig, "1", fig), _r(None, "t", fig)]
    assert _kinds(rows) == ["item", "item"]


@pytest.mark.parametrize("rows", [[], ()])
def test_empty_sheet_has_no_kinds(rows):
    assert _kinds(rows) == []


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


@pytest.mark.parametrize("a, b", [("5", "6"), ("120.00", "30.25")])
def test_aggregate_without_totals_reports_no_left_out_rows(a, b):
    rows = [_r(4, "pipe x", a, "1", a), _r(4, "pipe y", b, "1", b)]
    out = agg.aggregate(rows, terms=["pipe"], measure=agg.MEASURE_AMOUNT)
    assert out["sum"] == pytest.approx(float(a) + float(b))
    assert out["not_items"] == []
    assert out["rows_scanned"] == 2


# --- bill check: the bill's own printed total, found by arithmetic ----------

def _bill(n, p1, p2, total, *, first=10, extra=()):
    """Bill ``n``: two item pages, each closed by its page total, then a
    collection page that repeats both page totals and prints the bill total."""
    path = (f"Bill No. {n} - Works",)
    return [
        _r(first, "a", p1, "1", p1, path=path),
        _r(first, "page total", p1, path=path),
        _r(first + 1, "b", p2, "2", "1", path=path),
        *extra,
        _r(first + 1, "page total", p2, path=path),
        _r(first + 2, "page one", p1, path=path),
        _r(first + 2, "page two", p2, path=path),
        _r(first + 2, "bill total", total, path=path),
    ]


@pytest.mark.parametrize("n, p1, p2, total", [
    ("7", "300.00", "700.00", "1,000.00"),
    ("12", "1,250", "4,321", "5,571"),
])
def test_bill_whose_items_add_to_its_printed_total_reconciles(n, p1, p2, total):
    rows = _bill(n, p1, p2, total)
    [check] = agg.bill_checks(rows, {n})
    assert check["state"] == "RECONCILED"
    assert check["printed_total"] == agg.g.parse_number(total)
    assert check["printed_page"] == 12
    assert check["rows_summed"] == 2
    assert check["items_sum"] == agg.g.parse_number(p1) + agg.g.parse_number(p2)


@pytest.mark.parametrize("n, p1, p2, total", [
    ("7", "300.00", "700.00", "1,000.00"),
    ("12", "1,250", "4,321", "5,571"),
])
def test_bill_whose_items_do_not_add_up_fails(n, p1, p2, total):
    """An item the read changed: its page total no longer collects it, and the
    items no longer reach the printed bill total."""
    rows = _bill(n, p1, p2, total)
    rows[0] = _r(10, "a", "1.00", "1", "1", path=rows[0].path)
    [check] = agg.bill_checks(rows, {n})
    assert check["state"] == "FAILED"
    assert check["printed_total"] == agg.g.parse_number(total)


@pytest.mark.parametrize("b2, state", [
    ("400.00", "RECONCILED"),   # items short of the printed total by one unit
    ("399.99", "FAILED"),       # each page within its unit, the bill short by two
])
@pytest.mark.parametrize("unit", ["0.01", "1"])
def test_bill_tolerance_is_one_unit_of_the_printed_totals_last_place(b2, state, unit):
    """Each page total collects its items within one unit of its own last
    place; the drift adds up, and the bill total holds it to one unit too."""
    def fig(cents):   # a figure printed to the last place ``unit`` states
        return f"{cents / 100:,.2f}" if unit == "0.01" else f"{cents // 100:,}"
    k = 1 if unit == "0.01" else 100           # one unit, in cents
    b2c = round(float(b2) * 100) if unit == "0.01" else (40000 if b2 == "400.00" else 40000 - 100)
    path = ("Bill No. 4 - Works",)
    rows = [
        _r(1, "a1", fig(10000), "1", "1", path=path), _r(1, "a2", fig(20000 - k), "1", "1", path=path),
        _r(1, "pt", fig(30000), path=path),
        _r(2, "b1", fig(30000), "1", "1", path=path), _r(2, "b2", fig(b2c), "1", "1", path=path),
        _r(2, "pt", fig(70000), path=path),
        _r(3, "r1", fig(30000), path=path), _r(3, "r2", fig(70000), path=path),
        _r(3, "bill total", fig(100000), path=path),
    ]
    [check] = agg.bill_checks(rows, {"4"})
    assert check["printed_total"] == 1000.0
    assert check["rows_summed"] == 4
    assert check["state"] == state


@pytest.mark.parametrize("p1, p2, p3", [("300.00", "700.00", "50.00"), ("12", "88", "400")])
def test_the_bills_last_total_by_page_is_its_printed_total(p1, p2, p3):
    """A part summary inside the bill is a total of totals too; the bill's own
    total is the last one, which covers every page."""
    f = agg.g.parse_number
    path = ("Bill No. 6 - Works",)
    part = f"{f(p1) + f(p2):.2f}"
    whole = f"{f(p1) + f(p2) + f(p3):.2f}"
    rows = [
        _r(10, "a", p1, "1", p1, path=path), _r(10, "pt", p1, path=path),
        _r(11, "b", p2, "1", p2, path=path), _r(11, "pt", p2, path=path),
        _r(12, "r1", p1, path=path), _r(12, "r2", p2, path=path), _r(12, "part", part, path=path),
        _r(13, "c", p3, "1", p3, path=path), _r(13, "pt", p3, path=path),
        _r(14, "part", part, path=path), _r(14, "r3", p3, path=path), _r(14, "bill", whole, path=path),
    ]
    [check] = agg.bill_checks(rows, {"6"})
    assert (check["printed_page"], check["printed_total"]) == (14, f(whole))
    assert check["state"] == "RECONCILED"


@pytest.mark.parametrize("p1, p2", [("300.00", "700.00"), ("12", "88")])
def test_a_page_total_over_items_is_not_the_bill_total(p1, p2):
    """A collection of priced item lines is a page total; with no collection page the
    bill prints no total the check can use, so it is UNCHECKED, never RECONCILED."""
    path = ("Bill No. 3 - Works",)
    rows = [
        _r(1, "a", p1, "1", p1, path=path), _r(1, "b", p2, "1", p2, path=path),
        _r(1, "page total", str(agg.g.parse_number(p1) + agg.g.parse_number(p2)), path=path),
    ]
    [check] = agg.bill_checks(rows, {"3"})
    assert check["state"] == "UNCHECKED"
    assert check["printed_total"] is None
    assert check["rows_summed"] == 2


@pytest.mark.parametrize("p1, p2", [("300.00", "700.00"), ("12", "88")])
def test_a_collection_mixing_a_repeat_and_an_item_is_not_the_bill_total(p1, p2):
    path = ("Bill No. 5 - Works",)
    rows = [
        _r(1, "a", p1, "1", p1, path=path), _r(1, "pt", p1, path=path),
        _r(2, "bf", p1, path=path), _r(2, "b", p2, "1", p2, path=path),
        _r(2, "pt", str(agg.g.parse_number(p1) + agg.g.parse_number(p2)), path=path),
    ]
    [check] = agg.bill_checks(rows, {"5"})
    assert check["state"] == "UNCHECKED"


@pytest.mark.parametrize("n, other", [("7", "8"), ("12", "2")])
def test_rows_of_another_bill_are_not_in_the_check(n, other):
    rows = _bill(n, "300.00", "700.00", "1,000.00") + _bill(other, "50.00", "25.00", "75.00", first=20)
    checks = {c["bill"]: c for c in agg.bill_checks(rows, {n, other})}
    assert checks[n]["items_sum"] == 1000.0 and checks[n]["state"] == "RECONCILED"
    assert checks[other]["items_sum"] == 75.0 and checks[other]["printed_page"] == 22
    [only] = agg.bill_checks(rows, {other})
    assert only["bill"] == other


@pytest.mark.parametrize("desc", ["provisional", "to be priced"])
def test_item_rows_without_an_amount_are_counted_not_summed(desc):
    path = ("Bill No. 9 - Works",)
    extra = (_r(11, desc, "", "4", "", path=path), _r(11, desc + " too", "", "", "", path=path))
    rows = _bill("9", "300.00", "700.00", "1,000.00", extra=extra)
    [check] = agg.bill_checks(rows, {"9"})
    assert check["rows_without_amount"] == 2
    assert check["rows_summed"] == 2
    assert check["state"] == "RECONCILED"


@pytest.mark.parametrize("bills", [set(), frozenset()])
def test_no_bill_asked_gives_no_checks(bills):
    assert agg.bill_checks(_bill("7", "1", "2", "3"), bills) == []


@pytest.mark.parametrize("n", ["7", "1,313"])
def test_bill_with_no_rows_is_unchecked_with_nothing_summed(n):
    [check] = agg.bill_checks(_bill("2", "1", "2", "3"), {n})
    assert check["bill"] == n.replace(",", "")
    assert (check["rows_summed"], check["items_sum"], check["state"]) == (0, 0, "UNCHECKED")


@pytest.mark.parametrize("n, p1, p2, total", [
    ("7", "300.00", "700.00", "1,000.00"),
    ("12", "1,250", "4,321", "5,571"),
])
def test_aggregate_returns_the_asked_bills_check_with_the_sum(n, p1, p2, total):
    rows = _bill(n, p1, p2, total) + _bill("99", "5", "6", "11", first=40)
    res = agg.aggregate(rows, terms="", measure="amount", ask=f"What is the total amount of Bill No. {n}?")
    assert res["sum"] == agg.g.parse_number(total)
    assert [c["bill"] for c in res["bill_checks"]] == [n]
    assert res["bill_checks"][0]["state"] == "RECONCILED"


@pytest.mark.parametrize("terms", ["a", "b"])
def test_aggregate_without_a_bill_in_the_ask_has_no_checks(terms):
    res = agg.aggregate(_bill("7", "300.00", "700.00", "1,000.00"), terms=terms, measure="amount",
                        ask="how much are the works")
    assert res["bill_checks"] == []


@pytest.mark.parametrize("p1, p2", [("39,150,000.00", "700,000.00"), ("120.00", "35.50")])
def test_a_page_total_is_not_joined_to_the_next_pages_figureless_line(p1, p2):
    """A page total at the foot of a page and a line with no figure at the head
    of the next stay two rows, so the total is collected on its own page and
    the collection page's repeat of it is not summed again."""
    f = agg.g.parse_number
    path = ("Bill No. 19 - Works",)
    total = f"{f(p1) + f(p2):,.2f}"
    rows = [
        _r(1, "a", p1, "1", p1, path=path), _r(1, "carried", p1, path=path),
        _r(2, "b", p2, "1", p2, path=path), _r(2, "carried", p2, path=path),
        _r(3, "carried", "", path=path),
        _r(4, "from page 1", p1, path=path), _r(4, "from page 2", p2, path=path),
        _r(4, "bill total", total, path=path),
    ]
    items = agg.join_page_continuations(rows)
    assert [r.pages for r in items if r.description == "carried" and r.amount] == [[1], [2]]
    [check] = agg.bill_checks(items, {"19"})
    assert (check["rows_summed"], check["state"]) == (2, "RECONCILED")
    assert check["items_sum"] == f(total)


@pytest.mark.parametrize("head", [("", "4", "25.00", "100.00"), ("", "2", "7", "14")])
def test_a_description_split_by_the_page_break_still_joins(head):
    """The guard does not stop a real continuation: a description with no
    figure at the foot of a page takes the next page's figures."""
    ref, qty, rate, amount = head
    rows = [_r(1, "Supply and fix", ref="A"), _r(2, "pipes complete", amount, qty, rate, ref=ref)]
    [one] = agg.join_page_continuations(rows)
    assert (one.amount, one.pages) == (amount, [1, 2])


# --- a bill the document prints no item prices for ---------------------------

def _unpriced(n, qtys, *, total=None):
    """Bill ``n``: item rows with a quantity (or none) and blank money cells."""
    path = (f"Bill No. {n} - Works",)
    rows = [_r(30 + k, f"item {k}", "", q, "", path=path) for k, q in enumerate(qtys)]
    if total is not None:   # a figure printed for the bill with no item price under it
        rows.append(_r(40, "bill total", total, path=path))
    return rows


@pytest.mark.parametrize("n, qtys", [("1", ["1", "1", ""]), ("14", ["", "250"])])
def test_bill_with_no_rate_or_amount_on_any_item_is_not_priced(n, qtys):
    [check] = agg.bill_checks(_unpriced(n, qtys), {n})
    assert check["state"] == "NOT_PRICED"
    assert (check["rows_summed"], check["rows_without_amount"]) == (0, len(qtys))
    assert check["items_sum"] == 0


@pytest.mark.parametrize("total", ["5,000", "123.45"])
def test_an_amount_only_line_prices_the_bill(total):
    """A line that prints only an amount, with nothing above it to collect, is
    an item (a lump sum): the bill prints a price, so it is never NOT_PRICED."""
    [check] = agg.bill_checks(_unpriced("2", ["1", "1"], total=total), {"2"})
    assert check["state"] == "UNCHECKED"
    assert check["items_sum"] == agg.g.parse_number(total)


@pytest.mark.parametrize("cell", [("rate", "10.00"), ("amount", "75")])
def test_one_priced_item_makes_the_bill_priced(cell):
    """A single item that prints a rate or an amount: the document prices the
    bill, so the blanks are counted as rows without an amount, never NOT_PRICED."""
    rows = _unpriced("3", ["1", "2", "3"])
    rows[1] = agg.SheetRow(**{**rows[1].__dict__, cell[0]: cell[1]})
    [check] = agg.bill_checks(rows, {"3"})
    assert check["state"] == "UNCHECKED"


@pytest.mark.parametrize("n, other", [("1", "7"), ("20", "4")])
def test_not_priced_is_per_bill(n, other):
    rows = _unpriced(n, ["1", "1"]) + _bill(other, "300.00", "700.00", "1,000.00", first=50)
    checks = {c["bill"]: c for c in agg.bill_checks(rows, {n, other})}
    assert checks[n]["state"] == "NOT_PRICED"
    assert checks[other]["state"] == "RECONCILED"
