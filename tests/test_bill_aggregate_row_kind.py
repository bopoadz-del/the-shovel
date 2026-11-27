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


def _spans(rows):
    """The page range of each bill the fixture rows' paths name: the ranges
    a derived copy stores from the document's own summary pages."""
    out = {}
    for row in rows:
        for bill in agg.row_bills(row):
            if row.pages:
                lo, hi = out.get(bill, (row.pages[0], row.pages[0]))
                out[bill] = (min(lo, row.pages[0]), max(hi, row.pages[0]))
    return out


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
    rows = [_r(1, "x", cell, "1", cell), _r(1, "y", "0", "1", "0"), _r(1, "t", total)]
    assert _kinds(rows)[-1] == kind


@pytest.mark.parametrize("a, b", [("40", "60"), ("1,250.00", "750.00")])
def test_lines_on_another_page_are_not_collected(a, b):
    tot = f"{float(a.replace(',', '')) + float(b.replace(',', '')):.2f}"
    rows = [_r(1, "x", a, "1", a), _r(2, "y", b, "1", b), _r(2, "t", tot)]
    assert _kinds(rows) == ["item", "item", "item"]


@pytest.mark.parametrize("a, b, both", [("10", "5", "15"), ("200.00", "35.50", "235.50")])
def test_a_collection_resets_the_running_sum(a, b, both):
    rows = [_r(1, "a", a, "1", a), _r(1, "s1", a),
            _r(1, "b", b, "1", b), _r(1, "s2", b), _r(1, "lump", both),
            _r(2, "brought 1", a), _r(2, "brought 2", b)]
    assert _kinds(rows) == ["item", "collection", "item", "collection", "item",
                            "repeat", "repeat"]


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
    rows = [_r(1, "lump", fig), _r(2, "a", fig, "1", fig), _r(2, "b", "0", "1", "0"),
            _r(2, "t", fig)]
    assert _kinds(rows) == ["item", "item", "item", "collection"]


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
    [check] = agg.bill_checks(rows, {n}, _spans(rows))
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
    [check] = agg.bill_checks(rows, {n}, _spans(rows))
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
    [check] = agg.bill_checks(rows, {"4"}, _spans(rows))
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
    [check] = agg.bill_checks(rows, {"6"}, _spans(rows))
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
    [check] = agg.bill_checks(rows, {"3"}, _spans(rows))
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
    [check] = agg.bill_checks(rows, {"5"}, _spans(rows))
    assert check["state"] == "UNCHECKED"


@pytest.mark.parametrize("n, other", [("7", "8"), ("12", "2")])
def test_rows_of_another_bill_are_not_in_the_check(n, other):
    rows = _bill(n, "300.00", "700.00", "1,000.00") + _bill(other, "50.00", "25.00", "75.00", first=20)
    checks = {c["bill"]: c for c in agg.bill_checks(rows, {n, other}, _spans(rows))}
    assert checks[n]["items_sum"] == 1000.0 and checks[n]["state"] == "RECONCILED"
    assert checks[other]["items_sum"] == 75.0 and checks[other]["printed_page"] == 22
    [only] = agg.bill_checks(rows, {other}, _spans(rows))
    assert only["bill"] == other


@pytest.mark.parametrize("desc", ["provisional", "to be priced"])
def test_item_rows_without_an_amount_are_counted_not_summed(desc):
    path = ("Bill No. 9 - Works",)
    extra = (_r(11, desc, "", "4", "", path=path), _r(11, desc + " too", "", "", "", path=path))
    rows = _bill("9", "300.00", "700.00", "1,000.00", extra=extra)
    [check] = agg.bill_checks(rows, {"9"}, _spans(rows))
    assert check["rows_without_amount"] == 2
    assert check["rows_summed"] == 2
    assert check["state"] == "RECONCILED"


@pytest.mark.parametrize("bills", [set(), frozenset()])
def test_no_bill_asked_gives_no_checks(bills):
    assert agg.bill_checks(_bill("7", "1", "2", "3"), bills, _spans(_bill("7", "1", "2", "3"))) == []


@pytest.mark.parametrize("n", ["7", "1,313"])
def test_bill_with_no_rows_is_unchecked_with_nothing_summed(n):
    [check] = agg.bill_checks(_bill("2", "1", "2", "3"), {n}, {n.replace(",", ""): (50, 60)})
    assert check["bill"] == n.replace(",", "")
    assert (check["rows_summed"], check["items_sum"], check["state"]) == (0, 0, "UNCHECKED")


@pytest.mark.parametrize("n, p1, p2, total", [
    ("7", "300.00", "700.00", "1,000.00"),
    ("12", "1,250", "4,321", "5,571"),
])
def test_aggregate_returns_the_asked_bills_check_with_the_sum(n, p1, p2, total):
    rows = _bill(n, p1, p2, total) + _bill("99", "5", "6", "11", first=40)
    ask = f"What is the total amount of Bill No. {n}?"
    res = agg.aggregate(rows, terms="", measure="amount", ask=ask, ranges=_spans(rows))
    assert res["sum"] == agg.g.parse_number(total)
    assert [c["bill"] for c in res["bill_checks"]] == [n]
    assert res["bill_checks"][0]["state"] == "RECONCILED"
    # no stored page ranges: the check says membership was not read, no verdict
    [bare] = agg.aggregate(rows, terms="", measure="amount", ask=ask)["bill_checks"]
    assert (bare["membership"], bare["state"], bare["items_sum"]) == ("not_read", "UNCHECKED", None)


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
    [check] = agg.bill_checks(items, {"19"}, _spans(items))
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
    [check] = agg.bill_checks(_unpriced(n, qtys), {n}, _spans(_unpriced(n, qtys)))
    assert check["state"] == "NOT_PRICED"
    assert (check["rows_summed"], check["rows_without_amount"]) == (0, len(qtys))
    assert check["items_sum"] == 0


@pytest.mark.parametrize("total", ["5,000", "123.45"])
def test_an_amount_only_line_prices_the_bill(total):
    """A line that prints only an amount, with nothing above it to collect, is
    an item (a lump sum): the bill prints a price, so it is never NOT_PRICED."""
    [check] = agg.bill_checks(_unpriced("2", ["1", "1"], total=total), {"2"}, _spans(_unpriced("2", ["1", "1"], total=total)))
    assert check["state"] == "UNCHECKED"
    assert check["items_sum"] == agg.g.parse_number(total)


@pytest.mark.parametrize("cell", [("rate", "10.00"), ("amount", "75")])
def test_one_priced_item_makes_the_bill_priced(cell):
    """A single item that prints a rate or an amount: the document prices the
    bill, so the blanks are counted as rows without an amount, never NOT_PRICED."""
    rows = _unpriced("3", ["1", "2", "3"])
    rows[1] = agg.SheetRow(**{**rows[1].__dict__, cell[0]: cell[1]})
    [check] = agg.bill_checks(rows, {"3"}, _spans(rows))
    assert check["state"] == "UNCHECKED"


@pytest.mark.parametrize("n, other", [("1", "7"), ("20", "4")])
def test_not_priced_is_per_bill(n, other):
    rows = _unpriced(n, ["1", "1"]) + _bill(other, "300.00", "700.00", "1,000.00", first=50)
    checks = {c["bill"]: c for c in agg.bill_checks(rows, {n, other}, _spans(rows))}
    assert checks[n]["state"] == "NOT_PRICED"
    assert checks[other]["state"] == "RECONCILED"


# --- one line equal to the line above it: a collection only when restated ---

def _num(text):
    return agg.g.parse_number(text)


@pytest.mark.parametrize("fig", ["50,000", "1,234.50"])
def test_two_equal_lump_sums_are_both_items(fig):
    """Nothing later restates the second figure: it cannot be told from a
    second lump sum, so it is summed, and the bill has no printed total."""
    path = ("Bill No. 5 - Works",)
    rows = [_r(10, "provisional sum one", fig, path=path),
            _r(10, "provisional sum two", fig, path=path)]
    assert _kinds(rows) == ["item", "item"]
    out = agg.aggregate(rows, terms=[], ask="What is the total amount of Bill No. 5?",
                        measure=agg.MEASURE_AMOUNT)
    assert out["sum"] == pytest.approx(2 * _num(fig))
    assert out["count"] == 2
    [check] = out["bill_checks"]
    assert check["state"] == "UNCHECKED" and check["printed_total"] is None


@pytest.mark.parametrize("q, rate, amount", [("10", "100", "1,000"), ("4", "2.50", "10.00")])
def test_priced_item_then_an_equal_lump_sum_are_both_items(q, rate, amount):
    rows = [_r(3, "priced", amount, q, rate), _r(3, "allow for", amount)]
    assert _kinds(rows) == ["item", "item"]


@pytest.mark.parametrize("q, rate, amount", [("10", "100", "1,000"), ("4", "2.50", "10.00")])
def test_one_line_cover_restated_on_a_later_page_is_a_collection(q, rate, amount):
    rows = [_r(3, "priced", amount, q, rate), _r(3, "carried", amount),
            _r(4, "brought forward", amount)]
    assert _kinds(rows) == ["item", "collection", "repeat"]


@pytest.mark.parametrize("pages, page_total", [(5, "20,000"), (3, "7,500.50")])
def test_equal_page_totals_on_the_collection_page_add_to_the_bill_total(pages, page_total):
    path = ("Bill No. 5 - Works",)
    each = _num(page_total)
    half = f"{each / 2:.2f}"
    rows = []
    for p in range(1, pages + 1):
        rows += [_r(p, "a", half, "1", half, path=path), _r(p, "b", half, "1", half, path=path),
                 _r(p, "page total", page_total, path=path)]
    rows += [_r(pages + 1, f"page {p}", page_total, path=path) for p in range(1, pages + 1)]
    rows.append(_r(pages + 1, "bill total", f"{each * pages:.2f}", path=path))
    kinds = _kinds(rows)
    assert kinds[-1] == "collection"
    assert kinds[-1 - pages:-1] == ["repeat"] * pages
    [check] = agg.bill_checks(rows, {"5"}, _spans(rows))
    assert check["state"] == "RECONCILED"
    assert check["items_sum"] == pytest.approx(each * pages)
    assert check["printed_total"] == pytest.approx(each * pages)
    assert check["rows_summed"] == 2 * pages


# --- a repeat restates a collection of its OWN bill -------------------------

@pytest.mark.parametrize("same_bill, want", [(False, "item"), (True, "repeat")])
def test_a_figure_equal_to_another_bills_page_total_is_not_a_repeat(same_bill, want):
    five, six = ("Bill No. 5 - Works",), ("Bill No. 6 - Planting",)
    later = five if same_bill else six
    rows = [_r(10, "a", "30,000", "1", "30000", path=five),
            _r(10, "b", "20,000", "1", "20000", path=five),
            _r(10, "page total", "50,000", path=five),
            _r(12, "allow for irrigation", "50,000", path=later),
            _r(12, "plant trees", "7,000", "7", "1000", path=later)]
    assert _kinds(rows)[3] == want


@pytest.mark.parametrize("lump, trees", [("50,000", "7,000"), ("50,000", "12,500")])
def test_bill_total_keeps_a_lump_equal_to_another_bills_page_total(lump, trees):
    five, six = ("Bill No. 5 - Works",), ("Bill No. 6 - Planting",)
    rows = [_r(10, "a", "30,000", "1", "30000", path=five),
            _r(10, "b", "20,000", "1", "20000", path=five),
            _r(10, "page total", "50,000", path=five),
            _r(12, "allow for irrigation", lump, path=six),
            _r(12, "plant trees", trees, "1", trees.replace(",", ""), path=six)]
    out = agg.aggregate(rows, terms=[], ask="What is the total amount of Bill No. 6?",
                        measure=agg.MEASURE_AMOUNT)
    assert out["sum"] == pytest.approx(_num(lump) + _num(trees))
    assert out["count"] == 2


# --- not_items lists what THIS selection left out, not the whole sheet ------

def _two_bills():
    rows = []
    for n, first in (("5", 10), ("6", 20)):
        path = (f"Bill No. {n} - Works",)
        rows += [_r(first, "a", "100", "1", "100", path=path),
                 _r(first, "b", "200", "1", "200", path=path),
                 _r(first, "page total", "300", path=path)]
    return rows


@pytest.mark.parametrize("bill, page", [("5", 10), ("6", 20)])
def test_not_items_of_a_bill_ask_are_that_bills_own(bill, page):
    out = agg.aggregate(_two_bills(), terms=[], ask=f"What is the total amount of Bill No. {bill}?",
                        measure=agg.MEASURE_AMOUNT)
    assert [(n["kind"], n["pages"]) for n in out["not_items"]] == [("collection", [page])]
    assert out["rows_scanned"] == 6
    assert out["sum"] == pytest.approx(300)


@pytest.mark.parametrize("terms, want", [(["page"], 2), (["nothing"], 0)])
def test_not_items_of_a_terms_ask_are_the_matching_lines(terms, want):
    out = agg.aggregate(_two_bills(), terms=terms, measure=agg.MEASURE_AMOUNT)
    assert len(out["not_items"]) == want


# --- tolerances and the page a joined row stands on -------------------------

@pytest.mark.parametrize("earlier, cell, want", [
    ("1,234.50", "1,235", "repeat"),    # the coarser cell's place decides
    ("1,234.50", "1,236", "item"),
])
def test_repeat_tolerance_is_the_coarser_of_the_two_cells(earlier, cell, want):
    rows = [_r(1, "a", "1,000.00", "1", "1000"), _r(1, "b", "234.50", "1", "234.5"),
            _r(1, "page total", earlier), _r(2, "brought", cell)]
    assert _kinds(rows)[3] == want


@pytest.mark.parametrize("a, b", [("40", "60"), ("125.50", "74.50")])
def test_a_joined_row_stands_on_its_last_page(a, b):
    tot = f"{_num(a) + _num(b):.2f}"
    joined = agg.SheetRow(path=("Bill A",), ref="", description="joined", unit="",
                          quantity="1", rate=a, amount=a, pages=[5, 6])
    rows = [joined, _r(6, "y", b, "1", b), _r(6, "page total", tot)]
    assert _kinds(rows) == ["item", "item", "collection"]


@pytest.mark.parametrize("text, want", [("(100.50)", 0.01), ("-1,000.5", 0.1), ("(7)", 1.0)])
def test_last_place_reads_through_brackets_and_signs(text, want):
    assert agg._last_place(text) == pytest.approx(want)


@pytest.mark.parametrize("later_bill, want", [
    ("Bill No. 6 - Planting", "item"),     # restated only by another bill
    ("Bill No. 5 - Works", "collection"),
])
def test_the_restatement_must_be_in_the_same_bill(later_bill, want):
    five = ("Bill No. 5 - Works",)
    rows = [_r(3, "priced", "1,000", "10", "100", path=five), _r(3, "line", "1,000", path=five),
            _r(4, "later line", "1,000", path=(later_bill,))]
    assert _kinds(rows)[1] == want


@pytest.mark.parametrize("n,carried", [("5", "4"), ("12", "11")])
def test_a_carried_heading_naming_the_bill_before_does_not_move_rows(n, carried):
    rows = _bill(n, "100", "50", "150")
    # the first item page's heading is carried down from the bill before
    rows[0] = _r(10, "a", "100", "1", "100", path=(f"Bill No. {carried} - Works",))
    [check] = agg.bill_checks(rows, {n}, {n: (10, 12)})
    assert (check["items_sum"], check["state"], check["membership"]) == (150.0, agg.STATE_RECONCILED, "page_range")
    [before] = agg.bill_checks(rows, {carried}, {carried: (1, 9), n: (10, 12)})
    assert (before["rows_summed"], before["state"]) == (0, agg.STATE_UNCHECKED)


@pytest.mark.parametrize("ranges", [None, {}, {"8": (1, 30)}])
def test_a_bill_with_no_page_range_states_no_sum_and_no_verdict(ranges):
    rows = _bill("3", "100", "50", "150")
    [check] = agg.bill_checks(rows, {"3"}, ranges)
    assert check["membership"] == agg.MEMBERSHIP_NOT_READ
    assert (check["items_sum"], check["rows_summed"], check["state"]) == (None, 0, agg.STATE_UNCHECKED)


@pytest.mark.parametrize("meta,expected", [
    ({"bill_states": {"bills": [{"label": "5", "pages": [101, 113]},
                                {"label": "D/6/23", "pages": [137, 115]}]}},
     {"5": (101, 113), "6": (115, 137)}),
    ({"bill_states": {"bills": [{"label": "Bill No. 1,313", "pages": [2, 4]},
                                {"label": "7", "pages": [1]},
                                {"label": "8", "pages": [True, 3]},
                                {"label": "", "pages": [5, 6]}]}},
     {"1313": (2, 4)}),
])
def test_bill_page_ranges_reads_the_stored_ranges(meta, expected):
    assert agg.bill_page_ranges(meta) == expected


@pytest.mark.parametrize("meta", [None, {}, {"bill_states": None}, {"bill_states": {"bills": "x"}},
                                  {"bill_states": {"bills": [{"label": "3", "pages": None}]}}])
def test_bill_page_ranges_is_none_when_nothing_usable_is_stored(meta):
    assert agg.bill_page_ranges(meta) is None


# --- selection: a named bill with a stored page range selects by page --------

def _carried(n, prev, a, b, *, first):
    """Bill ``n`` on pages ``first``..``first+1``; its first page still carries
    the heading of bill ``prev`` (the heading printed on the page before)."""
    return [
        _r(first, "carried item", a, "1", a, path=(f"Bill No. {prev} - Works",)),
        _r(first + 1, "own item", b, "1", b, path=(f"Bill No. {n} - Works",)),
    ]


@pytest.mark.parametrize("n, prev, a, b", [
    ("3", "2", "120", "80"),
    ("8", "7", "1,500", "250"),
])
def test_a_named_bill_with_a_range_selects_its_rows_by_page(n, prev, a, b):
    rows = ([_r(10, "previous item", "9", "1", "9", path=(f"Bill No. {prev} - Works",))]
            + _carried(n, prev, a, b, first=20))
    ranges = {prev: (10, 19), n: (20, 21)}
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask=f"Total amount of Bill No. {n}?", ranges=ranges)
    want = agg.g.parse_number(a) + agg.g.parse_number(b)
    assert res["sum"] == want
    assert sorted(e["description"] for e in res["rows"]) == ["carried item", "own item"]
    # the rows summed are the rows checked
    assert res["bill_checks"][0]["items_sum"] == want
    assert res["scope"]["bill_pages"] == {n: [20, 21]}
    # the bill before keeps only its own page: the carried row is not its row
    prev_res = agg.aggregate(rows, terms="", measure="amount",
                             ask=f"Total amount of Bill No. {prev}?", ranges=ranges)
    assert [e["description"] for e in prev_res["rows"]] == ["previous item"]


@pytest.mark.parametrize("first, last, inside", [
    (20, 20, ["carried item"]),
    (21, 21, ["own item"]),
    (22, 30, []),
])
def test_page_range_ends_are_inclusive_and_outside_rows_are_left_out(first, last, inside):
    rows = _carried("3", "2", "120", "80", first=20)
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask="Total amount of Bill No. 3?", ranges={"3": (first, last)})
    assert [e["description"] for e in res["rows"]] == inside


@pytest.mark.parametrize("n, other", [("3", "2"), ("8", "7")])
def test_a_named_bill_without_a_range_still_selects_by_its_path(n, other):
    rows = _carried(n, other, "120", "80", first=20)
    # a range is stored for another bill only: bill n falls back to its path
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask=f"Total amount of Bill No. {n}?", ranges={other: (1, 5)})
    assert [e["description"] for e in res["rows"]] == ["own item"]
    assert res["scope"]["bill_pages"] == {}
    assert res["bill_checks"][0]["membership"] == "not_read"


@pytest.mark.parametrize("path", [("Bill No. 3 - Works",), ("Bill No. 4 - Works",)])
def test_a_row_with_no_page_is_not_in_a_ranged_bill(path):
    rows = [_r(None, "no page", "5", "1", "5", path=path),
            _r(20, "paged", "7", "1", "7", path=path)]
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask="Total amount of Bill No. 3?", ranges={"3": (20, 21)})
    assert [e["description"] for e in res["rows"]] == ["paged"]


@pytest.mark.parametrize("ranges, spans", [
    (None, ()),
    ({"3": (20, 21), "9": (1, 2)}, (("3", 20, 21),)),
])
def test_ask_scope_keeps_the_range_of_each_named_bill_only(ranges, spans):
    rows = agg.join_page_continuations(_carried("3", "2", "120", "80", first=20))
    scope = agg.ask_scope("Total amount of Bill No. 3?", rows, ranges)
    assert scope.spans == spans


# --- one placement for every bill reader: the page range, not the path ------

def _money(value):
    return f"{value:,.0f}"


def _two_ranged(lump, title_b5="Bill No. 5 - External Works", carried="Bill No. 4 - Earthworks"):
    """Bill 4 on pages 1-3 and Bill 5 on pages 4-6, each with page totals and
    a collection page. The first page of Bill 5 still carries the Bill 4
    heading; its first line is an amount-only lump of ``lump``."""
    b4, b5 = (carried,), (title_b5,)
    first = agg.g.parse_number(lump) + 10_000
    return [
        _r(1, "dig", "30,000", "10", "3,000", path=b4), _r(1, "fill", "20,000", "10", "2,000", path=b4),
        _r(1, "page", "50,000", path=b4),
        _r(2, "cart", "40,000", "10", "4,000", path=b4), _r(2, "tamp", "10,000", "10", "1,000", path=b4),
        _r(2, "page", "50,000", path=b4),
        _r(3, "p1", "50,000", path=b4), _r(3, "p2", "50,000", path=b4), _r(3, "bill", "100,000", path=b4),
        _r(4, "allow lump", lump, path=b4), _r(4, "kerb", "7,000", "7", "1,000", path=b4),
        _r(4, "paving", "3,000", "3", "1,000", path=b5),
        _r(4, "page", _money(first), path=b5),
        _r(5, "lights", "12,000", "12", "1,000", path=b5), _r(5, "bollards", "8,000", "8", "1,000", path=b5),
        _r(5, "page", "20,000", path=b5),
        _r(6, "p4", _money(first), path=b5), _r(6, "p5", "20,000", path=b5),
        _r(6, "bill", _money(first + 20_000), path=b5),
    ]


_RANGES_4_5 = {"4": (1, 3), "5": (4, 6)}


@pytest.mark.parametrize("lump", ["50,000", "100,000"])
def test_a_lump_under_a_carried_heading_is_not_a_repeat_of_the_bill_before(lump):
    # the lump equals a page total (50,000) or the bill total (100,000) of the
    # bill before; placed by page it is an item of Bill 5, never a repeat
    rows = _two_ranged(lump)
    want = agg.g.parse_number(lump) + 30_000
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask="What is the total amount of Bill No. 5?", ranges=_RANGES_4_5)
    assert res["sum"] == want
    check = res["bill_checks"][0]
    assert (check["items_sum"], check["printed_total"], check["state"]) == (want, want, "RECONCILED")
    assert agg.row_kinds(rows, _RANGES_4_5)[9] == agg.ROW_ITEM
    # without ranges the path decides, and the carried heading makes it a repeat
    assert agg.row_kinds(rows)[9] == agg.ROW_REPEAT


@pytest.mark.parametrize("lump", ["50,000", "100,000"])
def test_a_repeat_of_the_same_ranged_bill_is_still_a_repeat(lump):
    # Bill 4's collection page lists its own page totals: still repeats
    kinds = agg.row_kinds(_two_ranged(lump), _RANGES_4_5)
    assert kinds[6] == agg.ROW_REPEAT and kinds[7] == agg.ROW_REPEAT


@pytest.mark.parametrize("ask, title", [
    ("What is the total amount of Bill No. 5 External Works?", "Bill No. 5 - External Works"),
    ("Total amount of Bill No. 5 site landscaping?", "Bill No. 5 - Site Landscaping"),
])
def test_the_bills_title_words_do_not_drop_its_carried_heading_rows(ask, title):
    rows = _two_ranged("1,000", title_b5=title)
    res = agg.aggregate(rows, terms="", measure="amount", ask=ask, ranges=_RANGES_4_5)
    assert res["sum"] == 31_000
    assert res["sum"] == res["bill_checks"][0]["items_sum"]
    assert "allow lump" in {e["description"] for e in res["rows"]}


@pytest.mark.parametrize("ask, want", [
    # a word of a sub-heading inside the bill still scopes (not a title word)
    ("Total amount of Bill No. 5 lighting?", ["lights"]),
    ("Total amount of Bill No. 5 bollard?", ["bollards"]),
])
def test_a_word_that_is_not_the_bills_title_still_scopes(ask, want):
    rows = _two_ranged("1,000")
    rows[13] = _r(5, "lights", "12,000", "12", "1,000",
                  path=("Bill No. 5 - External Works", "Lighting"))
    res = agg.aggregate(rows, terms="", measure="amount", ask=ask, ranges=_RANGES_4_5)
    assert [e["description"] for e in res["rows"]] == want


@pytest.mark.parametrize("ranges", [_RANGES_4_5, {"4": (1, 3)}])
def test_title_words_are_dropped_only_for_a_ranged_named_bill(ranges):
    rows = agg.join_page_continuations(_two_ranged("1,000"))
    scope = agg.ask_scope("Total amount of Bill No. 5 External Works?", rows, ranges)
    assert ("external" in scope.level) is ("5" not in ranges)


@pytest.mark.parametrize("asked, want", [("4", 100_000), ("5", 31_000)])
def test_two_named_bills_each_select_by_their_own_range(asked, want):
    rows = _two_ranged("1,000")
    both = agg.aggregate(rows, terms="", measure="amount",
                         ask="What is the total amount of Bill No. 4 and Bill No. 5?",
                         ranges=_RANGES_4_5)
    assert both["sum"] == 131_000
    one = agg.aggregate(rows, terms="", measure="amount",
                        ask=f"What is the total amount of Bill No. {asked}?", ranges=_RANGES_4_5)
    assert one["sum"] == want


@pytest.mark.parametrize("a, b", [("300", "50"), ("1,200", "75")])
def test_a_row_joined_across_a_bill_boundary_is_in_its_first_pages_bill(a, b):
    rows = [_r(3, "dig", "100", "1", "100", path=("Bill No. 4",)),
            agg.SheetRow(path=("Bill No. 4",), ref="", description="fill, continued", unit="",
                         quantity="1", rate=a, amount=a, pages=[3, 4]),
            _r(4, "kerb", b, "1", b, path=("Bill No. 5",))]
    for bill, want in (("4", 100 + agg.g.parse_number(a)), ("5", agg.g.parse_number(b))):
        res = agg.aggregate(rows, terms="", measure="amount",
                            ask=f"Total amount of Bill No. {bill}?", ranges=_RANGES_4_5)
        assert res["sum"] == want
        assert res["bill_checks"][0]["items_sum"] == want


@pytest.mark.parametrize("amounts", [("100", "200"), ("7", "9", "11")])
def test_rows_with_no_page_under_a_ranged_bill_are_named_not_lost(amounts):
    rows = [_r(None, f"item {k}", a, "1", a, path=("Bill No. 5",)) for k, a in enumerate(amounts)]
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask="What is the total amount of Bill No. 5?", ranges={"5": (1, 3)})
    assert res["sum"] == 0.0
    assert [e["amount"] for e in res["without_page"]] == [agg.g.parse_number(a) for a in amounts]
    check = res["bill_checks"][0]
    assert (check["rows_without_page"], check["state"]) == (len(amounts), "UNCHECKED")


@pytest.mark.parametrize("path, listed", [(("Bill No. 5",), 1), (("Bill No. 6",), 0)])
def test_one_blank_page_cell_is_listed_only_under_the_bill_its_path_names(path, listed):
    rows = [_r(2, "paged", "40", "1", "40", path=("Bill No. 5",)),
            _r(None, "no page", "60", "1", "60", path=path)]
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask="What is the total amount of Bill No. 5?", ranges={"5": (1, 3)})
    assert res["sum"] == 40.0
    assert len(res["without_page"]) == listed
    # a path naming an unranged bill places the row there; it is not unplaced
    assert res["bill_checks"][0]["rows_without_page"] == listed


@pytest.mark.parametrize("ranges", [None, {"9": (1, 2)}])
def test_without_a_range_for_the_named_bill_no_row_is_listed_without_page(ranges):
    rows = [_r(None, "no page", "60", "1", "60", path=("Bill No. 5",)),
            _r(3, "paged", "40", "1", "40", path=("Bill No. 5",))]
    res = agg.aggregate(rows, terms="", measure="amount",
                        ask="What is the total amount of Bill No. 5?", ranges=ranges)
    assert res["without_page"] == []
    assert res["sum"] == 100.0


@pytest.mark.parametrize("bills", [
    [{"label": "Bill No. 5", "pages": [4, 6, 8]}],
    [None, 3, "Bill No. 5", {"label": "Bill No. 5", "pages": [4]}],
])
def test_bill_page_ranges_skips_malformed_entries(bills):
    assert agg.bill_page_ranges({"bill_states": {"bills": bills}}) is None


@pytest.mark.parametrize("bills", [
    [None, {"label": "Bill No. 5", "pages": [4, 6]}],
    [7, {"label": "Bill No. 5", "pages": [6, 4]}],
])
def test_bill_page_ranges_reads_past_a_malformed_entry(bills):
    assert agg.bill_page_ranges({"bill_states": {"bills": bills}}) == {"5": (4, 6)}


@pytest.mark.parametrize("stored, want", [
    ([{"label": "Bill No. 4", "pages": [1, 3]}, {"label": "Bill No. 5", "pages": [4, 6]}],
     ({"5": [4, 6]}, "page_range", 31_000)),
    # no stored range: the path decides (carried rows go to Bill 4) and the
    # check says not_read, never a verdict
    (None, ({}, "not_read", 23_000)),
])
def test_project_bills_read_the_stored_ranges_from_the_copys_metadata(monkeypatch, stored, want):
    from app.core import projects
    meta = {"derivation": "x"}
    if stored is not None:
        meta["bill_states"] = {"bills": stored}
    rows = _two_ranged("1,000")
    monkeypatch.setattr(agg, "derived_bill_documents",
                        lambda pid, did=None: [{"id": "d", "original_name": "b.xlsx", "metadata": meta}])
    monkeypatch.setattr(projects, "materialize_document_path", lambda doc: "sheet.xlsx")
    monkeypatch.setattr(projects, "coerce_document_metadata", lambda m: m or {})
    monkeypatch.setattr(agg, "read_sheet_rows", lambda path: rows)
    out = agg.aggregate_project_bills("p", terms="", measure="amount",
                                      ask="What is the total amount of Bill No. 5?")
    bill = out["bills"][0]
    pages, membership, total = want
    assert bill["scope"]["bill_pages"] == pages
    assert bill["bill_checks"][0]["membership"] == membership
    assert bill["sum"] == total
