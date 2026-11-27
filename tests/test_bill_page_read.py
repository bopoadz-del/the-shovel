"""Rows from a model's read of bill pages (``app.core.bill_page_read``)."""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_page_read as bpr
from app.core import bill_table_geometry as g


def _item(ref, q, r, a, desc="work", unit="m2", **kw):
    return {"type": "item", "ref": ref, "description": desc, "unit": unit,
            "quantity": q, "rate": r, "amount": a, **kw}


def _head(text):
    return {"type": "heading", "description": text}


def _carried(a):
    return {"type": "carried", "description": "to next", "amount": a}


@pytest.mark.parametrize("q,r,a", [(10, 5, 50), (3, 2.5, 7.5)])
def test_item_that_multiplies_is_accepted(q, r, a):
    out = bpr.rows_from_page_reads({"7": [_item("A", q, r, a)]})
    (row,) = out.rows
    assert row.check == ""
    assert row.source_page == 7
    assert out.flagged == []


@pytest.mark.parametrize("q,r,a", [(10, 5, 51), (3, 2.5, 9)])
def test_item_that_does_not_multiply_is_flagged_and_kept_as_read(q, r, a):
    out = bpr.rows_from_page_reads({"7": [_item("A", q, r, a)]})
    (row,) = out.rows
    assert row.check == "qty x rate != amount"
    assert row.fields["total"] == str(a)  # never repaired
    assert out.flagged == [row]


@pytest.mark.parametrize("bad", ["12a", True])
def test_figure_that_is_not_a_number_is_flagged(bad):
    out = bpr.rows_from_page_reads({"1": [_item("A", 2, bad, 10)]})
    assert out.rows[0].check == bpr.CHECK_NOT_A_NUMBER


@pytest.mark.parametrize("value,cell", [(188996.0, "188996"), (1234.5, "1234.5"),
                                        ("1,234.50", "1,234.50"), (7, "7")])
def test_figure_cell_text(value, cell):
    out = bpr.rows_from_page_reads({"1": [{"type": "carried", "amount": value}]})
    assert out.rows[0].fields["total"] == cell


@pytest.mark.parametrize("pages", [{"3": []}, {"3": None}, {"3": "no json"},
                                   {"3": [_item("A", 1, 1, 1), "junk"]}])
def test_page_without_a_list_of_lines_is_not_read(pages):
    out = bpr.rows_from_page_reads({**pages, "4": [_item("B", 2, 2, 4)]})
    assert out.pages_not_read == [3]
    assert [r.source_page for r in out.rows] == [4]


@pytest.mark.parametrize("key", ["p3", "", "-2", "3.5", "³"])
def test_key_that_is_not_a_page_number_is_listed_not_dropped(key):
    out = bpr.rows_from_page_reads({key: [_item("A", 1, 1, 1)],
                                    "4": [_item("B", 2, 2, 4)]})
    assert out.keys_not_pages == [key]
    assert [r.source_page for r in out.rows] == [4]
    assert out.pages_not_read == []


@pytest.mark.parametrize("key, page", [(" 12 ", 12), (7, 7), ("007", 7), ("٣", 3)])
def test_page_number_key_is_a_page(key, page):
    out = bpr.rows_from_page_reads({key: [_item("A", 1, 1, 1)]})
    assert out.keys_not_pages == []
    assert [r.source_page for r in out.rows] == [page]


@pytest.mark.parametrize("label", ["Bill 7", "Part B"])
def test_heading_run_is_the_path_under_the_bill_label(label):
    pages = {"2": [_head("Earth"), _head("Digging"), _item("A", 1, 2, 2),
                   _item("B", 1, 3, 3), _head("Fill"), _item("C", 1, 4, 4)]}
    rows = bpr.rows_from_page_reads(pages, bill_label=label).rows
    assert [r.path for r in rows] == [(label, "Earth", "Digging")] * 2 + [(label, "Fill")]
    assert rows[2].section == "Fill"


@pytest.mark.parametrize("label", ["", "   "])
def test_no_bill_label_gives_heading_path_only(label):
    rows = bpr.rows_from_page_reads({"1": [_head("H"), _item("A", 1, 1, 1)]},
                                    bill_label=label).rows
    assert rows[0].path == ("H",)


@pytest.mark.parametrize("first,second", [(9, 10), (99, 100)])
def test_pages_are_taken_in_page_order_and_the_path_carries_across(first, second):
    pages = {str(second): [_item("B", 1, 2, 2)],
             str(first): [_head("Roads"), _item("A", 1, 1, 1)],
             "x": [_item("Z", 1, 1, 1)]}
    rows = bpr.rows_from_page_reads(pages).rows
    assert [(r.source_page, r.path) for r in rows] == [
        (first, ("Roads",)), (second, ("Roads",))]


@pytest.mark.parametrize("cols", [{"Fixed Charge": "1,000", "Time Charge": 250},
                                  {"Labour": 40}])
def test_printed_money_columns_are_kept_as_their_own_columns(cols):
    line = {"type": "item", "ref": "A", "description": "x", "unit": "sum",
            "quantity": 1, "figures": cols}
    row = bpr.rows_from_page_reads({"1": [line]}).rows[0]
    for k, v in cols.items():
        assert row.extra[k] == str(v)
    assert row.fields["rate"] == "" and row.fields["total"] == ""
    assert row.check == ""


@pytest.mark.parametrize("cols", [{"Fixed Charge": "n/a"}, {"Time": False}])
def test_printed_money_column_not_a_number_is_flagged(cols):
    line = {"type": "item", "description": "x", "figures": cols}
    assert bpr.rows_from_page_reads({"1": [line]}).rows[0].check == bpr.CHECK_NOT_A_NUMBER


@pytest.mark.parametrize("kind", ["carried", "total"])
def test_non_heading_line_is_a_row_with_its_type(kind):
    line = {"type": kind, "description": "c", "amount": 9}
    row = bpr.rows_from_page_reads({"1": [line]}).rows[0]
    assert row.extra[bpr.LINE_TYPE_LABEL] == kind
    assert bpr.REF_LABEL not in row.extra


@pytest.mark.parametrize("a,b", [(100, 250), (1200, 37.5)])
def test_written_sheet_reads_back_through_the_aggregate_reader(tmp_path, a, b):
    """Public path: model lines -> rows -> the derived-sheet writer -> the sheet reader."""
    pages = {"5": [_head("Roads"), _item("A", 1, a, a), _item("B", 2, b / 2, b),
                   _carried(a + b)]}
    out = bpr.rows_from_page_reads(pages, bill_label="Bill 9")
    path = str(tmp_path / "d.xlsx")
    g.write_bill_workbook(out.rows, path)
    rows = agg.read_sheet_rows(path)
    assert [(r.ref, r.amount, r.pages) for r in rows] == [
        ("A", str(a), [5]), ("B", str(b), [5]), ("", str(a + b), [5])]
    assert all(r.path == ("Bill 9", "Roads") for r in rows)


@pytest.mark.parametrize("reads, want_pages, want_not_read", [
    ({0: [_item("A", 2, 3, 6)], 1: [_item("B", 4, 5, 20), _item("C", 1, 9, 9)]},
     [1, 2, 2], []),
    ({0: None, 1: "not lines", 2: [_head("Part"), _item("D", 7, 7, 49)]},
     [3], [1, 2]),
])
def test_rows_of_reads_each_page_and_numbers_it_from_one(reads, want_pages, want_not_read):
    from types import SimpleNamespace

    asked: list[int] = []

    def read_page(page_read):
        asked.append(page_read.index)
        return reads[page_read.index]

    rows_of = bpr.rows_of_model_reads(read_page, "Bill 9")
    rows, not_read, raw = rows_of([SimpleNamespace(index=i) for i in sorted(reads)])
    assert asked == sorted(reads)
    assert raw == {str(i + 1): reads[i] for i in reads}  # every reply, read or not
    assert [r.source_page for r in rows] == want_pages
    assert not_read == want_not_read
    assert all(tuple(r.path[:1]) == ("Bill 9",) for r in rows)


# --- per-bill state against the printed total -------------------------------

def _rows(pages):
    return bpr.rows_from_page_reads(pages).rows


_TWO_BILLS = {"1": [_item("A", 10, 5, 50), _item("B", 2, 3, 6)],
              "2": [_carried(56)],
              "3": [_item("C", 4, 25, 100), {"type": "item", "description": "lump"}],
              "5": [_item("D", 1, 7, 7)]}


@pytest.mark.parametrize("printed_one, printed_two, want_one, want_two", [
    (56, 100, bpr.STATE_RECONCILED, bpr.STATE_RECONCILED),
    (57, 99, bpr.STATE_FAILED, bpr.STATE_FAILED),
    (56.5, 100.5, bpr.STATE_RECONCILED, bpr.STATE_RECONCILED),
    (None, 100, bpr.STATE_UNCHECKED, bpr.STATE_RECONCILED),
])
def test_each_bill_is_checked_against_its_printed_total(printed_one, printed_two,
                                                        want_one, want_two):
    spans = [bpr.BillSpan("one", 1, 2, printed_one), bpr.BillSpan("two", 3, 4, printed_two)]
    got = bpr.bill_states(_rows(_TWO_BILLS), [], spans=spans, tolerance=0.5)
    one, two = got["bills"]
    assert (one["state"], two["state"]) == (want_one, want_two)
    # only item rows are summed: the carried line on page 2 is a row, not an item
    assert (one["rows"], one["rows_summed"], one["items_sum"]) == (3, 2, 56)
    assert (two["rows"], two["rows_summed"], two["rows_without_amount"]) == (2, 1, 1)
    assert two["items_sum"] == 100
    assert got["rows_unassigned"] == 1  # page 5 lies under no bill


@pytest.mark.parametrize("missing, want", [
    ([2], [bpr.STATE_NOT_READ, bpr.STATE_RECONCILED]),
    ([4], [bpr.STATE_RECONCILED, bpr.STATE_NOT_READ]),
    ([5], [bpr.STATE_RECONCILED, bpr.STATE_RECONCILED]),
])
def test_bill_with_a_page_not_read_is_not_read(missing, want):
    spans = [bpr.BillSpan("one", 1, 2, 56), bpr.BillSpan("two", 3, 4, 100)]
    got = bpr.bill_states(_rows(_TWO_BILLS), missing, spans=spans, tolerance=0)
    assert [b["state"] for b in got["bills"]] == want
    assert [b["pages_not_read"] for b in got["bills"]] == [
        [p for p in missing if 1 <= p <= 2], [p for p in missing if 3 <= p <= 4]]


@pytest.mark.parametrize("bad_amount", [57, "x"])
def test_bill_with_a_flagged_row_fails_even_when_the_sum_matches(bad_amount):
    pages = {"1": [_item("A", 10, 5, bad_amount)]}
    printed = 57 if bad_amount == 57 else 0
    got = bpr.bill_states(_rows(pages), [], spans=[bpr.BillSpan("b", 1, 1, printed)],
                          tolerance=0)
    (b,) = got["bills"]
    assert b["rows_flagged"] == 1
    assert b["state"] == bpr.STATE_FAILED


@pytest.mark.parametrize("tolerance, want", [(0.99, bpr.STATE_FAILED),
                                             (1, bpr.STATE_RECONCILED)])
def test_tolerance_boundary_is_inclusive(tolerance, want):
    got = bpr.bill_states(_rows({"1": [_item("A", 10, 5, 50)]}), [],
                          spans=[bpr.BillSpan("b", 1, 1, 51)], tolerance=tolerance)
    assert got["bills"][0]["state"] == want
    assert got["bills"][0]["diff"] == -1


@pytest.mark.parametrize("spans", [[], [bpr.BillSpan("far", 9, 12, 0)]])
def test_no_bill_holds_the_rows_they_are_unassigned(spans):
    rows = _rows(_TWO_BILLS)
    got = bpr.bill_states(rows, [], spans=spans, tolerance=0)
    assert got["rows_unassigned"] == len(rows)
    assert [b["rows"] for b in got["bills"]] == [0] * len(spans)
