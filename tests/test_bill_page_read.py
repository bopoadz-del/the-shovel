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
