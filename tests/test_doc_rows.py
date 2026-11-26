"""doc_rows: row shape, the four reconciliation states, rows_query, and the
spreadsheet path where row i and row chunk i are one row."""
from __future__ import annotations

import pytest

from app.core import doc_rows as dr
from app.core.bill_table_geometry import BillRow


def _row(table, n, desc, qty="", rate="", total="", path=("Bill A",), unit="m3"):
    fields = {"description": desc}
    for k, v in (("quantity", qty), ("unit", unit), ("rate", rate), ("total", total)):
        if v != "":
            fields[k] = v
    return dr.DocRow(document_id="doc-x", table_no=table, row_no=n, path=path,
                     fields=fields, page=n + 2)


# ── reconciliation states ──────────────────────────────────────────────────

@pytest.mark.parametrize("rows", [
    [_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Backfill", "4", "2.5", "10")],
    [_row(1, 1, "Formwork", "3", "100", "300.00")],
])
def test_every_checkable_row_agrees_reconciles(rows):
    assert dr.reconcile_table(rows).state == dr.RECONCILED


@pytest.mark.parametrize("rows, failed", [
    ([_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Backfill", "4", "2.5", "11")], 1),
    ([_row(1, 1, "Formwork", "3", "100", "200"), _row(1, 2, "Rebar", "2", "7", "15")], 2),
])
def test_one_disagreeing_row_fails_the_table(rows, failed):
    check = dr.reconcile_table(rows)
    assert check.state == dr.FAILED
    assert check.failed == failed
    assert check.checked == 2


@pytest.mark.parametrize("rows", [
    [_row(1, 1, "Excavate", "10", "", "50")],          # no rate
    [_row(1, 1, "Rate only item", "", "12.5", ""), _row(1, 2, "Lump sum", "", "", "900")],
])
def test_rows_with_nothing_to_check_are_unchecked(rows):
    check = dr.reconcile_table(rows)
    assert check.state == dr.UNCHECKED
    assert check.checked == 0


@pytest.mark.parametrize("total", [None, 100.0])
def test_no_rows_is_not_read(total):
    assert dr.reconcile_table([], total).state == dr.NOT_READ


@pytest.mark.parametrize("rows, printed", [
    ([_row(1, 1, "Lump sum A", total="900"), _row(1, 2, "Lump sum B", total="100")], 1000.0),
    ([_row(1, 1, "Excavate", "10", "5", "50")], 50.0),
])
def test_printed_total_that_agrees_reconciles(rows, printed):
    check = dr.reconcile_table(rows, printed)
    assert check.state == dr.RECONCILED
    assert check.total_of_rows == pytest.approx(printed)


@pytest.mark.parametrize("rows, printed", [
    ([_row(1, 1, "Lump sum A", total="900"), _row(1, 2, "Lump sum B", total="100")], 1100.0),
    ([_row(1, 1, "Excavate", "10", "5", "50")], 60.0),          # rows agree, total does not
    ([_row(1, 1, "Rate only item", "", "12.5", "")], 10.0),    # no amount to sum
])
def test_printed_total_that_disagrees_fails(rows, printed):
    assert dr.reconcile_table(rows, printed).state == dr.FAILED


@pytest.mark.parametrize("rows, totals, states", [
    ([_row(1, 1, "Excavate", "10", "5", "50"), _row(2, 1, "Backfill", "4", "2.5", "11")],
     {3: 75.0}, {1: dr.RECONCILED, 2: dr.FAILED, 3: dr.NOT_READ}),
    ([_row(1, 1, "Formwork", "3", "100", "350"), _row(2, 1, "Lump sum", total="40")],
     {2: 40.0}, {1: dr.FAILED, 2: dr.RECONCILED}),
])
def test_tables_reconcile_separately_and_a_total_with_no_rows_is_not_read(rows, totals, states):
    out = dr.reconcile_tables(rows, totals)
    assert {t: c.state for t, c in out.items()} == states


# ── rows_query ─────────────────────────────────────────────────────────────

ROWS = [
    _row(1, 1, "Excavate trench", "10", "5", "50", path=("Bill 2", "Earthworks")),
    _row(1, 2, "Backfill trench", "4", "2.5", "10", path=("Bill 2", "Earthworks")),
    _row(1, 3, "Blinding concrete", "2", "80", "160", path=("Bill 3", "Concrete")),
    _row(1, 4, "Provisional item", "", "", "", path=("Bill 3", "Concrete")),
]


@pytest.mark.parametrize("where, value, used", [
    ({"path": "Earthworks"}, 60.0, [1, 2]),
    ({"path": "Bill 3"}, 160.0, [3]),
    ({"description": "trench"}, 60.0, [1, 2]),
])
def test_sum_adds_the_matching_amounts_and_names_the_rows(where, value, used):
    out = dr.rows_query(ROWS, "sum", where)
    assert out["value"] == pytest.approx(value)
    assert [r["row_no"] for r in out["rows"]] == used


@pytest.mark.parametrize("where, value, skipped", [
    ({"path": "Concrete"}, 160.0, [4]),
    ({"path": "Bill 3"}, 160.0, [4]),
    ({"description": "Provisional"}, None, [4]),
    ({"path": "Earthworks"}, 60.0, []),
])
def test_sum_names_a_row_without_a_figure_as_skipped_not_zero(where, value, skipped):
    out = dr.rows_query(ROWS, "sum", where)
    assert out["value"] == (pytest.approx(value) if value is not None else None)
    assert [r["row_no"] for r in out["skipped"]] == skipped


@pytest.mark.parametrize("where, n", [({"path": "Bill 2"}, 2), ({}, 4), ({"description": "none such"}, 0)])
def test_count(where, n):
    assert dr.rows_query(ROWS, "count", where)["value"] == n


@pytest.mark.parametrize("where, row_no", [({"description": "Blinding"}, 3), ({"description": "Backfill"}, 2)])
def test_lookup_returns_the_one_row_with_its_page(where, row_no):
    out = dr.rows_query(ROWS, "lookup", where)
    assert out["row"]["row_no"] == row_no
    assert out["row"]["page"] == row_no + 2


@pytest.mark.parametrize("where", [{"description": "trench"}, {"description": "none such"}])
def test_lookup_refuses_when_not_exactly_one_row(where):
    out = dr.rows_query(ROWS, "lookup", where)
    assert "error" in out and "row" not in out


@pytest.mark.parametrize("a, b, diff", [
    ({"path": "Bill 3"}, {"path": "Bill 2"}, 100.0),
    ({"description": "Excavate"}, {"description": "Backfill"}, 40.0),
])
def test_compare_gives_first_minus_second(a, b, diff):
    out = dr.rows_query(ROWS, "compare", a, other=b)
    assert out["difference"] == pytest.approx(diff)


def test_compare_with_an_empty_side_has_no_difference():
    out = dr.rows_query(ROWS, "compare", {"path": "Bill 2"}, other={"description": "none such"})
    assert out["difference"] is None


@pytest.mark.parametrize("column, value", [("quantity", 16.0), ("rate", 87.5)])
def test_sum_reads_the_column_asked(column, value):
    assert dr.rows_query(ROWS, "sum", {}, column=column)["value"] == pytest.approx(value)


def test_unknown_operation_is_an_error_not_a_guess():
    out = dr.rows_query(ROWS, "average")
    assert "error" in out and out["operations"] == list(dr.OPERATIONS)


# ── readers ────────────────────────────────────────────────────────────────

NUMBERED = [
    _row(1, 1, "Strip topsoil", total="120", path=("Bill 1", "Site")),
    _row(1, 2, "Remove debris", total="80", path=("Bill 1", "Site")),
    _row(1, 3, "Formwork", total="300", path=("Bill 10",)),
    _row(1, 4, "Rebar", total="400", path=("Bill 11",)),
    _row(1, 5, "Blockwork", total="500", path=("Bill 12",)),
]


@pytest.mark.parametrize("where, value, count", [
    ({"path": "Bill 1"}, 200.0, 2),
    ({"path": "bill 10"}, 300.0, 1),
    ({"path": "BILL 1 > site"}, 200.0, 2),
    ({"description": "REBAR"}, 400.0, 1),
    ({"path": "Bill"}, 1400.0, 5),
])
def test_where_matches_whole_tokens_not_a_prefix(where, value, count):
    assert dr.rows_query(NUMBERED, "sum", where)["value"] == pytest.approx(value)
    assert dr.rows_query(NUMBERED, "count", where)["value"] == count


@pytest.mark.parametrize("item, picked", [("1", [1]), ("1.1", [3]), ("10", [2])])
def test_item_reference_matches_itself_only(item, picked):
    rows = [dr.DocRow("d", 1, n, fields={"description": "x", "total": "1"}, extra={"Item": ref})
            for n, ref in ((1, "1"), (2, "10"), (3, "1.1"), (4, "11"))]
    got = dr.rows_query(rows, "filter", {"Item": item})
    assert ([r["row_no"] for r in got["rows"]], got["count"]) == (picked, len(picked))


@pytest.mark.parametrize("value", ["", " - ", "  "])
def test_where_value_without_a_token_matches_nothing(value):
    assert dr.rows_query(NUMBERED, "count", {"path": value})["value"] == 0


@pytest.mark.parametrize("amounts, printed, state", [
    (["500000", "495500"], 1_000_000.0, dr.FAILED),      # one row's worth missing
    (["400", "600"], 1004.0, dr.FAILED),
    (["1000", "2000", "3000"], 6001.0, dr.RECONCILED),   # three halves + the total's half
    (["1000", "2000", "3000"], 6003.0, dr.FAILED),
    (["100.25", "200.25"], 300.5, dr.RECONCILED),
    (["100.25", "200.25"], 300.52, dr.FAILED),
    (["400", "600"], float("inf"), dr.FAILED),
])
def test_printed_total_allows_only_the_rounding_of_its_figures(amounts, printed, state):
    rows = [_row(1, i + 1, f"Lump sum {i}", total=a) for i, a in enumerate(amounts)]
    assert dr.reconcile_table(rows, printed).state == state


@pytest.mark.parametrize("where, other, missing", [
    ({"description": "Excavate"}, None, "other"),
    (None, {"description": "Backfill"}, "where"),
    (None, None, "where, other"),
    ({"description": "Excavate"}, {}, "other"),
])
def test_compare_without_both_sides_is_an_error_not_a_figure(where, other, missing):
    got = dr.rows_query(ROWS, "compare", where, other=other)
    assert got["error"].endswith(f"missing: {missing}")
    assert "difference" not in got and got["rows"] == []


@pytest.mark.parametrize("where, n", [({"description": "trench"}, 2), ({"description": "blinding"}, 1)])
def test_filter_counts_only_the_rows_that_match(where, n):
    assert dr.rows_query(ROWS, "filter", where)["count"] == n


@pytest.mark.parametrize("bill, expect, state", [
    ([BillRow(section="Earthworks", source_page=7, path=("Bill 2", "Earthworks"),
              fields={"description": "Excavate", "quantity": "10", "rate": "5", "total": "50"}),
      BillRow(section="Concrete", source_page=9,
              fields={"description": "Blinding", "quantity": "2", "rate": "80", "total": "150"},
              check="qty x rate != amount")],
     [(7, ("Bill 2", "Earthworks"), ""), (9, ("Concrete",), "qty x rate != amount")], dr.FAILED),
    ([BillRow(section="", source_page=3,
              fields={"description": "Provisional sum", "total": "500"})],
     [(3, (), "")], dr.UNCHECKED),
])
def test_bill_rows_keep_page_path_and_check(bill, expect, state):
    rows = dr.rows_from_bill("doc-b", bill)
    assert [(r.page, r.path, r.check) for r in rows] == expect
    assert dr.reconcile_table(rows).state == state


# ── locator: a citation names what the row has, and nothing it lacks ──────

@pytest.mark.parametrize("row, loc", [
    (dr.DocRow("d1", 2, 5, path=("Bill 4",), page=12),
     {"document_id": "d1", "table_no": 2, "row_no": 5, "path": ["Bill 4"], "page": 12}),
    (dr.DocRow("d2", 1, 3, sheet="Bill 1", sheet_row=9, chunk_ordinal=0),
     {"document_id": "d2", "table_no": 1, "row_no": 3, "sheet": "Bill 1",
      "sheet_row": 9, "chunk_ordinal": 0}),
    (dr.DocRow("d3", 1, 1),
     {"document_id": "d3", "table_no": 1, "row_no": 1}),
])
def test_locator_names_only_what_the_row_has(row, loc):
    assert row.locator() == loc


def _xlsx(tmp_path, name, sheets):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    path = tmp_path / name
    wb.save(path)
    return str(path)


HEADER = ["Item", "Description", "Unit", "Qty", "Rate", "Amount"]


@pytest.mark.parametrize("sheets, states, total", [
    ({"Bill 1": [["BILL No. 1 - SITE WORKS"], HEADER,
                 ["A", "Excavate to reduce levels", "m3", 120, 15, 1800],
                 ["B", "Fill with imported material", "m3", 40, 22.5, 900]]},
     {1: dr.RECONCILED}, 2700.0),
    ({"Bill 2": [["BILL No. 2 - CONCRETE"], HEADER,
                 ["A", "Blinding 50 thick", "m2", 300, 12, 3600]],
      "Bill 3": [["BILL No. 3 - MASONRY"], HEADER,
                 ["A", "Blockwork 200 thick", "m2", 50, 40, 2100]]},
     {1: dr.RECONCILED, 2: dr.FAILED}, 5700.0),
])
def test_spreadsheet_public_path_rows_match_chunks(tmp_path, sheets, states, total):
    from app.core import doc_index

    path = _xlsx(tmp_path, "priced.xlsx", sheets)
    read = doc_index.spreadsheet_rows(path, "priced.xlsx", ".xlsx")
    chunks = doc_index._spreadsheet_row_chunks(path, "priced.xlsx", ".xlsx")
    rows = dr.rows_from_spreadsheet("doc-s", read)
    assert len(rows) == len(chunks) > 0
    for r in rows:
        chunk = chunks[r.chunk_ordinal]
        assert f"Item: {r.fields['description']}" in chunk
        assert f"Sheet: {r.sheet}" in chunk
        assert r.sheet_row is not None and r.sheet_row >= 3   # banner + header above
    assert {t: c.state for t, c in dr.reconcile_tables(rows).items()} == states
    assert dr.rows_query(rows, "sum")["value"] == pytest.approx(total)


ROW_A = ["A", "Excavate", "m3", 10, 5, 50]
ROW_B = ["B", "Backfill", "m3", 4, 2.5, 10]


@pytest.mark.parametrize("sheets, numbered", [
    ({"S1": [["BILL 1"], HEADER, ROW_A, ROW_B], "S2": [["BILL 2"], HEADER, ROW_A]},
     [(1, 1), (1, 2), (2, 1)]),
    ({"S1": [["BILL 1"], HEADER, ROW_A], "S2": [["BILL 2"], HEADER, ROW_A, ROW_B, ROW_A]},
     [(1, 1), (2, 1), (2, 2), (2, 3)]),
])
def test_spreadsheet_row_no_counts_within_its_table(tmp_path, sheets, numbered):
    from app.core import doc_index

    path = _xlsx(tmp_path, "two.xlsx", sheets)
    rows = dr.rows_from_spreadsheet("d", doc_index.spreadsheet_rows(path, "two.xlsx", ".xlsx"))
    assert [(r.table_no, r.row_no) for r in rows] == numbered
    assert [r.chunk_ordinal for r in rows] == list(range(len(numbered)))


@pytest.mark.parametrize("sheet, sheet_rows", [
    ([["BILL 1"], HEADER, ROW_A, ROW_B], [3, 4]),
    ([[None], [None], [None], ["BILL 1"], HEADER, ROW_A, ROW_B], [6, 7]),
    ([["BILL 1"], HEADER, ROW_A, [None], ROW_B], [3, 5]),
])
def test_spreadsheet_sheet_row_is_the_row_the_sheet_prints(tmp_path, sheet, sheet_rows):
    from app.core import doc_index

    path = _xlsx(tmp_path, "one.xlsx", {"S1": sheet})
    rows = dr.rows_from_spreadsheet("d", doc_index.spreadsheet_rows(path, "one.xlsx", ".xlsx"))
    assert [r.sheet_row for r in rows] == sheet_rows


@pytest.mark.parametrize("sheets", [
    {"S1": [["BILL No. 1"], HEADER, ROW_A], "S2": [HEADER, ROW_B]},
    {"S2": [HEADER, ROW_B, ROW_A]},
])
def test_sheet_without_its_own_heading_has_no_path(tmp_path, sheets):
    from app.core import doc_index

    path = _xlsx(tmp_path, "h.xlsx", sheets)
    rows = dr.rows_from_spreadsheet("d", doc_index.spreadsheet_rows(path, "h.xlsx", ".xlsx"))
    assert [r.path for r in rows if r.sheet == "S2"] == [()] * len(sheets["S2"][1:])


@pytest.mark.parametrize("remark", ["by hand", "machine dig"])
def test_spreadsheet_keeps_unmapped_cells_in_extra(tmp_path, remark):
    from app.core import doc_index

    sheet = [["BILL 1"], [*HEADER, "Remarks"], [*ROW_A, remark]]
    path = _xlsx(tmp_path, "r.xlsx", {"S1": sheet})
    (row,) = dr.rows_from_spreadsheet("d", doc_index.spreadsheet_rows(path, "r.xlsx", ".xlsx"))
    assert dict(row.extra) == {"Item": "A", "Remarks": remark}


# ── printed total lines (PR-A step 3) ──────────────────────────────────────

@pytest.mark.parametrize("sheets, totals, states", [
    # a subtotal per page, then a running total that the next page brings forward
    ({"Bill 1": [["BILL No. 1 - SITE WORKS"], HEADER,
                 ["A", "Excavate to reduce levels", "m3", 120, 15, 1800],
                 ["B", "Fill with imported material", "m3", 40, 22.5, 900],
                 ["C/F", "", "", "", "", 2700],
                 ["B/F", "", "", "", "", 2700],
                 ["C", "Topsoil strip", "m2", 100, 2, 200],
                 ["To collection", "", "", "", "", 2900]]},
     [(1, 2, 2700.0), (1, 2, 2700.0), (1, 3, 2900.0)], {1: dr.RECONCILED}),
    # two sheets; the second prints a total its rows do not add up to
    ({"Bill 2": [["BILL No. 2 - CONCRETE"], HEADER,
                 ["A", "Blinding 50 thick", "m2", 300, 12, 3600],
                 ["", "", "", "", "", 3600]],
      "Bill 3": [["BILL No. 3 - MASONRY"], HEADER,
                 ["A", "Blockwork 200 thick", "m2", 50, 40, 2000],
                 ["B", "Lintels", "nr", 10, 15, 150],
                 ["", "", "", "", "", 2400]]},
     [(1, 1, 3600.0), (2, 2, 2400.0)], {1: dr.RECONCILED, 2: dr.FAILED}),
])
def test_spreadsheet_total_lines_are_read_and_checked(tmp_path, sheets, totals, states):
    from app.core import doc_index

    path = _xlsx(tmp_path, "priced.xlsx", sheets)
    read = doc_index.spreadsheet_rows(path, "priced.xlsx", ".xlsx")
    chunks = doc_index._spreadsheet_row_chunks(path, "priced.xlsx", ".xlsx")
    rows = dr.rows_from_spreadsheet("doc-t", read)
    # a total line makes no chunk and no row; every row still names its chunk
    assert len(rows) == len(chunks)
    assert [r.chunk_ordinal for r in rows] == list(range(len(rows)))
    for r in rows:
        assert f"Item: {r.fields['description']}" in chunks[r.chunk_ordinal]
    lines = dr.spreadsheet_printed_totals(read)
    assert [(ln.table_no, ln.after_row, ln.value) for ln in lines] == totals
    got = dr.reconcile_tables_with_lines(rows, lines)
    assert {t: c.state for t, c in got.items()} == states
    assert {t: c.total_printed for t, c in got.items()} == {t: v for t, _, v in totals}


@pytest.mark.parametrize("closing", [
    ["", "", "", 999, "", ""],       # a quantity alone: not a printed amount
    ["", "Carried forward", "", "", "", ""],   # a caption alone: a label
    ["", "", "", "", "", "Included"],          # an amount cell with no number
])
def test_a_line_without_a_printed_amount_is_not_a_total(tmp_path, closing):
    from app.core import doc_index

    sheets = {"Bill 1": [["BILL No. 1"], HEADER,
                         ["A", "Excavate", "m3", 10, 5, 50], closing]}
    path = _xlsx(tmp_path, "t.xlsx", sheets)
    read = doc_index.spreadsheet_rows(path, "t.xlsx", ".xlsx")
    assert dr.spreadsheet_printed_totals(read) == []
    rows = dr.rows_from_spreadsheet("doc-n", read)
    assert dr.reconcile_tables_with_lines(rows, [])[1].state == dr.RECONCILED


_LINE_ROWS = [_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Fill", "2", "10", "20"),
              _row(1, 3, "Lump sum", total="30"), _row(2, 1, "Blinding", "1", "8", "8")]


@pytest.mark.parametrize("lines, agrees", [
    ([dr.PrintedTotal(1, 2, 70.0), dr.PrintedTotal(1, 3, 30.0)], [True, True]),    # subtotals
    ([dr.PrintedTotal(1, 2, 70.0), dr.PrintedTotal(1, 3, 100.0)], [True, True]),   # running
    ([dr.PrintedTotal(1, 2, 70.0), dr.PrintedTotal(1, 3, 65.0)], [True, False]),   # neither
    ([dr.PrintedTotal(1, 0, 10.0), dr.PrintedTotal(2, 1, 8.0)], [None, True]),     # nothing above
    ([dr.PrintedTotal(2, 1, 8.6)], [False]),                    # past half a unit of "8"
    ([dr.PrintedTotal(2, 1, 8.5)], [True]),                     # within half a unit of "8"
])
def test_total_line_checks_the_rows_it_closes(lines, agrees):
    assert [c.agrees for c in dr.check_total_lines(_LINE_ROWS, lines)] == agrees


@pytest.mark.parametrize("amounts, printed, agrees", [
    (["50.25", "20.10"], 70.36, True),       # 3 x 0.005 allowed
    (["50.25", "20.10"], 70.40, False),      # 0.05 off: past the printed cents
    (["50", "20"], 71.0, True),              # 3 x 0.5 allowed
    (["50", "20"], 72.0, False),             # 2 off: past the printed units
    (["995,500"], 1_000_000.0, False),       # short by a share of the total
    (["1,000,000"], 1_000_000.0, True),
])
def test_total_line_allows_printed_rounding_not_a_share(amounts, printed, agrees):
    rows = [_row(1, i + 1, f"Item {i}", total=a) for i, a in enumerate(amounts)]
    (check,) = dr.check_total_lines(rows, [dr.PrintedTotal(1, len(rows), printed)])
    assert check.agrees is agrees
    assert check.running == pytest.approx(sum(r.figure("total") for r in rows))



@pytest.mark.parametrize("rows, printed, running, agrees", [
    # a measured row with no amount is left out of the sum, not counted as 0
    ([_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Fill", "4")], 50.0, 50.0, True),
    ([_row(1, 1, "Fill", "4"), _row(1, 2, "Cart", "2", "3", "6"),
      _row(1, 3, "Lump sum", total="30")], 37.0, 36.0, True),
    ([_row(1, 1, "Fill", "4"), _row(1, 2, "Topsoil", "9")], 0.0, None, None),
])
def test_total_line_sums_only_rows_that_print_an_amount(rows, printed, running, agrees):
    (check,) = dr.check_total_lines(rows, [dr.PrintedTotal(1, len(rows), printed)])
    assert check.running == running
    assert check.agrees is agrees


@pytest.mark.parametrize("sheets, numbered, lines", [
    ({"S1": [["BILL 1"], HEADER, ["A", "Excavate", "m3", 10, 5, 50], ["", "", "", "", "", 50]],
      "S2": [["BILL 2"], HEADER, ["A", "Fill", "m3", 2, 10, 20], ["B", "Cart", "m3", 1, 4, 4],
             ["", "", "", "", "", 24]]},
     [(1, 1, 0), (2, 1, 1), (2, 2, 2)], [(1, 1), (2, 2)]),
    ({"S1": [["BILL 1"], HEADER, ["A", "Excavate", "m3", 10, 5, 50], ["B", "Cart", "m3", 1, 4, 4]],
      "S2": [["BILL 2"], HEADER, ["", "", "", "", "", 9], ["A", "Fill", "m3", 2, 10, 20]]},
     [(1, 1, 0), (1, 2, 1), (2, 1, 2)], [(2, 0)]),
])
def test_row_no_counts_within_its_table_and_a_total_line_closes_its_own(tmp_path, sheets, numbered, lines):
    from app.core import doc_index

    path = _xlsx(tmp_path, "n.xlsx", sheets)
    read = doc_index.spreadsheet_rows(path, "n.xlsx", ".xlsx")
    rows = dr.rows_from_spreadsheet("doc-n", read)
    assert [(r.table_no, r.row_no, r.chunk_ordinal) for r in rows] == numbered
    got = [(ln.table_no, ln.after_row) for ln in dr.spreadsheet_printed_totals(read)]
    assert got == lines


@pytest.mark.parametrize("lines, states", [
    ([dr.PrintedTotal(1, 3, 100.0)], {1: dr.RECONCILED, 2: dr.RECONCILED}),
    ([dr.PrintedTotal(1, 3, 103.0)], {1: dr.FAILED, 2: dr.RECONCILED}),     # past 4 x 0.5
    ([dr.PrintedTotal(1, 3, 102.0)], {1: dr.RECONCILED, 2: dr.RECONCILED}), # at 4 x 0.5
    ([dr.PrintedTotal(1, 0, 5.0)], {1: dr.RECONCILED, 2: dr.RECONCILED}),
    ([dr.PrintedTotal(3, 0, 5.0)], {1: dr.RECONCILED, 2: dr.RECONCILED, 3: dr.NOT_READ}),
])
def test_table_state_takes_its_total_lines(lines, states):
    got = dr.reconcile_tables_with_lines(_LINE_ROWS, lines)
    assert {t: c.state for t, c in got.items()} == states


@pytest.mark.parametrize("rows, line, state", [
    # a failing row stays FAILED even when the printed total matches the amounts
    ([_row(1, 1, "Excavate", "10", "5", "60"), _row(1, 2, "Fill", "2", "10", "20")],
     dr.PrintedTotal(1, 2, 80.0), dr.FAILED),
    ([_row(1, 1, "Formwork", "3", "100", "350")], dr.PrintedTotal(1, 1, 350.0), dr.FAILED),
    # rows that print amounts only: the agreeing total is the one check, so RECONCILED
    ([_row(1, 1, "Lump sum A", total="40"), _row(1, 2, "Lump sum B", total="60")],
     dr.PrintedTotal(1, 2, 100.0), dr.RECONCILED),
    ([_row(1, 1, "Provisional sum", total="500")], dr.PrintedTotal(1, 1, 500.0), dr.RECONCILED),
    # the same amounts-only rows with no agreeing total stay UNCHECKED
    ([_row(1, 1, "Provisional sum", total="500")], dr.PrintedTotal(1, 0, 9.0), dr.UNCHECKED),
])
def test_a_total_line_neither_hides_a_failing_row_nor_leaves_lump_sums_unchecked(rows, line, state):
    assert dr.reconcile_tables_with_lines(rows, [line])[1].state == state
