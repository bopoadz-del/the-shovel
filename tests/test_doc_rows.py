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

@pytest.mark.parametrize("rows, printed", [
    ([_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Backfill", "4", "2.5", "10")], 60.0),
    ([_row(1, 1, "Formwork", "3", "100", "300.00")], "300.00"),
])
def test_every_checkable_row_agrees_reconciles(rows, printed):
    assert dr.reconcile_table(rows, printed).state == dr.RECONCILED
    assert dr.reconcile_table(rows).state == dr.UNCHECKED


@pytest.mark.parametrize("rows, failed, printed, state", [
    ([_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Backfill", "4", "2.5", "11")], 1,
     61.0, dr.RECONCILED),
    ([_row(1, 1, "Formwork", "3", "100", "200"), _row(1, 2, "Rebar", "2", "7", "15")], 2,
     214.0, dr.FAILED),
])
def test_disagreeing_rows_are_counted_and_the_state_is_the_printed_totals(rows, failed, printed, state):
    # Row arithmetic is counted and reported; the state is the printed total's.
    check = dr.reconcile_table(rows, printed)
    assert check.state == state
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
     {1: 50.0, 3: 75.0}, {1: dr.RECONCILED, 2: dr.UNCHECKED, 3: dr.NOT_READ}),
    ([_row(1, 1, "Formwork", "3", "100", "350"), _row(2, 1, "Lump sum", total="40")],
     {1: 300.0, 2: 40.0}, {1: dr.FAILED, 2: dr.RECONCILED}),
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


@pytest.mark.parametrize("where, value, used, match", [
    ({"path": "Earthworks"}, 60.0, [1, 2], "exact"),
    ({"path": "Bill 3"}, 160.0, [3], "exact"),
    ({"description": "trench"}, 60.0, [1, 2], "contains"),
])
def test_sum_adds_the_matching_amounts_and_names_the_rows(where, value, used, match):
    out = dr.rows_query(ROWS, "sum", where, match=match)
    assert out["value"] == pytest.approx(value)
    assert [r["row_no"] for r in out["rows"]] == used


@pytest.mark.parametrize("where, value, skipped", [
    ({"path": "Concrete"}, 160.0, [4]),
    ({"path": "Bill 3"}, 160.0, [4]),
    ({"description": "Provisional item"}, None, [4]),
    ({"path": "Earthworks"}, 60.0, []),
])
def test_sum_names_a_row_without_a_figure_as_skipped_not_zero(where, value, skipped):
    out = dr.rows_query(ROWS, "sum", where)
    assert out["value"] == (pytest.approx(value) if value is not None else None)
    assert [r["row_no"] for r in out["skipped"]] == skipped


@pytest.mark.parametrize("where, n", [({"path": "Bill 2"}, 2), ({}, 4), ({"description": "none such"}, 0)])
def test_count(where, n):
    assert dr.rows_query(ROWS, "count", where)["value"] == n


@pytest.mark.parametrize("where, row_no", [({"description": "Blinding concrete"}, 3),
                                           ({"description": "backfill  TRENCH"}, 2)])
def test_lookup_returns_the_one_row_with_its_page(where, row_no):
    out = dr.rows_query(ROWS, "lookup", where)
    assert out["row"]["row_no"] == row_no
    assert out["row"]["page"] == row_no + 2


@pytest.mark.parametrize("where", [{"description": "trench"}, {"description": "none such"}])
def test_lookup_refuses_when_not_exactly_one_row(where):
    out = dr.rows_query(ROWS, "lookup", where, match="contains")
    assert "error" in out and "row" not in out


@pytest.mark.parametrize("a, b, diff", [
    ({"path": "Bill 3"}, {"path": "Bill 2"}, 100.0),
    ({"description": "Excavate trench"}, {"description": "Backfill trench"}, 40.0),
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
    ({"path": "Bill"}, None, 0),
])
def test_where_matches_whole_tokens_not_a_prefix(where, value, count):
    got = dr.rows_query(NUMBERED, "sum", where)["value"]
    assert got == (pytest.approx(value) if value is not None else None)
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
    (["1000", "2000", "3000"], 6000.0, dr.RECONCILED),
    (["1000", "2000", "3000"], 6001.0, dr.FAILED),       # the allowance does not grow per row
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
    assert dr.rows_query(ROWS, "filter", where, match="contains")["count"] == n


@pytest.mark.parametrize("bill, expect, printed, state", [
    ([BillRow(section="Earthworks", source_page=7, path=("Bill 2", "Earthworks"),
              fields={"description": "Excavate", "quantity": "10", "rate": "5", "total": "50"}),
      BillRow(section="Concrete", source_page=9,
              fields={"description": "Blinding", "quantity": "2", "rate": "80", "total": "150"},
              check="qty x rate != amount")],
     [(7, ("Bill 2", "Earthworks"), ""), (9, ("Concrete",), "qty x rate != amount")],
     210.0, dr.FAILED),
    ([BillRow(section="", source_page=3,
              fields={"description": "Provisional sum", "total": "500"})],
     [(3, (), "")], None, dr.UNCHECKED),
])
def test_bill_rows_keep_page_path_and_check(bill, expect, printed, state):
    rows = dr.rows_from_bill("doc-b", bill)
    assert [(r.page, r.path, r.check) for r in rows] == expect
    assert dr.reconcile_table(rows, printed).state == state


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


@pytest.mark.parametrize("sheets, printed, states, total", [
    ({"Bill 1": [["BILL No. 1 - SITE WORKS"], HEADER,
                 ["A", "Excavate to reduce levels", "m3", 120, 15, 1800],
                 ["B", "Fill with imported material", "m3", 40, 22.5, 900]]},
     {1: 2700.0}, {1: dr.RECONCILED}, 2700.0),
    ({"Bill 2": [["BILL No. 2 - CONCRETE"], HEADER,
                 ["A", "Blinding 50 thick", "m2", 300, 12, 3600]],
      "Bill 3": [["BILL No. 3 - MASONRY"], HEADER,
                 ["A", "Blockwork 200 thick", "m2", 50, 40, 2100]]},
     {1: 3600.0, 2: 2000.0}, {1: dr.RECONCILED, 2: dr.FAILED}, 5700.0),
])
def test_spreadsheet_public_path_rows_match_chunks(tmp_path, sheets, printed, states, total):
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
    assert {t: c.state for t, c in dr.reconcile_tables(rows, printed).items()} == states
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


# ── second verdict: RECONCILED is the printed total only; where is typed ──

def _amounts(amounts):
    return [_row(1, i + 1, f"Lump sum {i}", total=a) for i, a in enumerate(amounts)]


@pytest.mark.parametrize("amounts, printed", [
    (["1000"] * 100 + ["40"], 100_040.0),              # integer figures
    (["1000.00"] * 100 + ["0.95"], 100_000.95),        # two-decimal figures
])
def test_a_table_short_by_one_row_fails_and_the_whole_table_reconciles(amounts, printed):
    assert dr.reconcile_table(_amounts(amounts[:-1]), printed).state == dr.FAILED
    assert dr.reconcile_table(_amounts(amounts), printed).state == dr.RECONCILED


@pytest.mark.parametrize("amounts, printed, state", [
    (["1000.00"] * 100, 100_001.0, dr.FAILED),          # figures print cents: a 1.00 gap is not rounding
    (["1000.00"] * 100, 100_000.0, dr.RECONCILED),
    (["1000", "2000", "3000"], 6000.5, dr.FAILED),      # the total prints tenths
    (["1000", "2000", "3000"], 6000.0, dr.RECONCILED),
    (["1000.00", "999.70"], 2000.0, dr.FAILED),          # amounts print cents: a 0.30 gap is not rounding
    (["1000.00", "1000.00"], 2000.0, dr.RECONCILED),
])
def test_a_numeric_total_is_held_to_the_finest_digit_the_table_prints(amounts, printed, state):
    assert dr.reconcile_table(_amounts(amounts), printed).state == state


@pytest.mark.parametrize("amounts, printed, state", [
    (["1000", "2000", "3000.50"], "6,000", dr.RECONCILED),    # the total's own half unit, inclusive
    (["1000", "2000", "3000.51"], "6,000", dr.FAILED),
    (["1000", "2000", "3000.50"], "6,000.00", dr.FAILED),     # same sum, total printed to cents
    (["1000", "2000", "3000.004"], "6,000.00", dr.RECONCILED),
])
def test_a_printed_total_as_text_is_held_to_its_own_last_digit(amounts, printed, state):
    assert dr.reconcile_table(_amounts(amounts), printed).state == state


@pytest.mark.parametrize("rows, failed", [
    ([_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Backfill", "4", "2.5", "10")], 0),
    ([_row(1, 1, "Excavate", "10", "5", "50"), _row(1, 2, "Backfill", "4", "2.5", "11")], 1),
])
def test_without_a_printed_total_a_table_is_unchecked_whatever_its_rows_say(rows, failed):
    check = dr.reconcile_table(rows)
    assert (check.state, check.checked, check.failed) == (dr.UNCHECKED, 2, failed)


@pytest.mark.parametrize("text, half", [("100.25", 0.005), ("7.5", 0.05), ("1,200", 0.5)])
def test_half_unit_is_half_the_last_printed_digit(text, half):
    assert dr._half_unit(text) == pytest.approx(half)


FIGURES = [
    _row(1, 1, "Pipe", "1", "1,200", "1000", path=("Bill 1", "Section 2")),
    _row(1, 2, "Valve", "1", "200", "50", path=("Bill 2", "Section 1")),
    _row(1, 3, "Valve spindle", "2", "1 200.00", "2400", path=("BILL No.1",)),
]


@pytest.mark.parametrize("where, rows", [
    ({"rate": "200"}, [2]),                  # not the 200 inside 1,200
    ({"rate": "1200"}, [1, 3]),              # one number however it is grouped
    ({"rate": "1,200.00"}, [1, 3]),
    ({"total": "50.0"}, [2]),
    ({"rate": "abc"}, []),                   # a figure column never matches text
])
def test_a_figure_column_compares_numbers(where, rows):
    got = dr.rows_query(FIGURES, "filter", where)
    assert [r["row_no"] for r in got["rows"]] == rows
    assert got["match"] == "exact"


@pytest.mark.parametrize("where, rows", [
    ({"path": "Bill 1"}, [1]),               # one heading, whole
    ({"path": "1 Bill"}, []),                # order counts
    ({"path": "bill   1 > SECTION 2"}, [1]),  # the whole path, case and spacing normalised
    ({"path": "Section"}, []),               # no partial heading
    ({"path": "BILL No. 1"}, [3]),           # spacing inside a heading does not count
    ({"description": "Valve"}, [2]),         # whole cell: not "Valve spindle"
    ({"description": "valve  SPINDLE"}, [3]),
])
def test_a_text_column_compares_the_whole_cell(where, rows):
    got = dr.rows_query(FIGURES, "filter", where)
    assert [r["row_no"] for r in got["rows"]] == rows


@pytest.mark.parametrize("where, rows", [
    ({"description": "valve"}, [2, 3]),
    ({"path": "section"}, [1, 2]),
])
def test_contains_is_used_only_when_asked_and_is_reported(where, rows):
    got = dr.rows_query(FIGURES, "filter", where, match="contains")
    assert [r["row_no"] for r in got["rows"]] == rows
    assert got["match"] == "contains"
    assert dr.rows_query(FIGURES, "count", where)["value"] < len(rows)


@pytest.mark.parametrize("match", ["fuzzy", ""])
def test_unknown_match_is_an_error(match):
    got = dr.rows_query(FIGURES, "count", {"description": "Valve"}, match=match)
    assert "error" in got and got["matches"] == list(dr.MATCHES)


@pytest.mark.parametrize("value, match", [("", "contains"), ("   ", "contains"), ("", "exact"), (" ", "exact")])
def test_an_empty_value_matches_no_cell_even_an_empty_one(value, match):
    rows = [dr.DocRow("d", 1, n, fields={"description": d, "total": "1"}, extra={"Item": ""})
            for n, d in ((1, "Pipe"), (2, ""))]
    assert dr.rows_query(rows, "count", {"description": value}, match=match)["value"] == 0
    assert dr.rows_query(rows, "count", {"Item": value}, match=match)["value"] == 0


BIG = [_row(1, 1, "Tunnel", "1", "10,000,000,000,000,000", "10,000,000,000,000,000"),
       _row(1, 2, "Survey", "0", "100", "0")]


@pytest.mark.parametrize("where, rows", [
    ({"rate": 1e16}, [1]),                   # a JSON number is compared as the number it is
    ({"rate": 100}, [2]),
    ({"quantity": True}, []),                # a boolean is not the number 1
    ({"quantity": False}, []),               # nor 0
])
def test_a_figure_column_takes_a_number_value_and_refuses_a_boolean(where, rows):
    assert [r["row_no"] for r in dr.rows_query(BIG, "filter", where)["rows"]] == rows


@pytest.mark.parametrize("printed, state", [(True, dr.FAILED), (1, dr.RECONCILED), ("1", dr.RECONCILED)])
def test_a_boolean_printed_total_is_not_the_number_1(printed, state):
    assert dr.reconcile_table(_amounts(["1"]), printed).state == state


# ── third verdict: a total the reader cannot take is not FAILED; columns ──

@pytest.mark.parametrize("printed", [None, "", "  ", "	"])
@pytest.mark.parametrize("amounts", [["1000", "2000", "3000"], ["400.50", "99.50"]])
def test_a_blank_printed_total_is_no_total(amounts, printed):
    check = dr.reconcile_table(_amounts(amounts), printed)
    assert (check.state, check.total_printed, check.total_unread) == (dr.UNCHECKED, None, False)
    assert dr.reconcile_tables(_amounts(amounts), {1: printed})[1].state == dr.UNCHECKED


@pytest.mark.parametrize("printed, number", [
    ("AED 6,000", 6000.0), ("6,000 AED", 6000.0), ("$6,000", 6000.0), ("6,000/-", 6000.0),
    ("Rs. 6,000", 6000.0), ("6000.", 6000.0), ("1 200.00", 1200.0), ("AED 6,000.00", 6000.0),
])
def test_the_one_figure_of_a_printed_total_is_read_beside_its_marks(printed, number):
    amounts = ["1000", "2000", "3000"] if number == 6000.0 else ["1000", "200"]
    check = dr.reconcile_table(_amounts(amounts), printed)
    assert (check.state, check.total_printed, check.total_unread) == (dr.RECONCILED, number, False)
    short = dr.reconcile_table(_amounts(amounts[:-1]), printed)
    assert (short.state, short.total_printed) == (dr.FAILED, number)


@pytest.mark.parametrize("printed, amounts, state", [
    ("AED 6,000", ["1000", "2000", "3000.50"], dr.RECONCILED),   # held to the figure's own half unit
    ("AED 6,000", ["1000", "2000", "3000.51"], dr.FAILED),
    ("AED 6,000.00", ["1000", "2000", "3000.50"], dr.FAILED),
    ("AED 6,000.00", ["1000", "2000", "3000.004"], dr.RECONCILED),
])
def test_a_marked_total_is_held_to_its_figures_own_last_digit(printed, amounts, state):
    assert dr.reconcile_table(_amounts(amounts), printed).state == state


@pytest.mark.parametrize("printed", ["AED", "nil", "see summary", "6,000 / 7,000", "Bill 3: 6,000"])
@pytest.mark.parametrize("amounts", [["1000", "2000", "3000"], ["3", "6000"]])
def test_a_printed_total_with_no_one_figure_is_not_read_not_failed(amounts, printed):
    check = dr.reconcile_table(_amounts(amounts), printed)
    assert (check.state, check.total_printed, check.total_unread) == (dr.NOT_READ, None, True)
    assert check.rows == len(amounts)
    assert dr.reconcile_table([], printed).total_unread is True
    assert dr.reconcile_tables(_amounts(amounts), {1: printed})[1].state == dr.NOT_READ


# The state table of the module docstring, row by row, two inputs per state.
@pytest.mark.parametrize("amounts, printed, state", [
    (["1000", "2000"], None, dr.UNCHECKED),               # no printed total
    (["400.50", "99.50"], "  ", dr.UNCHECKED),
    (["1000", "2000"], "see summary", dr.NOT_READ),       # total the checker cannot parse
    (["400.50", "99.50"], "500 / 600", dr.NOT_READ),
    ([], 3000.0, dr.NOT_READ),                            # no rows
    ([], "500.00", dr.NOT_READ),
    (["1000", "2000"], "3,000", dr.RECONCILED),           # rows add up
    (["400.50", "99.50"], 500.0, dr.RECONCILED),
    (["1000", "2000"], "3,001", dr.FAILED),               # rows do not add up
    (["400.50", "99.50"], "AED 499.00", dr.FAILED),
])
def test_the_state_table(amounts, printed, state):
    assert dr.reconcile_table(_amounts(amounts), printed).state == state


@pytest.mark.parametrize("amounts, printed", [([""], 0.0), (["", ""], 0.0), ([""], 10.0)])
def test_a_table_with_no_amount_read_fails_and_sums_to_nothing(amounts, printed):
    check = dr.reconcile_table(_amounts(amounts), printed)
    assert (check.state, check.total_of_rows) == (dr.FAILED, None)


def _sheet_rows(tmp_path, body):
    from app.core import doc_index

    path = _xlsx(tmp_path, "cols.xlsx", {"Bill 1": [["BILL No. 1"], HEADER, *body]})
    return dr.rows_from_spreadsheet("doc-c", doc_index.spreadsheet_rows(path, "cols.xlsx", ".xlsx"))


BODY = [["A", "Excavate", "m3", 120, 15, 1800], ["B", "Valve", "nr", 4, 2.5, 11],
        ["C", "Blockwork", "m2", 50, 40, 2000]]


@pytest.mark.parametrize("where, n, column", [
    ({"Amount": 1800}, 1, {"Amount": "total"}),
    ({"Qty": 4}, 1, {"Qty": "quantity"}),
    ({"Description": "Excavate"}, 1, {"Description": "description"}),
    ({"Amount (AED)": 2000}, 1, {"Amount (AED)": "total"}),
    ({"Unit Rate": 40}, 1, {"Unit Rate": "rate"}),     # a whole alias wins over a part
    ({"Item": "B"}, 1, None),                          # the sheet's own unmapped header
    ({"total": 9999}, 0, None),
])
def test_a_where_on_the_sheets_printed_header_reads_its_column(tmp_path, where, n, column):
    out = dr.rows_query(_sheet_rows(tmp_path, BODY), "count", where)
    assert out["value"] == n
    assert out.get("columns") == column


@pytest.mark.parametrize("where, column", [
    ({"descriptoin": "Valve"}, "total"), ({"Remarks": "x"}, "total"),
    ({"Amount": 1800}, "Amont"), ({}, "Profit"),
])
def test_an_unknown_column_is_refused_by_name_not_counted_as_zero(tmp_path, where, column):
    rows = _sheet_rows(tmp_path, BODY)
    for op in ("count", "sum", "filter"):
        out = dr.rows_query(rows, op, where, column=column)
        bad = next(k for k in [*where, column] if k not in ("Amount", "total"))
        assert out["error"] == f"unknown column {bad!r}"
        assert {"total", "Item", "path"} <= set(out["columns"])
        assert out["rows"] == []


@pytest.mark.parametrize("where", [{"Amount": 1800, "total": 1800}, {"Qty": 4, "quantity": 4}])
def test_two_names_for_one_column_are_refused(where):
    rows = [_row(1, 1, "Excavate", "4", "450", "1800"), _row(1, 2, "Fill", "4", "5", "20")]
    out = dr.rows_query(rows, "count", where)
    assert "name the same column" in out["error"] and out["rows"] == []
    assert dr.rows_query(rows, "count", dict([next(iter(where.items()))]))["value"] >= 1


@pytest.mark.parametrize("column, value", [("Amount", 1800 + 11 + 2000), ("Qty", 120 + 4 + 50)])
def test_sum_reads_the_column_its_printed_header_names(tmp_path, column, value):
    out = dr.rows_query(_sheet_rows(tmp_path, BODY), "sum", column=column)
    assert out["value"] == value


@pytest.mark.parametrize("body, checks", [
    (BODY, ["", "qty x rate != amount", ""]),
    ([["A", "Formwork", "m2", 10, 5, 50], ["B", "Rebar", "t", 2, 7, 15]], ["", "qty x rate != amount"]),
])
def test_a_spreadsheet_row_carries_its_own_arithmetic_check(tmp_path, body, checks):
    assert [r.check for r in _sheet_rows(tmp_path, body)] == checks


# ── the doc_rows table ─────────────────────────────────────────────────────

@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    yield


@pytest.mark.parametrize("rows", [
    [dr.DocRow("doc-p", 1, 1, path=("Bill 2", "Earthworks"), page=7, check="",
               fields={"description": "Excavate", "quantity": "10", "rate": "5", "total": "50"}),
     dr.DocRow("doc-p", 1, 2, path=("Bill 2",), page=8, check="qty x rate != amount",
               fields={"description": "Fill", "quantity": "2", "rate": "3", "total": "9"})],
    [dr.DocRow("doc-s", 2, 1, sheet="Bill 3", sheet_row=4, chunk_ordinal=11,
               fields={"description": "Blockwork"}, extra={"Remarks": "by others"})],
])
def test_stored_rows_read_back_as_written(fresh_db, rows):
    doc = rows[0].document_id
    assert dr.store_document_rows("proj-1", doc, rows) == len(rows)
    assert dr.load_rows("proj-1", doc) == rows
    assert dr.load_rows("proj-other") == []


@pytest.mark.parametrize("keep_existing, kept", [(False, ["Second"]), (True, ["First"])])
def test_reindex_replaces_and_backfill_keeps(fresh_db, keep_existing, kept):
    first = [dr.DocRow("doc-r", 1, 1, fields={"description": "First"}),
             dr.DocRow("doc-r", 1, 2, fields={"description": "First b"})]
    dr.store_document_rows("proj-2", "doc-r", first)
    written = dr.store_document_rows(
        "proj-2", "doc-r", [dr.DocRow("doc-r", 1, 1, fields={"description": "Second"})],
        keep_existing=keep_existing)
    assert written == (0 if keep_existing else 1)
    got = [r.fields["description"] for r in dr.load_rows("proj-2", "doc-r")][:1]
    assert got == kept


@pytest.mark.parametrize("docs", [("doc-a", "doc-b"), ("doc-c",)])
def test_load_rows_scopes_to_the_document_asked(fresh_db, docs):
    for d in docs:
        dr.store_document_rows("proj-3", d, [dr.DocRow(d, 1, 1, fields={"description": d})])
    for d in docs:
        assert [r.document_id for r in dr.load_rows("proj-3", d)] == [d]
    assert len(dr.load_rows("proj-3")) == len(docs)
