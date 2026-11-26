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


# ── the bill's own skeleton ─────────────────────────────────────────────────

def _footer(label, words=("Tender", "Issue", "Page")):
    return [*words, label]


def _doc(bodies, labels, words=("Tender", "Issue", "Page")):
    """Pages as lines of tokens: each body, then the running footer with the
    page's label (None: a divider page with no footer)."""
    return [[*body, *([_footer(lb, words)] if lb else [])] for body, lb in zip(bodies, labels)]


def _priced_doc(prefix, words, amounts):
    """Two parts, each with content pages, a collection page, then a summary.

    ``amounts[k]`` are part k's page amounts; the collection lists them and
    the summary carries their sum (or what ``amounts`` says in a
    ``("printed", value)`` override at the end)."""
    bodies, labels, summary = [], [], []
    for k, pages in enumerate(amounts, 1):
        printed = None
        if pages and isinstance(pages[-1], tuple):
            printed, pages = pages[-1][1], pages[:-1]
        coll = []
        for i, a in enumerate(pages, 1):
            bodies.append([["Item", "work", str(a)]])
            labels.append(f"{prefix}/{k}/{i}")
            coll.append(["From", f"{prefix}/{k}/{i}", "Included" if a is None else f"{a:,.2f}"])
        bodies.append(coll + [["Carried"]])
        labels.append(f"{prefix}/{k}/{len(pages) + 1}")
        total = printed if printed is not None else sum(a for a in pages if a is not None)
        summary.append(["Part", str(k), f"{prefix}/{k}/{len(pages) + 1}", f"{total:,.0f}"])
    bodies.append(summary + [["SUB-TOTAL", "0"]])
    labels.append(f"{prefix.upper()}S/1")
    return _doc(bodies, labels, words)


@pytest.mark.parametrize("prefix,words", [("D", ("Tender", "Issue", "Page")),
                                          ("Q", ("Contract", "Sheet"))])
def test_page_labels_read_the_running_footer(prefix, words):
    pages = _doc([[["Cover"]], [["Index", f"{prefix}/1/2"]], [["body"]], [["body"]]],
                 [None, None, f"{prefix}/1/1", f"{prefix}/1/2"], words)
    # the index names a page in its body, not on a running line: not its own label
    assert dr.page_labels(pages) == {3: f"{prefix}/1/1", 4: f"{prefix}/1/2"}


@pytest.mark.parametrize("labels,share,expect", [
    # the footer words recur on 2 of 4 pages: at the share, kept; above it, dropped
    (["A/1", "A/2", None, None], 0.5, {1: "A/1", 2: "A/2"}),
    (["B/7", "B/8", None, None], 0.51, {}),
    (["12/05/2023", "02/06/2023", None, None], 0.5, {}),   # dates are not labels
])
def test_page_label_needs_a_recurring_line_and_a_letter(labels, share, expect):
    pages = _doc([[["x"]], [["y"]], [["z"]], [["w"]]], labels)
    assert dr.page_labels(pages, furniture_share=share) == expect
    assert dr.page_labels([]) == {}


@pytest.mark.parametrize("labels,expect", [(["E/1", "E/1", "E/2"], {3: "E/2"}),
                                           (["F/3", "F/4", "F/4"], {1: "F/3"})])
def test_a_label_two_pages_claim_is_dropped(labels, expect):
    pages = _doc([[["a"]], [["b"]], [["c"]]], labels)
    assert dr.page_labels(pages) == expect


@pytest.mark.parametrize("p", ["K", "M/2"])
def test_page_listings_take_the_figure_after_the_label(p):
    labels = {1: f"{p}/1", 2: f"{p}/2", 3: f"{p}/3"}
    pages = [[["a"]], [["b"]], [["From", f"{p}/1", "1,250.50"], ["From", f"{p}/2", "Included"],
                                 ["Both", f"{p}/1", f"{p}/2", "9"], ["Self", f"{p}/3", "4"]]]
    assert dr.page_listings(pages, labels) == {3: [(f"{p}/1", 1250.5), (f"{p}/2", None)]}
    one = [[["a"]], [["b"]], [["From", f"{p}/1", "7"]]]
    assert dr.page_listings(one, labels) == {}


@pytest.mark.parametrize("prefix,words,amounts", [
    ("D", ("Tender", "Issue", "Page"), [[100.0, 250.0], [40.0, None, 60.0]]),
    ("R", ("Contract", "Sheet"), [[7.0, 8.0, 9.0], [1000.0, 2000.0]]),
])
def test_bill_skeleton_reads_parts_and_printed_totals(prefix, words, amounts):
    pages = _priced_doc(prefix, words, amounts)
    root, parts = dr.bill_skeleton(pages)
    assert root == len(pages)
    assert [p.printed_total for p in parts] == [sum(a for a in pg if a is not None) for pg in amounts]
    first = len(amounts[0]) + 1
    assert parts[0].pages == tuple(range(1, first + 1))
    assert parts[0].summary_page == first
    assert [a for _, a in parts[1].listed] == amounts[1]
    assert parts[1].pages[0] == first + 1


@pytest.mark.parametrize("labels", [["P/1", "P/2"], ["H/4/1", "H/4/2"]])
def test_bill_skeleton_without_listing_pages_is_empty(labels):
    pages = _doc([[["a"]], [["b", labels[0], "5"]]], labels)
    assert dr.bill_skeleton(pages) == (None, [])
    assert dr.bill_skeleton([]) == (None, [])


def _page_row(n, page, desc, qty="", rate="", total=""):
    return dr.DocRow(document_id="doc-x", table_no=1, row_no=n, page=page,
                     fields={"description": desc, "quantity": qty, "rate": rate, "total": total})


@pytest.mark.parametrize("words", [("Carried", "To collection"), ("", "Page total")])
def test_page_total_lines_are_the_running_sum_whatever_they_say(words):
    rows = [_page_row(1, 5, "a", "2", "10", "20"), _page_row(2, 5, "b", "1", "30", "30"),
            _page_row(3, 5, words[0], total="50"),
            _page_row(4, 6, "lump sum", total="50"),          # first on its page: an item
            _page_row(5, 6, "c", "1", "5", "5"),
            _page_row(6, 6, "lump sum 2", total="40"),        # not the running 55: an item
            _page_row(7, 6, words[1], total="95")]
    assert dr.page_total_lines(rows) == {2, 6}
    assert dr.page_total_lines([]) == set()



@pytest.mark.parametrize("left, item, page_no", [(70, 15, 9), (3, 400, 22)])
def test_page_total_lines_count_only_the_items_on_their_own_page(left, item, page_no):
    # The page before prints items but no carried line; its amounts must not
    # be counted into the next page's total, nor make that page's first
    # amount-only row a total.
    rows = [_page_row(1, page_no - 1, "a", "1", str(left), str(left)),
            _page_row(2, page_no, "carried in", total=str(left)),     # first on its page
            _page_row(3, page_no, "b", "1", str(item), str(item)),
            _page_row(4, page_no, "Carried", total=str(item + left)),  # own page runs to item+left
            _page_row(5, page_no, "Carried", total=str(item + left))]
    assert dr.page_total_lines(rows) == {3}
    only = [_page_row(1, page_no - 1, "a", "1", str(left), str(left)),
            _page_row(2, page_no, "b", "1", str(item), str(item)),
            _page_row(3, page_no, "Carried", total=str(item))]
    assert dr.page_total_lines(only) == {2}

def _rows_for(pages, amounts):
    """Item rows on each content page of a ``_priced_doc``, one per amount,
    plus that page's carried total line."""
    rows, n, page = [], 0, 0
    for part in amounts:
        for a in part:
            page += 1
            if a is not None:
                n += 1
                rows.append(_page_row(n, page, "item", "1", str(a), str(a)))
                n += 1
                rows.append(_page_row(n, page, "Carried", total=str(a)))
        page += 1  # the collection page
    return rows


@pytest.mark.parametrize("amounts", [[[100.0, 250.0], [40.0, None, 60.0]],
                                     [[7.0, 8.0, 9.0], [1000.0, 2000.0]]])
def test_reconcile_skeleton_states(amounts):
    pages = _priced_doc("D", ("Tender", "Issue", "Page"), amounts)
    _, parts = dr.bill_skeleton(pages)
    rows = _rows_for(pages, amounts)
    checks = dr.reconcile_skeleton(parts, rows)
    assert [c.state for c in checks] == [dr.RECONCILED, dr.RECONCILED]
    assert [c.total_of_rows for c in checks] == [p.printed_total for p in parts]
    assert checks[0].pages_agree == len(amounts[0]) and checks[0].pages_disagree == 0
    assert checks[1].pages_unpriced == sum(a is None for a in amounts[1])
    # drop one item of the first part: the part fails, the page disagrees
    lost = dr.reconcile_skeleton(parts, rows[2:])
    assert lost[0].state == dr.FAILED and lost[0].pages_disagree == 1
    assert lost[1].state == dr.RECONCILED
    # no rows on a part's pages: NOT READ, never RECONCILED
    assert [c.state for c in dr.reconcile_skeleton(parts, [])] == [dr.NOT_READ, dr.NOT_READ]


def test_reconcile_skeleton_part_without_printed_total_is_unchecked():
    part = dr.SkeletonPart("Z/1/2", 2, (1, 2), None, ((1, None),))
    rows = [_page_row(1, 1, "a", "1", "5", "5")]
    (c,) = dr.reconcile_skeleton([part], rows)
    assert c.state == dr.UNCHECKED and c.pages_unpriced == 1 and c.listed_sum is None
    # "5" against 7: past half a unit of each printed figure
    priced = dr.SkeletonPart("Z/1/2", 2, (1, 2), 7.0, ((1, 7.0),))
    (d,) = dr.reconcile_skeleton([priced], rows)
    assert d.state == dr.FAILED and d.pages_disagree == 1
    # "5" against 5.4: within the rounding "5" prints
    near = dr.SkeletonPart("Z/1/2", 2, (1, 2), 5.4, ((1, 5.4),))
    (e,) = dr.reconcile_skeleton([near], rows)
    assert e.state == dr.RECONCILED and e.pages_agree == 1


@pytest.mark.parametrize("words,expect", [
    ([{"text": "B", "left": 50, "top": 10, "height": 10}, {"text": "A", "left": 5, "top": 12, "height": 10},
      {"text": "C", "left": 5, "top": 30, "height": 10}], [["A", "B"], ["C"]]),
    ([{"text": "x", "left": 0, "top": 0, "height": 10}, {"text": "y", "left": 0, "top": 7, "height": 10}],
     [["x"], ["y"]]),
])
def test_lines_of_words(words, expect):
    assert dr.lines_of_words(words) == expect
    assert dr.lines_of_words([]) == []


@pytest.mark.parametrize("extra,amounts", [
    (2, [[1.0, 2.0], [5.0, 6.0], [8.0, 9.0]]),
    (3, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]),
])
def test_bill_skeleton_summary_is_the_largest_unlisted_listing(extra, amounts):
    pages = _priced_doc("D", ("Tender", "Issue", "Page"), amounts)
    summary = len(pages)
    # a second, smaller listing nothing names (a rate-only schedule's collection)
    side = [[["rates"]] for _ in range(extra)]
    side.append([["From", f"X/9/{i}", "0"] for i in range(1, extra + 1)])
    pages += _doc(side, [f"X/9/{i}" for i in range(1, extra + 2)])
    root, parts = dr.bill_skeleton(pages)
    assert root == summary and len(parts) == len(amounts)


@pytest.mark.parametrize("zero", ["0", "0.00"])
def test_a_zero_amount_first_on_its_page_is_not_a_total_line(zero):
    rows = [_page_row(1, 3, "Provisional", total=zero), _page_row(2, 3, "a", "1", "4", "4"),
            _page_row(3, 3, "Carried", total="4")]
    assert dr.page_total_lines(rows) == {2}


@pytest.mark.parametrize("amounts", [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0, 7.0], [8.0, 9.0]]])
def test_bill_skeleton_survives_collections_that_name_each_other(amounts):
    pages = _priced_doc("D", ("Tender", "Issue", "Page"), amounts)
    c1, c2 = len(amounts[0]) + 1, len(amounts[0]) + len(amounts[1]) + 2
    pages[c1 - 1].insert(0, ["From", f"D/2/{len(amounts[1]) + 1}", "0"])
    pages[c2 - 1].insert(0, ["From", f"D/1/{len(amounts[0]) + 1}", "0"])
    root, parts = dr.bill_skeleton(pages)
    assert root == len(pages) and len(parts) == 2
    assert c2 in parts[0].pages and c1 in parts[1].pages


@pytest.mark.parametrize("amounts,extra", [([[100.0, 250.0], [40.0, 60.0]], 45.0),
                                           ([[7.0, 8.0], [1000.0, 2000.0, 3.0]], 302.0)])
def test_summary_totals_read_sub_total_addition_and_grand_total(amounts, extra):
    pages = _priced_doc("D", ("Tender", "Issue", "Page"), amounts)
    sub = sum(sum(p) for p in amounts)
    pages[-1][-2:-1] = [["Sub", f"{sub:,.0f}"], ["Allowance", "18%", f"{extra:,.0f}"],
                        ["Total", f"{sub + extra:,.0f}"], ["Wrong", f"{sub + 2 * extra:,.0f}"]]
    assert dr.summary_totals(pages) == [sub, sub + extra]


@pytest.mark.parametrize("amounts", [[[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0, 7.0, 8.0]]])
def test_summary_totals_none_without_a_summary(amounts):
    assert dr.summary_totals(_doc([[["a"]], [["b"]]], ["P/1", "P/2"])) == []
    pages = _priced_doc("D", ("Tender", "Issue", "Page"), amounts)
    sub = sum(sum(p) for p in amounts)
    # a figure before the first entry (a heading number) is not a total
    pages[-1].insert(0, ["Section", "0"])
    pages[-1][-2:-1] = [["Sub", f"{sub:g}"]]
    assert dr.summary_totals(pages) == [sub]


# additions more than one printed unit apart: whole numbers one apart are
# within half a unit of each other, so either could be the other's total
@pytest.mark.parametrize("amounts,adds", [([[1.0, 2.0], [3.0, 4.0]], (5.0, 9.0)),
                                          ([[10.0], [20.0, 30.0]], (7.0, 12.0))])
def test_summary_totals_subtotal_counts_from_the_previous_total(amounts, adds):
    pages = _priced_doc("D", ("Tender", "Issue", "Page"), amounts)
    sub = sum(sum(p) for p in amounts)
    a, b = adds
    pages[-1][-2:-1] = [["Sub", f"{sub:g}"], ["Add", f"{a:g}"], ["Add", f"{b:g}"],
                        ["Adds", f"{a + b:g}"], ["See", "Q/7/1", f"{sub + a + b:g}"],
                        ["Grand", f"{sub + a + b:g}"]]
    # the line naming another reference is not a total, even when its figure matches
    assert dr.summary_totals(pages) == [sub, a + b, sub + a + b]


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
