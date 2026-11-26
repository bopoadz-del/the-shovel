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
