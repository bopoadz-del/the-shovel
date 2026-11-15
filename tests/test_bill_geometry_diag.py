"""P-B3 counts-only geometry diagnostic over a plan's word boxes.

Every input is a synthetic page laid out here from word boxes (the layout
helpers of ``test_bill_table_geometry``); no document, bill or page is named.
Each decision is exercised with at least two inputs.
"""

from __future__ import annotations

import functools
import json

import pytest

from app.core import bill_derived as bd
from app.core import bill_geometry_diag as d
from app.core import bill_table_geometry as g
from scripts import bill_plan_once as bpo
from tests.test_bill_derived import _BILL, _bill_a, _by_page, _Docs, _source
from tests.test_bill_plan_once import _deps, _summary
from tests.test_bill_table_geometry import COLS_A, COLS_B, LINES_A, LINES_B, _page

# --- the shared tolerance ------------------------------------------------

@pytest.mark.parametrize("q,r,a,want", [
    (120, 45.5, 5460.0, True), (4, 1200, 4800, True),
    (120, 45.5, 5560.0, False), (3, 3, 10, False),
])
def test_reconciles_is_the_check_row_tolerance(q, r, a, want):
    assert g.reconciles(q, r, a) is want
    fields = {"quantity": str(q), "rate": str(r), "total": str(a)}
    assert (g.check_row(fields) == "") is want


# --- 1. figure cells per row ---------------------------------------------

def _row(**fields):
    row = g.BillRow(section="", source_page=0, fields=fields)
    row.check = g.check_row(fields)
    return row


@pytest.mark.parametrize("rows,want", [
    ([_row(description="x", quantity="12", rate="", total=""),
      _row(description="y", quantity="5", rate="", total="")],
     {"rows": 2, "quantity": 2, "rate": 0, "total": 0, "all_three": 0, "reconciled": 0}),
    ([_row(description="x", quantity="2", rate="3.00", total="6.00"),
      _row(description="y", quantity="2", rate="3.00", total="7.00"),
      _row(description="z", quantity="m", rate="", total="9")],
     {"rows": 3, "quantity": 2, "rate": 2, "total": 3, "all_three": 2, "reconciled": 1}),
])
def test_field_counts_count_numbers_not_text(rows, want):
    assert d.field_counts(rows) == want


# --- 2 and 5. header columns ----------------------------------------------

def _lines(words):
    return d._pages(words)


@pytest.mark.parametrize("cols,lines,by_field", [
    (COLS_A, LINES_A, {"description": 1, "quantity": 1, "rate": 1, "total": 1, "unit": 1}),
    (COLS_B, LINES_B, {"description": 1, "quantity": 1, "rate": 1, "total": 1, "unit": 1}),
])
def test_header_counts_resolve_each_field_and_count_unmapped(cols, lines, by_field):
    out = d.header_counts(_lines(_page(cols, lines, page=0)), g._resolver())
    assert out["header_lines"] == 1
    assert out["columns_by_field"] == by_field
    assert out["columns_unmapped"] == 1  # the item / ref column
    assert out["per_header"][0]["page"] == 0


@pytest.mark.parametrize("ref_cells,want", [
    (["101", "102", "103", "104"], 1),  # an unmapped column that holds figures
    (["A", "B", "C", "D"], 0),          # an unmapped column of references
])
def test_unmapped_column_holding_mostly_numbers_is_counted(ref_cells, want):
    lines = [[ref, f"Item {i}", "m", "2", "3.00", "6.00"] for i, ref in enumerate(ref_cells)]
    out = d.header_counts(_lines(_page(COLS_A, lines, page=0)), g._resolver())
    assert out["unmapped_mostly_numeric"] == want


@pytest.mark.parametrize("shift,spread", [(0, 0.0), (37, 37.0)])
def test_left_edge_spread_per_field_over_header_lines(shift, spread):
    second = [(label, x + shift) for label, x in COLS_A]
    words = _page(COLS_A, LINES_A, page=0) + _page(second, LINES_A, page=1)
    out = d.header_counts(_lines(words), g._resolver())
    assert out["header_lines"] == 2
    assert out["header_left_spread_px"]["rate"] == {"headers": 2, "spread": spread}
    assert out["header_left_spread_px"]["description"] == {"headers": 2, "spread": spread}


# --- 3. header-free bands ----------------------------------------------------

@pytest.mark.parametrize("header", [True, False])
def test_bands_find_priced_rows_without_any_header(header):
    """The count reads geometry only: with or without the header line, the
    three priced lines of the page are found and reconcile."""
    out = d.band_counts(_lines(_page(COLS_A, LINES_A, page=0, header=header)))
    assert out["lines_three_bands"] == 3
    assert out["lines_product"] == 3
    assert out["lines_product_nontrivial"] == 3
    assert out["pages_with_product_line"] == 1


@pytest.mark.parametrize("lines,three,product,nontrivial", [
    # aligned in bands but the amount is not the product
    ([["A", "Thing", "m", "2", "3.00", "7.00"]] * 3, 3, 0, 0),
    # a product only through a factor of one (trivial)
    ([["A", "Thing", "m", "1", "5.00", "5.00"]] * 3, 3, 3, 0),
])
def test_bands_separate_aligned_from_reconciling_lines(lines, three, product, nontrivial):
    out = d.band_counts(_lines(_page(COLS_A, lines, page=0)))
    assert (out["lines_three_bands"], out["lines_product"], out["lines_product_nontrivial"]) == (
        three, product, nontrivial)


@pytest.mark.parametrize("n_lines,want", [(2, 0), (3, 3)])
def test_a_band_needs_enough_lines_to_be_vertical(n_lines, want):
    lines = [["A", "Thing", "m", "2", "3.00", "6.00"]] * n_lines
    assert d.band_counts(_lines(_page(COLS_A, lines, page=0)))["lines_three_bands"] == want


# --- 4. labels -----------------------------------------------------------

def _events(words):
    events: list[g.LabelEvent] = []
    rows = g.extract_bill_rows(words, on_label=events.append)
    return rows, events


@pytest.mark.parametrize("lines,reason,figure_number", [
    # a figure-only line with no description: a measured row read as a label
    ([["", "", "m", "2", "3.00", "6.00"], "Next section"], g.LABEL_NO_DESCRIPTION, True),
    # a heading line: no figure cell at all
    (["Earthworks", ["A", "Dig", "m3", "2", "3.00", "6.00"]], g.LABEL_NO_FIGURE, False),
])
def test_label_events_carry_reason_and_figure_number(lines, reason, figure_number):
    _rows, events = _events(_page(COLS_A, lines, page=0))
    assert events[0].reason == reason and events[0].figure_number is figure_number
    assert events[0].page == 0 and events[0].lines == 1


@pytest.mark.parametrize("cols,lines,reason,figure_number", [
    # the margin label "Bill 3 ..." opens a chain under the item column that
    # no figure line closes
    (COLS_A, LINES_A, g.LABEL_UNCLOSED_CHAIN, False),
    # the margin label "Section 7 ..." prints its number under a figure column
    (COLS_B, LINES_B, g.LABEL_NO_DESCRIPTION, True),
])
def test_the_observer_changes_no_row(cols, lines, reason, figure_number):
    words = _page(cols, lines, page=0)
    plain = g.extract_bill_rows(words)
    observed, events = _events(words)
    assert observed == plain
    assert events == [g.LabelEvent(0, reason, 1, figure_number)]


@pytest.mark.parametrize("pre", [["Cover title"], ["Cover title", "Second line"]])
def test_labels_before_any_header_are_counted_by_reason(pre):
    words = _page(COLS_A, pre, page=0, header=False) + _page(COLS_A, LINES_A, page=1)
    _rows, events = _events(words)
    out = d.label_counts(events, [0, 1, 2])
    assert out["by_reason"][g.LABEL_BEFORE_HEADER] == len(pre)
    assert out["per_page_min"] == 0 and out["per_page_max"] == len(pre)
    assert out["label_lines"] == len(pre) + 1


# --- whole diagnostic, counts only ----------------------------------------

def _texts(words):
    return {w.text for w in words if any(ch.isalpha() for ch in w.text)}


@pytest.mark.parametrize("cols,lines", [(COLS_A, LINES_A), (COLS_B, LINES_B)])
def test_geometry_diag_carries_no_word_text(cols, lines):
    words = _page(cols, lines, page=0)
    out = d.geometry_diag(words, [0])
    blob = json.dumps(out)
    assert not any(t in blob for t in _texts(words) if len(t) > 5)
    assert out["fields"]["rows"] == len(lines) - 1
    assert out["fields"]["reconciled"] == len(lines) - 1


# --- through the plan script -------------------------------------------

def _real_plan(words, seen):
    return functools.partial(bd.plan_bill_sheet, page_source=_by_page(words, seen))


@pytest.mark.parametrize("pages", ["", "0:2"])
def test_plan_reports_geometry_from_saved_boxes_only(tmp_path, monkeypatch, pages):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=3)
    docs.install(monkeypatch, tmp_path)
    seen: list[int] = []
    lines: list[str] = []
    deps = _deps(docs.rows, _real_plan(words, seen), lines, classify=_BILL,
                 geometry=bpo.saved_geometry)
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path / "wd"), "BILL_PLAN_PAGES": pages}

    assert bpo.main_from_env(env, deps=deps) == 0

    rep = _summary(lines)
    n = 3 if not pages else 2
    assert len(seen) == n  # the diagnostic read no page again
    geo = rep["geometry"]
    assert geo["pages_read"] == n and geo["pages_missing"] == 0
    assert geo["fields"]["rows"] == rep["rows"]
    assert "redacted" not in json.dumps(geo)


@pytest.mark.parametrize("missing", [[1], [0, 2]])
def test_saved_geometry_counts_missing_pages_and_reads_none(tmp_path, missing):
    words, _, _ = _bill_a()
    wd = tmp_path / "wd"
    wd.mkdir()
    for index in range(3):
        if index in missing:
            continue
        page_words = [w for w in words if w.page == index]
        (wd / f"page_{index:05d}.json").write_text(json.dumps(
            {"source": "ocr", "words": [w.__dict__ for w in page_words]}))
    out = bpo.saved_geometry(str(wd), 0, 3)
    assert out["pages_missing"] == len(missing)
    assert out["pages_read"] == 3 - len(missing)


@pytest.mark.parametrize("exc", [RuntimeError, KeyError])
def test_a_failing_diagnostic_leaves_the_plan_ok(tmp_path, monkeypatch, exc):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=3)
    docs.install(monkeypatch, tmp_path)
    lines: list[str] = []

    def boom(*_a):
        raise exc("secret text from a page")

    deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL, geometry=boom)
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path / "wd")}
    assert bpo.main_from_env(env, deps=deps) == 0
    rep = _summary(lines)
    assert rep["ok"] is True
    assert rep["geometry"] == {"error_class": exc.__name__}
    assert "secret" not in "".join(lines)
