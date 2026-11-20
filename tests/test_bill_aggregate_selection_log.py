"""The bill count/sum tool logs which term set and which rows each call
selected, as digests only, so two asks can be compared from the log.

Every test builds its own invented bill; nothing here is a real bill, row or
question.
"""

from __future__ import annotations

import logging
import re

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_table_geometry as g


def _row(path, page, ref, desc, unit, qty, amount):
    return g.BillRow(section=path[-1], source_page=page,
                     fields={"description": desc, "unit": unit, "quantity": qty, "total": amount},
                     extra={"Item": ref}, path=tuple(path))


WALLS = ("Masonry", "Walls")
FLOORS = ("Masonry", "Floors")

BILL_A = [
    _row(WALLS, 1, "W1", "Blockwork wall hollow 200 thick", "m2", "120", "4800"),
    _row(WALLS, 1, "W2", "Blockwork wall solid 100 thick", "m2", "35", "1750"),
    _row(FLOORS, 2, "S1", "Screed to floor 50 thick", "m2", "410", "6150"),
]
BILL_B = [
    _row(("Drainage", "Runs"), 3, "D1", "Clay pipe 150 diameter in trench", "m", "73", "2190"),
    _row(("Drainage", "Runs"), 3, "D2", "Clay pipe 225 diameter in trench", "m", "18", "900"),
    _row(("Drainage", "Chambers"), 4, "C1", "Inspection chamber 600 deep", "nr", "6", "3600"),
]

#: Every word and figure the invented bills print: none may reach the log.
_PRINTED = {w for bill in (BILL_A, BILL_B) for r in bill
            for v in (*r.fields.values(), *r.path, *r.extra.values())
            for w in re.findall(r"[^\W_]+", v) if len(w) > 2}


def _rows(tmp_path, bill, name):
    path = tmp_path / f"{name}.xlsx"
    g.write_bill_workbook(bill, str(path))
    return agg.read_sheet_rows(str(path))


def _line(rows, *, terms, section=None, measure=agg.MEASURE_QUANTITY, doc="d1"):
    result = agg.aggregate(rows, terms=terms, section=section, measure=measure)
    out = {"status": "success", "bills": [{"document_id": doc, "status": "success", **result}]}
    return agg.selection_log_line(out, terms=terms, section=section, measure=measure)


def _field(line, key):
    return re.search(rf"{key}=(\S+)", line).group(1)


def _rows_digest(line):
    return re.search(r"rows=(\S+?)\]", line).group(1)


@pytest.mark.parametrize("bill, one, two", [
    (BILL_A, "Blockwork wall", "WALL blockwork"),
    (BILL_B, "clay pipe trench", ["trench", "Pipe", "clay"]),
], ids=["a", "b"])
def test_same_word_set_in_any_order_or_case_logs_the_same(tmp_path, bill, one, two):
    rows = _rows(tmp_path, bill, "x")
    a, b = _line(rows, terms=one), _line(rows, terms=two)
    assert _field(a, "terms") == _field(b, "terms")
    assert _rows_digest(a) == _rows_digest(b)


@pytest.mark.parametrize("bill, narrow, wide, n_narrow, n_wide", [
    (BILL_A, "blockwork hollow", "blockwork", 1, 2),
    (BILL_B, "pipe 150", "pipe", 1, 2),
], ids=["a", "b"])
def test_a_different_term_set_and_row_set_log_differently(tmp_path, bill, narrow, wide, n_narrow, n_wide):
    rows = _rows(tmp_path, bill, "x")
    a, b = _line(rows, terms=narrow), _line(rows, terms=wide)
    assert _field(a, "terms") != _field(b, "terms")
    assert _rows_digest(a) != _rows_digest(b)
    assert (_field(a, "count").rstrip("]"), _field(b, "count").rstrip("]")) == (str(n_narrow), str(n_wide))


@pytest.mark.parametrize("bill, one, two", [
    (BILL_A, "blockwork", "wall"),
    (BILL_B, "clay", "diameter"),
], ids=["a", "b"])
def test_different_terms_that_pick_the_same_rows_share_a_row_digest(tmp_path, bill, one, two):
    """The case the log exists to separate: the term choice differs, the
    selection does not."""
    rows = _rows(tmp_path, bill, "x")
    a, b = _line(rows, terms=one), _line(rows, terms=two)
    assert _field(a, "terms") != _field(b, "terms")
    assert _rows_digest(a) == _rows_digest(b) != "-"


@pytest.mark.parametrize("bill, terms, section, measure", [
    (BILL_A, "blockwork wall", "walls", agg.MEASURE_AMOUNT),
    (BILL_B, "pipe", "drainage runs", agg.MEASURE_QUANTITY),
], ids=["a", "b"])
def test_the_line_holds_no_printed_word_or_figure(tmp_path, bill, terms, section, measure):
    line = _line(_rows(tmp_path, bill, "x"), terms=terms, section=section, measure=measure)
    logged = set(re.findall(r"[^\W_]+", line))
    assert not (logged & _PRINTED), logged & _PRINTED
    assert _field(line, "section").startswith(f"{len(agg._terms(section))}:")


@pytest.mark.parametrize("bill, terms", [(BILL_A, "nothing matches"), (BILL_B, "absent word")],
                         ids=["a", "b"])
def test_no_match_and_no_section_are_logged_as_empty(tmp_path, bill, terms):
    line = _line(_rows(tmp_path, bill, "x"), terms=terms)
    assert _field(line, "section") == "0:-"
    assert _field(line, "count") == "0"
    assert _rows_digest(line) == "-"


@pytest.fixture
def project(tmp_path, monkeypatch):
    from app.core import bill_derived, projects

    derived = {"derivation": bill_derived.DERIVATION}
    docs = []
    for name, bill in (("a", BILL_A), ("b", BILL_B)):
        path = tmp_path / f"{name}.xlsx"
        g.write_bill_workbook(bill, str(path))
        docs.append({"id": f"doc-{name}", "original_name": f"{name}.xlsx", "path": str(path),
                     "metadata": {**derived, "derived_from": f"src-{name}"}})
    docs.append({"id": "doc-gone", "original_name": "gone.xlsx", "path": "",
                 "metadata": {**derived, "derived_from": "src-gone"}})
    monkeypatch.setattr(projects, "list_documents", lambda pid, **_kw: docs)
    monkeypatch.setattr(projects, "materialize_document_path", lambda d: d["path"])


def _logged(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == agg.__name__ and r.getMessage().startswith("bill_rows_aggregate ")]


@pytest.mark.parametrize("document_id, bills, statuses", [
    (None, 3, ["doc-a success", "doc-b success", f"doc-gone error reason={agg.NO_READABLE_BYTES}"]),
    ("src-b", 1, ["doc-b success"]),
], ids=["every-bill", "one-bill"])
def test_each_project_call_logs_one_line_naming_every_bill(project, caplog, document_id, bills, statuses):
    caplog.set_level(logging.INFO, logger=agg.__name__)
    agg.aggregate_project_bills("p", terms="pipe", document_id=document_id)
    lines = _logged(caplog)
    assert len(lines) == 1
    assert _field(lines[0], "bills") == str(bills)
    for status in statuses:
        assert f"[{status}" in lines[0]


@pytest.mark.parametrize("document_id", ["src-none", "doc-none"])
def test_a_call_with_no_sheet_in_scope_is_logged_with_its_reason(project, caplog, document_id):
    caplog.set_level(logging.INFO, logger=agg.__name__)
    out = agg.aggregate_project_bills("p", terms="pipe", document_id=document_id)
    assert out == {"status": "error", "reason": agg.NO_DERIVED_SHEET}
    lines = _logged(caplog)
    assert len(lines) == 1
    assert f"status=error reason={agg.NO_DERIVED_SHEET} bills=0" in lines[0]
