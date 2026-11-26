"""rows_backfill_once: live Sources wiring and the plan/execute entry point."""
from __future__ import annotations

import json

import pytest

from app.core import doc_rows_backfill as rb
from app.core.bill_table_geometry import BillRow
from app.core.doc_index import _spreadsheet_row_chunks
from scripts import rows_backfill_once as rbo

SHEET_EXTS = (".xlsx", ".xls", ".csv")


@pytest.mark.parametrize("name,saved,kind", [
    ("a.xlsx", 0, rb.SHEET), ("B.CSV", 7, rb.SHEET),
    ("c.pdf", 3, rb.BILL), ("d.PDF", 1, rb.BILL),
    ("e.pdf", 0, ""), ("f.docx", 4, ""), ("", 0, ""), ("noext", 9, ""),
])
def test_document_kind_reads_extension_and_saved_pages(name, saved, kind):
    assert rbo.document_kind(name, saved, SHEET_EXTS) == kind


@pytest.mark.parametrize("rows,stored,n", [
    (["a", "b", "c"], ["a", "b", "c", "x"], 3),
    (["a", "b", "c"], ["a", "c"], 2),
    (["a", "a"], ["a"], 2),
    (["a", "b"], [], 0),
    ([], ["a"], 0),
])
def test_stored_row_chunk_count_counts_rows_whose_chunk_is_stored(rows, stored, n):
    assert rbo.stored_row_chunk_count(rows, stored) == n


def _csv(path, n, start=1):
    lines = ["Description,Quantity,Unit,Rate,Amount"]
    for i in range(n):
        q = start + i
        lines.append(f"Concrete grade {q},{q},m3,10,{q * 10}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _bill(n):
    return [BillRow(section="Works", source_page=1 + i,
                    fields={"description": f"Line {i}", "quantity": "2", "rate": "5", "total": "10"})
            for i in range(n)]


class _World:
    """The stores a LiveSources reads, built from the test's own inputs."""

    def __init__(self, tmp_path, docs, chunks_drop=0, saved=None, bills=None):
        self.docs = docs
        self.paths = {}
        self.chunks = {}
        for d in docs:
            if d["original_name"].endswith(".csv"):
                p = _csv(tmp_path / f"{d['id']}.csv", d["_n"], d.get("_start", 1))
                self.paths[d["id"]] = p
                made = _spreadsheet_row_chunks(p, d["original_name"], ".csv")
                self.chunks[d["id"]] = made[chunks_drop:]
        self.saved = saved or {}
        self.bills = bills or {}
        self.stored = {}

    def sources(self):
        return rbo.LiveSources(
            data_dir="/data",
            list_documents=lambda pid: list(self.docs),
            load_rows=lambda pid, did: self.stored.get(did, []),
            materialize=lambda doc: self.paths.get(doc.get("id")),
            chunk_texts=lambda pid, did: self.chunks.get(did, []),
            cached_pages=lambda wd: self.saved.get(wd.replace("\\", "/").rsplit("/", 1)[-1], 0),
            read_saved_rows=lambda wd, n: self.bills.get(wd.replace("\\", "/").rsplit("/", 1)[-1], []),
        )

    def store(self, project_id, doc_id, rows, *, keep_existing=False):
        if keep_existing and self.stored.get(doc_id):
            return 0
        self.stored[doc_id] = list(rows)
        return len(rows)


def _entry(plan, doc_id):
    return next(d for d in plan["documents"] if d["doc_id"] == doc_id)


@pytest.mark.parametrize("n,start", [(3, 1), (5, 40)])
def test_live_sheet_is_planned_when_every_row_chunk_is_stored(tmp_path, n, start):
    w = _World(tmp_path, [{"id": "s1", "original_name": "rates.csv", "_n": n, "_start": start}])
    plan = rb.plan_rows_backfill("p", w.sources())
    e = _entry(plan, "s1")
    assert (e["kind"], e["skip_reason"], e["rows_to_write"], e["row_chunks"]) == (rb.SHEET, "", n, n)


@pytest.mark.parametrize("n,drop", [(3, 1), (4, 4)])
def test_live_sheet_with_a_missing_row_chunk_is_skipped(tmp_path, n, drop):
    w = _World(tmp_path, [{"id": "s1", "original_name": "rates.csv", "_n": n}], chunks_drop=drop)
    e = _entry(rb.plan_rows_backfill("p", w.sources()), "s1")
    assert (e["skip_reason"], e["rows_to_write"], e["row_chunks"]) == (rb.CHUNK_MISMATCH, 0, n - drop)


@pytest.mark.parametrize("pages,saved,reason,rows", [
    (4, 4, "", 4), (6, 6, "", 6), (5, 4, rb.PAGES_MISSING, 0), (None, 3, rb.PAGES_MISSING, 0),
])
def test_live_bill_reads_saved_boxes_only_when_every_page_is_saved(tmp_path, pages, saved, reason, rows):
    w = _World(tmp_path, [{"id": "b1", "original_name": "bill.pdf", "pages_in_file": pages}],
               saved={"b1": saved}, bills={"b1": _bill(saved)})
    e = _entry(rb.plan_rows_backfill("p", w.sources()), "b1")
    assert (e["kind"], e["skip_reason"], e["rows_to_write"]) == (rb.BILL, reason, rows)


@pytest.mark.parametrize("pages_in_file", [True, "4", 4.0])
def test_page_count_that_is_not_an_int_is_no_coverage(tmp_path, pages_in_file):
    w = _World(tmp_path, [{"id": "b1", "original_name": "bill.pdf", "pages_in_file": pages_in_file}],
               saved={"b1": 4}, bills={"b1": _bill(4)})
    e = _entry(rb.plan_rows_backfill("p", w.sources()), "b1")
    assert (e["pages_cached"], e["pages_in_file"], e["skip_reason"]) == (4, 0, rb.PAGES_MISSING)


@pytest.mark.parametrize("docs,listed", [
    ([{"id": "x", "original_name": "notes.docx"}, {"id": "b", "original_name": "bill.pdf"},
      {"id": "s", "original_name": "rates.csv", "_n": 2}], ["s"]),
    ([{"id": "", "original_name": "rates.csv", "_n": 2},
      {"id": "s2", "original_name": "more.csv", "_n": 1}], ["s2"]),
])
def test_live_documents_lists_only_table_documents_with_an_id(tmp_path, docs, listed):
    w = _World(tmp_path, docs)
    assert [d.doc_id for d in w.sources().documents("p")] == listed


@pytest.mark.parametrize("meta,derived", [
    ({"derived_from": "b1"}, "b1"), ({}, ""), (None, ""),
])
def test_live_documents_carry_derived_from_from_metadata(tmp_path, meta, derived):
    doc = {"id": "s", "original_name": "rates.csv", "_n": 1}
    if meta is not None:
        doc["metadata"] = meta
    w = _World(tmp_path, [doc])
    (d,) = w.sources().documents("p")
    assert d.derived_from == derived


def _summary(lines):
    (line,) = lines
    assert line.startswith(rbo.SUMMARY_PREFIX)
    return json.loads(line[len(rbo.SUMMARY_PREFIX):])


@pytest.mark.parametrize("env,error", [
    ({}, rbo.ERR_MISSING_PROJECT), ({"ROWS_BACKFILL_PROJECT_ID": "  "}, rbo.ERR_MISSING_PROJECT),
    ({"ROWS_BACKFILL_PROJECT_ID": "p", "ROWS_BACKFILL_EXECUTE_HASH": "abc"}, rbo.ERR_BAD_HASH),
    ({"ROWS_BACKFILL_PROJECT_ID": "p", "ROWS_BACKFILL_EXECUTE_HASH": "A" * 64}, rbo.ERR_BAD_HASH),
])
def test_main_refuses_bad_selectors_before_reading_anything(env, error):
    lines = []

    class _Boom:
        def documents(self, project_id):
            raise AssertionError("read before the selectors were checked")

    assert rbo.main_from_env(env, sources=_Boom(), log=lines.append) == 2
    assert _summary(lines) == {"ok": False, "error": error}


@pytest.mark.parametrize("n", [2, 5])
def test_main_plan_then_execute_writes_and_completes(tmp_path, n):
    w = _World(tmp_path, [{"id": "s1", "original_name": "rates.csv", "_n": n}])
    lines = []
    env = {"ROWS_BACKFILL_PROJECT_ID": "p"}
    assert rbo.main_from_env(env, sources=w.sources(), log=lines.append, store=w.store) == 0
    plan = _summary(lines)
    assert (plan["mode"], plan["rows_to_write"], w.stored) == ("plan", n, {})
    lines.clear()
    env["ROWS_BACKFILL_EXECUTE_HASH"] = plan["plan_hash"]
    assert rbo.main_from_env(env, sources=w.sources(), log=lines.append, store=w.store) == 0
    done = _summary(lines)
    assert (done["mode"], done["written"], done["complete"], done["refused"]) == (
        "execute", {"s1": n}, True, False)
    assert done["after"]["rows_to_write"] == 0 and len(w.stored["s1"]) == n


@pytest.mark.parametrize("wrong", ["0" * 64, "f" * 64])
def test_main_execute_with_a_stale_hash_writes_nothing(tmp_path, wrong):
    w = _World(tmp_path, [{"id": "s1", "original_name": "rates.csv", "_n": 2}])
    lines = []
    env = {"ROWS_BACKFILL_PROJECT_ID": "p", "ROWS_BACKFILL_EXECUTE_HASH": wrong}
    assert rbo.main_from_env(env, sources=w.sources(), log=lines.append, store=w.store) == 1
    out = _summary(lines)
    assert (out["refused"], out["written"], out["after"], w.stored) == (True, {}, None, {})


@pytest.mark.parametrize("exc", [RuntimeError, OSError])
def test_main_reports_a_raise_by_class_only(exc):
    class _Raises:
        def documents(self, project_id):
            raise exc("secret path /data/x")

    lines = []
    rc = rbo.main_from_env({"ROWS_BACKFILL_PROJECT_ID": "p"}, sources=_Raises(), log=lines.append)
    assert rc == 1
    assert _summary(lines) == {"ok": False, "error": rbo.ERR_RAISED, "error_class": exc.__name__}


@pytest.mark.parametrize("name", ["Concrete grade 1", "rates.csv"])
def test_summary_carries_no_row_text_or_file_name(tmp_path, name):
    w = _World(tmp_path, [{"id": "s1", "original_name": "rates.csv", "_n": 1}])
    lines = []
    rbo.main_from_env({"ROWS_BACKFILL_PROJECT_ID": "p"}, sources=w.sources(), log=lines.append)
    assert name not in lines[0]
