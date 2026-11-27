"""doc_rows backfill: additive plan/execute over injected sources."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.core import doc_rows as dr
from app.core import doc_rows_backfill as bf
from app.core.bill_table_geometry import BillRow


@dataclass
class _SheetRow:
    sheet: str
    row_no: int
    fields: dict
    path: tuple = ()
    others: dict = field(default_factory=dict)


def _sheet(n, sheet="Bill 1"):
    return [_SheetRow(sheet, i + 2, {"description": f"Item {i}", "quantity": str(i + 1),
                                      "rate": "10", "total": str((i + 1) * 10)})
            for i in range(n)]


def _bill(n):
    return [BillRow(section="Works", source_page=3 + i,
                    fields={"description": f"Line {i}", "quantity": "2", "rate": "5", "total": "10"})
            for i in range(n)]


class _Fake:
    """Sources built from plain dicts; ``stored`` stands for doc_rows."""

    def __init__(self, docs, sheets=None, chunks=None, coverage=None, bills=None, stored=None):
        self.docs = docs
        self.sheets = sheets or {}
        self.chunks = chunks or {}
        self.coverage = coverage or {}
        self.bills = bills or {}
        self.stored = dict(stored or {})

    def documents(self, project_id):
        return list(self.docs)

    def has_rows(self, project_id, doc_id):
        return bool(self.stored.get(doc_id))

    def sheet_rows(self, doc):
        return self.sheets.get(doc.doc_id, [])

    def row_chunk_count(self, doc):
        return self.chunks.get(doc.doc_id, 0)

    def page_coverage(self, doc):
        return self.coverage.get(doc.doc_id, (0, 0))

    def bill_rows(self, doc):
        return self.bills.get(doc.doc_id, [])

    def store(self, project_id, doc_id, rows, *, keep_existing=False):
        if keep_existing and self.stored.get(doc_id):
            return 0
        self.stored[doc_id] = list(rows)
        return len(rows)


def _entry(plan, doc_id):
    return next(e for e in plan["documents"] if e["doc_id"] == doc_id)


# ── plan: what is written and why not ──────────────────────────────────────

@pytest.mark.parametrize("n", [1, 4])
def test_sheet_whose_rows_match_its_chunks_is_planned(n):
    src = _Fake([bf.BackfillDoc("s1", bf.SHEET)], sheets={"s1": _sheet(n)}, chunks={"s1": n})
    plan = bf.plan_rows_backfill("p", src)
    e = _entry(plan, "s1")
    assert (e["rows_to_write"], e["skip_reason"]) == (n, "")
    assert plan["rows_to_write"] == n and plan["documents_to_write"] == 1
    # the plan reports doc_rows' own verdict: a sheet that prints no total is
    # UNCHECKED (RECONCILED is the printed total only, #746)
    assert e["states"] == {"1": dr.UNCHECKED}


@pytest.mark.parametrize("n, chunks", [(3, 2), (2, 5), (0, 3)])
def test_sheet_whose_rows_differ_from_its_chunks_is_skipped(n, chunks):
    src = _Fake([bf.BackfillDoc("s1", bf.SHEET)], sheets={"s1": _sheet(n)}, chunks={"s1": chunks})
    e = _entry(bf.plan_rows_backfill("p", src), "s1")
    assert (e["rows_to_write"], e["skip_reason"], e["row_chunks"]) == (0, bf.CHUNK_MISMATCH, chunks)


@pytest.mark.parametrize("cached, in_file, reason, written", [
    (5, 5, "", 5), (6, 6, "", 5),             # every page saved: rows planned
    (4, 5, bf.PAGES_MISSING, 0),               # one page short: never read again
    (0, 0, bf.PAGES_MISSING, 0),               # nothing known about the file
])
def test_bill_is_read_only_when_its_saved_pages_cover_the_file(cached, in_file, reason, written):
    src = _Fake([bf.BackfillDoc("b1", bf.BILL)], coverage={"b1": (cached, in_file)},
                bills={"b1": _bill(5)})
    e = _entry(bf.plan_rows_backfill("p", src), "b1")
    assert (e["skip_reason"], e["rows_to_write"]) == (reason, written)
    assert (e["pages_cached"], e["pages_in_file"]) == (cached, in_file)


@pytest.mark.parametrize("kind, src_kw", [
    (bf.SHEET, {"sheets": {"d": _sheet(2)}, "chunks": {"d": 2}}),
    (bf.BILL, {"coverage": {"d": (3, 3)}, "bills": {"d": _bill(2)}}),
])
def test_document_that_already_has_rows_is_kept(kind, src_kw):
    src = _Fake([bf.BackfillDoc("d", kind)], stored={"d": ["x"]}, **src_kw)
    e = _entry(bf.plan_rows_backfill("p", src), "d")
    assert (e["skip_reason"], e["rows_to_write"]) == (bf.KEPT, 0)


@pytest.mark.parametrize("derived_from, reason", [
    ("b1", bf.DERIVED_OF),    # its source is planned too: one reading, not two
    ("gone", ""),             # its source is not in the project: the sheet is the reading
    ("s1", ""),               # names itself: no other reading to defer to
    ("", ""),
])
def test_sheet_derived_from_a_planned_document_is_not_counted_twice(derived_from, reason):
    docs = [bf.BackfillDoc("b1", bf.BILL), bf.BackfillDoc("s1", bf.SHEET, derived_from)]
    src = _Fake(docs, coverage={"b1": (2, 2)}, bills={"b1": _bill(2)},
                sheets={"s1": _sheet(2)}, chunks={"s1": 2})
    plan = bf.plan_rows_backfill("p", src)
    e = _entry(plan, "s1")
    assert e["skip_reason"] == reason
    assert e.get("derived_of") == (derived_from if reason else None)
    assert _entry(plan, "b1")["rows_to_write"] == 2
    assert plan["rows_to_write"] == (2 if reason else 4)


@pytest.mark.parametrize("doc, src_kw, reason", [
    (bf.BackfillDoc("s", bf.SHEET), {"chunks": {"s": 0}}, bf.NO_ROWS),
    (bf.BackfillDoc("b", bf.BILL), {"coverage": {"b": (2, 2)}}, bf.NO_ROWS),
    (bf.BackfillDoc("x", "drawing"), {}, bf.UNKNOWN_KIND),
])
def test_nothing_read_writes_nothing(doc, src_kw, reason):
    e = _entry(bf.plan_rows_backfill("p", _Fake([doc], **src_kw)), doc.doc_id)
    assert (e["skip_reason"], e["rows_to_write"]) == (reason, 0)


@pytest.mark.parametrize("n_a, n_b", [(2, 3), (4, 1)])
def test_plan_hash_follows_the_rows_and_needs_a_project(n_a, n_b):
    a = _Fake([bf.BackfillDoc("s1", bf.SHEET)], sheets={"s1": _sheet(n_a)}, chunks={"s1": n_a})
    b = _Fake([bf.BackfillDoc("s1", bf.SHEET)], sheets={"s1": _sheet(n_b)}, chunks={"s1": n_b})
    ha, hb = (bf.plan_rows_backfill("p", s)["plan_hash"] for s in (a, b))
    assert ha != hb
    assert ha == bf.plan_rows_backfill("p", a)["plan_hash"]
    assert ha != bf.plan_rows_backfill("q", a)["plan_hash"]
    with pytest.raises(ValueError):
        bf.plan_rows_backfill("", a)


@pytest.mark.parametrize("n", [1, 3])
def test_rows_digest_follows_row_content_at_the_same_count(n):
    rows = dr.rows_from_spreadsheet("s1", _sheet(n))
    other = dr.rows_from_spreadsheet("s1", _sheet(n, sheet="Bill 2"))
    d = bf._rows_digest(rows)
    assert isinstance(d, str) and len(d) == 64
    assert d == bf._rows_digest(dr.rows_from_spreadsheet("s1", _sheet(n)))
    assert d != bf._rows_digest(other)
    # Same row count, different rows: the plan hash still moves.
    a = _Fake([bf.BackfillDoc("s1", bf.SHEET)], sheets={"s1": _sheet(n)}, chunks={"s1": n})
    b = _Fake([bf.BackfillDoc("s1", bf.SHEET)], sheets={"s1": _sheet(n, sheet="Bill 2")},
              chunks={"s1": n})
    assert bf.plan_rows_backfill("p", a)["plan_hash"] != bf.plan_rows_backfill("p", b)["plan_hash"]


# ── execute ────────────────────────────────────────────────────────────────

def _mixed():
    return _Fake([bf.BackfillDoc("b1", bf.BILL), bf.BackfillDoc("s1", bf.SHEET),
                  bf.BackfillDoc("s2", bf.SHEET)],
                 coverage={"b1": (3, 3)}, bills={"b1": _bill(3)},
                 sheets={"s1": _sheet(2), "s2": _sheet(1)}, chunks={"s1": 2, "s2": 1},
                 stored={"s2": ["already"]})


@pytest.mark.parametrize("bad", ["", "0" * 64])
def test_execute_refuses_a_hash_that_is_not_the_live_plan(bad):
    src = _mixed()
    res = bf.execute_rows_backfill("p", bad, src, store=src.store)
    assert res["refused"] and not res["executed"] and res["written"] == {}
    assert src.stored == {"s2": ["already"]}


@pytest.mark.parametrize("project", ["p", "other-project"])
def test_execute_writes_planned_rows_keeps_stored_and_ends_with_nothing_left(project):
    src = _mixed()
    h = bf.plan_rows_backfill(project, src)["plan_hash"]
    res = bf.execute_rows_backfill(project, h, src, store=src.store)
    assert res["executed"] and res["refused"] is None and res["complete"]
    assert res["written"] == {"b1": 3, "s1": 2}
    assert src.stored["s2"] == ["already"]
    assert [r.page for r in src.stored["b1"]] == [3, 4, 5]
    assert [r.chunk_ordinal for r in src.stored["s1"]] == [0, 1]
    assert res["after"]["rows_to_write"] == 0
    # A second run plans nothing and writes nothing.
    again = bf.execute_rows_backfill(project, res["after"]["plan_hash"], src, store=src.store)
    assert again["written"] == {} and again["complete"]


@pytest.mark.parametrize("project", ["", None])
def test_execute_needs_a_project_and_writes_nothing_without_one(project):
    src = _mixed()
    h = bf.plan_rows_backfill("p", src)["plan_hash"]
    with pytest.raises(ValueError):
        bf.execute_rows_backfill(project, h, src, store=src.store)
    assert src.stored == {"s2": ["already"]}


def test_execute_reports_a_document_the_store_kept_and_an_incomplete_run():
    src = _mixed()
    h = bf.plan_rows_backfill("p", src)["plan_hash"]
    res = bf.execute_rows_backfill("p", h, src, store=lambda *a, **k: 0)
    assert res["kept"] == ["b1", "s1"] and res["written"] == {}
    assert not res["complete"] and res["after"]["rows_to_write"] == 5


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    yield


@pytest.mark.parametrize("n", [2, 3])
def test_execute_through_the_real_store_is_additive(fresh_db, n):
    src = _Fake([bf.BackfillDoc("s1", bf.SHEET), bf.BackfillDoc("s2", bf.SHEET)],
                sheets={"s1": _sheet(n), "s2": _sheet(1)}, chunks={"s1": n, "s2": 1})
    first = dr.DocRow("s2", 1, 1, fields={"description": "Stored before"})
    dr.store_document_rows("p", "s2", [first])
    src.has_rows = lambda p, d: bool(dr.load_rows(p, d))
    h = bf.plan_rows_backfill("p", src)["plan_hash"]
    res = bf.execute_rows_backfill("p", h, src)
    assert res["written"] == {"s1": n} and res["complete"]
    assert len(dr.load_rows("p", "s1")) == n
    assert dr.load_rows("p", "s2") == [first]
