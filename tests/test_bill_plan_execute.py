"""P-B3 ops entrypoint, EXECUTE mode: write a document's derived bill sheet
only for the plan hash given, and only when every run condition holds on
the plan rebuilt from the saved word boxes.

Fakes only: the plan, the derive call, the document rows and the chunk
counts are built here. No document, project or count of any real bill is
named; each decision has at least two inputs.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from app.core import bill_classify as bc
from scripts import bill_plan_once as bpo

H1 = "a" * 64
H2 = "0123456789abcdef" * 4


def _plan(plan_hash: str = H1, **over: Any) -> dict[str, Any]:
    """A whole-document plan of 10 pages, all read back from the cache."""
    cols = {
        "rows_priced_reconciled": 400, "rows_figures_not_reconciling": 1,
        "priced_rows_empty_description": 3, "priced_rows_empty_path": 0,
        "header_conflicts": 0,
    }
    cols.update(over.pop("columns", {}))
    plan = {
        "derived": False, "reason": bc.BILL_PRICED, "source_id": "doc1",
        "pages": [0, 10], "pages_in_file": 10,
        "pages_read_cache": 10, "pages_read_ocr": 0, "pages_read_text_layer": 0,
        "rows": 500, "unreconciled": 1, "sections": [{"path": ["A"], "rows": 500}],
        "columns": cols, "plan_hash": plan_hash,
    }
    plan.update(over)
    return plan


class Recorder:
    def __init__(self, plan: dict[str, Any], result: dict[str, Any] | None = None):
        self.plan_out = plan
        self.result = result or {
            "derived": True, "reason": bc.BILL_PRICED, "document_id": "new1",
            "superseded": None, "rows": 500, "rows_flagged": 1, "sheets": 2,
            "chunks": 520, "index_status": "indexed",
        }
        self.plan_calls: list[dict[str, Any]] = []
        self.derive_calls: list[dict[str, Any]] = []

    def plan(self, project_id, document_id, **kw):
        self.plan_calls.append(kw)
        return self.plan_out

    def derive(self, project_id, document_id, **kw):
        self.derive_calls.append(kw)
        return self.result


DOCS = {
    "doc1": {"id": "doc1", "project_id": "p1"},
    "new1": {"id": "new1", "project_id": "p1", "status": "indexed"},
    "new2": {"id": "new2", "project_id": "p1", "status": "partial"},
}


def _deps(rec: Recorder, lines: list[str], over_window: int = 0) -> bpo.Deps:
    return bpo.Deps(
        get_document=DOCS.get, plan=rec.plan, derive=rec.derive, log=lines.append,
        build_sha=lambda: "sha1", peak_rss_mb=lambda: 12.5, clock=lambda: 0.0,
        chunk_bounds=lambda pid, did: {"chunks": 520, "over_ceiling": over_window},
    )


def _env(tmp_path, **kw: str) -> dict[str, str]:
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "doc1",
           "BILL_PLAN_WORK_DIR": str(tmp_path)}
    env.update(kw)
    return env


def _summary(lines: list[str]) -> dict[str, Any]:
    (line,) = [ln for ln in lines if ln.startswith(bpo.SUMMARY_PREFIX)]
    return json.loads(line[len(bpo.SUMMARY_PREFIX):])


# --- hash unset: the run plans and never derives ---------------------------

@pytest.mark.parametrize("extra", [{}, {"BILL_EXECUTE_MIN_RECONCILED": "100"}])
def test_without_an_execute_hash_the_run_only_plans(tmp_path, extra):
    rec, lines = Recorder(_plan()), []
    code = bpo.main_from_env(_env(tmp_path, **extra), _deps(rec, lines))
    assert code == 0
    assert rec.derive_calls == []
    assert "mode" not in _summary(lines)


# --- parameters -------------------------------------------------------------

@pytest.mark.parametrize("env", [
    {"BILL_PLAN_EXECUTE_HASH": "A" * 64},                       # upper case
    {"BILL_PLAN_EXECUTE_HASH": "a" * 63},                       # short
    {"BILL_PLAN_EXECUTE_HASH": H1, "BILL_PLAN_PAGES": "0:5"},   # not whole-document
    {"BILL_PLAN_EXECUTE_HASH": H1, "BILL_EXECUTE_MIN_RECONCILED": "-1"},
    {"BILL_PLAN_EXECUTE_HASH": H1, "BILL_EXECUTE_MAX_UNRECONCILED_SHARE": "1.5"},
    {"BILL_PLAN_EXECUTE_HASH": H1, "BILL_EXECUTE_MAX_EMPTY_DESCRIPTION_SHARE": "x"},
])
def test_bad_execute_params_are_refused_before_any_io(tmp_path, env):
    rec, lines = Recorder(_plan()), []
    assert bpo.main_from_env(_env(tmp_path, **env), _deps(rec, lines)) == 2
    assert rec.plan_calls == [] and rec.derive_calls == []
    assert _summary(lines)["error"] == bpo.ERR_BAD_EXECUTE


@pytest.mark.parametrize("env,want", [
    ({"BILL_PLAN_EXECUTE_HASH": H1}, bpo.Gates()),
    ({"BILL_PLAN_EXECUTE_HASH": H2, "BILL_EXECUTE_MIN_RECONCILED": "861",
      "BILL_EXECUTE_MAX_UNRECONCILED_SHARE": "0.01",
      "BILL_EXECUTE_MAX_EMPTY_DESCRIPTION_SHARE": "0.05"},
     bpo.Gates(min_reconciled_rows=861, max_unreconciled_share=0.01,
               max_empty_description_share=0.05)),
])
def test_execute_params_are_read_with_their_defaults(env, want):
    assert bpo.parse_execute(env) == (env["BILL_PLAN_EXECUTE_HASH"], want)


def test_default_gates_are_shares_with_no_floor():
    g = bpo.Gates()
    assert (g.min_reconciled_rows, g.max_unreconciled_share, g.max_empty_description_share,
            g.max_header_conflicts, g.max_priced_rows_without_path) == (0, 0.005, 0.02, 0, 0)


# --- every condition holds: the write goes ahead ----------------------------

@pytest.mark.parametrize("plan_hash,new_id,status,over", [
    (H1, "new1", "indexed", 0),
    (H2, "new2", "partial", 3),
])
def test_a_passing_plan_is_derived_with_its_hash(tmp_path, plan_hash, new_id, status, over):
    rec = Recorder(_plan(plan_hash))
    rec.result = {**rec.result, "document_id": new_id, "superseded": "old9", "chunks": 7}
    lines: list[str] = []
    env = _env(tmp_path, BILL_PLAN_EXECUTE_HASH=plan_hash, BILL_EXECUTE_MIN_RECONCILED="400")
    assert bpo.main_from_env(env, _deps(rec, lines, over_window=over)) == 0
    assert rec.plan_calls[0]["pages"] is None  # whole document
    assert rec.derive_calls[0]["plan_hash"] == plan_hash
    assert rec.derive_calls[0]["work_dir"] == str(tmp_path)
    out = _summary(lines)
    assert out["ok"] is True and out["mode"] == bpo.MODE_EXECUTE
    assert out["refused"] == []
    assert (out["derived_document_id"], out["document_status"]) == (new_id, status)
    assert (out["rows_written"], out["chunks_added"], out["chunks_over_window"]) == (500, 7, over)
    assert out["superseded"] == "old9"
    assert out["gates"]["min_reconciled_rows"] == 400


# --- hash mismatch ----------------------------------------------------------

@pytest.mark.parametrize("given,rebuilt", [(H1, H2), (H2, H1)])
def test_a_hash_mismatch_never_derives(tmp_path, given, rebuilt):
    rec, lines = Recorder(_plan(rebuilt)), []
    assert bpo.main_from_env(_env(tmp_path, BILL_PLAN_EXECUTE_HASH=given), _deps(rec, lines)) == 1
    assert rec.derive_calls == []
    out = _summary(lines)
    assert out["error"] == bpo.ERR_REFUSED and out["refused"] == [bpo.REFUSE_HASH]


# --- each condition failing -------------------------------------------------

@pytest.mark.parametrize("over,gates,code", [
    # not the whole document
    ({"pages": [0, 9]}, bpo.Gates(), bpo.REFUSE_NOT_WHOLE),
    ({"pages": [1, 10]}, bpo.Gates(), bpo.REFUSE_NOT_WHOLE),
    # not a priced verdict
    ({"reason": bc.BILL_UNPRICED}, bpo.Gates(), bpo.REFUSE_NOT_PRICED),
    ({"reason": bc.NOT_BILL_UNRECONCILED}, bpo.Gates(), bpo.REFUSE_NOT_PRICED),
    # a page not read back from the cache
    ({"pages_read_cache": 9, "pages_read_ocr": 1}, bpo.Gates(), bpo.REFUSE_NOT_CACHED),
    ({"pages_read_cache": 9, "pages_read_text_layer": 1}, bpo.Gates(), bpo.REFUSE_NOT_CACHED),
    # below the floor given for the run
    ({}, bpo.Gates(min_reconciled_rows=401), bpo.REFUSE_FEW_RECONCILED),
    ({"columns": {"rows_priced_reconciled": 99, "rows_figures_not_reconciling": 0,
                  "priced_rows_empty_description": 0}}, bpo.Gates(min_reconciled_rows=100),
     bpo.REFUSE_FEW_RECONCILED),
    # unreconciled share over its ceiling (priced = 400 + n)
    ({"columns": {"rows_figures_not_reconciling": 3}}, bpo.Gates(), bpo.REFUSE_UNRECONCILED),
    ({"columns": {"rows_figures_not_reconciling": 1}}, bpo.Gates(max_unreconciled_share=0.001),
     bpo.REFUSE_UNRECONCILED),
    # empty-description share over its ceiling (2% of 401 = 8.02)
    ({"columns": {"priced_rows_empty_description": 9}}, bpo.Gates(), bpo.REFUSE_NO_DESCRIPTION),
    ({"columns": {"priced_rows_empty_description": 3}},
     bpo.Gates(max_empty_description_share=0.005), bpo.REFUSE_NO_DESCRIPTION),
    # header conflicts
    ({"columns": {"header_conflicts": 1}}, bpo.Gates(), bpo.REFUSE_CONFLICTS),
    ({"columns": {"header_conflicts": 4}}, bpo.Gates(max_header_conflicts=3), bpo.REFUSE_CONFLICTS),
    # priced rows without a section path
    ({"columns": {"priced_rows_empty_path": 1}}, bpo.Gates(), bpo.REFUSE_NO_PATH),
    ({"columns": {"priced_rows_empty_path": 2}}, bpo.Gates(max_priced_rows_without_path=1),
     bpo.REFUSE_NO_PATH),
    # counts the conditions read are missing
    ({"columns": {"priced_rows_empty_path": None}}, bpo.Gates(), bpo.REFUSE_COUNTS_MISSING),
    ({"columns": {"header_conflicts": "0"}}, bpo.Gates(), bpo.REFUSE_COUNTS_MISSING),
])
def test_each_failing_condition_refuses_the_write(tmp_path, over, gates, code):
    rec, lines = Recorder(_plan(**over)), []
    report = bpo.execute_one("p1", "doc1", execute_hash=H1, gates=gates,
                             work_dir=str(tmp_path), deps=_deps(rec, lines))
    assert report["refused"] == [code]
    assert report["error"] == bpo.ERR_REFUSED and report["ok"] is False
    assert rec.derive_calls == []


@pytest.mark.parametrize("cols,gates", [
    # exactly at each ceiling still passes (401 priced)
    ({"rows_figures_not_reconciling": 1, "priced_rows_empty_description": 8}, bpo.Gates()),
    ({"rows_priced_reconciled": 861, "rows_figures_not_reconciling": 4,
      "priced_rows_empty_description": 17}, bpo.Gates(min_reconciled_rows=861)),
])
def test_counts_at_the_ceilings_pass(cols, gates):
    assert bpo.gate_refusals(_plan(columns=cols), H1, gates) == []


@pytest.mark.parametrize("plan", [
    {"derived": False, "reason": bc.NOT_BILL_NO_HEADER},   # a skip
    {"derived": False, "reason": "no_rows_extracted", "sections": None},
])
def test_a_skip_is_refused_as_no_plan(plan):
    assert bpo.gate_refusals(plan, H1, bpo.Gates()) == [bpo.REFUSE_NO_PLAN]


# --- the derive call itself declines or raises ------------------------------

@pytest.mark.parametrize("reason", ["plan_hash_mismatch", "plan_not_whole_document"])
def test_a_declined_derive_is_not_ok(tmp_path, reason):
    rec = Recorder(_plan(), {"derived": False, "reason": reason})
    lines: list[str] = []
    assert bpo.main_from_env(_env(tmp_path, BILL_PLAN_EXECUTE_HASH=H1), _deps(rec, lines)) == 1
    out = _summary(lines)
    assert out["error"] == bpo.ERR_NOT_DERIVED and out["refused"] == [reason]
    assert "rows_written" not in out


@pytest.mark.parametrize("exc", [RuntimeError("doc text here"), OSError("secret path")])
def test_a_raising_derive_reports_only_its_class(tmp_path, exc):
    rec, lines = Recorder(_plan()), []

    def boom(*a, **k):
        raise exc

    rec.derive = boom
    assert bpo.main_from_env(_env(tmp_path, BILL_PLAN_EXECUTE_HASH=H1), _deps(rec, lines)) == 1
    out = _summary(lines)
    assert out["error"] == bpo.ERR_RAISED and out["error_class"] == type(exc).__name__
    assert str(exc) not in json.dumps(out)


# --- the counts the conditions read ----------------------------------------

def _row(path, desc="Pipe", q="2", r="3.00", t="6.00"):
    from app.core import bill_table_geometry as g

    return g.BillRow(section=path[-1] if path else "", source_page=0,
                     fields={"description": desc, "quantity": q, "rate": r, "total": t},
                     extra={}, path=tuple(path))


@pytest.mark.parametrize("rows,want", [
    ([_row(("A",)), _row(("A", "B"))], 0),
    ([_row(()), _row(("  ",)), _row(("A",)), _row((), r="", t="")], 2),  # the last is unpriced
])
def test_priced_rows_without_a_section_path_are_counted(rows, want):
    from app.core import bill_table_geometry as g

    assert g.ColumnStats().as_dict(rows)["priced_rows_empty_path"] == want


@pytest.mark.parametrize("lengths,ceiling,over", [
    ([100, 2500, 1999], "2000", 1),
    ([50, 60], "40", 2),
])
def test_document_chunk_bounds_counts_one_documents_chunks(monkeypatch, lengths, ceiling, over):
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool

    from app.core.chunk_bounds_plan import document_chunk_bounds
    from app.core.models import rag_chunk_table_name

    monkeypatch.setenv("CHUNK_MAX_CHARS", ceiling)
    # the active namespace table, whatever namespace the test session runs under
    table = rag_chunk_table_name(os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip())
    engine = create_engine("sqlite://", poolclass=StaticPool)
    with Session(engine) as session:
        session.execute(text(f"CREATE TABLE {table} (project_id TEXT, doc_id TEXT, text TEXT)"))
        for n in lengths:
            session.execute(text(f"INSERT INTO {table} VALUES ('p1', 'd1', :t)"), {"t": "x" * n})
        # another document and another project: never counted
        session.execute(text(f"INSERT INTO {table} VALUES ('p1', 'd2', :t)"), {"t": "x" * 9999})
        session.execute(text(f"INSERT INTO {table} VALUES ('p2', 'd1', :t)"), {"t": "x" * 9999})
        session.commit()  # stored chunks are committed rows
        out = document_chunk_bounds(session, "p1", "d1")
    assert out["chunks"] == len(lengths)
    assert out["over_ceiling"] == over
