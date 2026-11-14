"""P-B3 ops entrypoint: plan one document's bill sheet inside the ingest task.

``scripts/bill_plan_once.py`` only plans: it adds no row and indexes nothing,
prints one ``BILLPLANSUMMARY`` line of ids and counts, and resumes a long
document from the word boxes an earlier page range saved. Every input is
built here (synthetic word boxes from ``test_bill_derived``, in-memory
document rows); each decision has at least two inputs.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import pytest

from app.core import bill_classify as bc
from app.core import bill_derived as bd
from scripts import bill_plan_once as bpo
from tests.test_bill_derived import _BILL, _bill_a, _bill_b, _by_page, _Docs, _source


class Clock:
    def __init__(self, *ticks: float):
        self.ticks = list(ticks)

    def __call__(self) -> float:
        return self.ticks.pop(0)


def _deps(docs: dict[str, Any], plan, lines: list[str], **kw) -> bpo.Deps:
    return bpo.Deps(
        get_document=docs.get, plan=plan, log=lines.append,
        build_sha=lambda: "sha1", peak_rss_mb=lambda: 12.5,
        clock=kw.pop("clock", Clock(0.0, 3.0)), **kw,
    )


def _summary(lines: list[str]) -> dict[str, Any]:
    (line,) = [ln for ln in lines if ln.startswith(bpo.SUMMARY_PREFIX)]
    return json.loads(line[len(bpo.SUMMARY_PREFIX):])


# --- parsing --------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [("", None), ("  ", None), ("0:40", (0, 40)), ("40:80", (40, 80))])
def test_parse_pages_reads_a_half_open_range(raw, want):
    assert bpo.parse_pages(raw) == want


@pytest.mark.parametrize("raw", ["40", "a:b", "5:5", "9:3", "-1:4"])
def test_parse_pages_rejects_a_bad_range(raw):
    with pytest.raises(ValueError):
        bpo.parse_pages(raw)


@pytest.mark.parametrize("sections,want", [
    ([{"path": ["A"], "rows": 1}, {"path": ["A", "B"], "rows": 2},
      {"path": ["A", "B", "C"], "rows": 3}, {"path": ["A", "B", "C", "D"], "rows": 4}],
     {"1": 1, "2": 2, "3+": 7}),
    ([{"path": ["X", "Y"], "rows": 1}, {"path": ["X", "Z"], "rows": 2}], {"2": 3}),
])
def test_depth_distribution_counts_rows_per_path_depth(sections, want):
    assert bpo.depth_distribution(sections) == want


@pytest.mark.parametrize("names,want", [
    (["page_00000.json", "page_00001.json", "page_00002.json.part", "other.json"], 2),
    ([], 0),
])
def test_cached_pages_counts_only_finished_page_files(tmp_path, names, want):
    for name in names:
        (tmp_path / name).write_text("[]")
    assert bpo.cached_pages(str(tmp_path)) == want
    assert bpo.cached_pages(str(tmp_path / "absent")) == 0


@pytest.mark.parametrize("data_dir,want", [("/mnt/data", "/mnt/data/bill_plan/d9"), (None, "data/bill_plan/d9")])
def test_default_work_dir_is_per_document_under_data_dir(data_dir, want):
    assert bpo.default_work_dir("d9", data_dir).replace("\\", "/") == want


# --- plan_one decisions ---------------------------------------------------

@pytest.mark.parametrize("docs,error", [
    ({}, "document not found"),
    ({"d1": {"id": "d1", "project_id": "other"}}, "document does not belong to project"),
])
def test_plan_one_refuses_before_planning(tmp_path, docs, error):
    calls: list = []
    lines: list[str] = []
    rep = bpo.plan_one("p1", "d1", pages=None, work_dir=str(tmp_path),
                       deps=_deps(docs, lambda *a, **k: calls.append(a) or {}, lines))
    assert rep["ok"] is False and rep["error"] == error and calls == []


@pytest.mark.parametrize("reason", [bd.SKIP_NO_BYTES, bd.SKIP_NOT_PDF])
def test_plan_one_reports_a_skip_as_not_ok(tmp_path, reason):
    lines: list[str] = []
    plan_out = {"derived": False, "reason": reason}
    rep = bpo.plan_one("p1", "d1", pages=None, work_dir=str(tmp_path),
                       deps=_deps({"d1": {"project_id": "p1"}}, lambda *a, **k: plan_out, lines))
    assert rep["ok"] is False and rep["error"] == f"no plan: {reason}"
    assert rep["reason"] == reason and "section_paths" not in rep


@pytest.mark.parametrize("exc", [RuntimeError("ocr died"), bd.WebRequestRefused("web")])
def test_plan_one_reports_an_exception_and_keeps_timing(tmp_path, exc):
    def boom(*a, **k):
        raise exc
    lines: list[str] = []
    rep = bpo.plan_one("p1", "d1", pages=(0, 2), work_dir=str(tmp_path),
                       deps=_deps({"d1": {"project_id": "p1"}}, boom, lines, clock=Clock(1.0, 5.5)))
    assert rep["ok"] is False and type(exc).__name__ in rep["error"]
    assert rep["wall_s"] == 4.5 and rep["peak_rss_mb"] == 12.5


# --- end to end through the real plan_bill_sheet ---------------------------

def _real_plan(words, seen):
    return functools.partial(bd.plan_bill_sheet, page_words=_by_page(words, seen))


@pytest.mark.parametrize("build", [_bill_a, _bill_b])
def test_main_plans_through_the_real_planner_and_writes_no_row(tmp_path, monkeypatch, build):
    words, want, want_flagged = build()
    pages = 3 if build is _bill_a else 1
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=pages)
    docs.install(monkeypatch, tmp_path)
    seen: list[int] = []
    lines: list[str] = []
    deps = _deps(docs.rows, _real_plan(words, seen), lines, classify=_BILL)
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path / "wd")}

    assert bpo.main_from_env(env, deps=deps) == 0

    rep = _summary(lines)
    assert rep["ok"] is True and rep["build_sha"] == "sha1"
    assert rep["pages"] == [0, pages] and rep["pages_in_file"] == pages
    assert rep["rows"] == sum(n for n, _ in want.values())
    assert rep["unreconciled"] == want_flagged
    assert rep["section_paths"] == len(want)
    assert sum(rep["rows_by_path_depth"].values()) == rep["rows"]
    assert len(rep["plan_hash"]) == 64
    assert rep["verdict"]["is_bill"] is True and rep["verdict"]["reason"] == bc.BILL_PRICED
    assert rep["pages_cached_before"] == 0 and rep["pages_cached_after"] == pages
    assert seen == list(range(pages))
    assert docs.added == [] and docs.indexed == [] and docs.meta_updates == []
    # ids and counts only: no section label reaches the summary line
    line = [ln for ln in lines if ln.startswith(bpo.SUMMARY_PREFIX)][0]
    assert not any(label in line for label in want)


def test_ranges_resume_from_saved_boxes_and_the_whole_plan_matches(tmp_path, monkeypatch):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=3)
    docs.install(monkeypatch, tmp_path)
    wd = str(tmp_path / "wd")
    base = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1", "BILL_PLAN_WORK_DIR": wd}
    ocrd: list[int] = []
    reports = []
    for pages in ("0:2", "2:3", ""):
        lines: list[str] = []
        deps = _deps(docs.rows, _real_plan(words, ocrd), lines, classify=_BILL)
        assert bpo.main_from_env({**base, "BILL_PLAN_PAGES": pages}, deps=deps) == 0
        reports.append(_summary(lines))

    assert ocrd == [0, 1, 2]  # the whole-document run OCR'd nothing again
    assert [r["pages"] for r in reports] == [[0, 2], [2, 3], [0, 3]]
    assert [r["pages_cached_after"] for r in reports] == [2, 3, 3]
    fresh = bd.plan_bill_sheet("p1", "d1", classify=_BILL, page_words=_by_page(words))
    assert reports[-1]["plan_hash"] == fresh["plan_hash"]
    assert reports[0]["plan_hash"] != fresh["plan_hash"]


@pytest.mark.parametrize("env", [
    {"BILL_PLAN_PROJECT_ID": "p1"},
    {"BILL_PLAN_DOCUMENT_ID": "d1"},
    {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1", "BILL_PLAN_PAGES": "7:2"},
])
def test_main_rejects_bad_env_before_any_io(env):
    calls: list = []
    lines: list[str] = []
    deps = _deps({"d1": {"project_id": "p1"}}, lambda *a, **k: calls.append(a), lines)
    assert bpo.main_from_env(env, deps=deps) == 2
    assert calls == [] and _summary(lines)["ok"] is False


@pytest.mark.parametrize("value,hands_over", [("d1", True), ("  ", False)])
def test_p1b_main_hands_over_only_when_document_id_is_set(monkeypatch, value, hands_over):
    from scripts import p1b_ingest_drive_server as p1b

    calls: list[str] = []
    monkeypatch.delenv("DOC_REINDEX_DOCUMENT_ID", raising=False)
    monkeypatch.setenv("BILL_PLAN_DOCUMENT_ID", value)
    monkeypatch.setattr(bpo, "main_from_env", lambda: calls.append("plan") or 7)

    def parser_reached():
        calls.append("walk")
        raise SystemExit(0)

    monkeypatch.setattr(p1b, "build_parser", parser_reached)
    try:
        code = p1b.main()
    except SystemExit:
        code = None
    assert calls == (["plan"] if hands_over else ["walk"])
    assert code == (7 if hands_over else None)
