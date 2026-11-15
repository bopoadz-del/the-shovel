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
    ({}, bpo.ERR_NOT_FOUND),
    ({"d1": {"id": "d1", "project_id": "other"}}, bpo.ERR_WRONG_PROJECT),
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
    assert rep["ok"] is False and rep["error"] == bpo.ERR_NO_PLAN
    assert rep["reason"] == reason and "section_paths" not in rep


@pytest.mark.parametrize("exc", [RuntimeError("ocr died"), bd.WebRequestRefused("web")])
def test_plan_one_reports_an_exception_and_keeps_timing(tmp_path, exc):
    def boom(*a, **k):
        raise exc
    lines: list[str] = []
    rep = bpo.plan_one("p1", "d1", pages=(0, 2), work_dir=str(tmp_path),
                       deps=_deps({"d1": {"project_id": "p1"}}, boom, lines, clock=Clock(1.0, 5.5)))
    assert rep["ok"] is False and rep["error"] == bpo.ERR_RAISED
    assert rep["error_class"] == type(exc).__name__ and str(exc) not in json.dumps(rep)
    assert rep["wall_s"] == 4.5 and rep["peak_rss_mb"] == 12.5
    # the message is kept privately, beside the word boxes
    kept = json.loads((tmp_path / "last_error.json").read_text(encoding="utf-8"))
    assert kept == {"error_class": type(exc).__name__, "message": str(exc)}


# --- the public summary carries counts, codes and hashes only ---------------

#: Free text a plan, a verdict or an exception could carry; none may reach the summary.
_LEAKS = [
    "Bill No. 4 - Substructure / Concrete works",
    "C:/projects/Tender Package/priced_bill_final.pdf",
    "Excavate to reduce levels, n.e. 2m deep",
    "x" * (bpo.HASH_LEN + 1),
]


def _assert_public(summary: dict[str, Any], leaks: list[str]) -> None:
    """Every string (key or value) is a short code token; no leak survives."""
    def walk(value):
        if isinstance(value, str):
            assert len(value) <= bpo.HASH_LEN and bpo._TOKEN.fullmatch(value), value
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(k)
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        else:
            assert value is None or isinstance(value, (bool, int, float)), value
    walk(summary)
    text = json.dumps(summary)
    assert not any(leak in text for leak in leaks)
    assert set(summary) <= set(bpo.SUMMARY_KEYS)


@pytest.mark.parametrize("leak", _LEAKS)
def test_public_summary_redacts_free_text_in_every_field(leak):
    report = {
        "ok": False, "error": leak, "error_class": leak, "document_id": "d1",
        "reason": leak, "plan_hash": "a" * 64,
        "verdict": {"is_bill": True, "reason": leak, "measured_rows": 7},
        "rows_by_path_depth": {"1": 3, leak: 2},
        "sections": [{"path": [leak], "rows": 3}],  # not a summary key: dropped
        "file_name": leak,
    }
    out = bpo.public_summary(report)
    _assert_public(out, [leak])
    assert out["plan_hash"] == "a" * 64 and out["document_id"] == "d1"
    assert out["error"] == bpo.REDACTED and out["verdict"]["measured_rows"] == 7
    assert "sections" not in out and "file_name" not in out


@pytest.mark.parametrize("value,want", [
    ("priced_rows_reconcile", "priced_rows_reconcile"),
    ("b" * 64, "b" * 64),
    ("b" * 65, bpo.REDACTED),
    ("two words", bpo.REDACTED),
    ("", bpo.REDACTED),
    (object(), bpo.REDACTED),
    (12, 12),
    (None, None),
])
def test_public_value_passes_only_tokens_and_numbers(value, want):
    assert bpo._public_value(value) == want


def _leaky_plan(*a, **k):
    """A plan whose sections, reason and extras carry free text."""
    return {
        "derived": False, "reason": "priced_rows_reconcile", "source_id": "d1",
        "pages": [0, 2], "pages_in_file": 9, "rows": 5, "unreconciled": 1,
        "sections": [{"path": [_LEAKS[0], _LEAKS[2]], "rows": 4, "unreconciled": 1,
                      "pages": [0, 1]},
                     {"path": [_LEAKS[0]], "rows": 1, "unreconciled": 0, "pages": [1, 1]}],
        "plan_hash": "c" * 64, "original_name": _LEAKS[1],
    }


@pytest.mark.parametrize("plan,code", [
    (_leaky_plan, None),
    (lambda *a, **k: (_ for _ in ()).throw(ValueError(_LEAKS[2])), bpo.ERR_RAISED),
])
def test_summary_line_never_carries_plan_or_exception_text(tmp_path, plan, code):
    lines: list[str] = []
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path), "BILL_PLAN_PAGES": "0:2"}
    deps = _deps({"d1": {"project_id": "p1"}}, plan, lines)
    rc = bpo.main_from_env(env, deps=deps)
    assert len(lines) == 1  # the script prints nothing but the summary
    summary = _summary(lines)
    _assert_public(summary, _LEAKS)
    if code is None:
        assert rc == 0 and summary["rows_by_path_depth"] == {"1": 1, "2": 4}
        assert summary["private_plan_saved"] is True
        # the full plan, labels included, is kept privately in the work dir
        kept = json.loads((tmp_path / "plan_0_2.json").read_text(encoding="utf-8"))
        assert kept["sections"][0]["path"] == [_LEAKS[0], _LEAKS[2]]
    else:
        assert rc == 1 and summary["error"] == code and summary["error_class"] == "ValueError"
        assert not (tmp_path / "plan_0_2.json").exists()


@pytest.mark.parametrize("blocker", ["file", "dir"])
def test_a_plan_that_cannot_be_kept_privately_fails_the_run(tmp_path, blocker):
    target = tmp_path / "plan_0_2.json"
    if blocker == "dir":  # the plan's file name is taken by a directory
        target.mkdir()
        wd = tmp_path
    else:  # the work directory is a regular file
        wd = tmp_path / "wd"
        wd.write_text("x", encoding="utf-8")
    lines: list[str] = []
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(wd), "BILL_PLAN_PAGES": "0:2"}
    rc = bpo.main_from_env(env, deps=_deps({"d1": {"project_id": "p1"}}, _leaky_plan, lines))
    summary = _summary(lines)
    assert rc == 1 and summary["ok"] is False and summary["error"] == bpo.ERR_NOT_SAVED
    assert summary["private_plan_saved"] is False and summary["error_class"]
    _assert_public(summary, _LEAKS)


@pytest.mark.parametrize("durable", [True, False])
def test_summary_reports_whether_the_work_dir_outlives_the_task(tmp_path, durable):
    seen: list[str] = []
    lines: list[str] = []
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path), "BILL_PLAN_PAGES": "0:2"}
    deps = _deps({"d1": {"project_id": "p1"}}, _leaky_plan, lines,
                 durable=lambda p: seen.append(p) or durable)
    assert bpo.main_from_env(env, deps=deps) == 0
    assert _summary(lines)["work_dir_durable"] is durable and seen == [str(tmp_path)]


@pytest.mark.parametrize("pages,name", [([0, 40], "plan_0_40.json"), ([40, 80], "plan_40_80.json")])
def test_private_plan_is_named_by_its_page_range(pages, name):
    assert bpo.private_plan_name({"pages": pages}) == name


# --- end to end through the real plan_bill_sheet ---------------------------

def _real_plan(words, seen):
    return functools.partial(bd.plan_bill_sheet, page_source=_by_page(words, seen))


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
    _assert_public(rep, [str(label) for label in want])
    assert (tmp_path / "wd" / f"plan_0_{pages}.json").is_file()


@pytest.mark.parametrize("ranges,cached", [(("0:2", "2:3"), [2, 3]), (("0:1", "1:3"), [1, 3])])
def test_ranges_resume_from_saved_boxes_and_the_whole_plan_matches(
    tmp_path, monkeypatch, ranges, cached,
):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=3)
    docs.install(monkeypatch, tmp_path)
    wd = str(tmp_path / "wd")
    base = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1", "BILL_PLAN_WORK_DIR": wd}
    ocrd: list[int] = []
    reports = []
    for pages in (*ranges, ""):
        lines: list[str] = []
        deps = _deps(docs.rows, _real_plan(words, ocrd), lines, classify=_BILL)
        assert bpo.main_from_env({**base, "BILL_PLAN_PAGES": pages}, deps=deps) == 0
        reports.append(_summary(lines))

    assert ocrd == [0, 1, 2]  # the whole-document run OCR'd nothing again
    want_pages = [[int(n) for n in r.split(":")] for r in ranges] + [[0, 3]]
    assert [r["pages"] for r in reports] == want_pages
    assert [r["pages_cached_after"] for r in reports] == [*cached, 3]
    assert [r["pages_cached_before"] for r in reports] == [0, cached[0], 3]
    fresh = bd.plan_bill_sheet("p1", "d1", classify=_BILL, page_source=_by_page(words))
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


@pytest.mark.parametrize("counts", [
    {"pages_read": 5, "pages_sampled": 5, "lines_tested": 0, "pages_without_words": 0,
     "pages_text_layer": 5, "pages_ocr": 0, "header_lines": 0,
     "rejected_fewer_than_3_cells": 0, "rejected_no_description_column": 0,
     "rejected_no_figure_column": 0},
    {"pages_read": 20, "pages_sampled": 18, "lines_tested": 41, "pages_without_words": 2,
     "pages_text_layer": 1, "pages_ocr": 19, "header_lines": 0,
     "rejected_fewer_than_3_cells": 30, "rejected_no_description_column": 7,
     "rejected_no_figure_column": 4},
])
def test_a_refused_verdict_still_reports_what_its_pages_were_read_from(tmp_path, counts):
    lines: list[str] = []

    def plan(project_id, document_id, *, pages, classify, work_dir):
        verdict = classify([])
        return {"derived": False, "reason": verdict.reason}

    def classify(reads):
        return bc.BillVerdict(False, bc.NOT_BILL_NO_HEADER, **counts)

    deps = _deps({"d1": {"project_id": "p1"}}, plan, lines, classify=classify)
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path / "wd"), "BILL_PLAN_PAGES": "0:5"}
    assert bpo.main_from_env(env, deps=deps) == 1
    rep = _summary(lines)
    assert rep["error"] == bpo.ERR_NO_PLAN
    assert rep["verdict"]["reason"] == bc.NOT_BILL_NO_HEADER
    for name, value in counts.items():
        assert rep["verdict"][name] == value
    assert set(counts) <= set(bpo.VERDICT_FIELDS)
