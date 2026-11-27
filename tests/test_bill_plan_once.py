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

import httpx
import pytest

from app.core import bill_classify as bc
from app.core import bill_derived as bd
from app.core import bill_page_read as bpr
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
    # two titles; a title's rows are summed over its sub-paths
    ([{"path": ["T1", "a"], "rows": 2}, {"path": ["T1", "b"], "rows": 3},
      {"path": ["T2"], "rows": 4}],
     {"section_count": 2, "rows_per_section": [5, 4], "rows_empty_title": 0}),
    # an empty path and a blank first element are rows with no title
    ([{"path": [], "rows": 2}, {"path": ["  ", "x"], "rows": 1},
      {"path": ["T"], "rows": 6}],
     {"section_count": 1, "rows_per_section": [6], "rows_empty_title": 3}),
    ([], {"section_count": 0, "rows_per_section": [], "rows_empty_title": 0}),
])
def test_section_counts_are_by_title(sections, want):
    assert bpo.section_counts(sections) == want


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
        # section counts reach the public line as numbers only
        # (both sections carry the same title: one section of 5 rows)
        assert (summary["section_count"], summary["rows_per_section"],
                summary["rows_empty_title"]) == (1, [5], 0)
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


# --- the model reader (BILL_PLAN_READER=model) ------------------------------




@pytest.mark.parametrize("env,want", [
    ({}, bpo.Reader()),
    ({"BILL_PLAN_READER": " geometry "}, bpo.Reader()),
    ({"BILL_PLAN_READER": "model"}, bpo.Reader(bpo.READER_MODEL, 0.0)),
    ({"BILL_PLAN_READER": " MODEL ", "BILL_PLAN_TOTAL_TOLERANCE": "0.5"},
     bpo.Reader(bpo.READER_MODEL, 0.5)),
    ({"BILL_PLAN_READER": "model", "BILL_PLAN_TOTAL_TOLERANCE": "0"},
     bpo.Reader(bpo.READER_MODEL, 0.0)),
])
def test_parse_reader_reads_the_reader_and_its_tolerance(env, want):
    assert bpo.parse_reader(env) == want


@pytest.mark.parametrize("env", [
    {"BILL_PLAN_READER": "ocr"},
    {"BILL_PLAN_TOTAL_TOLERANCE": "0.5"},
    {"BILL_PLAN_READER": "geometry", "BILL_PLAN_TOTAL_TOLERANCE": "1"},
    {"BILL_PLAN_READER": "model", "BILL_PLAN_TOTAL_TOLERANCE": "-0.01"},
    {"BILL_PLAN_READER": "model", "BILL_PLAN_TOTAL_TOLERANCE": "nan"},
    {"BILL_PLAN_READER": "model", "BILL_PLAN_TOTAL_TOLERANCE": "inf"},
    {"BILL_PLAN_READER": "model", "BILL_PLAN_TOTAL_TOLERANCE": "half"},
])
def test_parse_reader_rejects_a_bad_reader_or_tolerance(env):
    with pytest.raises(ValueError):
        bpo.parse_reader(env)


@pytest.mark.parametrize("env", [
    {"BILL_PLAN_READER": "ocr"},
    {"BILL_PLAN_READER": "model", "BILL_PLAN_TOTAL_TOLERANCE": "-1"},
])
def test_main_rejects_a_bad_reader_before_any_io(env):
    calls: list = []
    lines: list[str] = []
    deps = _deps({"d1": {"project_id": "p1"}}, lambda *a, **k: calls.append(a), lines)
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1", **env}
    assert bpo.main_from_env(env, deps=deps) == 2
    assert calls == [] and _summary(lines) == {"ok": False, "error": bpo.ERR_BAD_READER}


@pytest.mark.parametrize("tolerance", ["", "0.5"])
def test_main_refuses_the_model_reader_when_it_is_not_wired(tolerance):
    calls: list = []
    lines: list[str] = []
    deps = _deps({"d1": {"project_id": "p1"}}, lambda *a, **k: calls.append(a), lines)
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_READER": "model", "BILL_PLAN_TOTAL_TOLERANCE": tolerance}
    assert bpo.main_from_env(env, deps=deps) == 2
    assert calls == [] and _summary(lines)["error"] == bpo.ERR_BAD_READER


@pytest.mark.parametrize("tolerance", [0.0, 0.5])
def test_model_reader_hands_one_pair_to_plan_and_derive(tolerance):
    got: list = []
    pair = (object(), object())

    def model_read(doc, wd, tol):
        got.append((doc, wd, tol))
        return pair

    def plan(*a, **k):
        return k

    def derive(*a, **k):
        return k

    deps = bpo.Deps(get_document={}.get, plan=plan, derive=derive, model_read=model_read)
    out = bpo.with_reader(deps, bpo.Reader(bpo.READER_MODEL, tolerance), "d9", "/w")
    assert got == [("d9", "/w", tolerance)]
    want = {"rows_of_reads": pair[0], "states_of_rows": pair[1]}
    assert out.plan() == want and out.derive() == want
    no_derive = bpo.with_reader(
        bpo.Deps(get_document={}.get, plan=plan, model_read=model_read),
        bpo.Reader(bpo.READER_MODEL, tolerance), "d9", "/w")
    assert no_derive.derive is None


@pytest.mark.parametrize("reader", [bpo.Reader(), bpo.Reader(bpo.READER_GEOMETRY, 0.0)])
def test_geometry_reader_leaves_deps_as_they_are(reader):
    def model_read(*a):
        raise AssertionError("model reader asked for")

    deps = bpo.Deps(get_document={}.get, plan=lambda *a, **k: k, model_read=model_read)
    assert bpo.with_reader(deps, reader, "d1", "/w") is deps


@pytest.mark.parametrize("plan,want", [
    ({"rows": 3}, {}),
    ({"bill_states": None}, {}),
    ({"pages_not_read": [2, 5], "bill_states": {
        "rows_unassigned": 4,
        "bills": [{"state": "RECONCILED"}, {"state": "NOT READ"}, {"state": "RECONCILED"}]}},
     {"pages_not_read": 2, "rows_unassigned": 4,
      "bill_state_counts": {"reconciled": 2, "not_read": 1}}),
    ({"bill_states": {"rows_unassigned": 0, "bills": [{"state": "TOTAL ONLY"}]}},
     {"pages_not_read": 0, "rows_unassigned": 0, "bill_state_counts": {"total_only": 1}}),
])
def test_state_counts_are_counts_of_the_plan_states(plan, want):
    assert bpo.state_counts(plan) == want


class _Page:
    def __init__(self, index):
        self.index = index


@pytest.mark.parametrize("first,second", [
    ([{"type": "item", "amount": "5"}], [{"type": "item", "amount": "7"}]),
    ([{"type": "heading", "description": "x"}], []),
])
def test_saved_page_reader_saves_lines_and_gives_them_back(tmp_path, first, second):
    replies = [first, second]
    asked: list[int] = []

    def inner(page):
        asked.append(page.index)
        return replies.pop(0)

    read = bpr.saved_page_reader(inner, str(tmp_path / "wd"))
    assert read(_Page(3)) == first
    assert read(_Page(3)) == first  # given back, not asked again
    assert read(_Page(4)) == second
    assert asked == [3, 4]
    assert (tmp_path / "wd" / bpr.SAVED_READ_NAME.format(index=3)).is_file()


@pytest.mark.parametrize("not_read", [
    {bpr.NOT_READ_KEY: True, "attempts": 2, "replies": ["", "x"]},
    None,
    # a list that holds no line object is not a read either: never saved
    ["page too faint to read"],
    [None],
])
def test_saved_page_reader_asks_a_page_not_read_again(tmp_path, not_read):
    asked: list[int] = []

    def inner(page):
        asked.append(page.index)
        return not_read

    read = bpr.saved_page_reader(inner, str(tmp_path / "wd"))
    assert read(_Page(0)) == not_read and read(_Page(0)) == not_read
    assert asked == [0, 0]
    assert not (tmp_path / "wd").exists() or not any((tmp_path / "wd").iterdir())


def _line(kind, bill, amount, **kw):
    return {"type": kind, "bill": bill, "amount": amount, **kw}


def _two_bill_reads(total_b: str):
    """Two bills over three pages; the summary on page 3 prints each total."""
    return {
        0: [_line("heading", "Bill A", None, description="h"),
            _line("item", "Bill A", "10", quantity="2", rate="5", ref="1")],
        1: [_line("item", "Bill B", "6", quantity="3", rate="2", ref="1"),
            _line("item", "Bill B", "4", quantity="1", rate="4", ref="2")],
        2: [_line("summary", "Bill A", "10"), _line("summary", "Bill B", total_b)],
    }


@pytest.mark.parametrize("total_b,tolerance,want_counts", [
    ("10", 0.0, {"reconciled": 2}),
    ("10.5", 0.0, {"reconciled": 1, "failed": 1}),
    ("10.5", 0.5, {"reconciled": 2}),
])
def test_main_plans_with_the_model_reader_and_checks_each_bill(
    tmp_path, monkeypatch, total_b, tolerance, want_counts,
):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=3)
    docs.install(monkeypatch, tmp_path)
    reads = _two_bill_reads(total_b)
    asked: list[int] = []

    def model(page):
        asked.append(page.index)
        return reads[page.index]

    def model_read(doc, wd, tol):
        return bpr.model_bill_read(bpr.saved_page_reader(model, wd), tolerance=tol)

    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path / "wd"), "BILL_PLAN_READER": "model",
           "BILL_PLAN_TOTAL_TOLERANCE": str(tolerance)}
    reports = []
    for _ in range(2):
        lines: list[str] = []
        deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL,
                     model_read=model_read)
        assert bpo.main_from_env(env, deps=deps) == 0
        reports.append(_summary(lines))

    rep = reports[0]
    assert rep["reader"] == bpo.READER_MODEL
    # every non-heading line is a row with its line type; only items are summed
    assert rep["rows"] == 5
    assert rep["bill_state_counts"] == want_counts
    # the summary page belongs to no bill: its 2 lines are counted, never moved
    assert rep["pages_not_read"] == 0 and rep["rows_unassigned"] == 2
    assert asked == [0, 1, 2]  # the second run gave the saved reads back
    assert reports[1]["plan_hash"] == rep["plan_hash"]
    assert docs.added == [] and docs.indexed == []
    saved = json.loads((tmp_path / "wd" / "plan_0_3.json").read_text(encoding="utf-8"))
    assert [b["label"] for b in saved["bill_states"]["bills"]] == ["Bill A", "Bill B"]
    assert saved["bill_states"]["tolerance"] == tolerance
    _assert_public(rep, ["Bill A", "Bill B"])


@pytest.mark.parametrize("bad_page", [0, 1])
def test_model_reader_page_not_read_is_counted_and_asked_again(tmp_path, monkeypatch, bad_page):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=3)
    docs.install(monkeypatch, tmp_path)
    reads = _two_bill_reads("10")
    # The bill of the page not read has no page left: no span, its summary
    # total is listed as a bill without pages; the other bill still checks.
    reads[bad_page] = {bpr.NOT_READ_KEY: True, "attempts": 2, "replies": []}

    def model_read(doc, wd, tol):
        return bpr.model_bill_read(bpr.saved_page_reader(lambda p: reads[p.index], wd),
                                   tolerance=tol)

    lines: list[str] = []
    deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL, model_read=model_read)
    env = {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
           "BILL_PLAN_WORK_DIR": str(tmp_path / "wd"), "BILL_PLAN_READER": "model"}
    assert bpo.main_from_env(env, deps=deps) == 0
    rep = _summary(lines)
    assert rep["pages_not_read"] == 1
    saved = json.loads((tmp_path / "wd" / "plan_0_3.json").read_text(encoding="utf-8"))
    lost = ["Bill A", "Bill B"][bad_page]
    assert saved["bill_states"]["summary_without_pages"] == [lost]
    states = {b["label"]: b for b in saved["bill_states"]["bills"]}
    # the bill whose only page was not read still has a state: NOT READ
    assert states[lost]["state"] == bpr.STATE_NOT_READ and states[lost]["pages"] is None
    assert states[lost]["pages_not_read"] == [bad_page + 1]
    # The bill left on the other page could reach the unread page too (no
    # other page names a bill between them), so it cannot be checked either.
    assert rep["bill_state_counts"] == {"not_read": 2}
    kept = states[["Bill B", "Bill A"][bad_page]]
    assert kept["state"] == bpr.STATE_NOT_READ and kept["pages_not_read"] == [bad_page + 1]
    assert not (tmp_path / "wd" / bpr.SAVED_READ_NAME.format(index=bad_page)).exists()


def _bill_a_three_pages(missing: int):
    """Bill A on pages 1-3, an item of 10 on each; the summary on page 4 prints 30."""
    reads = {i: [_line("item", "Bill A", "10", quantity="1", rate="10", ref=str(i))]
             for i in range(3)}
    reads[3] = [_line("summary", "Bill A", "30")]
    reads[missing] = {bpr.NOT_READ_KEY: True, "attempts": 2, "replies": []}
    return reads


def _model_env(tmp_path):
    return {"BILL_PLAN_PROJECT_ID": "p1", "BILL_PLAN_DOCUMENT_ID": "d1",
            "BILL_PLAN_WORK_DIR": str(tmp_path / "wd"), "BILL_PLAN_READER": "model"}


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_bill_with_its_first_middle_or_last_page_not_read_is_not_read(
        tmp_path, monkeypatch, missing):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=4)
    docs.install(monkeypatch, tmp_path)
    reads = _bill_a_three_pages(missing)

    def model_read(doc, wd, tol):
        return bpr.model_bill_read(bpr.saved_page_reader(lambda p: reads[p.index], wd),
                                   tolerance=tol)

    lines: list[str] = []
    deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL, model_read=model_read)
    assert bpo.main_from_env(_model_env(tmp_path), deps=deps) == 0
    rep = _summary(lines)
    assert rep["pages_not_read"] == 1
    assert rep["bill_state_counts"] == {"not_read": 1}
    saved = json.loads((tmp_path / "wd" / "plan_0_4.json").read_text(encoding="utf-8"))
    (bill,) = saved["bill_states"]["bills"]
    assert bill["pages_not_read"] == [missing + 1] and bill["items_sum"] == 20


@pytest.mark.parametrize("junk", ['["page too faint to read"]', "[null]", "[[]]"])
@pytest.mark.parametrize("edge", [0, 2])
def test_junk_reply_on_a_bills_edge_page_is_not_read_and_asked_again(
        tmp_path, monkeypatch, junk, edge):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=4)
    docs.install(monkeypatch, tmp_path)
    good = _bill_a_three_pages(1)
    good[1] = [_line("item", "Bill A", "10", quantity="1", rate="10", ref="1")]
    asked: list[int] = []

    def model_read(doc, wd, tol):
        def ask_for(index):
            def ask(messages):
                asked.append(index)
                if index == edge:
                    return '{"lines": ' + junk + '}'
                return json.dumps({"lines": good[index]})
            return ask

        def page(p):
            return bpr.model_page_reader(ask_for(p.index), lambda i: b"png")(p)
        return bpr.model_bill_read(bpr.saved_page_reader(page, wd), tolerance=tol)

    for _run in range(2):
        lines: list[str] = []
        deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL,
                     model_read=model_read)
        assert bpo.main_from_env(_model_env(tmp_path), deps=deps) == 0
        rep = _summary(lines)
        assert rep["pages_not_read"] == 1
        assert rep["bill_state_counts"] == {"not_read": 1}
        saved = json.loads((tmp_path / "wd" / "plan_0_4.json").read_text(encoding="utf-8"))
        (bill,) = saved["bill_states"]["bills"]
        assert bill["pages_not_read"] == [edge + 1] and bill["items_sum"] == 20
    # the junk page is never saved: each run asks it again (twice per run,
    # PAGE_READ_ATTEMPTS), the read pages only once
    attempts = bpr.PAGE_READ_ATTEMPTS
    assert asked.count(edge) == 2 * attempts
    assert sorted(set(asked) - {edge}) == sorted({0, 1, 2, 3} - {edge})
    assert all(asked.count(i) == 1 for i in set(asked) - {edge})
    assert not (tmp_path / "wd" / bpr.SAVED_READ_NAME.format(index=edge)).exists()


@pytest.mark.parametrize("junk", [["page too faint"], [None]])
@pytest.mark.parametrize("good", [[{"type": "item", "amount": "5"}],
                                  [{"type": "item", "amount": "8"}]])
def test_saved_file_that_is_not_a_read_is_asked_again(tmp_path, junk, good):
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / bpr.SAVED_READ_NAME.format(index=2)).write_text(json.dumps(junk), encoding="utf-8")
    asked: list[int] = []

    def inner(page):
        asked.append(page.index)
        return good

    read = bpr.saved_page_reader(inner, str(wd))
    assert read(_Page(2)) == good and read(_Page(2)) == good
    assert asked == [2]


@pytest.mark.parametrize("printed, want", [("30", {"reconciled": 1}), ("31", {"failed": 1})])
def test_bill_with_every_page_read_is_checked_against_its_total(
        tmp_path, monkeypatch, printed, want):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=4)
    docs.install(monkeypatch, tmp_path)
    reads = _bill_a_three_pages(0)
    reads[0] = [_line("item", "Bill A", "10", quantity="1", rate="10", ref="0")]
    reads[3] = [_line("summary", "Bill A", printed)]

    def model_read(doc, wd, tol):
        return bpr.model_bill_read(lambda p: reads[p.index], tolerance=tol)

    lines: list[str] = []
    deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL, model_read=model_read)
    assert bpo.main_from_env(_model_env(tmp_path), deps=deps) == 0
    rep = _summary(lines)
    assert rep["pages_not_read"] == 0 and rep["bill_state_counts"] == want


@pytest.mark.parametrize("pages, error", [(3, RuntimeError), (2, httpx.ConnectError)])
def test_provider_failure_on_every_page_is_pages_not_read_not_no_rows(
        tmp_path, monkeypatch, pages, error):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=pages)
    docs.install(monkeypatch, tmp_path)
    calls: list[int] = []

    def ask(messages):
        calls.append(1)
        raise error("provider down")

    def model_read(doc, wd, tol):
        reader = bpr.model_page_reader(ask, lambda i: b"png")
        return bpr.model_bill_read(bpr.saved_page_reader(reader, wd), tolerance=tol)

    lines: list[str] = []
    deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL, model_read=model_read)
    assert bpo.main_from_env(_model_env(tmp_path), deps=deps) == 1
    rep = _summary(lines)
    assert rep["error"] == bpo.ERR_NO_PLAN
    assert rep["reason"] == bd.SKIP_PAGES_NOT_READ != bd.SKIP_NO_ROWS
    assert rep["pages_not_read"] == pages
    assert len(calls) == pages * bpr.PAGE_READ_ATTEMPTS


@pytest.mark.parametrize("pages", [2, 3])
def test_pages_read_with_no_rows_is_still_no_rows(tmp_path, monkeypatch, pages):
    words, _, _ = _bill_a()
    docs = _Docs({**_source("d1"), "project_id": "p1"}, pages=pages)
    docs.install(monkeypatch, tmp_path)

    def model_read(doc, wd, tol):
        return bpr.model_bill_read(lambda p: [{"type": "heading", "description": "h"}],
                                   tolerance=tol)

    lines: list[str] = []
    deps = _deps(docs.rows, _real_plan(words, []), lines, classify=_BILL, model_read=model_read)
    assert bpo.main_from_env(_model_env(tmp_path), deps=deps) == 1
    rep = _summary(lines)
    assert rep["reason"] == bd.SKIP_NO_ROWS and "pages_not_read" not in rep


@pytest.mark.parametrize("states, not_read, want", [
    ({"bills": [{"state": "RECONCILED"}], "rows_unassigned": 2}, [5], {"reconciled": 1}),
    ({"bills": [{"state": "NOT READ"}, {"state": "FAILED"}], "rows_unassigned": 0}, [],
     {"not_read": 1, "failed": 1}),
])
def test_execute_summary_carries_the_bill_state_counts(tmp_path, states, not_read, want):
    plan = {"reason": "bill", "pages": [0, 1], "plan_hash": "h1", "rows": 1,
            "sections": {}, "pages_not_read": not_read, "bill_states": states}
    deps = bpo.Deps(get_document={"d1": {"project_id": "p1"}}.get,
                    plan=lambda *a, **k: plan, derive=lambda *a, **k: {},
                    log=lambda _line: None)
    rep = bpo.execute_one("p1", "d1", execute_hash="other", gates=bpo.Gates(),
                          work_dir=str(tmp_path), deps=deps)
    assert rep["pages_not_read"] == len(not_read)
    assert rep["rows_unassigned"] == states["rows_unassigned"]
    assert rep["bill_state_counts"] == want
