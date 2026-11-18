"""scripts/trace_page.py: one static page per run from its trace lines (Q-2).

Every test builds its own trace lines and rows. Two inputs per decision.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "trace_page", Path(__file__).resolve().parents[1] / "scripts" / "trace_page.py"
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)

T0 = 1_800_000_000.0  # an epoch second; rows and lines are built around it


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload(**over):
    p = {
        "request_id": "req-1", "path": "pre_injection", "query": "user_message",
        "query_chars": 42, "retrieved_k": 3, "handed_k": 2, "threshold_fired": False,
        "top_score": 0.81,
        "fused_candidates": {
            "n_candidates": 3, "n_both_legs": 1, "fused_k": 3,
            "leg_top": {"vector": [{"chunk_id": "c1", "leg_rank": 1, "other_leg_rank": 4,
                                    "fused_rank": 2, "chars": 900, "handed_position": 2}]},
        },
        "chunks": [
            {"rank": 1, "chunk_id": "c9", "doc_id": "d7", "score": 0.81, "fused_rank": 1, "chars": 500},
            {"rank": 2, "chunk_id": "c1", "doc_id": "d3", "score": 0.7, "fused_rank": 2, "chars": 900},
        ],
        "not_handed": [
            {"rank": 3, "chunk_id": "c5", "doc_id": "d3", "score": 0.4, "fused_rank": 3,
             "chars": 1200, "dropped_by": "token_cap"},
        ],
    }
    p.update(over)
    return p


def _event(ts: float, payload, wrapped: bool = False):
    text = "rag_retrieval_trace " + json.dumps(payload)
    msg = json.dumps({"level": "INFO", "message": text}) if wrapped else text
    return {"timestamp": int(ts * 1000), "message": msg}


def _row(rid, start, t_end, **over):
    rec = {"id": rid, "rep": 1, "time_utc": _iso(start), "sha_before": "abc1234",
           "sha_after": "abc1234", "question": "Q?", "expected": "E", "answer": "x" * 120,
           "sources": {"list": "S"}, "timing": {"t_end": t_end}}
    rec.update(over)
    return rec


# ── parsing: export shapes ───────────────────────────────────────────────

@pytest.mark.parametrize("shape", ["events_plain", "events_wrapped", "jsonl", "list"])
def test_every_export_shape_parses_and_skips_other_lines(tmp_path, shape):
    wrapped = shape == "events_wrapped"
    evs = [_event(T0 + 3, _payload(handed_k=3), wrapped), {"timestamp": 1, "message": "other line"},
           _event(T0 + 1, _payload(handed_k=1), wrapped)]
    f = tmp_path / "ev"
    if shape == "jsonl":
        f.write_text("\n".join(json.dumps(e) for e in evs) + "\nnot json\n")
    elif shape == "list":
        f.write_text(json.dumps(evs))
    else:
        f.write_text(json.dumps({"events": evs}))
    got = tp.load_events(str(f))
    assert [p["handed_k"] for _, p in got] == [1, 3]  # trace lines only, by time


# ── join by time window ──────────────────────────────────────────────────

@pytest.mark.parametrize("offset,expected", [
    (5, (1, 0, 0)),      # inside A1
    (110, (0, 1, 0)),    # inside A2
    (60, (0, 0, 1)),     # past A1's 30 s + 10 s slack, before A2 starts
    (-5, (0, 0, 1)),     # before any row
])
def test_each_line_goes_to_the_row_whose_window_covers_it(offset, expected):
    rows = [tp.row_from_record(_row("A1", T0, 30)), tp.row_from_record(_row("A2", T0 + 100, 30))]
    by_row, unmatched = tp.assign([(T0 + offset, _payload())], rows, slack_s=10)
    assert (len(by_row[0]), len(by_row[1]), unmatched) == expected


@pytest.mark.parametrize("slack,unmatched", [(10, 0), (5, 1)])
def test_slack_decides_a_line_just_after_the_end(slack, unmatched):
    rows = [tp.row_from_record(_row("A1", T0, 30))]
    assert tp.assign([(T0 + 38, _payload())], rows, slack_s=slack)[1] == unmatched


# ── nothing but ids, ranks, scores ───────────────────────────────────────

@pytest.mark.parametrize("planted", [
    "Secret Bill of Quantities.pdf", "PLANTED QUESTION", "PLANTED ANSWER",
    "PLANTED EXPECTED", "PLANTED CHUNK TEXT", "what is the slab depth",
])
def test_names_and_question_text_never_reach_the_page(planted):
    payload = _payload(doc_names=["Secret Bill of Quantities.pdf"], query="what is the slab depth")
    payload["chunks"][0]["doc_name"] = "Secret Bill of Quantities.pdf"
    payload["chunks"][0]["doc_id"] = "Secret Bill of Quantities.pdf"
    payload["chunks"][1]["text"] = "PLANTED CHUNK TEXT"
    rows = [tp.row_from_record(_row("A1", T0, 30, question="PLANTED QUESTION",
                                    answer="PLANTED ANSWER", expected="PLANTED EXPECTED"))]
    by_row, _ = tp.assign([(T0 + 1, payload)], rows)
    page = tp.render("run", rows, by_row, 0)
    assert page.count(planted) == 0
    assert page.count(tp.WITHHELD) == 2  # the name-shaped doc_id: handed + fused order
    assert ("c9" in page, "d3" in page, "token_cap" in page) == (True, True, True)


@pytest.mark.parametrize("value,expected", [
    ("4f1c-d7_x:9", "4f1c-d7_x:9"),
    ("chunk.0012", "chunk.0012"),
    ("Site Plan Rev B", tp.WITHHELD),
    ("drawing_A101.pdf", tp.WITHHELD),
    (None, None),
])
def test_ids_pass_and_names_are_withheld(value, expected):
    assert tp._safe_str(value) == expected


@pytest.mark.parametrize("label,expected", [
    ("followup_context", "followup_context"),
    ("tool_query", "tool_query"),
    ("pipe", "other"),
    ("what is the slab depth", "other"),
])
def test_query_label_is_kept_only_when_it_is_a_product_code(label, expected):
    assert tp.project_trace(_payload(query=label))["summary"]["query"] == expected


@pytest.mark.parametrize("markup", ["<b>", "<script>"])
def test_markup_is_escaped(markup):
    payload = _payload()
    payload["fused_candidates"]["leg_top"] = {markup: []}
    rows = [tp.row_from_record(_row("A1", T0, 30))]
    by_row, _ = tp.assign([(T0 + 1, payload)], rows)
    page = tp.render(markup, rows, by_row, 0)
    assert page.count(markup) == 0


# ── what was cut and why; fused order ────────────────────────────────────

@pytest.mark.parametrize("cut,expected", [
    (None, {"token_cap": 1}),
    ([{"chunk_id": "c5", "dropped_by": "token_cap"},
      {"chunk_id": "c6", "dropped_by": "per_doc_cap"},
      {"chunk_id": "c7"}], {"token_cap": 1, "per_doc_cap": 1, "unexplained": 1}),
])
def test_cut_is_counted_by_rule(cut, expected):
    payload = _payload() if cut is None else _payload(not_handed=cut)
    assert tp.project_trace(payload)["cut_by"] == expected


@pytest.mark.parametrize("handed,expected", [
    (None, [("c9", "yes"), ("c1", "yes"), ("c5", "no")]),
    ([{"chunk_id": "c9", "fused_rank": 4}, {"chunk_id": "c2"}], [("c5", "no"), ("c9", "yes")]),
])
def test_fused_order_interleaves_handed_and_cut_by_rank(handed, expected):
    payload = _payload() if handed is None else _payload(chunks=handed)
    view = tp.project_trace(payload)
    assert [(r["chunk_id"], r["handed"]) for r in tp._fused_order(view)] == expected


# ── Q-1 scores ride along, numbers only ──────────────────────────────────

@pytest.mark.parametrize("record,expected", [
    ({"row": "A1", "rep": 1, "numbers_found": 3, "flag": True, "note": "PLANTED NOTE"},
     {("A1", 1): {"numbers_found": 3, "flag": True}}),
    ({"row": "B2", "rep": 2, "overlap": 0.25, "cites": [1, 2]},
     {("B2", 2): {"overlap": 0.25}}),
])
def test_scores_keep_numbers_and_drop_text(tmp_path, record, expected):
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps(record) + "\nnot json\n")
    assert tp.load_scores(str(f)) == expected


@pytest.mark.parametrize("pattern,code,sections", [("*_v2_*.json", 0, 2), ("none_*.json", 2, None)])
def test_cli_writes_one_page(tmp_path, pattern, code, sections):
    (tmp_path / "A1_v2_x.json").write_text(json.dumps(_row("A1", T0, 30)))
    (tmp_path / "A2_v2_x.json").write_text(json.dumps(_row("A2", T0 + 100, 30)))
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"events": [_event(T0 + 2, _payload()), _event(T0 + 101, _payload())]}))
    out = tmp_path / "page.html"
    assert tp.main(["--events", str(ev), "--rows", str(tmp_path / pattern),
                    "--out", str(out), "--label", "run1"]) == code
    if sections is not None:
        page = out.read_text(encoding="utf-8")
        assert (page.count("<h2>row "), "unmatched: 0" in page) == (sections, True)


@pytest.mark.parametrize("line,parsed,reason", [
    ("rag_retrieval_trace {broken", False, "undecodable_trace_line"),
    ("rag_retrieval_trace " + json.dumps({"handed_k": 1}), True, None),
])
def test_unreadable_trace_lines_are_counted_not_silent(line, parsed, reason):
    tp.SKIPPED.clear()
    assert (tp._trace_payload(line) is not None) == parsed
    assert dict(tp.SKIPPED) == ({reason: 1} if reason else {})


@pytest.mark.parametrize("stamp,kept", [("not a time", False), ("2026-11-18T06:26:46Z", True)])
def test_rows_without_a_start_time_are_counted_not_silent(stamp, kept):
    tp.SKIPPED.clear()
    row = tp.row_from_record({"id": "A1", "time_utc": stamp})
    assert (row is not None, tp.SKIPPED["row_without_start_time"]) == (kept, 0 if kept else 1)
