"""scripts/trace_page.py: one static page per run from its trace lines (Q-2).

Every test builds its own trace lines and rows. Two inputs per decision.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "trace_page", Path(__file__).resolve().parents[1] / "scripts" / "trace_page.py"
)
tp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tp)

T0 = 1_800_000_000.0  # an epoch second; rows and lines are built around it


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone
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


# ── parsing: plain and JSON-wrapped messages ─────────────────────────────

def test_plain_and_wrapped_messages_both_parse(tmp_path):
    for wrapped in (False, True):
        f = tmp_path / f"ev_{wrapped}.json"
        f.write_text(json.dumps({"events": [_event(T0, _payload(), wrapped),
                                            {"timestamp": 1, "message": "other line"}]}))
        got = tp.load_events(str(f))
        assert len(got) == 1 and got[0][1]["handed_k"] == 2


def test_one_event_per_line_export_parses(tmp_path):
    f = tmp_path / "ev.jsonl"
    f.write_text("\n".join(json.dumps(_event(T0 + i, _payload(handed_k=i))) for i in (3, 1)))
    got = tp.load_events(str(f))
    assert [p["handed_k"] for _, p in got] == [1, 3]  # sorted by time


# ── join by time window ──────────────────────────────────────────────────

def test_each_line_goes_to_the_row_whose_window_covers_it():
    rows = [tp.row_from_record(_row("A1", T0, 30)), tp.row_from_record(_row("A2", T0 + 100, 30))]
    events = [(T0 + 5, _payload()), (T0 + 110, _payload()), (T0 + 60, _payload())]
    by_row, unmatched = tp.assign(events, rows, slack_s=10)
    assert len(by_row[0]) == 1 and len(by_row[1]) == 1
    assert unmatched == 1  # +60 s is past A1's 30 s + 10 s and before A2 starts


def test_slack_decides_a_line_just_after_the_end():
    rows = [tp.row_from_record(_row("A1", T0, 30))]
    events = [(T0 + 38, _payload())]
    assert tp.assign(events, rows, slack_s=10)[1] == 0
    assert tp.assign(events, rows, slack_s=5)[1] == 1


# ── nothing but ids, ranks, scores ───────────────────────────────────────

def test_names_and_question_text_never_reach_the_page():
    payload = _payload(doc_names=["Secret Bill of Quantities.pdf"], query="what is the slab depth")
    payload["chunks"][0]["doc_name"] = "Secret Bill of Quantities.pdf"
    payload["chunks"][0]["doc_id"] = "Secret Bill.pdf"
    payload["chunks"][1]["text"] = "PLANTED CHUNK TEXT"
    rows = [tp.row_from_record(_row("A1", T0, 30, question="PLANTED QUESTION",
                                    answer="PLANTED ANSWER", expected="PLANTED EXPECTED"))]
    by_row, _ = tp.assign([(T0 + 1, payload)], rows)
    page = tp.render("run", rows, by_row, 0)
    for planted in ("Secret", "PLANTED", "slab depth", "Bill of Quantities"):
        assert planted not in page
    # what the page is for is still there
    assert "c9" in page and "d3" in page and "token_cap" in page and "other" in page


def test_ids_pass_and_names_are_withheld():
    assert tp._safe_str("4f1c-d7_x:9") == "4f1c-d7_x:9"
    assert tp._safe_str("chunk.0012") == "chunk.0012"
    assert tp._safe_str("Site Plan Rev B") == tp.WITHHELD
    assert tp._safe_str("drawing_A101.pdf") == tp.WITHHELD


def test_query_label_is_kept_only_when_it_is_a_product_code():
    assert tp.project_trace(_payload(query="followup_context"))["summary"]["query"] == "followup_context"
    assert tp.project_trace(_payload(query="pipe"))["summary"]["query"] == "other"


def test_markup_in_an_id_is_escaped():
    payload = _payload()
    payload["fused_candidates"]["leg_top"] = {"<b>": []}
    rows = [tp.row_from_record(_row("A1", T0, 30))]
    by_row, _ = tp.assign([(T0 + 1, payload)], rows)
    page = tp.render("<run>", rows, by_row, 0)
    assert "<b>" not in page and "<run>" not in page


# ── what was cut and why; fused order ────────────────────────────────────

def test_cut_is_counted_by_rule():
    one = _payload()
    two = _payload(not_handed=[
        {"chunk_id": "c5", "dropped_by": "token_cap"},
        {"chunk_id": "c6", "dropped_by": "per_doc_cap"},
        {"chunk_id": "c7"},
    ])
    assert tp.project_trace(one)["cut_by"] == {"token_cap": 1}
    assert tp.project_trace(two)["cut_by"] == {"token_cap": 1, "per_doc_cap": 1, "unexplained": 1}


def test_fused_order_interleaves_handed_and_cut_by_rank():
    view = tp.project_trace(_payload())
    assert [(r["chunk_id"], r["handed"]) for r in tp._fused_order(view)] == [
        ("c9", "yes"), ("c1", "yes"), ("c5", "no")]
    view = tp.project_trace(_payload(chunks=[{"chunk_id": "c9", "fused_rank": 4}]))
    assert [r["chunk_id"] for r in tp._fused_order(view)] == ["c5", "c9"]


# ── Q-1 scores ride along, numbers only ──────────────────────────────────

def test_scores_keep_numbers_and_drop_text(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps({"row": "A1", "rep": 1, "numbers_found": 3, "flag": True,
                             "note": "PLANTED NOTE"}) + "\n")
    got = tp.load_scores(str(f))
    assert got == {("A1", 1): {"numbers_found": 3, "flag": True}}
    assert tp.load_scores(None) == {}


def test_cli_writes_one_page(tmp_path):
    (tmp_path / "A1_v2_x.json").write_text(json.dumps(_row("A1", T0, 30)))
    (tmp_path / "A2_v2_x.json").write_text(json.dumps(_row("A2", T0 + 100, 30)))
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"events": [_event(T0 + 2, _payload()), _event(T0 + 101, _payload())]}))
    out = tmp_path / "page.html"
    assert tp.main(["--events", str(ev), "--rows", str(tmp_path / "*_v2_*.json"),
                    "--out", str(out), "--label", "run1"]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.count("<h2>row ") == 2 and "unmatched: 0" in page
    assert tp.main(["--events", str(ev), "--rows", str(tmp_path / "none_*.json"),
                    "--out", str(out)]) == 2


def test_unreadable_inputs_are_counted_not_silent():
    tp.SKIPPED.clear()
    assert tp._trace_payload("rag_retrieval_trace {broken") is None
    assert tp._trace_payload("rag_retrieval_trace " + json.dumps(_payload())) is not None
    assert tp.SKIPPED == {"undecodable_trace_line": 1}
    assert tp.row_from_record({"id": "A1", "time_utc": "not a time"}) is None
    assert tp.row_from_record(_row("A1", T0, 30)) is not None
    assert tp.SKIPPED["row_without_start_time"] == 1
