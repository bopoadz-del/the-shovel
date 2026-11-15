"""Exam 17 for exam 11: every exam-11 checker passes a known-good input and
fails a known-bad one (SPEC_EXAM_PLAN rule 3). All inputs are invented here.
"""
from __future__ import annotations

import json

import pytest

from tests.exam11_checks import (
    activity_mismatch,
    citation_leaks,
    process_talk,
    ran_labels,
    sse_events,
)

_CLEAN = [
    "The slab needs 945 m³ of concrete [1].",
    "Retention is 5% of each interim payment, released at completion [2].",
]


@pytest.mark.parametrize("text", _CLEAN)
def test_a_clean_answer_has_no_citation_leak(text):
    assert citation_leaks(text) == []


@pytest.mark.parametrize(
    "text,kind",
    [
        ("The slab needs 945 m³.\n\nSource: pour_schedule_rev3.xlsx", "source line"),
        ("Retention is 5% (source: Particular Conditions, clause 14.3).", "inline source"),
        ("Per north_wing_spec.pdf the cover is 40 mm.", "file name"),
        ("Cover is 40 mm (document_id: 3f2a9c1e-77b0-4d2a-9e1f-0c5d8b2a6e41).", "document id"),
        ("Cover is 40 mm. 12 of 30 documents are searchable.", "coverage count"),
    ],
)
def test_a_citation_in_the_answer_text_is_caught(text, kind):
    found = citation_leaks(text)
    assert any(f.startswith(kind) for f in found), found


@pytest.mark.parametrize("text", _CLEAN)
def test_a_clean_answer_has_no_process_talk(text):
    assert process_talk(text) == []


@pytest.mark.parametrize(
    "text,kind",
    [
        ("I ran construction_calc and got 945 m³.", "machine identifier"),
        ("The figure is deterministic and certified.", "machinery word"),
        ("I searched the project documents and found nothing on cover.", "search narration"),
    ],
)
def test_process_talk_in_the_answer_is_caught(text, kind):
    found = process_talk(text)
    assert any(f.startswith(kind) for f in found), found


def _stream(*events: dict) -> str:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


@pytest.mark.parametrize(
    "body,shown",
    [
        (_stream({"type": "status", "label": "Searching project documents"},
                 {"type": "token", "content": "x"}),
         ["Searching project documents"]),
        (_stream({"type": "tool_call", "tool": "read_bill", "args_preview": ""},
                 {"type": "done"}),
         ["Running read bill…"]),
    ],
)
def test_a_status_line_backed_by_an_event_of_this_turn_holds(body, shown):
    assert activity_mismatch(shown, ran_labels(sse_events(body))) == []


@pytest.mark.parametrize(
    "body,shown,expect",
    [
        (_stream({"type": "status", "label": "Searching project documents"}),
         ["Calculating"], "no backend event"),
        (_stream({"type": "tool_call", "tool": "read_bill"}),
         ["Running read bill…", "<after answer>"], "after the answer"),
    ],
)
def test_a_false_or_lingering_status_line_is_caught(body, shown, expect):
    found = activity_mismatch(shown, ran_labels(sse_events(body)))
    assert any(expect in f for f in found), found


@pytest.mark.parametrize(
    "body,types",
    [
        (_stream({"type": "start"}, {"type": "heartbeat"}, {"type": "done"}),
         ["start", "heartbeat", "done"]),
        ("data: not json\n\n" + _stream({"type": "token", "content": "a"}), ["token"]),
    ],
)
def test_the_stream_reader_keeps_typed_events_in_order(body, types):
    assert [e["type"] for e in sse_events(body)] == types
