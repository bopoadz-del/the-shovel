"""A per-hop compaction must say what it lost, and lose as little as possible.

Live on 554e0b9, Master Corpus. Asked how many cubic metres of demolition are
in the BOQ, the platform answered:

    "Neither the injected excerpts nor the extracted raw pages (pages 1-6 of
     16) contain any demolition line items measured in cubic metres (m3). ...
     The BOQ processor output was truncated after page 6, so I cannot see the
     remainder of the document."

The model behaved well -- it disclosed the truncation and did not fabricate.
The platform did not. boq_processor returned the whole document and
_compact_messages_for_tpm cut it to a 1000-character preview whose only
explanation was "Compacted for Groq TPM 8000". The model had to infer the
extent of the loss from the content, and inferred "pages 1-6 of 16".

This is the SECOND truncation a large tool result passes through.
_tool_result_content caps it at _TOOL_RESULT_MAX_CHARS when the tool
message is built (tests/test_tool_result_truncation.py); this pass then
cuts what survived, per Groq hop. Neither layer said how much was gone.

Two defects, one test file:

1. The loss was not stated, and the absence claim was not forbidden.
   inject.py has applied SCOPE OF ABSENCE to retrieval excerpts since the
   Saudi Building Code incident, for exactly this reason. Tool results never
   got the rule, and they carry more authority than excerpts do.

2. Every tool result over 1200 chars was cut to the same 1000 chars whether
   or not the budget required it, and the LAST one -- the one the question is
   about -- was cut as hard as a stale one from iteration 0.

Groq is the production provider, so this ran on every large tool result.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _LAST_TOOL_PREVIEW,
    _TOOL_PREVIEW_FLOOR,
    _approx_message_chars,
    _compact_messages_for_tpm,
    _compacted_tool_payload,
)


def _boq(rows: int = 400) -> str:
    """A BOQ-shaped result whose answer is past any small preview."""
    body = [
        {"code": f"D{500 + i}", "desc": f"Breakout item {i}", "unit": "m", "qty": i}
        for i in range(rows)
    ]
    body.append({"code": "D550.9", "desc": "Demolition, concrete", "unit": "m3",
                 "qty": 945})
    return json.dumps({"status": "success", "items": body})


def _turn(*tool_contents: str, user: str = "How many m3 of demolition?"):
    msgs: list = [
        {"role": "system", "content": "you are a construction assistant"},
        {"role": "user", "content": user},
    ]
    for n, content in enumerate(tool_contents):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": str(n), "type": "function",
             "function": {"name": "boq_processor", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": str(n),
                     "name": "boq_processor", "content": content})
    return msgs


def _tools(out):
    return [m for m in out if isinstance(m, dict) and m.get("role") == "tool"]


# -- the payload says what it lost ----------------------------------------


def test_the_payload_states_the_loss_as_a_number():
    """Mutation killed: keeping the old note-only payload, which is what
    made the model infer "pages 1-6 of 16"."""
    content = "x" * 40000
    payload = json.loads(_compacted_tool_payload(content, 1000))
    assert payload["truncated"] is True
    assert payload["chars_shown"] == 1000
    assert payload["chars_dropped"] == 39000
    assert payload["chars_total"] == 40000
    assert payload["preview"] == content[:1000]


def test_the_payload_forbids_the_absence_claim():
    """The dangerous output is not "I could not see it all" -- the model
    said that. It is "the BOQ contains no m3 demolition", said with the
    tool's authority, on the strength of 2.5% of the document."""
    text = json.loads(_compacted_tool_payload("y" * 9000, 1000))["scope_of_absence"]
    assert "NEVER state" in text
    assert "absent" in text
    assert "1000" in text and "9000" in text and "8000" in text
    # And it must say what to do instead, or the instruction is just a scold.
    assert "re-run" in text or "narrow" in text


def test_a_result_shorter_than_the_keep_reports_no_loss():
    payload = json.loads(_compacted_tool_payload("short", 1000))
    assert payload["chars_dropped"] == 0
    assert payload["chars_shown"] == len("short")
    assert payload["preview"] == "short"


# -- only compact what the budget requires --------------------------------


def test_a_hop_already_under_budget_is_untouched():
    msgs = _turn(_boq(5))
    assert _compact_messages_for_tpm(msgs, budget=1_000_000) is msgs


def test_the_last_tool_result_keeps_the_larger_share():
    """It is the one the answer is about. Mutation killed: giving every tool
    result the same floor, which is what cut the just-parsed BOQ to 1000
    chars alongside a stale result from iteration 0."""
    stale, fresh = _boq(300), _boq(300)
    out = _compact_messages_for_tpm(_turn(stale, fresh), budget=8000)
    tools = _tools(out)
    assert len(tools) == 2
    first = json.loads(tools[0]["content"])
    second = json.loads(tools[1]["content"])
    assert first["chars_shown"] == _TOOL_PREVIEW_FLOOR
    # A relationship, not the constant: the last result's share is capped at
    # _LAST_TOOL_PREVIEW but is trimmed further when the hop still does not
    # fit -- tool results yield before retrieved evidence does. Asserting
    # the constant here would fail the moment that trimming is correct.
    assert first["chars_shown"] < second["chars_shown"] <= _LAST_TOOL_PREVIEW


def test_an_older_compacted_result_still_carries_data():
    """The floor is not decoration: it is the promise that a compacted
    result is still worth reading. Cut to nothing, the model reasons about
    the compaction note instead of the tool output -- which is the failure
    the whole envelope exists to prevent.

    Mutation killed: _TOOL_PREVIEW_FLOOR = 0, which every other assertion
    here survives because they compare against the constant itself.
    """
    stale, fresh = _boq(400), _boq(400)
    out = _compact_messages_for_tpm(_turn(stale, fresh), budget=8000)
    payload = json.loads(_tools(out)[0]["content"])
    assert payload["chars_shown"] >= 500, "a preview must be worth reading"
    # Recognisable content, not just length: real rows the model can use.
    assert '"code"' in payload["preview"]
    assert '"qty"' in payload["preview"]
    assert payload["preview"].startswith(stale[:200])


def test_compaction_stops_as_soon_as_the_hop_fits():
    """A turn 200 chars over budget must not lose 39k characters.

    Mutation killed: compacting every oversized tool result unconditionally.
    """
    small_over = "a" * 3000
    # Bigger than the last-result share, so "compact everything" would
    # visibly shrink it and this assertion can tell the two apart.
    fresh = _boq(400)
    assert len(fresh) > _LAST_TOOL_PREVIEW
    msgs = _turn(small_over, fresh)
    total = _approx_message_chars(msgs)
    # Over budget, but only just: compacting the OLDEST alone must suffice.
    budget = total - 1500
    out = _compact_messages_for_tpm(msgs, budget=budget)
    tools = _tools(out)
    assert json.loads(tools[0]["content"])["truncated"] is True
    # The result the question is about is untouched -- still raw JSON with
    # its data in it, not a compaction envelope.
    assert tools[1]["content"] == fresh
    assert "chars_dropped" not in tools[1]["content"]


def test_the_last_result_is_cut_to_the_floor_when_nothing_else_is_left():
    """Correctness before generosity: a hop that still will not fit must
    shrink, even if that costs the fresh result its larger share."""
    a, b = _boq(500), _boq(500)
    out = _compact_messages_for_tpm(_turn(a, b), budget=3000)
    tools = _tools(out)
    for t in tools:
        assert json.loads(t["content"])["chars_shown"] == _TOOL_PREVIEW_FLOOR


def test_a_single_tool_result_gets_the_larger_share():
    """The common case: one tool, one answer. It is both first and last."""
    out = _compact_messages_for_tpm(_turn(_boq(400)), budget=9000)
    payload = json.loads(_tools(out)[0]["content"])
    assert payload["chars_shown"] == _LAST_TOOL_PREVIEW
    assert payload["chars_dropped"] > 0


def _boq_with_answer_at(offset_rows: int, tail_rows: int) -> str:
    """A BOQ whose 945 m3 line sits after ``offset_rows`` and is followed by
    ``tail_rows`` more, so the answer's position in the string is controlled."""
    rows = [
        {"code": f"D{500 + i}", "desc": f"Breakout item {i}", "unit": "m", "qty": i}
        for i in range(offset_rows)
    ]
    rows.append({"code": "D550.9", "desc": "Demolition, concrete",
                 "unit": "m3", "qty": 945})
    rows.extend(
        {"code": f"D{900 + i}", "desc": f"Tail item {i}", "unit": "nr", "qty": i}
        for i in range(tail_rows)
    )
    return json.dumps({"status": "success", "items": rows})


def test_the_answer_past_the_old_cut_survives_the_new_one():
    """The E4 regression as data rather than prose.

    The 945 m3 line is placed past 1000 characters and inside 6000: the old
    pass cut it away and the model reported the item did not exist; the new
    one keeps it. Mutation killed: setting _LAST_TOOL_PREVIEW back to the
    floor.
    """
    content = _boq_with_answer_at(offset_rows=25, tail_rows=400)
    at = content.index('"945"') if '"945"' in content else content.index("945")
    assert _TOOL_PREVIEW_FLOOR < at < _LAST_TOOL_PREVIEW, at
    assert len(content) > _LAST_TOOL_PREVIEW

    out = _compact_messages_for_tpm(_turn(content), budget=9000)
    payload = json.loads(_tools(out)[0]["content"])
    assert payload["truncated"] is True
    assert "945" in payload["preview"]
    assert "m3" in payload["preview"]
    # And the model is told how much it still cannot see.
    assert payload["chars_dropped"] > 0


def test_a_result_that_fits_the_larger_share_is_kept_whole():
    """Not compacted at all -- the old pass cut it to 1000 chars anyway."""
    content = _boq(60)
    assert _TOOL_PREVIEW_FLOOR < len(content) <= _LAST_TOOL_PREVIEW
    out = _compact_messages_for_tpm(_turn(content), budget=9000)
    assert _tools(out)[0]["content"] == content
    assert "945" in _tools(out)[0]["content"]


# -- everything the old pass guaranteed still holds -------------------------


def test_the_hop_actually_fits_afterwards():
    out = _compact_messages_for_tpm(_turn(_boq(400), _boq(400)), budget=4000)
    assert _approx_message_chars(out) <= 4000


def test_roles_and_pairing_survive():
    msgs = _turn(_boq(200), _boq(200))
    out = _compact_messages_for_tpm(msgs, budget=5000)
    assert [m.get("role") for m in out] == [m.get("role") for m in msgs]
    assert [m.get("tool_call_id") for m in _tools(out)] == ["0", "1"]


def test_non_dict_and_non_string_entries_pass_through():
    msgs = _turn(_boq(300))
    msgs.insert(1, "not-a-dict")
    msgs.insert(2, {"role": "assistant", "content": ["not-a-string"]})
    out = _compact_messages_for_tpm(msgs, budget=3000)
    assert "not-a-dict" in out
    assert any(
        isinstance(m, dict) and m.get("content") == ["not-a-string"] for m in out
    )


def test_a_tool_result_under_the_floor_is_never_compacted():
    """Compacting it would spend more characters on the envelope than the
    result contains."""
    tiny = "z" * (_TOOL_PREVIEW_FLOOR - 1)
    out = _compact_messages_for_tpm(_turn(tiny, _boq(400)), budget=4000)
    assert _tools(out)[0]["content"] == tiny


@pytest.mark.parametrize("budget", [400, 1200, 3000, 8000, 20000])
def test_no_budget_makes_it_crash_or_grow(budget):
    msgs = _turn(_boq(300), "m" * 5000, _boq(50))
    out = _compact_messages_for_tpm(msgs, budget=budget)
    assert _approx_message_chars(out) <= _approx_message_chars(msgs)
    assert isinstance(out, list) and out
