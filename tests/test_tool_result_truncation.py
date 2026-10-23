"""An over-long tool result must stay VALID JSON.

LIVE 2026-08-03, feature sweep against prod — two different actions, same
hard provider rejection:

    document_metadata        -> kimi HTTP 400
    parse_primavera_schedule -> kimi HTTP 400
    {"message": "Invalid request: tokenization failed",
     "type": "invalid_request_error"}

Cause: the `tool` message content was built as

    json.dumps(result, default=str)[:8000]

`json.dumps` emits ASCII with \\uXXXX escapes, so that raw slice can land
mid-escape or mid-string. The corpus makes it easy to hit — the client
filenames contain en dashes, which serialize to `\\u2013`:

    tail of cut: ' 2 \\u2013 specification (4 of 9).pdf", "'
    json.loads  -> Unterminated string starting at ... (char 7999)

The provider got malformed JSON and refused the whole request. Because a
failed turn also persisted nothing, the user just saw a question with no
answer (see tests/test_failed_turn_persistence.py).
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _TOOL_RESULT_MAX_CHARS,
    _requested_char_offset,
    _tool_result_content,
)


def _content(payload, offset=0):
    """Pin the Kimi/Groq 8k cap so these tests ignore LLM_PROVIDER."""
    return _tool_result_content(payload, offset=offset, max_chars=_TOOL_RESULT_MAX_CHARS)


def _en_dash_docs(n: int) -> dict:
    """The shape that actually broke it — en dashes become \\u2013."""
    return {
        "documents": [
            {"name": f"ab-2023-101 vol 2 – specification ({i} of 11).pdf",
             "type": "pdf"}
            for i in range(n)
        ]
    }


def test_small_result_is_passed_through_unchanged():
    payload = {"documents": [{"name": "a.pdf"}]}
    assert json.loads(_content(payload)) == payload


def test_oversized_result_is_still_valid_json():
    """The load-bearing assertion. The old code failed exactly here."""
    out = _content(_en_dash_docs(300))

    parsed = json.loads(out)  # must not raise
    assert parsed["truncated"] is True


def test_the_old_raw_slice_really_did_produce_invalid_json():
    """Pins WHY this helper exists, so nobody 'simplifies' it back.

    If this ever stops raising, the escape-splitting hazard is gone and the
    helper could be revisited — but that is a deliberate decision, not an
    accident.
    """
    raw = json.dumps(_en_dash_docs(300), default=str)[:_TOOL_RESULT_MAX_CHARS]

    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)


def test_truncated_result_stays_within_the_cap():
    out = _content(_en_dash_docs(500))
    assert len(out) <= _TOOL_RESULT_MAX_CHARS


def test_truncation_is_announced_to_the_model():
    """Silent truncation would let the model present a partial list as whole."""
    parsed = json.loads(_content(_en_dash_docs(300)))

    assert parsed["truncated"] is True
    assert "truncated" in parsed["note"].lower()
    assert parsed["preview"]


def test_non_serialisable_values_do_not_explode():
    """default=str must survive on both the small and truncated paths."""
    class Weird:
        def __repr__(self): return "<weird>"

    small = json.loads(_content({"v": Weird()}))
    assert isinstance(small["v"], str)

    big = json.loads(_content({"v": [Weird()] * 4000}))
    assert big["truncated"] is True


@pytest.mark.parametrize("n", [40, 120, 400, 1200])
def test_output_always_parses_across_sizes(n):
    """Sweep sizes so the cut lands at many different offsets — including
    inside an escape sequence, which is the whole hazard."""
    json.loads(_content(_en_dash_docs(n)))


# -- what the truncation SAYS, not just that it stays parseable ------------
#
# The tests above pin that an over-long result stays valid JSON. They do not
# pin what the model is told about the cut, and that turned out to matter as
# much. Live on 554e0b9: asked how many cubic metres of demolition are in the
# BOQ, the model answered "(pages 1-6 of 16) ... no demolition line items
# measured in cubic metres". The 945 m3 line is past the cut. It had been
# told only that the result "exceeded 8000 characters and was truncated" --
# not by how much -- so it inferred the extent from the content and inferred
# it wrong.
#
# Two layers cut the same result: this one when the tool message is built,
# and _compact_messages_for_tpm per Groq hop (tests/test_tpm_compaction.py).
# Both now state the loss and both forbid the absence claim.


def test_the_envelope_states_the_loss_as_a_number():
    parsed = json.loads(_content(_en_dash_docs(300)))
    assert parsed["chars_total"] > parsed["chars_shown"]
    assert parsed["chars_dropped"] == parsed["chars_total"] - parsed["chars_shown"]
    assert parsed["chars_shown"] == len(parsed["preview"])


def test_the_envelope_forbids_the_absence_claim():
    """inject.py has applied SCOPE OF ABSENCE to retrieval excerpts since the
    Saudi Building Code incident, because a partial view is evidence of what
    IS present and never of what is absent. A truncated tool result is the
    same shape of evidence and carries more authority, being the tool's own
    output rather than a search hit."""
    parsed = json.loads(_content(_en_dash_docs(300)))
    rule = parsed["scope_of_absence"]
    assert "NEVER state" in rule
    assert "absent" in rule
    assert str(parsed["chars_shown"]) in rule
    assert str(parsed["chars_total"]) in rule


def test_the_numbers_survive_the_shrink_loop():
    """keep is reduced until the ENVELOPE fits, so chars_shown must track the
    final preview rather than the first attempt -- otherwise the model is
    told it can see more than it can."""
    for n in (300, 500, 900):
        parsed = json.loads(_content(_en_dash_docs(n)))
        assert parsed["chars_shown"] == len(parsed["preview"])
        assert str(parsed["chars_shown"]) in parsed["scope_of_absence"]


def test_a_small_result_carries_no_scope_note():
    """Nothing was lost, so there is nothing to warn about -- and a warning
    on a complete result would teach the model to hedge answers it can
    fully support."""
    out = _content({"documents": [{"name": "a.pdf"}]})
    assert "scope_of_absence" not in out
    assert "chars_dropped" not in out


# -- a truncated result must be resumable ---------------------------------
#
# Saying what was lost stopped the model claiming absence. It did not let it
# find the answer. Live on 733ef49, after the loss numbers landed:
#
#     "The tool extracted text from all 16 pages, but the result was
#      truncated -- only the first 7,116 of 22,189 characters were returned
#      ... I cannot see whether an m3 demolition item exists later in the
#      bill, and I therefore cannot provide a quantity or CESMM code."
#
# Honest, correctly scoped, and still a dead end: the model knew exactly what
# it was missing and had no way to ask for it. char_offset is that way.


def _big(rows: int = 400) -> dict:
    payload = {"items": [{"code": f"D{i}", "desc": f"Breakout item {i}",
                          "unit": "m", "qty": i} for i in range(rows)]}
    payload["items"].append({"code": "D550.9", "desc": "Demolition, concrete",
                             "unit": "m3", "qty": 945})
    return payload


def _walk(payload: dict, limit: int = 20):
    """Every window the model would read, following next_char_offset."""
    out, offset = [], 0
    for _ in range(limit):
        parsed = json.loads(_content(payload, offset=offset))
        out.append(parsed)
        if not parsed["next_char_offset"]:
            return out
        offset = parsed["next_char_offset"]
    raise AssertionError("windows did not terminate")


def test_the_windows_tile_the_whole_result():
    """No gaps and no overlap: the model reading every window has read the
    serialized result exactly once.

    Mutation killed: slicing from 0 regardless of offset, which would loop
    forever on the first window.
    """
    payload = _big()
    windows = _walk(payload)
    assert len(windows) > 1
    rebuilt = "".join(w["preview"] for w in windows)
    assert rebuilt == json.dumps(payload, default=str)


def test_chars_remaining_reaches_zero_and_stops():
    windows = _walk(_big())
    assert windows[-1]["chars_remaining"] == 0
    assert windows[-1]["next_char_offset"] is None
    assert all(w["chars_remaining"] > 0 for w in windows[:-1])
    # Strictly decreasing, so a model following it always terminates.
    remaining = [w["chars_remaining"] for w in windows]
    assert remaining == sorted(remaining, reverse=True)
    assert len(set(remaining)) == len(remaining)


def test_the_answer_past_the_cap_is_reachable():
    """The E4 regression as data. The 945 m3 line sits past
    _TOOL_RESULT_MAX_CHARS, so no single window can hold it."""
    payload = _big()
    full = json.dumps(payload, default=str)
    assert full.index("945") > _TOOL_RESULT_MAX_CHARS
    windows = _walk(payload)
    assert "945" not in windows[0]["preview"], "not reachable in one call"
    assert any("945" in w["preview"] and "m3" in w["preview"] for w in windows)


def test_the_instruction_changes_on_the_last_window():
    """While there is more, the model is told to read it BEFORE answering
    about absence. On the last window that caveat would be wrong -- it has
    seen everything, and hedging a complete answer is its own failure."""
    windows = _walk(_big())
    first, last = windows[0], windows[-1]
    assert "char_offset=" in first["note"]
    assert "BEFORE telling the user" in first["note"]
    assert "Read the remaining windows" in first["scope_of_absence"]

    assert "LAST window" in last["note"]
    assert "char_offset=" not in last["note"]
    assert "read every window" in last["scope_of_absence"]
    # The absence rule itself never goes away, on any window.
    assert all("NEVER state" in w["scope_of_absence"] for w in windows)


def test_every_window_is_valid_json():
    """The original hazard: json.dumps emits \\uXXXX escapes and a raw slice
    can cut mid-escape. An offset can now START mid-escape too -- the preview
    is a JSON string value, so it is re-escaped and still parses."""
    for offset in (0, 1, 500, 4001, 7999, 8000, 12345):
        json.loads(_content(_en_dash_docs(300), offset=offset))


def test_an_offset_past_the_end_terminates_rather_than_looping():
    parsed = json.loads(_content(_big(), offset=10**9))
    assert parsed["chars_shown"] == 0
    assert parsed["chars_remaining"] == 0
    assert parsed["next_char_offset"] is None


def test_a_result_that_fits_is_still_returned_whole_at_offset_zero():
    small = {"documents": [{"name": "a.pdf"}]}
    assert json.loads(_content(small)) == small
    # ...but an explicit offset into it is honoured rather than ignored.
    assert json.loads(_content(small, offset=5))["char_offset"] == 5


@pytest.mark.parametrize(
    "args,expected",
    [
        ('{"file_path":"x.pdf","char_offset":7116}', 7116),
        ('{"file_path":"x.pdf"}', 0),
        ('{"char_offset":0}', 0),
        ('{"char_offset":-40}', 0),
        ('{"char_offset":"nope"}', 0),
        ('{"char_offset":null}', 0),
        ("not json at all", 0),
        ('["not","a","dict"]', 0),
    ],
)
def test_the_offset_is_read_defensively(args, expected):
    """A malformed offset must degrade to the first window, never raise --
    the model writes this argument and models mistype."""
    assert _requested_char_offset({"function": {"arguments": args}}) == expected


def test_no_tool_call_means_no_offset():
    assert _requested_char_offset(None) == 0
    assert _requested_char_offset({}) == 0
