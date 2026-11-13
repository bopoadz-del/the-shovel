"""S-A9: a tool's instruction to the model is never the user's answer.

Every input here is built in the test: the tool payloads are generated, the
envelopes come from the platform's own builders, and the "echo" drafts are
derived from whatever directive those builders wrote, so the tests follow
the envelope wording instead of pinning it.
"""

from __future__ import annotations

import json
import re

import pytest

from app.agents.runtime import (
    _TOOL_DIRECTIVE_KEYS,
    _compacted_tool_payload,
    _directive_words,
    _sanitize_final_text,
    _strip_tool_directive_echo,
    _tool_directive_texts,
    _tool_result_content,
)


def _big_payload(rows: int) -> dict:
    return {"rows": [{"item": f"line {i}", "qty": i * 7, "unit": "m2"} for i in range(rows)]}


def _fetch_envelope(rows: int, cap: int, offset: int = 0) -> str:
    return _tool_result_content(_big_payload(rows), offset=offset, max_chars=cap)


def _compaction_envelope(chars: int, keep: int) -> str:
    return _compacted_tool_payload("x" * chars, keep)


def _tool_msg(content: str) -> dict:
    return {"role": "tool", "tool_call_id": "t1", "name": "fetch_document", "content": content}


def _as_exit_check_words(text: str) -> str:
    """What the exit check does to identifiers: ``a_b=1`` reads ``a b 1``."""
    return re.sub(r"[_=]", " ", text)


@pytest.mark.parametrize(
    "envelope",
    [_fetch_envelope(400, 3000), _compaction_envelope(9000, 500)],
    ids=["fetch-window", "compaction"],
)
def test_every_envelope_text_key_is_a_listed_directive(envelope):
    payload = json.loads(envelope)
    assert payload["truncated"] is True
    text_keys = {k for k, v in payload.items() if isinstance(v, str) and k != "preview"}
    assert text_keys, "envelope carries no directive text"
    assert text_keys <= set(_TOOL_DIRECTIVE_KEYS)


@pytest.mark.parametrize(
    ("rows", "cap"),
    [(400, 3000), (900, 5000)],
)
def test_draft_that_copies_the_continuation_is_removed(rows, cap):
    env = _fetch_envelope(rows, cap)
    note = json.loads(env)["note"]
    next_offset = json.loads(env)["next_char_offset"]
    assert next_offset and str(next_offset) in note
    messages = [_tool_msg(env)]
    # The draft as it reached the user: identifiers already turned to words.
    draft = _as_exit_check_words(note)
    assert _strip_tool_directive_echo(draft, messages) == ""
    # Same copy with the figure changed is still the same instruction.
    other = draft.replace(str(next_offset), str(next_offset + 13))
    assert _strip_tool_directive_echo(other, messages) == ""


@pytest.mark.parametrize(
    ("answer", "rows", "cap"),
    [
        ("The billed quantity of line 12 is 84 m2, from the measured rows.", 400, 3000),
        ("Line 30 carries 210 m2; the next window of the bill was not needed.", 900, 5000),
    ],
)
def test_draft_that_does_not_copy_is_kept(answer, rows, cap):
    messages = [_tool_msg(_fetch_envelope(rows, cap))]
    assert _strip_tool_directive_echo(answer, messages) == answer


@pytest.mark.parametrize(
    ("answer", "rows", "cap"),
    [
        ("Line 5 is 35 m2.", 400, 3000),
        ("The schedule shows line 9 at 63 m2.", 900, 5000),
    ],
)
def test_mixed_draft_keeps_the_answer_and_drops_the_directive(answer, rows, cap):
    env = _fetch_envelope(rows, cap)
    note = _as_exit_check_words(json.loads(env)["note"])
    messages = [_tool_msg(env)]
    out = _strip_tool_directive_echo(f"{answer} {note}", messages)
    assert out == answer


@pytest.mark.parametrize(
    "truncated",
    [True, False],
)
def test_only_an_envelope_counts_as_a_directive(truncated):
    """The same sentence inside a tool's DATA is evidence and may be quoted."""
    sentence = "Call the supervisor again with the same form and repeat until all rows are signed."
    payload = {"truncated": truncated, "note": sentence, "preview": "{}"}
    messages = [_tool_msg(json.dumps(payload))]
    out = _strip_tool_directive_echo(sentence, messages)
    assert (out == "") is truncated
    assert (_tool_directive_texts(messages) == [sentence]) is truncated


@pytest.mark.parametrize(
    ("offset", "cap"),
    [(0, 3000), (3000, 3000)],
)
def test_continuing_and_last_window_notes_are_both_directives(offset, cap):
    env = _fetch_envelope(100, cap, offset=offset)
    payload = json.loads(env)
    assert (payload["next_char_offset"] is None) is (offset > 0)
    draft = _as_exit_check_words(payload["note"])
    assert _strip_tool_directive_echo(draft, [_tool_msg(env)]) == ""


@pytest.mark.parametrize(
    ("rows", "cap"),
    [(400, 3000), (900, 5000)],
)
def test_sanitize_final_text_empties_a_directive_only_answer(rows, cap):
    env = _fetch_envelope(rows, cap)
    note = json.loads(env)["note"]
    messages = [{"role": "user", "content": "how much?"}, _tool_msg(env)]
    assert _sanitize_final_text(note, messages=messages) == ""
    # Without the tool message the same words are not known to be a directive.
    assert _sanitize_final_text(note, messages=messages[:1]) != ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("char_offset=7098", ["char", "offset", "#"]),
        ("Chars-Remaining: 12 of 40", ["chars", "remaining", "#", "of", "#"]),
    ],
)
def test_directive_words_split_identifiers_and_mask_figures(text, expected):
    assert _directive_words(text) == expected
