"""`_bound_history` — defensive cap on the client-sent conversation thread.

The chat UI sends the full thread as `history` every turn. On a long pilot
session that balloons the prompt (slower TTFT). This bounds it: drop empty
(interrupted-stream) turns, keep only the most recent messages.
"""
from __future__ import annotations

from app.routers.agents import _bound_history, _MAX_HISTORY_MESSAGES


def test_non_list_returns_empty():
    assert _bound_history(None) == []
    assert _bound_history("nope") == []
    assert _bound_history({"role": "user"}) == []


def test_drops_empty_content_turns():
    hist = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},      # interrupted stream
        {"role": "assistant", "content": "   "},    # whitespace only
        {"role": "assistant", "content": "world"},
    ]
    out = _bound_history(hist)
    assert out == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_drops_malformed_entries():
    hist = [
        {"role": "user", "content": "ok"},
        {"role": "assistant"},                      # no content key
        {"content": 123},                            # non-str content
        "garbage",                                   # not a dict
    ]
    assert _bound_history(hist) == [{"role": "user", "content": "ok"}]


def test_caps_to_most_recent():
    hist = [{"role": "user", "content": f"m{i}"} for i in range(_MAX_HISTORY_MESSAGES + 10)]
    out = _bound_history(hist)
    assert len(out) == _MAX_HISTORY_MESSAGES
    # Keeps the most recent, drops the oldest.
    assert out[-1]["content"] == f"m{_MAX_HISTORY_MESSAGES + 9}"
    assert out[0]["content"] == f"m{10}"


def test_short_history_passes_through():
    hist = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert _bound_history(hist) == hist
