"""A status line is not an answer, and it cannot make a promise into one.

Live on 2d221e6 -- the build that taught the model to read a truncated tool
result window by window. Asked how many cubic metres of demolition are in the
BOQ, the whole reply was:

    "I'll continue reading the BOQ to find any cubic-metre items. Let me pull
     the next window.

     Continuing BOQ extraction..."

_SEARCH_PROMISE_TAIL_RE matches "Let me pull the next window." on its own. It
did not match here because it is anchored to the END of the text and a
progress line follows it. So a dangling promise shipped as the answer -- the
exact failure #454 and #456 exist to prevent, reached through vocabulary the
platform itself had just taught the model.

This is finding 3 from the same battery becoming load-bearing: the detector
inspects the tail, and the promise had stopped being the tail.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _final_text_needs_forced_retry,
    _looks_like_search_preamble,
    _strip_progress_narration,
)

LIVE = (
    "I'll continue reading the BOQ to find any cubic-metre items. Let me "
    "pull the next window.\n\nContinuing BOQ extraction..."
)


# -- the live failure ------------------------------------------------------


def test_the_live_answer_is_refused():
    """Mutation killed: not stripping narration before re-testing."""
    assert _looks_like_search_preamble(LIVE) is True
    assert _final_text_needs_forced_retry(LIVE, user_message="how many m3?") is True


def test_the_promise_alone_was_already_caught():
    """Not redundant -- it shows the detector was working and the trailing
    line is what defeated it, so the fix targets the right thing."""
    assert _looks_like_search_preamble("Let me pull the next window.") is True


def test_narration_alone_is_not_an_answer():
    """A status line answers no question at all."""
    for text in (
        "Continuing BOQ extraction...",
        "Reading the next window.",
        "Fetching the remaining pages…",
        "Analyzing the schedule",
    ):
        assert _looks_like_search_preamble(text) is True, text


# -- the stripper ----------------------------------------------------------


def test_trailing_narration_is_removed_and_the_rest_kept():
    assert _strip_progress_narration(LIVE).endswith("next window.")
    assert "Continuing" not in _strip_progress_narration(LIVE)


def test_several_trailing_narration_lines_go_together():
    text = "Let me pull the next window.\nContinuing extraction...\n\nReading page 6."
    assert _strip_progress_narration(text) == "Let me pull the next window."


def test_narration_in_the_MIDDLE_is_left_alone():
    """Only the tail is status. A line like this between two real statements
    is part of the prose, and cutting it would change the meaning of an
    answer that is otherwise fine."""
    text = "The rate is 0.1%.\nReading the cap now.\nThe cap is 10%."
    assert _strip_progress_narration(text) == text


def test_a_real_answer_survives_stripping_unchanged():
    text = "Delay Damages are 0.1% of the Contract Price per calendar day."
    assert _strip_progress_narration(text) == text


# -- and the guard rails ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        (
            "Delay Damages are 0.1% of the Contract Price per calendar day "
            "(Sub-Clause 8.8.1), capped at 10%."
        ),
        (
            "The rate is 0.1% per calendar day. Let me know if you want the "
            "SAR figure once the Contract Price is confirmed."
        ),
        (
            "Post Tender Clarifications: log dated 1-Aug-23. Tender Addenda: "
            "06 and 07. Schedule of Project Requirements: SOPR Rev B."
        ),
        "No m3 item appears in the windows I read; I read all four.",
    ],
)
def test_a_real_answer_is_never_refused(text):
    """The cost of a false positive is a working answer thrown away and a
    retry spent, so the common shapes get their own pins -- especially
    'Let me know', which contains 'let me' and must not trip."""
    assert _looks_like_search_preamble(text) is False
    assert _final_text_needs_forced_retry(text, user_message="q") is False


def test_a_long_answer_is_out_of_scope_entirely():
    """The detector is for short dangling replies. A long answer that happens
    to mention continuing is a different thing and must not be caught."""
    long_answer = (
        "Delay Damages are 0.1% of the Contract Price per calendar day. " * 12
        + "\nContinuing to monitor the milestone dates."
    )
    assert len(long_answer) > 500
    assert _looks_like_search_preamble(long_answer) is False


def test_empty_and_whitespace_are_handled():
    assert _looks_like_search_preamble("") is False
    assert _looks_like_search_preamble("   \n  ") is False
    assert _strip_progress_narration("") == ""
    assert _strip_progress_narration("\n\n") == ""


def test_the_length_bound_is_deliberate_and_pinned():
    """Pre-existing design, restated here because this change makes it load-
    bearing: the detector only judges SHORT dangling replies. A long answer
    that happens to end on "let me check" has already said something, and
    throwing it away to retry would cost more than it saves.

    Mutation killed: dropping the len/newline bound from
    _looks_like_search_preamble.
    """
    body = "Delay Damages are 0.1% of the Contract Price per calendar day. " * 10
    long_with_promise = body + "Let me check the cap."
    short_with_promise = "The rate is stated. Let me check the cap."

    assert len(long_with_promise) > 500
    assert _looks_like_search_preamble(long_with_promise) is False
    # The same tail in a short reply IS refused -- the bound is the only
    # difference, so the assertion above is about length, not phrasing.
    assert _looks_like_search_preamble(short_with_promise) is True
