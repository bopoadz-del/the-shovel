"""A promise to READ is a dangling promise too.

Live on 2d9d9c0, ``POST /v1/chat/stream``, project master_corpus:

    "Draft a variation order under FIDIC clause"

The turn was handed to heavy-reasoning, ran five iterations, and ended -- with
no error -- on:

    "I now have the project's own change-management procedure and VSR form.
     Let me read the final window of the VSR form to complete the picture."

The user asked for a document and was shown a status line.

A detector for exactly this already existed. It missed because it only knew
the verbs of SEARCHING -- search, run, pull, look up, fetch, retrieve, check,
validate -- and the list was written out twice, once in each regex. When the
platform taught the model to read a truncated tool result window by window,
the model started promising to READ, and neither copy had the verb.

This is the third time new vocabulary has carried a dangling promise past the
detector. The first two fixes each added the one phrase that had just been
seen. These tests pin the FAMILY -- everything the agent can promise to go
and do to a document -- so the fourth does not need a fourth fix.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _final_text_needs_forced_retry,
    _looks_like_search_preamble,
)

# Exact live wording. Do not paraphrase.
LIVE_VARIATION_ORDER_TURN = (
    "I now have the project's own change-management procedure and VSR form. "
    "Let me read the final window of the VSR form to complete the picture."
)


def test_the_live_turn_is_caught():
    assert _looks_like_search_preamble(LIVE_VARIATION_ORDER_TURN)
    assert _final_text_needs_forced_retry(
        LIVE_VARIATION_ORDER_TURN,
        user_message="Draft a variation order under FIDIC clause",
    )


@pytest.mark.parametrize(
    "text",
    [
        "Let me read the next window.",
        "I'll open the priced BOQ and go through the demolition section.",
        "The excerpt is truncated. I will review the remaining pages.",
        "I am continuing through the specification register.",
        "I need to examine the rest of the schedule before answering.",
        "I'm loading the second half of the document now.",
        "Let me scan the remaining clauses.",
        "I'll work through the rest of the bill.",
    ],
)
def test_every_kind_of_promised_work_is_a_promise(text):
    """Shape-invariance. The class, not the one verb caught this time."""
    assert _looks_like_search_preamble(text), text


@pytest.mark.parametrize(
    "text",
    [
        # The old vocabulary must not have been lost while widening it.
        "Let me search more precisely for the Engineer's Representative.",
        "I'll pull the WIR and check the hold points.",
        "I am running a targeted search to retrieve the clause details.",
    ],
)
def test_the_original_search_vocabulary_still_matches(text):
    assert _looks_like_search_preamble(text), text


@pytest.mark.parametrize(
    "text",
    [
        "The Performance Bond is 10% of the Accepted Contract Amount.",
        # A bare leading verb is a result, not a first-person promise.
        "Search results show three specifications covering variations.",
        "The schedule shows Milestone 5 at 731 days from right of access.",
        "Review of the register shows two open items.",
        "A read of clause 13 shows the Engineer must instruct in writing.",
    ],
)
def test_a_real_answer_is_never_mistaken_for_a_promise(text):
    """The control. Widening the verb list is only safe if it still needs a
    FIRST-PERSON promise; a detector that flagged answers would send every
    good turn round again."""
    assert not _looks_like_search_preamble(text), text


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 852 days. Let me know if you need anything else.",
        "Clause 13 covers Variations. Let me know if you want me to check the form.",
        "Let me know if you would like me to read the full clause.",
    ],
)
def test_let_me_know_is_a_courtesy_not_a_promise(text):
    """"Let me know if you want me to check..." was ALREADY reachable through
    "check" before this change. Adding "read", "review" and "open" widens that
    hole, so it is closed here rather than left to get worse."""
    assert not _looks_like_search_preamble(text), text


def test_a_long_real_answer_that_mentions_reading_is_left_alone():
    """The detector only judges short turns. A full answer that happens to
    contain "I will review" is an answer."""
    long_answer = (
        "The Variation Procedure is set out in the specification register. "
        * 12
        + "I will review any further instruction you issue under Clause 13."
    )
    assert len(long_answer) > 500
    assert not _looks_like_search_preamble(long_answer)
