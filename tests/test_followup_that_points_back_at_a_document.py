"""A follow-up that says "the letter" cannot be retrieved on by itself.

Live on 24d1c0c, master_corpus, one conversation, three runs:

    D1  Who signed the letter about the XBCC concrete batching plant at
        Wadi Example, and in what capacity?                          3/3
    D2  Per that letter, since when had the land been delivered
        to ACMC?                                                   3/3
    D3  What reason does the letter give for no longer needing a
        pre-cast factory?                                          1/3

D3's two failures retrieved precast-concrete SPECIFICATIONS -- tolerances,
rejection criteria, sampling -- and honestly reported no letter. The words
that carry signal in D3 are "pre-cast factory"; the word that says which
document is "the letter", and that points at turn one.

Follow-up expansion already existed, but only for a THIN message (under four
content terms). D3 has seven. Length was standing in for the real property:
the message refers to something it does not identify. D2 passed only because
its own words ("land delivered to ACMC") happen to sit in the letter.
"""
from __future__ import annotations

import pytest

from app.core.rag.inject import build_retrieval_query, message_points_back_at_a_document

D1 = ("Who signed the letter about the XBCC concrete batching plant at Wadi "
      "Example, and in what capacity?")
D2 = "Per that letter, since when had the land been delivered to ACMC?"
D3 = "What reason does the letter give for no longer needing a pre-cast factory?"


def _hist(*turns):
    return [{"role": role, "content": text} for role, text in turns]


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.delenv("RAG_FOLLOWUP_CONTEXT", raising=False)


def test_the_live_d3_turn_regains_the_letter_it_is_about():
    history = _hist(("user", D1), ("assistant", "Barry ... Engineer's Representative"),
                    ("user", D2), ("assistant", "Since March 2024."))

    query = build_retrieval_query(D3, history)

    assert query.endswith(D3)
    for subject in ("XBCC", "batching", "Wadi", "Example"):
        assert subject in query, query


def test_d2_is_recognised_too_not_passed_by_luck():
    assert "XBCC" in build_retrieval_query(D2, _hist(("user", D1)))


@pytest.mark.parametrize(
    "message",
    [
        D2,
        D3,
        "What date is on that email?",
        "Summarise the memo in three bullet points for the project director.",
        "Who else was copied on this notice and what did they reply?",
        "List every action item recorded in those minutes with its owner.",
        "Does the same report mention any delay to the culvert works?",
    ],
)
def test_pointing_back_is_recognised_whatever_the_document_kind(message):
    """Shape-invariance: the class is "refers to an unidentified document",
    not the word "letter"."""
    assert message_points_back_at_a_document(message), message


@pytest.mark.parametrize(
    "message",
    [
        # Names its own document: nothing to borrow.
        D1,
        "What does the letter regarding the precast yard at Gate 4 say about handover?",
        "Summarise the report on groundwater monitoring for March 2024.",
        "Open the email from the Engineer dated 16 June 2025 and list its attachments.",
        "What does letter QZ-ENG-320-0000-ACM-LTR-MN-000372 say about the land?",
        # "the contract" / "the Specification" / "the bill" are THE project's,
        # not something an earlier turn introduced. Expanding these would
        # splice the previous question into every Contract Data ask.
        "What is the approved method of electronic communication under the contract?",
        "What scheduling method does the Specification require for programmes?",
        "What is the Part Summary total for page d/3/1 of the Demolition bill?",
        "What is the value of the Performance Bond?",
    ],
)
def test_a_question_that_stands_alone_is_not_a_followup(message):
    assert not message_points_back_at_a_document(message), message


def test_a_standalone_question_mid_conversation_is_byte_identical():
    """The control that matters live: the A-series asked one after another in
    a single chat must each retrieve on their own words only."""
    ask = "What is the approved method of electronic communication under the contract?"
    history = _hist(("user", "What is the value of the Performance Bond?"),
                    ("assistant", "10% of the Accepted Contract Amount."))
    assert build_retrieval_query(ask, history) == ask


def test_pointing_back_with_nothing_behind_it_is_a_noop():
    assert build_retrieval_query(D3, []) == D3
    assert build_retrieval_query(D3, _hist(("assistant", "hello"))) == D3


def test_assistant_turns_are_still_never_borrowed():
    history = _hist(("user", D1), ("assistant", "ZZZWRONGANSWER about a gantry crane"))
    assert "ZZZWRONGANSWER" not in build_retrieval_query(D3, history)


def test_kill_switch_still_wins(monkeypatch):
    monkeypatch.setenv("RAG_FOLLOWUP_CONTEXT", "0")
    assert build_retrieval_query(D3, _hist(("user", D1))) == D3
