"""TERM EQUIVALENCE answer-mapping hint + its A9 leak guard.

Live gap (2026-09-13): the retriever surfaced the right Contract Data document
(AB-2023-101 Vol 1, High confidence) yet the model answered "I don't have the
contract sum before VAT" — it would not equate the everyday synonym with the
formal "Accepted Contract Amount". The synonym-expansion PR fixed RETENTION
(a genuine retrieval miss) but not this, because ACA was never a retrieval
problem in the chat path — it is answer mapping.

format_chunks_as_system_message now appends a TERM EQUIVALENCE note when the
question uses such a synonym. Like every routing note it is governed by the
INTERNAL GUIDANCE directive (never quoted), so it must (a) fire only on a
synonym, (b) never appear for a canonical-term question, and (c) be preceded by
INTERNAL GUIDANCE / NEVER quote so it cannot leak verbatim (the A9 failure mode).
"""
from __future__ import annotations

from app.core.rag.inject import (
    _term_equivalence_note,
    format_chunks_as_system_message,
)
from app.core.rag.vector_store import Chunk

CD_TEXT = (
    "CONTRACT DATA\n1.1.1\nAccepted Contract Amount (excluding VAT)\n"
    "SAR 1,754,504,456.25\n"
)


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="c1", project_id="p", doc_id="d1",
        chunk_index=0, text=text, score=0.9,
    )


def test_note_fires_for_contract_sum_synonym():
    note = _term_equivalence_note("What is the contract sum before VAT as a number?")
    assert "TERM EQUIVALENCE" in note
    assert "Accepted Contract Amount" in note


def test_note_fires_for_delay_cap_and_defects_synonyms():
    assert "Maximum Amount of Delay Damages" in _term_equivalence_note(
        "What is the cap on delay damages?"
    )
    assert "Defects Notification Period" in _term_equivalence_note(
        "What is the maintenance period?"
    )


def test_note_empty_for_canonical_or_unrelated_query():
    # Canonical wording needs no mapping.
    assert _term_equivalence_note(
        "What is the Accepted Contract Amount including VAT?"
    ) == ""
    # Unrelated question.
    assert _term_equivalence_note("What concrete grade is specified?") == ""


def test_note_is_governed_by_internal_guidance_and_never_leaks_bare():
    """A9 leak guard: the mapping note must sit UNDER the INTERNAL GUIDANCE /
    NEVER quote directive so the model cannot parrot it into the reply."""
    msg = format_chunks_as_system_message(
        [_chunk(CD_TEXT)], 1,
        query="What is the contract sum before VAT as a number?",
    )["content"]
    assert "TERM EQUIVALENCE" in msg
    assert "INTERNAL GUIDANCE" in msg
    assert "NEVER quote" in msg
    assert msg.index("INTERNAL GUIDANCE") < msg.index("TERM EQUIVALENCE")
    assert msg.index("NEVER quote") < msg.index("TERM EQUIVALENCE")


def test_note_absent_from_system_message_for_canonical_query():
    """A canonical-term question must not carry the synonym note at all — it
    would be dead weight and one more thing that could leak."""
    msg = format_chunks_as_system_message(
        [_chunk(CD_TEXT)], 1,
        query="What is the Accepted Contract Amount including VAT?",
    )["content"]
    assert "TERM EQUIVALENCE" not in msg
