"""A named standard absent from the project layer must consult the general layer.

B11/B15/B16 class: a standards-figure ask was refused ("not in project docs")
or answered with a project figure wearing the standard's name, when the same
figure lives in the curated general-knowledge library. The mechanism is the
inject-time steering note: when a named standard is absent from the retrieved
excerpts, the note must direct a general-knowledge search and give the two
outcomes — answer at General tier, or the explicit "not in your documents or
the general library" — instead of a refusal dressed as grounding.

Fixture wording only. No live client names. Two unrelated standard/figure
pairs per outcome, so nothing here depends on one code.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _answer_tier
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.named_standard_attribution import (
    _GENERAL_LIBRARY_ABSENT_PHRASE,
    absence_note,
)
from app.core.rag.vector_store import Chunk


def _chunk(cid, doc_id, text, *, source_name="", layer="own", score=0.81):
    return Chunk(
        chunk_id=cid,
        project_id="p_fixture",
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        score=score,
        source_name=source_name,
        layer=layer,
    )


# Two unrelated standards, each with a project figure and a library figure.
PRESENT_CASES = [
    (
        "Per NFPA 51B, how long must a fire watch be maintained after hot work?",
        "NFPA 51B",
        ("NFPA 51B standard for fire prevention. A fire watch shall be "
         "maintained for at least 60 minutes after hot work."),
        "NFPA-51B-2019.txt",
    ),
    (
        "Per ACI 305, what is the maximum temperature of fresh concrete?",
        "ACI 305",
        ("ACI 305 hot weather concreting. The maximum temperature of fresh "
         "concrete as placed shall not exceed 35°C."),
        "ACI-305R-20.txt",
    ),
]

ABSENT_CASES = [
    (
        "Per NFPA 51B, how long must a fire watch be maintained after hot work?",
        "NFPA 51B",
        ("Project hot-work permit. A fire watch shall be maintained for "
         "30 minutes after hot work is complete."),
        "project-hot-work-permit.txt",
    ),
    (
        "Per ACI 305, what is the maximum temperature of fresh concrete?",
        "ACI 305",
        ("Project concrete specification. The maximum temperature of fresh "
         "concrete at delivery is 32°C."),
        "project-concrete-spec.txt",
    ),
]


@pytest.mark.parametrize("ask,standard,text,source", PRESENT_CASES)
def test_standard_present_in_general_layer_is_not_refused(ask, standard, text, source):
    """A knowledge-base excerpt that states the standard must not trigger the
    absence note — the answer is free to use it and is credited General tier."""
    chunks = [
        _chunk("gk", "gk1", text, source_name=source, layer="general_knowledge"),
    ]
    content = format_chunks_as_system_message(chunks, 1, query=ask)["content"]
    assert "NAMED STANDARD ABSENT" not in content, content
    # The tier machinery credits a general-knowledge source as general.
    assert _answer_tier([{"layer": "general_knowledge"}]) == "general"


@pytest.mark.parametrize("ask,standard,text,source", ABSENT_CASES)
def test_standard_absent_everywhere_directs_consultation_and_refusal(
    ask, standard, text, source,
):
    """A project excerpt that does not back the standard must steer a
    general-knowledge search, credit the General tier when found, and refuse in
    both layers' own words when the library has nothing."""
    chunks = [_chunk("proj", "proj1", text, source_name=source)]
    content = format_chunks_as_system_message(chunks, 1, query=ask)["content"]
    assert "NAMED STANDARD ABSENT" in content, content
    assert standard in content, content
    assert "search_general_knowledge" in content, content
    assert "General tier" in content, content
    assert _GENERAL_LIBRARY_ABSENT_PHRASE in content, content


def test_absence_note_is_empty_without_a_named_standard():
    """A plain project ask never inherits the named-standard steering."""
    chunks = [_chunk("proj", "proj1", "A project figure.", source_name="x.txt")]
    assert absence_note("How long must the fire watch be maintained?", chunks) == ""
