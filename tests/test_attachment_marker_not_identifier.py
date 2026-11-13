"""S-A8: an attachment marker is evidence handed over, never a corpus reference.

The composer writes ``[attached: NAME]`` (photos add ``| observations: ...``)
into the message. A digit-heavy filename used to be extracted as an exact
reference, so an attachment-only turn was refused by the RAG-miss short
circuit before any model call. Every test builds its own message and audit
record.
"""
from __future__ import annotations

import pytest

from app.agents import runtime
from app.core.rag.retriever import (
    attachment_markers_carry_content,
    extract_query_identifiers,
    strip_attachment_markers,
)
from app.routers.chat import _parse_attached_markers


def _miss_audit(message: str) -> dict:
    """The audit record inject writes when retrieval missed."""
    return {
        "identifier_miss": True,
        "threshold_fired": True,
        "extracted_identifiers": extract_query_identifiers(message),
    }


ATTACHMENT_ONLY = [
    "[attached: IMG_20261113_113412.jpg | observations: 2 people visible; "
    "helmet detected; crack near grid C4, low confidence]",
    "[attached: 8841920377.pdf]",
]

TEXT_MISS = [
    ("What does RFI-4417 say about the slab?", "rfi-4417"),
    ("Summarise drawing PRC-90213 for me", "prc-90213"),
]


@pytest.mark.parametrize("message", ATTACHMENT_ONLY)
def test_attachment_marker_yields_no_identifier(message):
    assert extract_query_identifiers(message) == []
    assert strip_attachment_markers(message).strip() == ""


@pytest.mark.parametrize("empty", [None, ""])
def test_missing_message_yields_no_identifier(empty):
    assert extract_query_identifiers(empty) == []


@pytest.mark.parametrize(
    "message,kept",
    [
        ("Check RFI-4417 [attached: 5512.pdf] please", "RFI-4417"),
        ("[attached: a.jpg | observations: wet floor] grid C4 status", "grid C4"),
    ],
)
def test_strip_keeps_the_text_around_markers(message, kept):
    out = strip_attachment_markers(message)
    assert kept in out
    assert "attached" not in out


@pytest.mark.parametrize("message", ATTACHMENT_ONLY)
def test_attachment_only_turn_is_not_short_circuited(message):
    audit = _miss_audit(message)
    assert runtime._should_short_circuit_rag_miss(audit, None, message) is False
    assert runtime._rag_miss_identifiers(audit, None, message) == []


@pytest.mark.parametrize("message,ref", TEXT_MISS)
def test_text_only_missing_reference_still_short_circuits(message, ref):
    audit = _miss_audit(message)
    assert ref in audit["extracted_identifiers"]
    assert runtime._should_short_circuit_rag_miss(audit, None, message) is True
    assert runtime._turn_has_attachment_content(message, None) is False


@pytest.mark.parametrize("message,ref", TEXT_MISS)
def test_mixed_turn_with_observations_answers_and_refuses_the_reference(message, ref):
    full = (
        f"{message} [attached: 00{len(ref)}7731.jpg | observations: "
        "1 person visible; no vest detected]"
    )
    audit = _miss_audit(full)
    # The attachment name never becomes an identifier; the real one stays.
    assert ref in audit["extracted_identifiers"]
    assert not any("7731" in i for i in audit["extracted_identifiers"])
    # The turn is not refused whole...
    assert runtime._should_short_circuit_rag_miss(audit, None, full) is False
    # ...the missed reference is pinned for the model instead.
    missed = runtime._rag_miss_identifiers(audit, None, full)
    assert ref in missed
    note = runtime._build_reference_miss_note(missed)
    assert ref in note
    assert runtime._MISSING_REFERENCE_ANSWER in note
    assert "attach" in note.lower()


@pytest.mark.parametrize("message,ref", TEXT_MISS)
def test_mixed_turn_with_resolved_document_is_not_refused_whole(message, ref):
    full = f"{message} [attached: 5520193.pdf]"
    audit = _miss_audit(full)
    docs = [{"id": f"doc-{len(ref)}", "original_name": "5520193.pdf"}]
    assert runtime._should_short_circuit_rag_miss(audit, None, full, docs) is False
    # Without the resolved document and without a payload, the bare marker
    # carries no evidence, so the real missing reference still refuses.
    assert runtime._should_short_circuit_rag_miss(audit, None, full, None) is True


@pytest.mark.parametrize(
    "message,expected",
    [
        ("[attached: a.jpg | observations: wet floor]", True),
        ("[attached: 1234.pdf]", False),
    ],
)
def test_marker_payload_counts_as_attachment_content(message, expected):
    assert attachment_markers_carry_content(message) is expected
    assert runtime._turn_has_attachment_content(message, None) is expected


@pytest.mark.parametrize(
    "message,names",
    [
        ("see [attached: Plan 01.pdf] and [attached: x.jpg | observations: y]",
         ["Plan 01.pdf", "x.jpg"]),
        ("[attached: A.docx][attached: a.docx]", ["A.docx"]),
    ],
)
def test_router_parses_the_shared_marker(message, names):
    assert _parse_attached_markers(message) == names


@pytest.mark.parametrize("missed,empty", [([], True), (["rfi-31"], False)])
def test_note_only_when_a_reference_missed(missed, empty):
    note = runtime._build_reference_miss_note(missed)
    assert (note == "") is empty
    assert all(m in note for m in missed)


@pytest.mark.parametrize("message,ref", TEXT_MISS)
@pytest.mark.parametrize("with_attachment", [True, False])
def test_pin_reference_miss_appends_only_for_attachment_turns(message, ref, with_attachment):
    full = message + (" [attached: 31337.jpg | observations: ladder on site]"
                      if with_attachment else "")
    messages = [{"role": "user", "content": full}]
    runtime._pin_reference_miss(messages, full, None, _miss_audit(full), None)
    notes = [m["content"] for m in messages if m["role"] == "system"]
    assert len(notes) == (1 if with_attachment else 0)
    if with_attachment:
        assert ref in notes[0]
