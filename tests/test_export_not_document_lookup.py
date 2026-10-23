"""An export request is not a document lookup (UI-PHYS H1 / diagnostic D6).

D6 finding: the H1 export turn dispatched **zero tools** and returned
"could not confirm this reference" in ~0.6s. The export action never ran —
`_should_short_circuit_rag_miss` fired first, and the router's refusal was
displayed as though it were the answer.

That gate already excludes self-coding and self-contained calculations on the
grounds that neither is a document lookup. An export request is the same kind
of thing: it asks for a FILE to be produced, not for a passage to be found
inside one.
"""

from __future__ import annotations

from app.agents import runtime


def _short_circuits(msg: str) -> bool:
    """Would the RAG-miss fast refusal fire for this message?

    Conditions replicate the H1 turn: RAG produced no context, the audit shows
    an identifier miss, and the extracted identifier contains a digit.
    """
    audit = {"extracted_identifiers": ["Schedule 10"], "identifier_miss": True}
    return runtime._should_short_circuit_rag_miss(audit, None, msg)


def test_the_helper_actually_triggers_the_gate():
    """Self-check for the tests below.

    Every other assertion here is of the form `not _short_circuits(...)`. If the
    audit record failed to satisfy the gate's real conditions the helper would
    return False for everything, all of those would pass vacuously, and the
    mutation probe would never fire. An earlier version of this file did
    exactly that — it keyed on "reason" instead of the boolean the gate reads.
    """
    assert _short_circuits("What does the Specification document say about 003113?")


EXPORT_ASKS = [
    "Export this conversation as a Word document",
    "Can you download the BOQ to Excel?",
    "Save Schedule 10 as a PDF",
    "give me a docx copy",
    "Export A1-A9 answers as a docx report",
]

LOOKUP_ASKS = [
    "What does the Specification document say about 003113?",
    "Quote Sub-Clause 8.8.1 for me",
]


def test_export_requests_are_recognised():
    for msg in EXPORT_ASKS:
        assert runtime._asks_for_export(msg), msg


def test_export_request_is_not_refused_as_a_missing_reference():
    """The H1 failure itself: refused before the export could dispatch."""
    for msg in EXPORT_ASKS:
        assert not _short_circuits(msg), msg


def test_document_lookups_keep_the_fast_refusal():
    """The gate exists for a reason — a real reference miss should still
    short-circuit rather than spend a model call confirming it cannot answer."""
    for msg in LOOKUP_ASKS:
        assert not runtime._asks_for_export(msg), msg
        assert _short_circuits(msg), msg


def test_an_action_verb_alone_does_not_count():
    """Guard against over-broad matching quietly disabling the fast path.

    Both of these contain an export/download word in a non-export sense. If
    either matched, every turn mentioning them would lose the fast refusal.
    """
    assert not runtime._asks_for_export("What is the export value of the works?")
    assert not runtime._asks_for_export("Download times are slow on site")


def test_mutation_probe_the_exclusion_must_be_wired_into_the_gate():
    """MUTATION PROBE.

    Recognising an export request is useless unless the recognition is
    actually consulted by `_should_short_circuit_rag_miss`. Deleting
    `or _asks_for_export(user_message)` from that function leaves
    `test_export_requests_are_recognised` green and reintroduces H1 —
    this test is what fails.
    """
    msg = "Export this conversation as a Word document"
    assert runtime._asks_for_export(msg), "precondition: predicate recognises it"
    assert not _short_circuits(msg), (
        "export request still short-circuits: the predicate exists but is not "
        "wired into _should_short_circuit_rag_miss"
    )
