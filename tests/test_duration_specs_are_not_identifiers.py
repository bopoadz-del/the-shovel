"""A concrete-age / duration spec is not a document reference.

Live-fire find 2026-09-13. A grounded concrete-spec question was refused,
reproducibly, three times:

    "State the structural concrete strength class and 28-day cube strength."
    -> "I could not confirm this reference in the indexed project sources
        for Drive Archive. Please check whether the document is indexed."

...while the very same question with the "28-day" phrase removed answered
correctly ("Concrete grade: C35/45 ... 28-day cube strength ..."), and
/v1/rag/search returned the C35/45 chunk in ALL of the top-5. So retrieval
had the content; the answer layer false-declined.

Mechanism, probed locally: ``extract_query_identifiers`` returned
``['28-day']``. "28-day" is a concrete-age spec, not a document code, but the
standalone-alphanumeric pattern matched it; no chunk contains the exact token
"28-day" as a reference, so ``identifier_miss`` fired, RAG injection was
suppressed, and ``_should_short_circuit_rag_miss`` returned the canned
"could not confirm this reference" answer.

A small integer joined to a time word (28-day, 7 day, 90-days, 56 week) is now
excluded from identifier extraction, mirroring the existing decimal-quantity
and unit-ratio exclusions. Drawing/sheet refs (054-0009) and letter-bearing
codes are unaffected — their tail is not a time word.
"""
from __future__ import annotations

from app.core.rag.retriever import extract_query_identifiers


def test_the_live_concrete_spec_question_yields_no_identifiers():
    """RED before the fix: ['28-day']."""
    assert extract_query_identifiers(
        "State the structural concrete strength class and 28-day cube strength."
    ) == []
    assert extract_query_identifiers(
        "What is the 28-day cube strength of the structural concrete?"
    ) == []


def test_duration_specs_alone_are_not_references():
    for msg in (
        "7-day and 90-day compressive strength",
        "56 day curing regime",
        "the 28 days cube test result",
        "a 12-week programme float",
        "48-hour concrete cure",
    ):
        assert extract_query_identifiers(msg) == [], msg


def test_real_reference_codes_still_extract_alongside_durations():
    # A duration and a real drawing ref in the same query: keep only the ref.
    ids = extract_query_identifiers(
        "For the 28-day cube on drawing 054-0009, what grade applies?"
    )
    assert any("054-0009" in i for i in ids), ids
    assert "28-day" not in ids, ids


def test_labeled_and_dotted_codes_unaffected():
    assert any(
        "13.1" in i for i in extract_query_identifiers("what does clause 13.1 say")
    )
    assert "d999.46" in extract_query_identifiers("check drawing D999.46 please")
