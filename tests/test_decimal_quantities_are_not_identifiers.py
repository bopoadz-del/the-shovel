"""A measured quantity is not a document reference.

Live-fire find 2026-08-15 (F21). A grounded costing request carrying the
measured take-off quantities was refused three times in ~1 second each:

    "... floor screed, 1947.87 square metres; ceramic skirting, 2342.20
     metres ..." -> "I could not confirm this reference in the indexed
     project sources. Please check whether the document is indexed."

Mechanism, probed locally: ``extract_query_identifiers`` returned
``['1947.87', '2342.20']`` -- the standalone-alphanumeric-code pattern
matches letterless decimals, no chunk contains those exact strings, and the
exact-reference miss gate short-circuits the whole turn. Any costing or
take-off question that states its quantities was structurally un-askable on
a grounded project.

Pure numbers with a decimal separator are now excluded from identifier
extraction. Hyphenated digit codes (drawing/sheet refs) still match, and
codes that carry letters (D999.46) still match.
"""
from __future__ import annotations

from app.core.rag.retriever import extract_query_identifiers

LIVE_MESSAGE = (
    "Write a resource-based cost build-up for a measured finishes bill. "
    "The measured quantities are: forty millimetre cement sand floor screed, "
    "1947.87 square metres; ceramic floor tiling including adhesive and "
    "grout, 1947.87 square metres; ceramic skirting one hundred millimetres "
    "high, 2342.20 metres; ceiling preparation and emulsion paint, "
    "1947.87 square metres."
)


def test_the_live_costing_message_yields_no_identifiers():
    """RED before the fix: ['1947.87', '2342.20']."""
    assert extract_query_identifiers(LIVE_MESSAGE) == []


def test_decimal_quantities_alone_are_not_references():
    assert extract_query_identifiers("the slab needs 24.5 cubic metres") == []
    assert extract_query_identifiers("grand total 15184347.90 this month") == []
    assert extract_query_identifiers("progress is 87,5 percent") == []


def test_real_reference_codes_still_extract():
    # letter-bearing dotted code
    assert "d999.46" in extract_query_identifiers("check drawing D999.46 please")
    # hyphenated code
    ids = extract_query_identifiers("see document QZ-ENG-320-0009 for details")
    assert any("054-0009" in i or "qz-eng" in i for i in ids)
    # labeled reference with digits
    ids = extract_query_identifiers("what is the status of VO Ref 31?")
    assert any("31" in i for i in ids)


def test_hyphenated_digit_pairs_keep_matching():
    """Sheet refs like 054-0009 are digits-only but hyphen-joined -- still
    plausible document codes, deliberately NOT excluded."""
    ids = extract_query_identifiers("refer to sheet 054-0009 in the pack")
    assert any("054-0009" in i for i in ids)


def test_currency_rate_units_are_not_document_references():
    assert extract_query_identifiers(
        "Convert 1250 AED/m2 to USD/ft2 using 3.6725 AED = 1 USD"
    ) == []


def test_leftover_l7_arithmetic_is_not_a_document_identifier():
    """Live leftover L7: '(18.4-16)*2850' extracted '18.4-16' as a sheet
    ref, so the RAG-miss gate returned 'could not confirm this reference'
    in ~0s and sympy never ran."""
    msg = (
        "Stay on heavy-reasoning. Planned quantity 16, actual 18.4, rate 2850. "
        "Compute (18.4-16)*2850 with sympy_reasoning {expression}. "
        "Report the 2.4 overrun and the 6840 cost impact."
    )
    assert extract_query_identifiers(msg) == []
    assert extract_query_identifiers("see sheet 054-0009")  # still a code
