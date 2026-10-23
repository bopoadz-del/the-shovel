"""A typo must not be mistaken for a construction reference.

LIVE INCIDENT 2026-08-02 (operator's own session, project Chad):

    user: "You should find it in the project specif8cation not drawings"
    app:  "I could not confirm this reference in the indexed project sources
           for Chad. Please check whether the document is indexed or provide
           the exact filename."

That message was a CONVERSATIONAL CORRECTION, not a document lookup. The
stray '8' in the misspelling made the identifier extractor treat it as a
reference code; retrieval then missed on it, and the missing-reference
short-circuit in runtime.py answered with the "provide the exact filename"
template instead of letting the model respond.

Two rules had to be tightened, because the typo arrived through BOTH:
  * rule 4 (standalone alphanumeric codes) matched "specif8cation"
  * rule 3 (labeled references) split it into label "spec" + code
    "if8cation" — which carries a digit and so passed that rule's guard

The discriminator is the ALPHABETIC RUN. Real codes are built from short
abbreviations (M145, D999, PRC951, IP054); English words carry runs of 5+
letters, and a typo'd digit does not change that.
"""
from __future__ import annotations

import pytest

from app.core.rag.retriever import extract_query_identifiers


# ── the live failure ─────────────────────────────────────────────────────────

def test_the_live_typo_extracts_no_identifier():
    """The exact message from the incident."""
    q = "You should find it in the project specif8cation not drawings"
    assert extract_query_identifiers(q) == []


def test_the_typo_is_not_split_into_a_labeled_reference():
    """Regression guard for the rule-3 path specifically.

    Before the fix this returned ['spec if8cation', 'if8cation'] — the
    labeled-reference rule treating 'spec' as a label. Asserting emptiness
    alone would not prove WHICH rule was fixed, so pin the fragments.
    """
    ids = extract_query_identifiers("check the specif8cation please")
    assert not any("if8cation" in i for i in ids), ids
    assert not any(i.startswith("spec ") for i in ids), ids


@pytest.mark.parametrize(
    "query",
    [
        "the schedul3 is late",
        "review the draw1ngs tomorrow",
        "send me the spec1fication",
        "what about the found4tion works",
    ],
)
def test_other_digit_typos_are_ignored(query):
    assert extract_query_identifiers(query) == []


# ── real references must be UNAFFECTED ───────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected_member",
    [
        ("what is in QZ-ENG-320-0000-EXC-DWG-ST-100-0000951-05",
         "qz-eng-320-0000-exc-dwg-st-100-0000951-05"),
        ("show me drawing D999.46", "d999.46"),
        ("what does XYZ-501 say", "xyz-501"),
        ("give me RFI 42", "rfi 42"),
        ("VO Ref 31 status", "vo 31"),
        ("clause 13.1 of the contract", "clause 13.1"),
    ],
)
def test_genuine_references_still_extract(query, expected_member):
    """The guard must be surgical — these are what the feature exists for."""
    assert expected_member in extract_query_identifiers(query)


@pytest.mark.parametrize(
    "query",
    [
        "risk register for a 30-storey tower",   # prose compound
        "concrete 250 kg/cm2",                   # measurement unit
        "backfill compacted to 95% MDD",         # percentage prose
    ],
)
def test_previously_guarded_cases_remain_guarded(query):
    """Earlier incidents this file must not regress."""
    assert extract_query_identifiers(query) == []


# ── the unit under the fix ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "token,is_typo",
    [
        ("specif8cation", True),
        ("if8cation", True),     # the rule-3 fragment
        ("schedul3", True),
        ("M145", False),         # short alpha prefix -> real code
        ("A615", False),
        ("D999", False),
        ("PRC951", False),
        ("IP054", False),
        ("0000951", False),      # digits only
        ("D999.46", False),      # separator token -> handled elsewhere
    ],
)
def test_is_misspelled_word(token, is_typo):
    from app.core.rag.retriever import _is_misspelled_word
    assert _is_misspelled_word(token) is is_typo
