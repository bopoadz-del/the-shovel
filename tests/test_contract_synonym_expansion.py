"""Contract-term synonym expansion widens recall for FIDIC synonyms.

Live gap (2026-09-13): synonym phrasings honestly declined facts that ARE in
the corpus. "What is the contract sum before VAT?" retrieved the Accepted
Contract Amount chunk at NO rank in the top-15, while "What is the Accepted
Contract Amount excluding VAT?" answered SAR 1,754,504,456.25. Probing showed
that appending the canonical heading ("Accepted Contract Amount") lifts that
chunk to rank 0.

expand_contract_synonyms returns the canonical term(s) to APPEND to a
supplementary retrieval query for any synonym present — never altering the
primary query, and never firing on a definition question.
"""
from __future__ import annotations

from app.core.rag.retriever import expand_contract_synonyms


def test_contract_sum_expands_to_accepted_contract_amount():
    assert "Accepted Contract Amount" in expand_contract_synonyms(
        "What is the contract sum before VAT as a number?"
    )
    assert "Accepted Contract Amount" in expand_contract_synonyms(
        "net contract value before tax"
    )


def test_retention_synonyms_expand_to_canonical_heading():
    for q in (
        "how much is held back as retention each payment",
        "What is the retention rate withheld from the contractor?",
        "what percentage is retained from the certificate",
    ):
        assert "Percentage of Retention" in expand_contract_synonyms(q), q


def test_definition_questions_are_not_expanded():
    # Glossary path — must not be pulled onto a particulars lookup.
    assert expand_contract_synonyms("What does Accepted Contract Amount mean?") == ""
    assert expand_contract_synonyms("Define liquidated damages") == ""


def test_unrelated_queries_get_no_expansion():
    assert expand_contract_synonyms("What concrete grade is specified?") == ""
    assert expand_contract_synonyms("Who is the site security provider?") == ""


def test_expansion_is_deduplicated_and_never_repeats_present_terms():
    # If the canonical term is already in the query, it is not appended again.
    out = expand_contract_synonyms(
        "What is the Accepted Contract Amount contract value?"
    )
    # "contract value" triggers, but the canonical term is already present.
    assert out == "", out


def test_expansion_returns_plain_space_joined_terms():
    out = expand_contract_synonyms("cap on delay damages as a percentage")
    assert out and all(tok.strip() for tok in out.split())
    # No duplicate tokens.
    toks = out.lower().split()
    assert len(toks) == len(set(toks)), out
