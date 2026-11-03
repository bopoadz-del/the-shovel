"""A Contract Data lookup on a project stands on the PROJECT corpus, not Master
Corpus RAG.

Battery 41540262748 read Q4/5/6 (the Engineer, performance bond, milestones) as
general_rag because Master-Corpus chunks were kept in the sources the answer
stood on, even though the project holds the contract. should_suppress_master_
corpus_fallback now fires for a contract-data lookup on a non-master project
(as it already did for calculator-shaped asks), so MC chunks are dropped and the
project corpus remains -- mirroring AGENTS.md "Contract Data Q&A stays on
project-assistant + RAG." A non-contract, non-calc question is unaffected.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import should_suppress_master_corpus_fallback

# A real (non-master) project id is a hex slug, not a general-corpus name.
_PROJECT_ID = "4e9480af"


@pytest.mark.parametrize(
    "text,expected",
    [
        # Contract-data lookups on a project -> suppress MC (prefer project).
        ("What is the Time for Completion under the contract?", True),
        ("Who is named as the Engineer and what are their powers?", True),
        ("What performance bond or retention does the contract require?", True),
        ("What are the milestone dates or sectional completion terms?", True),
        # Structural classes (>=2 phrasings each), not the battery's four:
        # party roles beyond the Engineer...
        ("Who is the Employer under the contract?", True),
        ("What are the Contractor's obligations under the contract?", True),
        # security / obligations...
        ("What retention is held under the contract?", True),
        ("What insurance is the Contractor required to provide under the contract?", True),
        # money terms (a liquidated/delay-damages lookup, hyphen tolerant)...
        ("What is the cap on liquidated damages?", True),
        ("What delay-damages rate and cap does the contract state?", True),
        # A standards / HSE question is neither contract-data nor a calculator
        # ask -> Master Corpus stays allowed (it is a general-knowledge answer).
        ("What is the HSE lighting (lux) requirement for general site areas?", False),
        # Over-match controls: the class keywords alone, OUT of contract context,
        # must stay False (a bare milestone / site engineer / a van insurer /
        # water retention are not Contract Data lookups).
        ("The project reached a major milestone yesterday.", False),
        ("Ask the site engineer to check the rebar spacing on level 3.", False),
        ("Which insurance company offers the cheapest cover for a van?", False),
        ("How much retention of water does the slab need?", False),
        # A worded arithmetic ask that mentions "LD cap" is a CALCULATION, not a
        # Contract Data lookup: it must stay False so the cost-grounding gate
        # still refuses a fabricated rate (regression guard for
        # test_leftover_hat_ui_fails::test_fabricated_rate_still_refused...).
        ("ten percent of four million eight hundred thousand LD cap", False),
    ],
)
def test_contract_data_lookup_suppresses_master_corpus_on_a_project(text, expected):
    assert should_suppress_master_corpus_fallback(_PROJECT_ID, text) is expected
