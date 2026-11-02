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
        # A standards / HSE question is neither contract-data nor a calculator
        # ask -> Master Corpus stays allowed (it is a general-knowledge answer).
        ("What is the HSE lighting (lux) requirement for general site areas?", False),
    ],
)
def test_contract_data_lookup_suppresses_master_corpus_on_a_project(text, expected):
    assert should_suppress_master_corpus_fallback(_PROJECT_ID, text) is expected
