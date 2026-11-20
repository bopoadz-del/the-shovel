"""Answer-style check 3: the tester's 18 machinery terms never reach the user.

The terms are the authoritative tester list (TESTER_MACHINERY_WORDS.md); the
exit check removes them whole-word, case-insensitive. Two inputs per decision,
own inputs, no real document/answer strings.
"""

from __future__ import annotations

import pytest

from app.agents.answer_exit import _machine_wording


@pytest.mark.parametrize(
    ("term", "variant"),
    [
        ("embedding", "embedding"),
        ("embeddings", "EMBEDDINGS"),
        ("vector", "vector"),
        ("vectors", "Vectors"),
        ("top-k", "top-k"),
        ("rag", "RAG"),
        ("llm", "LLM"),
        ("token", "token"),
        ("tokens", "Tokens"),
        ("pipeline", "pipeline"),
        ("reranker", "reranker"),
    ],
)
def test_machinery_term_is_removed(term, variant):
    out = _machine_wording(f"The {variant} is 5.")
    assert term not in out.lower()


@pytest.mark.parametrize(
    "phrase",
    [
        "context window",
        "knowledge base",
        "semantic search",
    ],
)
def test_machinery_phrase_is_removed(phrase):
    out = _machine_wording(f"The {phrase} is used here.")
    assert phrase not in out.lower()
