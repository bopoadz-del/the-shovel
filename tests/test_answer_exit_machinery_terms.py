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
        ("indexed", "indexed"),
        ("index", "Index"),
        ("indexes", "indexes"),
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
        "semantic search",
    ],
)
def test_machinery_phrase_is_removed(phrase):
    out = _machine_wording(f"The {phrase} is used here.")
    assert phrase not in out.lower()


@pytest.mark.parametrize(
    "line",
    [
        "3 of 5 project documents indexed",
        "12 of 12 documents indexed",
        "This project has no indexed documents yet",
        "the indexed files are listed",
    ],
)
def test_document_status_indexed_is_preserved(line):
    out = _machine_wording(line)
    assert "indexed" in out.lower()


@pytest.mark.parametrize(
    "token_in_filename",
    [
        "WORD-SPEC-TOKEN-FENWICK-20260819-081916",
        "SPEC-VECTOR-9.docx",
    ],
)
def test_term_inside_a_filename_is_preserved(token_in_filename):
    out = _machine_wording(token_in_filename)
    assert token_in_filename.lower() in out.lower()
