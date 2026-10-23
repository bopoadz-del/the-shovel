"""The two database backends must implement the SAME retrieval semantics.

THE ROOT CAUSE THIS PINS
------------------------
Hybrid retrieval has two legs: semantic (pgvector/cosine) and lexical (BM25).
The lexical leg was implemented twice, and the two implementations disagreed:

* SQLite  — ``_sanitize_fts5_query`` OR-joins the tokens. Its docstring states
  the reason: "AND semantics return zero hits on natural-language queries
  because BM25 can never contribute and hybrid collapses to semantic-only."
* PostgreSQL — used ``plainto_tsquery``, which ANDs every term.

The diagnosis was right and was applied to one backend. Production runs the
other one. So in production:

* a chunk had to contain EVERY word of a question to be eligible;
* one word absent from the corpus (a typo, a plural, a product name) emptied
  the entire BM25 leg;
* ``search()`` then silently fell back to cosine-only.

Ranking across a corpus of thousands of chunks at k=5 therefore came down to
cosine alone, which is unstable under rephrasing — the same corpus answers a
question on one turn and reports the material missing on the next. That is the
mechanism behind the self-contradicting answers, and it is not specific to any
one query.

It was invisible to the suite because dev and CI run SQLite — the forgiving
path. No test could observe the semantics production used. These tests assert
the two backends agree, so the divergence cannot silently reopen.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest

from app.core.rag.vector_store import (
    _sanitize_fts5_query,
    _sanitize_websearch_query,
)


QUERIES = [
    "what is the soil backfilling specification",
    "Saudi buiding code",                       # the live typo
    "concrete cover to reinforcement in slabs",
    "geotechnical standards",
]


@pytest.mark.parametrize("query", QUERIES)
def test_both_backends_are_bag_of_words(query):
    """Neither backend may require every term to be present."""
    sqlite_q = _sanitize_fts5_query(query)
    postgres_q = _sanitize_websearch_query(query)

    assert " OR " in sqlite_q, f"SQLite leg is not bag-of-words: {sqlite_q!r}"
    assert " or " in postgres_q, (
        f"PostgreSQL leg is not bag-of-words: {postgres_q!r} — one absent term "
        f"would empty the BM25 leg and collapse retrieval to cosine-only"
    )


@pytest.mark.parametrize("query", QUERIES)
def test_both_backends_select_the_same_terms(query):
    """Parity of CONTENT, not just of operator: a term searched on one backend
    must be searched on the other, or the backends rank differently."""
    sqlite_terms = {t.lower() for t in _sanitize_fts5_query(query).split(" OR ")}
    postgres_terms = {t.lower() for t in _sanitize_websearch_query(query).split(" or ")}
    assert sqlite_terms == postgres_terms


def test_a_term_absent_from_the_corpus_cannot_empty_the_lexical_leg():
    """The exact production shape: a misspelt word among good ones.

    Under the old AND semantics the whole query matched nothing. Under
    bag-of-words the surviving terms still carry the lookup.
    """
    postgres_q = _sanitize_websearch_query("Saudi buiding code")
    assert postgres_q == "Saudi or buiding or code"
    # The correctly-spelled terms remain independently searchable, which is what
    # keeps a typo from costing the whole leg.
    assert "saudi" in postgres_q.lower() and "code" in postgres_q.lower()


def test_websearch_operator_words_are_not_passed_through_as_terms():
    """websearch_to_tsquery reads or/and/not as OPERATORS. Leaving a user's
    "and" in the token stream changes the meaning of the query."""
    assert _sanitize_websearch_query("slabs and beams") == "slabs or beams"
    assert _sanitize_websearch_query("cover not exceeding") == "cover or exceeding"


def test_punctuation_cannot_produce_an_unparseable_query():
    """Real user input carries quotes, slashes and dashes. The sanitizer must
    never emit something the tsquery parser rejects — a syntax error would take
    the lexical leg down for that request."""
    for raw in ['grade C40/50 "suspended" slabs', "rev-C -- superseded?", "!!!"]:
        out = _sanitize_websearch_query(raw)
        assert "  " not in out
        assert not out.startswith("or") and not out.endswith("or")


def test_empty_input_yields_no_lexical_leg():
    assert _sanitize_websearch_query("") == ""
    assert _sanitize_websearch_query("   ") == ""
    # A query of nothing but operator words has no searchable content.
    assert _sanitize_websearch_query("and or not") == ""
