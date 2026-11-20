"""The lexical leg matches numbers, units and grades as the user wrote them.

``lexical_query_terms`` is the one tokeniser both backends' query sanitizers
use. The rule is the digit/letter boundary only: a decimal and a slash-joined
run stay one token, a number followed by an alphabetic token also emits the
joined form, a number glued to letters also emits its parts, and every form
is OR'ed. These tests build their own corpora; none reads a project's data.
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest
from sqlalchemy import text

from app.core.rag import vector_store as vs
from app.core.rag.vector_store import (
    _sanitize_fts5_query,
    _sanitize_websearch_query,
    lexical_query_terms,
)

_PG = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")
pg_only = pytest.mark.skipif(not _PG, reason="the PostgreSQL text-search parser")


# ── the tokeniser (no database) ──────────────────────────────────────────


@pytest.mark.parametrize("query, kept", [
    ("depth 2.5 to 3m", "2.5"),
    ("strength 12.75 at 28 days", "12.75"),
])
def test_a_decimal_stays_one_token(query, kept):
    terms = lexical_query_terms(query)
    assert kept in terms
    _whole, frac = kept.split(".")
    # Not cut into the integer and the fraction as separate terms.
    assert frac not in terms


@pytest.mark.parametrize("query, kept", [
    ("compressive strength 12.5 N/mm2", "N/mm2"),
    ("concrete grade C50/40 to piles", "C50/40"),
])
def test_a_slash_joined_run_stays_one_token(query, kept):
    terms = lexical_query_terms(query)
    assert kept in terms
    for part in kept.split("/"):
        assert part not in terms


@pytest.mark.parametrize("query, number, unit", [
    ("blinding 1200 mm wide", "1200", "mm"),
    ("test pressure 10 bar", "10", "bar"),
])
def test_a_number_then_a_word_also_emits_the_joined_form(query, number, unit):
    terms = lexical_query_terms(query)
    assert number in terms and unit in terms
    assert number + unit in terms


@pytest.mark.parametrize("query, glued, number, unit", [
    ("blinding 1200mm wide", "1200mm", "1200", "mm"),
    ("dewater to 2.5m below", "2.5m", "2.5", "m"),
])
def test_a_number_glued_to_letters_also_emits_its_parts(query, glued, number, unit):
    terms = lexical_query_terms(query)
    assert glued in terms and number in terms and unit in terms


@pytest.mark.parametrize("query, not_joined", [
    ("see clause 5. Next the works", "5Next"),
    ("items 12, mm values", "12mm"),
])
def test_the_joined_form_needs_only_whitespace_between(query, not_joined):
    assert not_joined not in lexical_query_terms(query)


@pytest.mark.parametrize("query", [
    "Manhole spacing requirements?",
    "what is the soil backfilling specification",
])
def test_a_question_without_figures_is_unchanged(query):
    words = query.replace("?", "").split()
    assert lexical_query_terms(query) == words
    assert _sanitize_websearch_query(query) == " or ".join(words)
    assert _sanitize_fts5_query(query) == " OR ".join(words)


@pytest.mark.parametrize("query", ["12.5 N/mm2 grade C50/40", "pipe 2.5 dia"])
def test_fts5_quotes_a_form_its_tokenizer_would_split(query):
    out = _sanitize_fts5_query(query)
    for term in out.split(" OR "):
        bare = term.strip('"')
        if "." in bare or "/" in bare:
            assert term == f'"{bare}"', out
        else:
            assert term == bare, out


# ── end to end through the store (SQLite in the plain jobs, PostgreSQL on
#    the PG shards) ─────────────────────────────────────────────────────────


def _store_with(chunks):
    from app.core.rag.embeddings import get_embedder, reset_embedder_cache

    reset_embedder_cache()
    embedder = get_embedder()
    store = vs.get_store(dim=embedder.dim)
    project = f"lexq-{uuid.uuid4().hex[:10]}"
    for doc_id, body in chunks:
        store.upsert_chunks(project, doc_id, [body], embedder.encode([body]))
    return store, project


_NOISE = [
    ("noise-a", "The contractor shall submit the method statement for approval."),
    ("noise-b", "Backfill shall be placed in layers and compacted to the specified density."),
]


# Each question shares NO plain word with its target: only the figure, unit
# or grade can carry the match, so a tokeniser that splits it finds nothing.
@pytest.mark.parametrize("question, doc_text", [
    # spaced in the question, joined in the document ...
    ("width 1200 mm", "Blinding strip 1200mm under the footing."),
    ("excavate 3 m", "Excavation depth 2.5 to 3m in rock."),
    # ... and joined in the question, spaced in the document
    ("width 1200mm", "Blinding strip 1200 mm under the footing."),
    ("excavate 3m", "Excavation depth 2.5 to 3 m in rock."),
])
def test_units_match_whether_spaced_or_joined(question, doc_text):
    store, project = _store_with(_NOISE + [("target", doc_text)])
    hits = store.bm25_search(project, question, k=5)
    assert "target" in [h.doc_id for h in hits], [(h.doc_id, h.text) for h in hits]


@pytest.mark.parametrize("question, doc_text, decoy", [
    ("12.5 N/mm2", "Minimum cube strength 12.5 N/mm2 at 7 days.",
     "Table 12 row 5: N values and mm2 areas."),
    ("C50/40", "All piles shall be grade C50/40 concrete.",
     "Section C50 lists 40 items."),
])
def test_a_slash_grade_finds_its_chunk(question, doc_text, decoy):
    # The decoy holds the grade's parts apart; it must not outrank the grade.
    store, project = _store_with(_NOISE + [("target", doc_text), ("decoy", decoy)])
    hits = store.bm25_search(project, question, k=5)
    assert hits and hits[0].doc_id == "target", [(h.doc_id, h.text) for h in hits]


# ── what the PostgreSQL parser does with each form (CI PG shard) ─────────


def _lexemes(conn, body: str) -> list:
    return sorted(conn.execute(
        text("SELECT tsvector_to_array(to_tsvector('english', :b))"), {"b": body},
    ).scalar())


def _tsquery(conn, question: str) -> str:
    return conn.execute(
        text("SELECT websearch_to_tsquery('english', :q)::text"),
        {"q": _sanitize_websearch_query(question)},
    ).scalar()


def _matches(conn, question: str, body: str) -> bool:
    return bool(conn.execute(
        text(
            "SELECT to_tsvector('english', :b) "
            "@@ websearch_to_tsquery('english', :q)"
        ),
        {"b": body, "q": _sanitize_websearch_query(question)},
    ).scalar())


@pg_only
@pytest.mark.parametrize("body, want", [
    ("Depth 2.5 to 3m", ["2.5", "3m", "depth"]),          # decimal range + unit
    ("pressure 12.75 bar", ["12.75", "bar", "pressur"]),  # decimal
    ("Blinding 1200mm", ["1200mm", "blind"]),              # joined unit
    ("Blinding 1200 mm", ["1200", "blind", "mm"]),         # spaced unit
    ("strength 12.5 N/mm2", ["12.5", "n/mm2", "strength"]),  # slash unit
    ("grade C50/40", ["c50/40", "grade"]),                 # slash grade
])
def test_postgres_parser_keeps_each_form_as_written(body, want):
    from app.core.db import get_engine

    with get_engine().connect() as conn:
        assert _lexemes(conn, body) == want


@pg_only
@pytest.mark.parametrize("question, body", [
    # Only the figure, unit or grade is shared between question and body.
    ("2.5 to 3 m", "Depth 2.5 to 3m"),
    ("1200 mm", "Blinding 1200mm"),
    ("1200mm", "Blinding 1200 mm"),
    ("12.5 N/mm2", "strength 12.5 N/mm2"),
    ("C50/40", "grade C50/40"),
])
def test_postgres_question_form_meets_document_form(question, body):
    from app.core.db import get_engine

    with get_engine().connect() as conn:
        tsq = _tsquery(conn, question)
        assert _matches(conn, question, body), (question, tsq, _lexemes(conn, body))
