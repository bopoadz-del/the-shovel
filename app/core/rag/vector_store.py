"""Per-project chunk store backed by SQLAlchemy.

Storage shape: one row per chunk in the ``chunks`` table. On PostgreSQL
embeddings use ``pgvector`` ``vector(N)`` (N = the model's dimension, 384 for
the default BAAI/bge-small-en-v1.5) with cosine-distance ANN search
(``ORDER BY embedding <=> :q`` scoped by ``project_id``). On SQLite the
same table stores float32 BLOBs and search falls back to numpy cosine
similarity over the project's rows — slower but works everywhere.

Hybrid retrieval: when ``RAG_HYBRID_SEARCH`` is truthy and the caller
supplies ``query_text``, the store also runs a BM25 leg and fuses with
Reciprocal Rank Fusion. On PostgreSQL the BM25 leg is an idf-weighted,
length-normed score over the ``text_search`` tsvector column + GIN index (added by Alembic 0003).
On SQLite the BM25 leg is FTS5 over a ``chunks_fts`` external-content
virtual table maintained by AFTER INSERT/DELETE/UPDATE triggers on
``chunks``. See ``_ensure_fts5_sqlite`` for the trigger rationale.

SQLAlchemy-backed via app.core.db — unified The Fork schema.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
from contextlib import nullcontext
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, delete, false, func, or_, select, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

import json as json_lib

from app.core.db import _engine_for_url, _session_factory_for_url, get_database_url
from app.core.models import Document, Project
from app.core.models import make_rag_chunk_class, rag_chunk_table_name
# S1: the fallback table width derives from the one config dim — the resolved
# default embedder (BAAI/bge-small-en-v1.5, 384) — NOT the legacy models.py
# EMBEDDING_DIM (256, the old `chunks` table). get_store() still prefers the
# live embedder's actual dim; this only sets the width for a brand-new store.
from app.core.rag.embeddings import get_embedder
from app.core.rag.text_repair import repair_fake_bold

logger = logging.getLogger(__name__)


# ── Hybrid retrieval constants ────────────────────────────────────────────

# Reciprocal Rank Fusion constant per Cormack et al. 2009 — dampens the
# contribution of low-ranked items so the top of each list dominates.
RRF_K = 60

# Pre-fetch ceiling per leg before fusion. Wider than the caller's k so
# that a chunk that's #40 in one list but #2 in the other still has a
# shot at the final top-K. 50 is the spec value.
HYBRID_FETCH_PER_LEG = 50

# Strips non-word characters except spaces; whitespace then collapses
# into FTS5 tokens. The result is OR-joined so MATCH is bag-of-words.
_FTS5_SAFE_RE = re.compile(r"[^\w\s]+", re.UNICODE)

# OCR of CESMM/BOQ item codes inserts a space after the class letter
# (``D 549.2`` vs ``D549.2``). Collapse that gap so index text and query
# identifiers share one token shape. Only a single Latin letter is
# consumed — drawing codes (AB-CDE-001) and contract ids (AB-2002-202)
# are unchanged.
_CESMM_OCR_SPACE_RE = re.compile(r"\b([A-Za-z])[ \t]+(\d+\.\d+)\b")
_CESMM_COMPACT_TOKEN_RE = re.compile(r"^[a-z]\d+$")
# A letter x written straight after a number (optionally with its unit,
# "5 x 0.25", "5 m x 0.25 m") is the times sign of a dimension chain, not
# the class letter of an item code. Collapsing it turned "12 x 5 x 0.25 m"
# into "12 x 5 x0.25 m", and "x0.25" was then read as a document reference.
_TIMES_SIGN_LETTERS = frozenset("xX")
_NUMBER_BEFORE_RE = re.compile(r"\d(?:[ \t]*[A-Za-z]{1,3}\d?)?[ \t]*$")


def _collapse_item_code(match: re.Match) -> str:
    letter = match.group(1)
    start = match.start()
    if letter in _TIMES_SIGN_LETTERS and _NUMBER_BEFORE_RE.search(
        match.string[max(0, start - 16):start]
    ):
        return match.group(0)
    return letter + match.group(2)


def normalize_cesmm_item_codes(text: str) -> str:
    """Collapse ``[A-Z]\\s+\\d+`` to ``[A-Z]\\d+`` in ``text``.

    Live WAVE 2 B5: Neon ``ILIKE '%D549.2%'`` was 0 hits while
    ``ILIKE '%D 549.2%'`` was 2 — Tesseract wrote the class letter and
    the digits as separate tokens. Callers must run this at index time
    (chunk text) and at query time (identifiers + match text) so either
    form retrieves the other. Empty/None input is returned unchanged.
    """
    if not text:
        return text
    return _CESMM_OCR_SPACE_RE.sub(_collapse_item_code, text)


def _identifier_tokens(ident: str) -> List[str]:
    """Alphanumeric tokens of an identifier after CESMM space-collapse."""
    collapsed = normalize_cesmm_item_codes(ident or "").lower()
    return [t for t in re.split(r"[^a-z0-9]+", collapsed) if t]


def _hybrid_enabled() -> bool:
    """Read RAG_HYBRID_SEARCH live so tests / operators can flip it
    without re-importing the module. Truthy values: 1, true, yes, on
    (case-insensitive). Default: true."""
    raw = os.getenv("RAG_HYBRID_SEARCH", "true")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


#: Per leg, how many of its best candidates the OTHER leg did not return are
#: guaranteed a place in the fused order at their own leg rank (S-R7c). One
#: setting, the same number for both legs. 0 restores plain fusion.
FUSION_LEG_SHARE_DEFAULT = 2


def fusion_leg_share() -> int:
    """``RAG_FUSION_LEG_SHARE`` read live; default
    ``FUSION_LEG_SHARE_DEFAULT``. A malformed or negative value falls back
    to the default / 0 respectively."""
    raw = os.getenv("RAG_FUSION_LEG_SHARE", "")
    try:
        n = int(str(raw).strip()) if str(raw).strip() else FUSION_LEG_SHARE_DEFAULT
    except ValueError:
        n = FUSION_LEG_SHARE_DEFAULT
    return max(n, 0)


def guarantee_leg_share(
    order: List["Chunk"],
    sem_rank: Dict[str, int],
    bm_rank: Dict[str, int],
    share: int,
) -> List["Chunk"]:
    """Give each leg's best single-leg candidates a place by their own rank.

    Reciprocal-rank fusion with k=60 ranks a chunk found by both legs at
    ANY depth (2/(60+50) = 0.0182) above the best hit of one leg
    (1/61 = 0.0164), so a passage only exact words can find (a figure, a
    reference, a clause number) sits under every merely on-topic one. Here
    the ``share`` best-ranked candidates of each leg that the other leg did
    not return move up to the position of their own leg rank (never down:
    a candidate already fused higher keeps its place). Everything else
    keeps its fused order. When the legs agree there is no single-leg
    candidate and the order is unchanged. On a tie of positions
    the vector-leg candidate goes first, as fusion keeps the semantic
    instance on a tie.
    """
    fused_pos = {c.chunk_id: i for i, c in enumerate(order, 1)}
    picks: List[Tuple[int, int, str]] = []
    for leg_index, (mine, other) in enumerate(((sem_rank, bm_rank), (bm_rank, sem_rank))):
        alone = sorted(
            (r, cid) for cid, r in mine.items()
            if cid not in other and cid in fused_pos
        )[:share]
        picks += [(min(r, fused_pos[cid]), leg_index, cid) for r, cid in alone]
    picked = {cid for _, _, cid in picks}
    by_id = {c.chunk_id: c for c in order}
    out = [c for c in order if c.chunk_id not in picked]
    last = -1
    for target, _leg, cid in sorted(picks):
        at = min(max(target - 1, last + 1), len(out))
        out.insert(at, by_id[cid])
        last = at
    return out


# ── The lexical leg's query tokeniser (both backends) ──────────────────────
#
# Documents write a quantity, a unit and a grade in more than one shape:
# "2.5", "1200mm" or "1200 mm", "N/mm2", "C50/40". PostgreSQL's parser keeps a
# decimal and a slash-joined run as ONE lexeme in ``to_tsvector``, and keeps a
# number glued to letters as one lexeme ('1200mm'). Stripping every punctuation
# mark from the question split those into pieces no chunk's tsvector holds, so
# the exact figure a user typed could not be matched as written.
#
# The rule is the digit/letter boundary only -- no unit list, no word list:
#   * a decimal (digits '.' digits) and a run of word characters joined by '/'
#     stay one token;
#   * a number followed by an alphabetic token also emits the joined form
#     ("1200 mm" -> 1200, mm, 1200mm);
#   * a number glued to letters also emits its parts ("1200mm" -> 1200mm,
#     1200, mm).
# Every form is OR'ed, so whichever way the question writes it meets whichever
# way the document wrote it. The stored index is not touched.
_LEXICAL_TOKEN_RE = re.compile(
    r"\w+(?:/\w+)+"            # slash-joined: N/mm2, C50/40
    r"|\d+(?:\.\d+)+[^\W\d_]*"   # decimal, possibly glued to letters: 2.5, 2.5m
    r"|\w+",
    re.UNICODE,
)
_LEXICAL_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")
_LEXICAL_ALPHA_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_LEXICAL_NUMBER_LETTERS_RE = re.compile(r"(\d+(?:\.\d+)*)([^\W\d_]+)", re.UNICODE)
# A token FTS5 accepts unquoted in MATCH.
_FTS5_BAREWORD_RE = re.compile(r"\w+", re.UNICODE)


def lexical_query_terms(query: str) -> List[str]:
    """The lexical leg's search terms for ``query``, in order, each once.

    See the notes above ``_LEXICAL_TOKEN_RE``. Case is kept (both backends
    fold it); duplicates are dropped case-insensitively.
    """
    text_ = query or ""
    matches = list(_LEXICAL_TOKEN_RE.finditer(text_))
    out: List[str] = []
    seen: Set[str] = set()

    def emit(term: str) -> None:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)

    for i, m in enumerate(matches):
        tok = m.group(0)
        emit(tok)
        glued = _LEXICAL_NUMBER_LETTERS_RE.fullmatch(tok)
        if glued:
            emit(glued.group(1))
            emit(glued.group(2))
        if i + 1 < len(matches) and _LEXICAL_NUMBER_RE.fullmatch(tok):
            nxt = matches[i + 1]
            # Joined only across whitespace: "5. Next" is two sentences.
            gap = text_[m.end():nxt.start()]
            if gap.isspace() and _LEXICAL_ALPHA_RE.fullmatch(nxt.group(0)):
                emit(tok + nxt.group(0))
    return out


def _sanitize_fts5_query(query: str) -> str:
    """Strip punctuation, collapse whitespace, and re-join tokens with
    ``OR`` so the resulting MATCH clause is bag-of-words rather than
    "every token must appear."

    AND semantics return zero hits on natural-language queries because
    BM25 can never contribute and hybrid collapses to semantic-only.
    OR is the standard bag-of-words relaxation BM25 expects.

    Empty → empty (caller treats that as "no BM25 leg")."""
    if not query:
        return ""
    # FTS5's unicode61 tokenizer splits on '.' and '/', so a form carrying
    # them is quoted: inside quotes FTS5 reads it as the phrase of its parts,
    # which is how it indexed the same characters in a chunk.
    tokens = [
        t if _FTS5_BAREWORD_RE.fullmatch(t) else f'"{t}"'
        for t in lexical_query_terms(query)
    ]
    if not tokens:
        return ""
    return " OR ".join(tokens)


# Words websearch_to_tsquery reads as OPERATORS. Passing them through as if
# they were search terms changes the meaning of the query.
_WEBSEARCH_OPERATORS = frozenset({"or", "and", "not"})


def _doc_scope(
    doc_ids: Optional[Iterable[str]],
) -> Tuple[Optional[str], Dict[str, str]]:
    """``AND c.doc_id IN (...)`` with one bound parameter per document id.

    ``None`` -> ``("", {})`` (no restriction). An empty collection ->
    ``(None, {})``: a filter that can match nothing, which the caller answers
    with ``[]`` instead of a query. Ids are bound, never interpolated, and
    deduplicated in a fixed order so equal sets build equal statements.
    """
    if doc_ids is None:
        return "", {}
    ids = sorted({str(d) for d in doc_ids if d})
    if not ids:
        return None, {}
    params = {f"bm25_doc_{i}": doc_id for i, doc_id in enumerate(ids)}
    placeholders = ", ".join(f":{name}" for name in params)
    return f" AND c.doc_id IN ({placeholders})", params


def _sanitize_websearch_query(query: str) -> str:
    """OR-join tokens for PostgreSQL's ``websearch_to_tsquery``.

    THE DEFECT THIS FIXES — the production retrieval failure
    --------------------------------------------------------
    The SQLite leg above OR-joins its tokens, and its docstring states the
    reason plainly: AND semantics return zero hits on natural-language queries,
    so BM25 can never contribute and hybrid collapses to semantic-only. That
    diagnosis was correct — and it was only ever applied to SQLite.

    PostgreSQL, which is what production runs, used ``plainto_tsquery``, and
    ``plainto_tsquery`` ANDs every term. The consequences in prod:

      * A chunk had to contain EVERY word of the question to be eligible. Ask
        "what is the soil backfilling specification" and only a chunk carrying
        soil AND backfilling AND specification could match.
      * ONE word absent from the whole corpus — a typo ("buiding"), a plural, a
        product name — reduced the entire BM25 leg to zero rows.
      * ``search()`` then hits ``if not bm25_results: return sem_results[:k]``
        and silently degrades to pure cosine, with nothing logged and nothing
        in the answer to say retrieval ran on one leg.

    So in production the lexical half of "hybrid" retrieval was dead for most
    real questions, and ranking fell to cosine alone over a corpus of thousands
    of chunks at k=5. That is why the same corpus answered a question correctly
    on one turn and reported the material absent on the next: nothing about the
    retrieval was stable, only the phrasing of the question changed.

    And it was structurally invisible to the test suite: dev and CI run SQLite,
    which takes the forgiving OR path, so no test could observe the semantics
    production actually used. The backends have to agree, which is what this
    makes true.

    ``websearch_to_tsquery`` is used rather than a hand-built ``to_tsquery``
    because it is the one tsquery parser designed for untrusted user input: it
    never raises a syntax error on stray punctuation or an unbalanced quote,
    where ``to_tsquery`` does.
    """
    if not query:
        return ""
    tokens = [
        t for t in lexical_query_terms(query)
        if t.lower() not in _WEBSEARCH_OPERATORS
    ]
    if not tokens:
        return ""
    return " or ".join(tokens)


# ── The lexical leg's score: rarity over matched lexemes, mildly length-normed ─
#
# ``ts_rank`` gave every matched query lexeme the same weight and did not
# normalise for length, so a long chunk won by holding MORE of the question's
# common words ('works', 'water', 'base') over a short chunk holding its rare
# ones. The score is now
#
#     sum of idf(lexeme) over the query lexemes the chunk holds
#     ------------------------------------------------------------
#          1 + ln(number of distinct lexemes in the chunk)
#
# * idf = ln(1 + 1/f), f the fraction of chunks holding the lexeme, read from
#   the store's own planner statistics (``pg_stats.most_common_elem_freqs``).
#   A lexeme absent from that list is rarer than the least common listed one,
#   so it takes that floor frequency. No statistics at all -> every lexeme
#   weighs the same. No word list, no fixed threshold.
# * The divisor grows with the log of the chunk's vocabulary: mild, so a
#   long chunk that really holds the rare terms still ranks.
# * The sum is scaled by 0.1 / (sum of every query lexeme's idf), one constant
#   per query, so the order is unchanged and the value stays on ``ts_rank``'s
#   scale (at most 0.1) for the callers that read ``Chunk.score`` from this leg.
#   Fusion uses the leg's rank, never this value.
#
# ── Bounded BM25 (exact top-k without ranking every matching row) ─────────
#
# The bag-of-words tsquery above matches most of a large project: measured on
# live, "commencement date stated agreement pump station works" matched 57,069
# of the master corpus's 118,665 chunks. Ranking all of them reads every one of
# those tsvectors -- ~155k buffers (1.2 GB) per call and ~80% of all the
# database reading one question did.
#
# The top-k does not need every match ranked. Order the query lexemes by idf,
# rarest first, w1 >= w2 >= ... >= wn. Admit every chunk that holds one of the
# r rarest lexemes, or at least t of the others. A chunk left out holds none of
# the r rarest and at most t-1 of the rest, so its weight sum is at most
# w(r+1) + ... + w(r+t-1), and its divisor is at least 1: its score is at most
# that sum times the query's scale. Once the k-th score among the admitted
# chunks reaches that bound, no chunk left out can enter the top-k. "Holds one
# of these, or t of those" is something the GIN index answers without touching
# the heap for the rest, and the rarest lexemes admit few rows.
#
# ``_bm25_bounded`` ranks exactly the chunks the GIN index admits for (r, t),
# and accepts the result only when the k-th score proves the bound. Otherwise it
# admits more rare lexemes and lowers t, and in the end falls back to the full
# scan. The result is identical to ranking every match.

# The score's ceiling: ``ts_rank``'s for an unweighted tsvector (weight D).
_LEXICAL_SCORE_SCALE = 0.1
# Relative margin on the bound for float rounding in the database's sum.
_BM25_BOUND_MARGIN = 1e-9
# Most OR-of-AND groups one candidate tsquery may hold. The GIN index checks
# every group per candidate row, so a much larger tsquery costs more CPU than
# the reads it saves (measured on live: 600 groups took 1.2 s, 120 took 0.2 s
# for the same 26k buffers).
_BM25_MAX_CONJUNCTIONS = 120
# First threshold tried. On live the proving threshold was 4-5 lexemes for
# questions of 6 to 17 lexemes; starting higher only adds rounds.
_BM25_START_THRESHOLD = 5
# Most bounded attempts, and most attempts that came back with fewer than k
# rows (a small or empty project), before the full scan answers instead.
_BM25_MAX_ROUNDS = 6
_BM25_MAX_SHORT_ROUNDS = 2
# One operand of a tsquery's text form: a quoted lexeme with no weight/prefix.
_TSQUERY_LEXEME_RE = re.compile(r"^'((?:[^']|'')+)'$")
# Every quoted lexeme anywhere in a tsquery's text form.
_TSQUERY_ANY_LEXEME_RE = re.compile(r"'((?:[^']|'')+)'")


def tsquery_or_lexemes(tsquery_text: Optional[str]) -> Optional[List[str]]:
    """Distinct lexemes of a tsquery that is a plain OR of lexemes.

    ``"'commenc' | 'date'"`` -> ``['commenc', 'date']``; an empty query ->
    ``[]``; anything else (a phrase from a compound token, ``&``, ``!``, a
    weight or prefix marker) -> ``None``, which the caller answers with the
    full scan.
    """
    raw = (tsquery_text or "").strip()
    if not raw:
        return []
    out: Set[str] = set()
    for part in raw.split(" | "):
        m = _TSQUERY_LEXEME_RE.match(part.strip())
        if not m:
            return None
        out.add(m.group(1).replace("''", "'"))
    return sorted(out)


def tsquery_lexemes(tsquery_text: Optional[str]) -> List[str]:
    """Every distinct lexeme a tsquery names, whatever joins them.

    The score weighs the lexemes a chunk holds; a phrase's lexemes count one
    by one (the ``@@`` predicate still decides whether the chunk matches).
    """
    return sorted({
        m.group(1).replace("''", "'")
        for m in _TSQUERY_ANY_LEXEME_RE.finditer(tsquery_text or "")
    })


def lexeme_weights(
    lexemes: Iterable[str], frequency: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """idf = ln(1 + 1/f) per lexeme; see the notes above ``_LEXICAL_SCORE_SCALE``.

    ``frequency`` is the fraction of chunks holding each listed lexeme. An
    unlisted lexeme takes the least listed frequency (it is rarer than every
    listed one); with nothing listed every lexeme weighs 1.
    """
    import math

    freq = {k: float(v) for k, v in (frequency or {}).items() if v and v > 0}
    floor = min(freq.values()) if freq else None
    out: Dict[str, float] = {}
    for lx in lexemes:
        f = freq.get(lx, floor)
        out[lx] = 1.0 if f is None else math.log(1.0 + 1.0 / min(f, 1.0))
    return out


def lexical_score_scale(weights: Dict[str, float]) -> float:
    """The per-query constant that puts the score on ``ts_rank``'s scale."""
    total = sum(weights.values())
    return _LEXICAL_SCORE_SCALE / total if total > 0 else 0.0


def rarest_first(lexemes: Iterable[str], weights: Optional[Dict[str, float]] = None) -> List[str]:
    """``lexemes`` by idf, highest first; ties by the lexeme itself."""
    w = weights or {}
    return sorted(set(lexemes), key=lambda x: (-w.get(x, 0.0), x))


def _tsquery_literal(lexeme: str) -> str:
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


def bm25_candidate_tsquery(
    lexemes: List[str], t: int, weights: Optional[Dict[str, float]] = None,
    rare: int = 0,
) -> Optional[str]:
    """A tsquery matched by every chunk that holds one of the ``rare`` rarest
    ``lexemes``, or at least ``t`` of the others.

    A chunk with ``t`` of the ``m`` others holds at least ``j`` of any
    ``m - t + j`` of them, so OR-ing the ``j``-subsets of the ``m - t + j``
    rarest admits all such chunks while the common lexemes stay out of the
    index lookup. The largest ``j`` (up to 3) whose subset count fits
    ``_BM25_MAX_CONJUNCTIONS`` is used. When ``t`` exceeds the others, only
    the rare lexemes admit. ``weights`` only orders the lexemes.
    """
    from itertools import combinations
    from math import comb

    order = rarest_first(lexemes, weights)
    n = len(order)
    if n == 0 or t < 1 or rare < 0 or rare >= n:
        return None
    singles = [f"({_tsquery_literal(x)})" for x in order[:rare]]
    rest = order[rare:]
    m = len(rest)
    if t > m:
        return " | ".join(singles) if singles else None
    budget = _BM25_MAX_CONJUNCTIONS - len(singles)
    for j in (3, 2, 1):
        if j > t:
            continue
        r = m - t + j
        if comb(r, j) > budget:
            continue
        groups = (
            " & ".join(_tsquery_literal(x) for x in combo)
            for combo in combinations(rest[:r], j)
        )
        return " | ".join(singles + [f"({g})" for g in groups])
    return None


def bm25_excluded_bound(
    lexemes: List[str], weights: Dict[str, float], t: int, rare: int = 0,
) -> float:
    """Highest score a chunk left out by ``bm25_candidate_tsquery`` can have.

    It holds none of the ``rare`` rarest lexemes and at most ``t - 1`` of the
    rest, so its weight sum is at most the ``t - 1`` largest weights after the
    first ``rare``; its divisor is at least 1.
    """
    order = rarest_first(lexemes, weights)
    held = order[rare:rare + max(t - 1, 0)]
    return sum(weights.get(x, 0.0) for x in held) * lexical_score_scale(weights)


def bm25_rank_bound_holds(kth_rank: float, bound: float) -> bool:
    """True when no chunk scoring at most ``bound`` can displace the k-th."""
    return kth_rank >= bound * (1 + _BM25_BOUND_MARGIN)


# ── Public types ──────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """One indexed chunk. ``embedding`` is omitted from the public
    serializer to keep response payloads small.

    ``rrf_score`` is set on results from the hybrid path (debug-only,
    not serialized). ``score`` remains the primary ranking signal —
    cosine for semantic-only; for hybrid results it is the FUSED rank
    expressed on the cosine scale (see ``_rrf_combine``), with the raw
    leg scores kept beside it in ``vector_score`` / ``lexical_score``
    and the fused position in ``fused_rank`` (all debug-only)."""

    chunk_id: str
    project_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: Optional[float] = None  # set on search results, None when raw
    rrf_score: Optional[float] = field(default=None, repr=False, compare=False)
    # Hybrid path only, debug-only like ``rrf_score``: the raw score each leg
    # gave this chunk (cosine / lexical rank; None when that leg missed it)
    # and its 1-based position after fusion. Never serialized.
    vector_score: Optional[float] = field(default=None, repr=False, compare=False)
    lexical_score: Optional[float] = field(default=None, repr=False, compare=False)
    fused_rank: Optional[int] = field(default=None, repr=False, compare=False)
    # Retrieval layer this chunk came from, set by ``retrieve_with_filter``:
    # "own" (the active project), "general_knowledge" (a disclosed curated GK
    # project), or "master_corpus" (the labeled empty/thin fallback corpus).
    # STEP 0 isolation signal — never sent to the LLM or the wire; the chat
    # runtime reads it to disclose a Master-Corpus fallback in the answer +
    # sources panel. compare=False so it never affects Chunk equality in tests.
    layer: str = field(default="own", compare=False)
    # BOQ item references found in ``text``, set by ``retrieve_with_filter``.
    # These are LABELS ("D529.2" is an item number, not 529.2): retrieve,
    # display and cite them, never reason about them as figures. Not
    # serialized and compare=False, like ``layer`` -- it is a tag on the same
    # text, not new content.
    ref_codes: tuple = field(default=(), compare=False, repr=False)
    # Layered RAG (docs/rag-deployment-plan.md). ``knowledge_layer`` is the
    # PERSISTED L1/L2A/L2B/L3 layer (see app.core.rag.layers.LAYERS), hydrated
    # by ``search`` from the stored column. It is deliberately a SEPARATE field
    # from ``layer`` above: ``layer`` is the per-query STEP 0 isolation tag that
    # the retriever recomputes on every search, so it cannot also carry the
    # persisted layer. ``authority`` is the persisted authority label (one of
    # app.core.rag.layers.AUTHORITIES). Both drive the authority-precedence
    # re-rank when RAG_LAYERED is on, and both are None for unlayered rows.
    # compare=False so neither affects Chunk equality in tests.
    knowledge_layer: Optional[str] = field(default=None, compare=False)
    authority: Optional[str] = field(default=None, compare=False)
    # 1-based page of the source PDF where this chunk's text starts; None for
    # non-PDF sources and for rows indexed before pages were recorded.
    page: Optional[int] = field(default=None, compare=False)
    # OCR provenance (Alembic 0033): True when this chunk's text was captured
    # by OCR rather than the PDF text layer. Hydrated from the stored column;
    # compare=False so it does not affect Chunk equality.
    ocr: bool = field(default=False, compare=False)
    # Revision currency (app.core.rag.revision, audit §5.2). Filename-derived,
    # set by ``retrieve_with_filter`` on every search: ``revision`` is the
    # parsed single-char revision token ("" when none), ``drawing_number`` the
    # parsed sheet code, ``superseded`` True when the filename marks the doc
    # obsolete. Internal ranking + answer-disclosure signals — never serialized
    # to the wire; compare=False so none affects Chunk equality in tests.
    revision: str = field(default="", compare=False)
    drawing_number: str = field(default="", compare=False)
    superseded: bool = field(default=False, compare=False)
    # The uploader's original filename, resolved alongside the fields above.
    # Construction documents carry real facts in their NAMES that never reach
    # the text layer — expiry dates, revision letters, discipline codes,
    # status. Live example: "AM Rev Design NOC ... (exp. 17Jul25).pdf", where
    # 17 July 2025 appears nowhere in the extracted text, so the expiry was
    # unanswerable however good retrieval got. Passed to the model in the
    # context marker; internal like the rest, never serialized to the wire.
    source_name: str = field(default="", compare=False)
    # The document's type and role from its documents row (S-R11), filled by
    # ``inject.annotate_document_kinds`` before the excerpt is handed, so the
    # model can tell a specification from a drawing or a contractor's method
    # statement. "" until read; internal like ``source_name``, never on the wire.
    doc_type: str = field(default="", compare=False)
    doc_role: str = field(default="", compare=False)
    # ── photo_chunks fields (kind="photo") ─────────────────────────────
    # text chunks keep kind="text" and the photo fields default to None.
    kind: str = "text"
    sha256: Optional[str] = None
    photo_url: Optional[str] = None
    photo_metadata: Optional[dict] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # ONE choke point, not the eight places a Chunk is built from a row.
        # Some PDFs fake bold by printing each glyph twice and index as
        # "EEnnggiinneeeerr EEXXAAMMPPLLEECCOO"; every rule downstream then reads
        # junk (live: "Clause (as" was returned as the Engineer). Repairing
        # here fixes documents already in the index, with no re-index, and no
        # future read path can forget to. Clean text is returned untouched.
        if self.text:
            self.text = repair_fake_bold(self.text)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop None scores from API responses
        if d["score"] is None:
            d.pop("score")
        # rrf_score and the per-leg scores are debug-only; never on the wire
        d.pop("rrf_score", None)
        d.pop("vector_score", None)
        d.pop("lexical_score", None)
        d.pop("fused_rank", None)
        # layer/knowledge_layer/authority are internal ranking+isolation
        # signals; keep them off the API payload
        d.pop("layer", None)
        d.pop("knowledge_layer", None)
        d.pop("authority", None)
        if d.get("page") is None:
            d.pop("page", None)
        # Revision-currency signals are internal (ranking + disclosure), not wire
        d.pop("revision", None)
        d.pop("drawing_number", None)
        d.pop("superseded", None)
        d.pop("source_name", None)
        d.pop("doc_type", None)
        d.pop("doc_role", None)
        # Drop photo fields for plain text chunks to keep payloads small
        if d.get("kind") == "text":
            d.pop("sha256", None)
            d.pop("photo_url", None)
            d.pop("photo_metadata", None)
        return d


# ── Module-level cache (one store per database URL) ───────────────────────

_STORE_CACHE: dict = {}
_CACHE_LOCK = Lock()
_INITIALIZED_NAMESPACES: Set[Tuple[str, str]] = set()
_INIT_LOCK = Lock()


# pgvector 0.8 keeps scanning the index until enough rows survive the query's
# WHERE clause, instead of stopping after the first ef_search candidates.
# ``relaxed_order`` is the right mode for retrieval: results may come back
# slightly out of distance order, and the retriever re-ranks by score anyway.
_ITERATIVE_SCAN_MODE = os.getenv("RAG_HNSW_ITERATIVE_SCAN", "relaxed_order").strip()

# One-shot latch. The setting does not exist before pgvector 0.8, and retrying
# a statement that will always fail once per search is pure latency.
_ITERATIVE_SCAN_SUPPORTED: Optional[bool] = None


def _enable_iterative_scan(session) -> None:
    """Make filtered HNSW search return rows instead of silently returning few.

    Without this, ``ORDER BY embedding <=> :q ... WHERE project_id = :p`` is a
    post-filter: HNSW picks its nearest candidates across the WHOLE table and
    Postgres then discards the ones from other projects. On a multi-tenant
    table that can leave nothing at all -- measured on live, a project with
    13,922 of 172,809 chunks returned 0 rows with "Rows Removed by Filter: 39".

    Best-effort by design. On pgvector < 0.8 the GUC does not exist; the search
    still works, just with the old recall, so a failure here must never break
    retrieval. It is latched rather than swallowed silently: logged once, then
    skipped.
    """
    global _ITERATIVE_SCAN_SUPPORTED
    if _ITERATIVE_SCAN_SUPPORTED is False or not _ITERATIVE_SCAN_MODE:
        return
    if _ITERATIVE_SCAN_MODE not in ("relaxed_order", "strict_order", "off"):
        logger.warning(
            "invalid RAG_HNSW_ITERATIVE_SCAN=%r; expected relaxed_order / "
            "strict_order / off — leaving the default in place",
            _ITERATIVE_SCAN_MODE,
        )
        _ITERATIVE_SCAN_SUPPORTED = False
        return
    try:
        stmt = text(f"SET LOCAL hnsw.iterative_scan = {_ITERATIVE_SCAN_MODE}")
        if _ITERATIVE_SCAN_SUPPORTED or not hasattr(session, "begin_nested"):
            session.execute(stmt)
        else:
            # First probe in a savepoint: a failed SET aborts the transaction,
            # and the search that follows on this session would fail with it.
            with session.begin_nested():
                session.execute(stmt)
        _ITERATIVE_SCAN_SUPPORTED = True
    except Exception as exc:  # noqa: BLE001 — recall optimisation, not a gate
        if _ITERATIVE_SCAN_SUPPORTED is None:
            logger.warning(
                "hnsw.iterative_scan unavailable (%s); filtered vector search "
                "keeps pgvector's post-filter recall — upgrade to pgvector 0.8+",
                exc,
            )
        _ITERATIVE_SCAN_SUPPORTED = False


def _is_production() -> bool:
    env = os.getenv("ENV", os.getenv("ENVIRONMENT", "")).strip().lower()
    return env in ("prod", "production")


def _rag_vector_namespace() -> str:
    """Active RAG vector namespace. Legacy ``chunks`` table = empty string."""
    return os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()


def get_lexical_store(
    db_path: Optional[str] = None,
    namespace: Optional[str] = None,
) -> "VectorStore":
    """Open the store WITHOUT constructing an embedder — for BM25-only use.

    ``get_store`` always builds the embedder, purely to read ``identity["dim"]``
    for the table width. That made every read path depend on the embedding
    model, including BM25, which never touches the vector column: remove the
    model and keyword search over an already-indexed corpus died with it, even
    though the text was fully present and matchable.

    The width and model name are read from the table's OWN rows instead, so the
    identity check this constructs with is the identity already stored — it can
    neither trip the mixed-model guard nor mislabel what is written (nothing is
    written through this path; it is read-only by construction).

    Falls back to the configured/default identity for an empty table, where
    there is nothing to be inconsistent with.
    """
    path = db_path or _default_db_path()
    url = _database_url(path)
    ns = namespace if namespace is not None else _rag_vector_namespace()
    table = rag_chunk_table_name(ns)

    model_name: Optional[str] = None
    dim: Optional[int] = None
    try:
        with _engine_for_url(url).connect() as conn:
            row = conn.execute(
                text(f"SELECT embedding_model, embedding_dim FROM {table} LIMIT 1")
            ).first()
        if row and row[0]:
            model_name, dim = str(row[0]), int(row[1])
    except Exception:  # noqa: BLE001
        # Missing table, or the legacy namespace whose rows have no identity
        # columns. The configured defaults below are correct in both cases, so
        # this is recoverable — but not silent: if the identity probe fails for
        # any OTHER reason, the store opens on assumed defaults, and that is
        # worth being able to see.
        logger.info(
            "lexical store: could not read stored identity from %s; using "
            "configured defaults", table, exc_info=True,
        )

    if not model_name:
        model_name = (os.getenv("RAG_EMBEDDING_MODEL") or "").strip() or "unknown"
    if not dim:
        # No stored width (empty/new table): open at the CONFIGURED embedder's
        # actual dimension, never a hardcoded constant, so the table width is
        # the width that will be written. model_name "unknown" == env unset ->
        # resolve the default embedder.
        probe = None if model_name == "unknown" else model_name
        dim = get_embedder(model_name=probe).identity["dim"]

    key = (url, ns, model_name, dim)
    with _CACHE_LOCK:
        if key not in _STORE_CACHE:
            _STORE_CACHE[key] = VectorStore(
                db_path=path, dim=dim, namespace=ns, model_name=model_name,
            )
    return _STORE_CACHE[key]


def get_store(
    dim: Optional[int] = None,
    db_path: Optional[str] = None,
    namespace: Optional[str] = None,
    model_name: Optional[str] = None,
) -> "VectorStore":
    """Process-cached store. Different ``db_path``/namespace values get
    different cached instances, which keeps tests isolated when they swap
    DATA_DIR or embedder.

    ``dim`` is optional; when omitted the store uses the configured
    embedder's actual dimension so the table width always matches the
    vectors being written.
    """
    path = db_path or _default_db_path()
    url = _database_url(path)
    ns = namespace if namespace is not None else _rag_vector_namespace()
    embedder = get_embedder(model_name=model_name)
    identity = embedder.identity
    actual_dim = dim if dim is not None else identity["dim"]
    key = (url, ns, identity["model"], identity["dim"])
    with _CACHE_LOCK:
        if key not in _STORE_CACHE:
            _STORE_CACHE[key] = VectorStore(
                db_path=path,
                dim=actual_dim,
                namespace=ns,
                model_name=identity["model"],
            )
    return _STORE_CACHE[key]


def reset_store_cache() -> None:
    """Drop all cached stores. Used by tests to pick up a swapped DATA_DIR."""
    global _STORE_CACHE, _INITIALIZED_NAMESPACES
    with _CACHE_LOCK:
        for s in _STORE_CACHE.values():
            try:
                s.close()
            except Exception:
                logger.warning(
                    "swallowed %s in reset_store_cache() — continuing",
                    "Exception", exc_info=True,
                )
        _STORE_CACHE = {}
    with _INIT_LOCK:
        _INITIALIZED_NAMESPACES = set()


def _default_db_path() -> str:
    """Default backing path/URL for the unified schema database."""
    url = get_database_url()
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///") :]
    return url


def _database_url(db_path: str) -> str:
    """Map a legacy file path or explicit URL to a SQLAlchemy database URL."""
    if "://" in db_path:
        return db_path
    return f"sqlite:///{os.path.abspath(db_path)}"


def _ensure_sqlite_parent_dir(url: str) -> None:
    if url.startswith("sqlite:///"):
        parent = os.path.dirname(url[len("sqlite:///") :])
        if parent:
            os.makedirs(parent, exist_ok=True)


# ── One retrieval per question, not one per caller ──────────────────────────
# Live 2026-10-07: one question issued BM25 eight times. The dual search ran it
# for the raw question and for its wrapper-stripped variant, which
# websearch_to_tsquery reduces to the SAME tsquery; and the model's own
# search_project_documents call then repeated the turn's retrieval. Inside a
# chat turn (``start_turn_memo``) identical lexical and vector searches are
# answered from the turn's memo. The memo holds the database rows (immutable),
# never Chunk objects, which rerankers adjust in place. Outside a turn nothing
# is memoised.
_TURN_MEMO: "contextvars.ContextVar[Optional[Dict[tuple, list]]]" = contextvars.ContextVar(
    "rag_turn_memo", default=None)


#: A project with at most this many chunks is ranked exactly from its own rows
#: rather than through the HNSW graph (see VectorStore._search_pgvector). At
#: ~4 chunks per heap page that is ~1,000 pages, below a typical filtered HNSW
#: walk of the table.
_EXACT_SCAN_MAX_ROWS = 4000
#: The planner's estimate of a project's chunk rows (planning only: reads
#: no table page); see ``_project_is_small``.
_PROJECT_ROWS_ESTIMATE_SQL = "EXPLAIN (FORMAT JSON) SELECT 1 FROM {table} WHERE project_id = :p"


def start_turn_memo() -> "contextvars.Token":
    """Begin a turn's retrieval memo in the current context; returns the reset token."""
    return _TURN_MEMO.set({})


def end_turn_memo(token: "contextvars.Token") -> None:
    _TURN_MEMO.reset(token)


_MISS = object()


def _memo_get(key: tuple):
    """The turn's memoised value for ``key``, or ``_MISS``."""
    memo = _TURN_MEMO.get()
    return _MISS if memo is None else memo.get(key, _MISS)


def _query_ref(query: Optional[str]) -> str:
    """A query as a log may carry it: its length in characters and a short
    sha256, never its words. Equal queries give equal refs, so log lines
    still join on it."""
    q = query or ""
    digest = hashlib.sha256(q.encode("utf-8")).hexdigest()[:12]
    return f"chars={len(q)} sha256={digest}"


def _terms_ref(terms) -> str:
    """Terms taken from a question (needles, identifiers) as a log may carry
    them: how many and a short sha256 of the list, never the terms. Equal
    lists give equal refs, so log lines still join on it."""
    items = [str(t) for t in (terms or [])]
    digest = hashlib.sha256("".join(items).encode("utf-8")).hexdigest()[:12]
    return f"count={len(items)} sha256={digest}"


def _error_ref(exc: BaseException) -> str:
    """A database error as a log may carry it: its class, the driver's class
    and the SQLSTATE when there is one. Never its message: SQLAlchemy
    renders the bound parameters into it (the query) and FTS5 echoes the
    MATCH text."""
    orig = getattr(exc, "orig", None)
    parts = [type(exc).__name__]
    if orig is not None:
        parts.append(type(orig).__name__)
        state = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        if state:
            parts.append(f"sqlstate={state}")
    return " ".join(parts)


def _memo_put(key: tuple, value):
    memo = _TURN_MEMO.get()
    if memo is not None:
        memo[key] = value
    return value


def _retrieval_trace_on() -> bool:
    """The retrieval trace's own switch (``inject.retrieval_trace_enabled``),
    read lazily: ``inject`` imports this module."""
    from app.core.rag.inject import retrieval_trace_enabled

    return retrieval_trace_enabled()


def record_fusion(project_id: str, summary: Dict[str, Any]) -> None:
    """Append one fusion's leg summary (``fusion_trace.leg_summary``) to the
    turn's list for ``project_id``. Outside a turn nothing is kept."""
    memo = _TURN_MEMO.get()
    if memo is not None:
        memo.setdefault(("fusion", project_id), []).append(summary)


def fusion_records(project_id: str) -> list:
    """The turn's fusion summaries for ``project_id``, oldest first."""
    hit = _memo_get(("fusion", project_id))
    return [] if hit is _MISS else hit


def record_fused_candidates(project_id: str, record: Optional[Dict[str, Any]]) -> None:
    """The last retrieval's followed fusion summary (``fusion_trace.Follow``)
    for ``project_id``; the pre_injection trace line reads it."""
    _memo_put(("fused_candidates", project_id), record)


def fused_candidates_record(project_id: str) -> Optional[Dict[str, Any]]:
    hit = _memo_get(("fused_candidates", project_id))
    return None if hit is _MISS else hit


def _memoised(key: tuple, compute):
    hit = _memo_get(key)
    return _memo_put(key, compute()) if hit is _MISS else hit


_TERM_COUNT_CACHE_LOCK = Lock()
# The session-scope clause when the scope cannot be computed: no row.
_NO_ROW_SQL = " AND 1 = 0"


def _projects_scope_key():
    """The turn's chat-session scope, for memo keys of scoped searches."""
    from app.core import projects as _projects

    return _projects.current_session_scope()


def hidden_document_predicate(prefix: str = "", *, postgres: bool = True) -> str:
    """The condition that hides a document from retrieval.

    ``retrieval_visible`` false, or an ingest status that has no chunks
    (failed, empty, tombstoned, quarantined ...). One definition: the
    retrieval filter uses it, and alembic 0025 indexes exactly this condition
    (``idx_documents_hidden_from_retrieval``) so the filter is an index lookup
    over the few hidden documents instead of a scan of every documents row.
    """
    from app.core.ingest_status import NO_CHUNK_STATUSES

    flag = "IS FALSE" if postgres else "= 0"
    barred = ", ".join(f"'{s}'" for s in sorted(NO_CHUNK_STATUSES))
    return f"({prefix}retrieval_visible {flag} OR {prefix}ingest_status IN ({barred}))"


def _ensure_hnsw_index(eng, table_name: str) -> None:
    """Ensure the pgvector HNSW ANN index on ``{table}.embedding`` exists.

    Semantic search runs ``ORDER BY embedding <=> q``; WITHOUT this index that
    is an exact sequential scan — fine at ~10k chunks, catastrophic at 100k+
    (a full-Drive corpus layer is 133k: per-query full-scan hit tens of
    seconds and dropped the DB). HNSW + ``vector_cosine_ops`` matches the
    ``<=>`` cosine operator the retriever uses. ``IF NOT EXISTS`` makes this a
    one-time build; later boots skip it. Never raises — a missing index
    degrades to full-scan, it must not crash startup.
    """
    try:
        with eng.begin() as conn:
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw "
                f"ON {table_name} USING hnsw (embedding vector_cosine_ops)"
            ))
    except Exception:  # noqa: BLE001 — full-scan fallback, never block startup
        logger.warning(
            "could not ensure hnsw index on %s (semantic search will "
            "seq-scan until it exists)", table_name, exc_info=True,
        )


def _ensure_embedding_inline(eng, table_name: str) -> None:
    """Keep ``embedding`` in the heap row (``STORAGE MAIN``), not in TOAST.

    pgvector declares ``vector`` with EXTERNAL storage, so once a row passes
    the ~2 KB TOAST threshold the embedding is the column moved out of line.
    Every distance computed outside the HNSW index -- the exact
    ``ORDER BY embedding <=> q`` over one project's rows the planner picks
    for a small project -- then fetches each row's embedding through the
    TOAST index: about six buffers per row instead of a share of one heap
    page. MAIN makes the toaster move ``text`` / ``text_search`` out first
    and keep the vector inline. Rows written before the change keep their
    old layout until they are rewritten. Metadata-only; never raises.
    """
    try:
        with eng.begin() as conn:
            storage = conn.execute(text(
                "SELECT a.attstorage FROM pg_attribute a "
                "WHERE a.attrelid = CAST(:t AS regclass) AND a.attname = 'embedding'"
            ), {"t": table_name}).scalar()
            if storage is None or storage == "m":
                return
            conn.execute(text("SET LOCAL lock_timeout = '2s'"))
            conn.execute(text(
                f"ALTER TABLE {table_name} ALTER COLUMN embedding SET STORAGE MAIN"
            ))
    except Exception:  # noqa: BLE001 — a storage hint must not block startup
        logger.warning(
            "could not set inline storage for %s.embedding (exact vector "
            "scans read TOAST until it is set)", table_name, exc_info=True,
        )


#: Rows up to this size stay whole in the heap page instead of being TOASTed.
#: A chunk row -- text, its tsvector, the embedding -- is ~3-4 KB; the default
#: target (~2 KB) pushed the tsvector out of line once the embedding stayed in.
_CHUNK_TOAST_TUPLE_TARGET = 4080


def _ensure_tsvector_inline(eng, table_name: str) -> None:
    """Keep ``text_search`` in the heap row so BM25 ranking reads no TOAST.

    ``ts_rank`` reads each candidate's tsvector. Live 2026-10-07 the tsvector
    sat in TOAST (moved there once the embedding stayed inline, alembic
    0023), so ranking ~1,000 candidates cost ~3,800 buffers instead of ~880.
    The table's ``toast_tuple_target`` is raised so an ordinary chunk row
    stays whole, and ``text_search`` is ``STORAGE MAIN`` so an oversized row
    moves its ``text`` out first. Rows written earlier keep their layout until
    rewritten. Metadata-only; never raises.
    """
    try:
        with eng.begin() as conn:
            row = conn.execute(text(
                "SELECT c.reloptions, a.attstorage FROM pg_class c "
                "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'text_search' "
                "WHERE c.oid = CAST(:t AS regclass)"
            ), {"t": table_name}).first()
            if row is None:
                return
            opts, storage = list(row[0] or []), row[1]
            target_set = f"toast_tuple_target={_CHUNK_TOAST_TUPLE_TARGET}" in opts
            if target_set and storage == "m":
                return
            conn.execute(text("SET LOCAL lock_timeout = '2s'"))
            if not target_set:
                conn.execute(text(
                    f"ALTER TABLE {table_name} SET (toast_tuple_target = {_CHUNK_TOAST_TUPLE_TARGET})"
                ))
            if storage != "m":
                conn.execute(text(
                    f"ALTER TABLE {table_name} ALTER COLUMN text_search SET STORAGE MAIN"
                ))
    except Exception:  # noqa: BLE001 — a storage hint must not block startup
        logger.warning(
            "could not keep %s.text_search inline (BM25 ranking reads TOAST "
            "until it is set)", table_name, exc_info=True,
        )


def _ensure_trigram_index(eng, table_name: str) -> None:
    """Ensure the ``pg_trgm`` GIN index on ``lower(text)`` exists.

    ``chunks_containing_all`` and ``identifier_search`` filter on
    ``LOWER(text) LIKE '%needle%'``. A btree cannot serve a leading wildcard,
    so without this the project's whole text is read to find a rare phrase
    (~48k buffers per call on the live master corpus). Alembic 0022 creates
    it for the tables present at deploy; namespaced tables created later get
    it here. Never raises -- a missing index costs reads, not correctness.
    """
    try:
        with eng.begin() as conn:
            present = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
            ).first()
            if present is None:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        with eng.begin() as conn:
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {table_name}_text_trgm "
                f"ON {table_name} USING gin (lower(text) gin_trgm_ops)"
            ))
    except Exception:  # noqa: BLE001 — substring reads fall back to the project scan
        logger.warning(
            "could not ensure the trigram index on %s (substring recall will "
            "read the project's text until it exists)", table_name, exc_info=True,
        )


def table_already_exists(exc: BaseException) -> bool:
    """True when a CREATE lost a race rather than genuinely failing.

    ``ensure_table`` issues ``CREATE TABLE`` together with the table's own
    indexes and unique constraint as one batch, so a concurrent creator can
    leave ANY of those already present. The loser then sees the table
    ("table X already exists"), one of its indexes
    ("index idx_chunks_<ns>_doc already exists") or its constraint reported
    "already exists" -- every one means the object we asked for is now there,
    which is the outcome we wanted. SQLite phrases it "<kind> X already exists"
    (OperationalError); PostgreSQL uses SQLSTATE 42P07 (duplicate table or
    index) or 42710 (duplicate_object, e.g. a named constraint). A genuine
    failure -- "no such table", "database is locked", a disk error -- carries
    none of these words and still propagates.
    """
    code = getattr(getattr(exc, "orig", None), "pgcode", None)
    if code in {"42P07", "42710"}:
        return True
    msg = str(exc).lower()
    return "already exists" in msg and any(
        kind in msg for kind in ("table", "relation", "index", "constraint")
    )


def ensure_table(table, *, bind) -> None:
    """CREATE ``table`` if absent, tolerating a concurrent creator.

    ``checkfirst=True`` is check-THEN-create and therefore not atomic: two
    callers can both see "absent" and both issue CREATE, and the loser raises.
    ``_INIT_LOCK`` below only serialises threads inside ONE process, so it does
    not cover a second uvicorn worker, nor a test creating the table on its own
    engine while the app creates it on the request thread.

    That race is not theoretical. It failed CI on an unrelated PR as
    ``table chunks_t48b48c8a1360 already exists``, and had shown up earlier as
    ``database is locked`` from the same two writers -- one root cause, two
    symptoms, and easy to dismiss as flakiness because each looks like a
    different fault.

    Losing this race is a success: the table exists either way.
    """
    try:
        table.create(bind=bind, checkfirst=True)
    except SQLAlchemyError as exc:
        if not table_already_exists(exc):
            raise
        logger.debug(
            "table %s was created concurrently; continuing", table.name,
        )


def _ensure_page_column(eng, table_name: str) -> None:
    """Add the nullable ``page`` column to a chunk table that predates it.

    Namespaced tables are created here, not by Alembic, so a table created
    before 0021 must gain the column here too or every read that selects
    ``page`` fails. Idempotent; a failure is logged, never raised.
    """
    try:
        from sqlalchemy import inspect as _inspect

        columns = {c["name"] for c in _inspect(eng).get_columns(table_name)}
        if "page" in columns:
            return
        with eng.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN page INTEGER"))
    except Exception:  # noqa: BLE001 — never block startup; reads log their own failure
        logger.warning(
            "could not ensure the page column on %s", table_name, exc_info=True,
        )


def _ensure_ocr_column(eng, table_name: str) -> None:
    """Add the ``ocr`` provenance column to a chunk table that predates it.

    Namespaced tables are created here, not by Alembic, so a table created
    before 0033 must gain the column here too or every write that sets
    ``ocr`` fails. Idempotent; a failure is logged, never raised.
    """
    try:
        from sqlalchemy import inspect as _inspect

        columns = {c["name"] for c in _inspect(eng).get_columns(table_name)}
        if "ocr" in columns:
            return
        with eng.begin() as conn:
            conn.execute(text(
                f"ALTER TABLE {table_name} ADD COLUMN ocr BOOLEAN NOT NULL DEFAULT FALSE"
            ))
    except Exception:  # noqa: BLE001 — never block startup; reads log their own failure
        logger.warning(
            "could not ensure the ocr column on %s", table_name, exc_info=True,
        )


def _ensure_schema(url: str, rag_chunk_cls: type) -> None:
    global _INITIALIZED_NAMESPACES
    table_name = rag_chunk_cls.__tablename__
    init_key = (url, table_name)
    with _INIT_LOCK:
        if init_key in _INITIALIZED_NAMESPACES:
            return
        _ensure_sqlite_parent_dir(url)
        eng = _engine_for_url(url)
        ensure_table(rag_chunk_cls.__table__, bind=eng)
        _ensure_page_column(eng, table_name)
        _ensure_ocr_column(eng, table_name)
        # `checkfirst=True` above SKIPS the whole table create — indexes
        # included — when the table already exists. A prod table that predates
        # the idx_chunks_project declaration therefore never got the btree, so
        # COUNT/filter-by-project seq-scans the full table (~11s on the master
        # corpus; pgvector search stays fast via its own index). Create each
        # declared index explicitly + idempotently so legacy tables get it too.
        for index in rag_chunk_cls.__table__.indexes:
            try:
                index.create(bind=eng, checkfirst=True)
            except Exception:  # noqa: BLE001 — never block startup on an index
                # Not blocking startup is right; staying SILENT is not. The
                # comment above measures the cost of a missing btree here:
                # ~11s seq-scans on the master corpus. Swallowed, that shows
                # up as "retrieval got slow" with nothing to point at.
                logger.warning(
                    "could not create index %s on %s — queries filtering by "
                    "project_id may fall back to a sequential scan",
                    getattr(index, "name", "?"), rag_chunk_cls.__tablename__,
                    exc_info=True,
                )
        # PostgreSQL BM25 leg: the ``text_search`` GENERATED column + GIN
        # index. Alembic 0003 only covers the legacy ``chunks`` table;
        # namespaced tables (chunks_v2, ...) are created HERE, so they must
        # get the column here too or ``_bm25_postgres`` fails with
        # UndefinedColumn. Idempotent via IF NOT EXISTS.
        if eng.dialect.name == "postgresql":
            try:
                with eng.begin() as conn:
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS "
                        "text_search tsvector GENERATED ALWAYS AS "
                        "(to_tsvector('english', text)) STORED"
                    ))
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {table_name}_fts_gin "
                        f"ON {table_name} USING GIN (text_search)"
                    ))
            except Exception:  # noqa: BLE001 — BM25 degrades, never block startup
                logger.warning(
                    "could not ensure text_search column on %s", table_name,
                    exc_info=True,
                )
            _ensure_embedding_inline(eng, table_name)
            _ensure_tsvector_inline(eng, table_name)
            _ensure_hnsw_index(eng, table_name)
            _ensure_trigram_index(eng, table_name)
        _INITIALIZED_NAMESPACES.add(init_key)


# ── Store ────────────────────────────────────────────────────────────────


class VectorStore:
    """SQLAlchemy-backed chunk store with pgvector search on PostgreSQL.

    Thread-safety: one session per operation. On SQLite operations also take
    turns behind an internal lock; on PostgreSQL they run concurrently, each on
    its own pooled connection.
    """

    def __init__(
        self,
        db_path: str,
        dim: Optional[int] = None,
        namespace: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.db_path = db_path
        # No explicit width -> follow the configured embedder's actual dim, so
        # the store table width always matches the vectors written. Never a
        # hardcoded constant (all real callers pass an explicit dim via
        # get_store(); this resolves the safety-net default only).
        self.dim = dim if dim is not None else get_embedder(model_name=model_name).identity["dim"]
        self.namespace = namespace if namespace is not None else _rag_vector_namespace()
        if self.namespace == "" and _is_production():
            # namespace "" is the RETIRED legacy ``chunks`` table. Production
            # reads and writes the namespaced table only; a misconfigured
            # RAG_VECTOR_NAMESPACE (or a caller passing "") must fail loud
            # rather than silently serve an empty, contaminated table.
            raise RuntimeError(
                "RAG vector namespace '' (the retired legacy chunks table) is "
                "refused when ENV=production; set RAG_VECTOR_NAMESPACE to the "
                "active namespace"
            )
        self.model_name = model_name or os.getenv("RAG_EMBEDDING_MODEL") or "fake"
        self._database_url = _database_url(db_path)
        self._use_pgvector = self._database_url.startswith("postgresql")
        # SQLite (dev, tests) is one file: operations take turns. PostgreSQL
        # gives each operation its own pooled connection, so a process-wide
        # lock only serialised every user's retrieval -- live 2026-10-07, 15
        # users: one chunk count waited 12 s on it, on the event loop.
        self._lock = nullcontext() if self._use_pgvector else Lock()
        # Build the namespaced model class. The old ``chunks`` table is
        # retired in place: namespace="" maps to it, but prod defaults to
        # namespace="v2" so new writes never touch the contaminated table.
        self._rag_chunk_cls = make_rag_chunk_class(
            namespace=self.namespace,
            dim=dim,
            model_name=self.model_name,
        )
        self._table_name = rag_chunk_table_name(self.namespace)
        _ensure_schema(self._database_url, self._rag_chunk_cls)
        # Fail loud if the namespace already contains vectors from a
        # different embedder. Mixed-model contamination must be impossible.
        self._verify_embedding_identity()
        # FTS5 mirror (SQLite only). Idempotent.
        if not self._use_pgvector:
            self._ensure_fts5_sqlite()
        # Cached: isolated vector-store test DBs have no documents table.
        self._visibility_ready: Optional[bool] = None

    @property
    def fast_search(self) -> bool:
        """True when search uses pgvector ANN on PostgreSQL."""
        return self._use_pgvector

    _VISIBILITY_REPROBE_S = 30.0

    def _docs_visibility_ready(self, session: Optional[Session] = None) -> bool:
        """True when ``documents.retrieval_visible`` can gate this store.

        Isolated hybrid-test DBs create only the chunks table. A store the
        documents table cannot gate cannot compute the session scope either,
        so its project-wide reads return no candidate (``_session_hidden_ids``
        fails closed). ``True`` is cached per store; ``False`` is probed again
        after ``_VISIBILITY_REPROBE_S``, so one failed probe does not leave a
        store reading nothing for the life of the process.
        """
        if self._visibility_ready is True:
            return True
        now = time.monotonic()
        if (self._visibility_ready is False
                and now - getattr(self, "_visibility_probed_at", now)
                < self._VISIBILITY_REPROBE_S):
            return False
        self._visibility_probed_at = now
        close = False
        if session is None:
            session = self._session_factory()()
            close = True
        try:
            session.execute(text("SELECT retrieval_visible FROM documents LIMIT 0"))
            self._visibility_ready = True
        except Exception:
            try:
                session.rollback()
            except Exception:
                logger.debug(
                    "visibility probe rollback failed; treating documents "
                    "as ungated",
                    exc_info=True,
                )
            self._visibility_ready = False
        finally:
            if close:
                session.close()
        return self._visibility_ready

    def _hidden_doc_sql(self, chunk_alias: str = "c") -> str:
        """AND-clause that hides chunks no reader may see.

        Hidden: ``retrieval_visible = false``, or a document whose status is
        one of ``ingest_status.NO_CHUNK_STATUSES`` (failed, empty, tombstoned,
        quarantined ...). Chunks with no documents row stay visible (NOT
        EXISTS). Dialect-split because SQLite stores the boolean as 0/1.
        """
        if self._visibility_ready is not True:
            return ""
        return (
            f" AND NOT EXISTS (SELECT 1 FROM documents d "
            f"WHERE d.id = {chunk_alias}.doc_id "
            f"AND {hidden_document_predicate('d.', postgres=self._use_pgvector)})"
        )

    def _session_hidden_ids(
        self, session: Session, project_id: str,
    ) -> Optional[Set[str]]:
        """Chat-session files out of the turn's session scope.

        The document list's rule (``projects.document_in_scope``) applied to
        search: another session's files, or another user's, are never
        candidates. ``None`` when the set cannot be computed -- the documents
        table cannot gate this store, or reading it failed. The rule fails
        CLOSED: every caller then reads NO candidate of the project from this
        store, never every session's files."""
        if not self._docs_visibility_ready(session):
            return None
        from app.core import projects as _projects

        # One lookup per turn, not one per search leg: the turn's legs share
        # the scope, and a session file uploaded mid-turn is not this turn's.
        key = ("session_hidden", project_id, _projects.current_session_scope())
        try:
            return set(_memoised(key, lambda: frozenset(
                _projects.session_hidden_document_ids(project_id, db_session=session))))
        except SQLAlchemyError:
            session.rollback()
            # Not latched as "ungated": one failed read decides this call only.
            logger.warning("session scope unreadable; the project's search "
                           "returns no candidates this call", exc_info=True)
            return None

    def _session_visible(self, stmt: Any, session: Session, project_id: str) -> Any:
        """``stmt`` restricted to chunks in the turn's session scope.

        No row at all when the hidden set cannot be computed (fail closed)."""
        hidden = self._session_hidden_ids(session, project_id)
        if hidden is None:
            return stmt.where(false())
        if hidden:
            return stmt.where(self._rag_chunk_cls.doc_id.not_in(sorted(hidden)))
        return stmt

    def _session_hidden_sql(
        self, session: Session, project_id: str, column: str, params: Dict[str, Any],
    ) -> str:
        """AND-clause leaving out ``_session_hidden_ids``; binds into ``params``.

        ``AND 1 = 0`` (no row) when the hidden set cannot be computed."""
        hidden = self._session_hidden_ids(session, project_id)
        if hidden is None:
            return _NO_ROW_SQL
        ids = sorted(hidden)
        if not ids:
            return ""
        names = []
        for i, did in enumerate(ids):
            key = f"_session_hidden_{i}"
            params[key] = did
            names.append(f":{key}")
        return f" AND {column} NOT IN ({', '.join(names)})"

    def _hidden_doc_ids(
        self, session: Session, project_id: str,
    ) -> Optional[Set[str]]:
        """Ids no reader may see; ``None`` = every row is hidden (the session
        scope cannot be computed, ``_session_hidden_ids``)."""
        in_other_session = self._session_hidden_ids(session, project_id)
        if in_other_session is None:
            return None
        try:
            from app.core.ingest_status import NO_CHUNK_STATUSES

            rows = session.execute(
                select(Document.id).where(
                    Document.project_id == project_id,
                    or_(
                        Document.retrieval_visible.is_(False),
                        Document.ingest_status.in_(sorted(NO_CHUNK_STATUSES)),
                    ),
                )
            ).scalars().all()
        except SQLAlchemyError:
            session.rollback()
            return None
        return {str(r) for r in rows} | in_other_session

    def check_identity(self) -> Optional[Dict[str, Any]]:
        """Read-only: compare the stored corpus identity (model/dim/normalized)
        to this store's resolved embedder. Returns ``None`` when they match (or
        there is nothing to check — legacy namespace or empty table), else
        ``{"expected": ..., "stored": ...}`` (model names + dims only, no
        secrets). The dim is the load-bearing field: a 384-dim index queried by
        a 256-dim embedder returns garbage."""
        cls = self._rag_chunk_cls
        if not hasattr(cls, "embedding_model"):
            return None
        expected = {"model": self.model_name, "dim": self.dim, "normalized": True}
        with self._lock:
            with self._session_factory()() as session:
                row = session.execute(
                    select(
                        cls.embedding_model,
                        cls.embedding_dim,
                        cls.embedding_normalized,
                    ).limit(1)
                ).first()
        if row is None:
            return None
        stored = {
            "model": row.embedding_model,
            "dim": row.embedding_dim,
            "normalized": row.embedding_normalized,
        }
        return None if stored == expected else {"expected": expected, "stored": stored}

    def _verify_embedding_identity(self) -> None:
        """Fail loud if the namespace contains chunks from a different model.

        The structural guard that makes mixed-model contamination impossible,
        enforced on the write path. Skipped for the legacy namespace, whose
        original ``chunks`` table predates the identity columns.
        """
        mismatch = self.check_identity()
        if mismatch is not None:
            raise RuntimeError(
                f"Embedding identity mismatch in namespace {self.namespace!r}: "
                f"expected {mismatch['expected']}, found {mismatch['stored']}. "
                f"Mixed-model namespaces are not allowed."
            )

    def close(self) -> None:
        pass

    def _session_factory(self):
        return _session_factory_for_url(self._database_url)

    def _ensure_fk_parents(self, session: Session, project_id: str, doc_id: str) -> None:
        """Satisfy chunks FK on PostgreSQL when tests use bare project/doc ids."""
        if not self._use_pgvector:
            return
        if session.get(Project, project_id) is None:
            session.add(
                Project(
                    id=project_id,
                    name=project_id,
                    client=None,
                    status="active",
                    aconex_connected=False,
                    user_id="system",
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            session.flush()
        if session.get(Document, doc_id) is None:
            session.add(
                Document(
                    id=doc_id,
                    project_id=project_id,
                    original_name=doc_id,
                    stored_as=None,
                    file_path=None,
                    doc_type="document",
                    doc_role="other",
                    size=0,
                    uploaded_at=datetime.now(timezone.utc).isoformat(),
                    content_sha256=None,
                )
            )
            session.flush()

    # ── FTS5 mirror (SQLite path) ────────────────────────────────────────

    def _ensure_fts5_sqlite(self) -> None:
        """Create the FTS5 mirror + AI/AD/AU sync triggers on first init,
        then backfill any rows that pre-date the FTS5 table.

        Schema: external-content (``content='<table>'``,
        ``content_rowid='rowid'``). Text isn't duplicated; FTS5 reads it
        from the source table at query time via ``rowid``. Joining back to
        the source table by rowid restores the chunk tuple bm25_search
        returns.

        Idempotent: presence-check short-circuits subsequent calls.
        """
        table = self._table_name
        fts_table = f"{table}_fts"
        with self._lock:
            with self._session_factory()() as session:
                conn = session.connection()
                row = conn.exec_driver_sql(
                    "SELECT sql FROM sqlite_master "
                    f"WHERE type='table' AND name='{fts_table}'"
                ).fetchone()
                if row is not None:
                    # Already exists. Verify it's the external-content
                    # shape we expect; if a future change introduces a
                    # different shape, surface it so we don't silently
                    # bm25 against the wrong schema.
                    existing_sql = (row[0] or "").lower()
                    if f"content='{table}'" not in existing_sql and f'content="{table}"' not in existing_sql:
                        logger.warning(
                            "%s exists but is not external-content; "
                            "BM25 results may be unreliable. sql=%r",
                            fts_table, row[0],
                        )
                    return
                conn.exec_driver_sql(
                    f"CREATE VIRTUAL TABLE {fts_table} USING fts5("
                    f"text, content='{table}', content_rowid='rowid')"
                )
                conn.exec_driver_sql(
                    f"CREATE TRIGGER {table}_ai AFTER INSERT ON {table} BEGIN "
                    f"INSERT INTO {fts_table}(rowid, text) VALUES (new.rowid, new.text); "
                    "END"
                )
                conn.exec_driver_sql(
                    f"CREATE TRIGGER {table}_ad AFTER DELETE ON {table} BEGIN "
                    f"INSERT INTO {fts_table}({fts_table}, rowid, text) "
                    "VALUES('delete', old.rowid, old.text); "
                    "END"
                )
                conn.exec_driver_sql(
                    f"CREATE TRIGGER {table}_au AFTER UPDATE ON {table} BEGIN "
                    f"INSERT INTO {fts_table}({fts_table}, rowid, text) "
                    "VALUES('delete', old.rowid, old.text); "
                    f"INSERT INTO {fts_table}(rowid, text) VALUES (new.rowid, new.text); "
                    "END"
                )
                # One-time backfill — bulk insert via SELECT is fast
                # (a few seconds even for 140k rows on local SSD).
                cur = conn.exec_driver_sql(
                    f"INSERT INTO {fts_table}(rowid, text) "
                    f"SELECT rowid, text FROM {table}"
                )
                n = cur.rowcount if cur.rowcount is not None else 0
                session.commit()
                # One-line operator log on first init.
                print(f"vector_store: FTS5 backfilled {n} rows", flush=True)

    # ── Writes ───────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        project_id: str,
        doc_id: str,
        chunks: List[str],
        embeddings: np.ndarray,
        *,
        knowledge_layer: Optional[str] = None,
        authority: Optional[str] = None,
        pages: Optional[List[Optional[int]]] = None,
        ocr_flags: Optional[List[bool]] = None,
    ) -> int:
        """Replace all existing chunks for ``(project_id, doc_id)`` with
        the supplied set. Idempotent — calling twice with the same input
        is a no-op net change.

        ``pages`` (aligned with ``chunks``) is each chunk's 1-based source
        page; omitted, or of another length, every page is stored as NULL.

        ``embeddings`` must be a 2-D array of shape ``(len(chunks), dim)``.
        Returns the number of chunks written.

        FTS5 sync (SQLite path): the AI/AD/AU triggers on ``chunks``
        keep ``chunks_fts`` in lock-step automatically. PostgreSQL path:
        the ``text_search`` tsvector column is GENERATED ALWAYS — the
        DB recomputes it on write, no application-side work.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if len(chunks) == 0:
            self.delete_doc(project_id, doc_id)
            return 0

        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim {emb.shape[1]} != store dim {self.dim}"
            )

        if pages is None or len(pages) != len(chunks):
            pages = [None] * len(chunks)
        now = _now()
        with self._lock:
            with self._session_factory()() as session:
                self._ensure_fk_parents(session, project_id, doc_id)
                session.execute(
                    delete(self._rag_chunk_cls).where(
                        self._rag_chunk_cls.project_id == project_id,
                        self._rag_chunk_cls.doc_id == doc_id,
                    )
                )
                for i, (txt, vec, page) in enumerate(zip(chunks, emb, pages)):
                    if "\x00" in txt:
                        # PostgreSQL rejects NUL bytes in text columns with
                        # DataError, which would abort this whole document's
                        # insert. Extraction already strips NUL; this is the
                        # last line of defence for any other write path.
                        txt = txt.replace("\x00", "")
                    session.add(
                        self._rag_chunk_cls(
                            chunk_id=f"{project_id}:{doc_id}:{i}",
                            project_id=project_id,
                            doc_id=doc_id,
                            chunk_index=i,
                            text=txt,
                            embedding=vec,
                            embedding_model=self.model_name,
                            embedding_dim=self.dim,
                            embedding_normalized=True,
                            created_at=now,
                            knowledge_layer=knowledge_layer,
                            authority=authority,
                            page=int(page) if page else None,
                            ocr=bool(ocr_flags[i]) if ocr_flags and i < len(ocr_flags) else False,
                        )
                    )
                session.commit()
        return len(chunks)

    def rewrite_chunks(
        self,
        updates: List[Tuple[str, str, np.ndarray]],
    ) -> int:
        """In-place text + embedding rewrite for existing rows.

        ``updates`` is ``(chunk_id, new_text, embedding)``. The PostgreSQL
        ``text_search`` tsvector is GENERATED ALWAYS and recomputes on
        write. Returns how many rows were actually updated. Unknown ids
        are skipped.
        """
        if not updates:
            return 0
        written = 0
        with self._lock:
            with self._session_factory()() as session:
                for chunk_id, txt, vec in updates:
                    row = session.get(self._rag_chunk_cls, chunk_id)
                    if row is None:
                        continue
                    clean = (txt or "").replace("\x00", "")
                    emb = np.asarray(vec, dtype=np.float32)
                    if emb.ndim != 1 or emb.shape[0] != self.dim:
                        raise ValueError(
                            f"embedding shape {emb.shape} != ({self.dim},)"
                        )
                    row.text = clean
                    row.embedding = emb
                    written += 1
                session.commit()
        return written

    def delete_doc(self, project_id: str, doc_id: str) -> int:
        with self._lock:
            with self._session_factory()() as session:
                result = session.execute(
                    delete(self._rag_chunk_cls).where(
                        self._rag_chunk_cls.project_id == project_id,
                        self._rag_chunk_cls.doc_id == doc_id,
                    )
                )
                session.commit()
                return int(result.rowcount or 0)

    # ── Read ─────────────────────────────────────────────────────────────

    def count(
        self, project_id: Optional[str] = None, *, all_sessions: bool = False,
    ) -> int:
        """Chunk rows, of ``project_id`` when given.

        A project's count leaves out the chunks of chat-session files out of
        the current session scope (``_session_hidden_ids``), as search does;
        ``all_sessions`` counts every row."""
        with self._lock:
            with self._session_factory()() as session:
                stmt = select(func.count()).select_from(self._rag_chunk_cls)
                if project_id is not None:
                    stmt = stmt.where(self._rag_chunk_cls.project_id == project_id)
                    if not all_sessions:
                        stmt = self._session_visible(stmt, session, project_id)
                return int(session.scalar(stmt) or 0)

    def has_chunks(self, project_id: str) -> bool:
        """Whether the project has any chunk in the current session scope:
        one index probe, not a count. Another session's file's chunks do
        not make the project searchable (``_session_hidden_ids``)."""
        with self._lock:
            with self._session_factory()() as session:
                stmt = (
                    select(self._rag_chunk_cls.chunk_id)
                    .where(self._rag_chunk_cls.project_id == project_id)
                )
                stmt = self._session_visible(stmt, session, project_id)
                return session.execute(stmt.limit(1)).first() is not None

    def count_by_doc(
        self, project_id: str, *, all_sessions: bool = False,
    ) -> Dict[str, int]:
        """Return a ``{doc_id: chunk_count}`` map for ``project_id``.

        Uses the ``idx_chunks_doc`` index; one small GROUP BY regardless of
        corpus size. This is the source of truth for per-document chunk
        counts, including chunks written by ``/v1/admin/corpus/bulk-insert``
        (which bypasses the legacy ``doc_index`` JSON blob).

        Chat-session files out of the current session scope are left out
        (``_session_hidden_ids``, the rule every search leg applies), so a
        project-wide count never counts another session's file.
        ``all_sessions`` is for maintenance that must see every row
        (reconcile, ingest bookkeeping, the admin write check).
        """
        with self._lock:
            with self._session_factory()() as session:
                stmt = (
                    select(self._rag_chunk_cls.doc_id, func.count())
                    .where(self._rag_chunk_cls.project_id == project_id)
                    .group_by(self._rag_chunk_cls.doc_id)
                )
                counts = {row[0]: int(row[1]) for row in session.execute(stmt).all()}
                if all_sessions or not counts:
                    return counts
                hidden = self._session_hidden_ids(session, project_id)
                if hidden is None:
                    return {}
                return {d: n for d, n in counts.items() if str(d) not in hidden}

    def doc_chunk_texts(
        self, project_id: str, doc_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Return ``{doc_id: [chunk_text, ...]}`` (ordered by chunk_index) for
        the given docs in ``project_id``.

        Used by the corpus delete-docs admin endpoint to export a restore
        bundle *before* deleting — so the exact text leaving the RAG is
        captured. Bounded by the caller's doc-id list, not the corpus size.
        """
        if not doc_ids:
            return {}
        out: Dict[str, List[str]] = {d: [] for d in doc_ids}
        with self._lock:
            with self._session_factory()() as session:
                stmt = (
                    select(
                        self._rag_chunk_cls.doc_id,
                        self._rag_chunk_cls.chunk_index,
                        self._rag_chunk_cls.text,
                    )
                    .where(
                        self._rag_chunk_cls.project_id == project_id,
                        self._rag_chunk_cls.doc_id.in_(doc_ids),
                    )
                    .order_by(
                        self._rag_chunk_cls.doc_id,
                        self._rag_chunk_cls.chunk_index,
                    )
                )
                for did, _idx, txt in session.execute(stmt).all():
                    out.setdefault(did, []).append(txt or "")
        return out

    def chunks_for_docs(
        self,
        project_id: str,
        doc_ids: List[str],
        *,
        k_per_doc: int = 12,
        from_end: bool = False,
        all_rows: bool = False,
        offset: int = 0,
    ) -> List[Chunk]:
        """Return indexed chunks for specific documents, ordered by chunk_index.

        Letter/signatory (D1) and specification-title (C2) retrieval use
        this to pull a filename-matched document into the candidate pool
        without competing against a Volume 5 / demolition-spec flood in
        ``identifier_search``. Empty ``doc_ids`` or a store miss returns
        ``[]``; failures never raise into the answer path.

        ``from_end=True`` takes the last ``k_per_doc`` rows per file
        (Contract Data appendix after a long Conditions body). Default
        stays first-N so existing callers are unchanged.

        ``all_rows=True`` returns every chunk of those docs (the SQL
        already loads them). Daily delay damages: a filled 1.1.1 excl-VAT row
        can sit in the middle of a combined volume, past first-400 and
        before last-400, so prefix+tail still miss.

        ``offset`` skips that many rows per file (from the start, or
        from the end when ``from_end``). After #541: a
        store that ignores ``all_rows`` and caps ``k_per_doc`` at 400
        still misses chunk 500 of a 1200-row volume; windowed offsets
        walk the middle.
        """
        if not project_id or not doc_ids:
            return []
        unique = []
        seen: Set[str] = set()
        for did in doc_ids:
            if did and did not in seen:
                seen.add(did)
                unique.append(did)
        if not unique:
            return []
        per = max(1, int(k_per_doc or 12))
        off = max(0, int(offset or 0))
        cls = self._rag_chunk_cls
        try:
            with self._lock:
                with self._session_factory()() as session:
                    # A caller's ids come from a filename lookup; they are not
                    # a licence to read a retired document (live d8d9573: the
                    # superseded Contract Data came back as results #3-#5).
                    # Hidden-ness is per document, so dropping those ids up
                    # front is the same as dropping their rows afterwards.
                    hidden = self._hidden_doc_ids_among(session, project_id, unique)
                    visible = [d for d in unique if d not in hidden]
                    if not visible:
                        return []
                    where = (cls.project_id == project_id, cls.doc_id.in_(visible))
                    if all_rows:
                        stmt = (
                            select(*self._chunk_row_columns())
                            .where(*where)
                            .order_by(cls.doc_id, cls.chunk_index)
                        )
                    else:
                        # The per-document window is cut in SQL, so only the
                        # rows asked for are read and sent -- not the whole
                        # volume to be sliced here (a 1,200-chunk volume read
                        # in full for a 12-row window, several times a turn).
                        order = cls.chunk_index.desc() if from_end else cls.chunk_index
                        pos = func.row_number().over(
                            partition_by=cls.doc_id, order_by=order,
                        ).label("pos")
                        inner = (
                            select(*self._chunk_row_columns(), pos)
                            .where(*where)
                            .subquery()
                        )
                        stmt = (
                            select(*[inner.c[c.key] for c in self._chunk_row_columns()])
                            .where(inner.c.pos > off, inner.c.pos <= off + per)
                            .order_by(inner.c.doc_id, inner.c.chunk_index)
                        )
                    rows = session.execute(stmt).all()
        except Exception as exc:  # noqa: BLE001 — rescue must not break the turn
            logger.warning(
                "chunks_for_docs failed for project=%s docs=%s: %s",
                project_id, unique, exc,
            )
            return []

        chunks = [self._row_as_chunk(r) for r in rows]
        if all_rows or not from_end:
            return chunks
        # from_end keeps the caller's document order, each window ascending.
        by_doc: Dict[str, List[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        return [c for did in unique for c in by_doc.get(did, [])]

    def _chunk_row_columns(self) -> tuple:
        """Every chunk column a read returns -- never the embedding.

        The embedding is ~1.5 KB per row stored out of line; loading the
        whole row (``select(cls)``) read it from TOAST for every chunk of a
        document only for it to be dropped.
        """
        cls = self._rag_chunk_cls
        return (
            cls.chunk_id, cls.project_id, cls.doc_id, cls.chunk_index, cls.text,
            cls.knowledge_layer, cls.authority, cls.page, cls.ocr,
        )

    @staticmethod
    def _row_as_chunk(r) -> Chunk:
        return Chunk(
            chunk_id=r.chunk_id,
            project_id=r.project_id,
            doc_id=r.doc_id,
            chunk_index=int(r.chunk_index),
            text=r.text or "",
            score=0.0,
            knowledge_layer=getattr(r, "knowledge_layer", None),
            authority=getattr(r, "authority", None),
            page=getattr(r, "page", None),
            ocr=bool(getattr(r, "ocr", False)),
        )

    def _hidden_doc_ids_among(
        self, session: Session, project_id: str, doc_ids: List[str],
    ) -> Set[str]:
        """``_hidden_doc_ids`` restricted to ``doc_ids`` -- primary-key probes
        instead of reading every document row of the project. Every one of
        ``doc_ids`` when the session scope cannot be computed (fail closed)."""
        if not doc_ids:
            return set()
        wanted = {str(d) for d in doc_ids}
        in_other_session = self._session_hidden_ids(session, project_id)
        if in_other_session is None:
            return wanted
        try:
            from app.core.ingest_status import NO_CHUNK_STATUSES

            rows = session.execute(
                select(Document.id).where(
                    Document.id.in_(list(doc_ids)),
                    Document.project_id == project_id,
                    or_(
                        Document.retrieval_visible.is_(False),
                        Document.ingest_status.in_(sorted(NO_CHUNK_STATUSES)),
                    ),
                )
            ).scalars().all()
        except SQLAlchemyError:
            session.rollback()
            return wanted
        return {str(r) for r in rows} | (in_other_session & wanted)

    def chunks_containing_all(
        self,
        project_id: str,
        needles: List[str],
        *,
        k: int = 20,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Chunk]:
        """Chunks whose text contains every ``needle`` (case-insensitive).

        Used by C2 spec-identity rescue: the register line
        ``…-SPE-012650-1.0 Variation Procedure`` is a short TOC chunk that
        cosine never ranks, while ``Section 012650 — Variation and
        Adjustments`` in the same volume floods the semantic pool. Exact
        substring AND on the needles (``SPE-`` + the title phrase) is the
        discriminator a CSI section heading cannot fake.

        Needles are sanitised (no LIKE wildcards, min length 3). Empty
        ``needles`` or a store miss returns ``[]``; failures never raise.
        Optional ``doc_ids`` scopes the LIKE to those files so a
        first-N prefix cannot hide a later appendix row.
        """
        if not project_id or not needles:
            return []
        cleaned: List[str] = []
        seen: Set[str] = set()
        for raw in needles:
            tok = " ".join((raw or "").lower().split())
            if len(tok) < 3 or tok in seen:
                continue
            if any(ch in tok for ch in "%_"):
                continue
            seen.add(tok)
            cleaned.append(tok)
            if len(cleaned) >= 6:
                break
        if not cleaned:
            return []
        limit = max(1, int(k or 20))
        params: Dict[str, Any] = {"project_id": project_id, "k": limit}
        clauses: List[str] = []
        for i, tok in enumerate(cleaned):
            clauses.append(f"LOWER(text) LIKE :n{i}")
            params[f"n{i}"] = f"%{tok}%"
        extra_doc = ""
        if doc_ids:
            unique_docs: List[str] = []
            seen_d: Set[str] = set()
            for did in doc_ids:
                if did and did not in seen_d:
                    seen_d.add(did)
                    unique_docs.append(did)
            if unique_docs:
                ph = []
                for i, did in enumerate(unique_docs):
                    key = f"did{i}"
                    ph.append(f":{key}")
                    params[key] = did
                extra_doc = f" AND doc_id IN ({', '.join(ph)}) "
        try:
            with self._lock:
                with self._session_factory()() as session:
                    # Probe first: _hidden_doc_sql is inert until the store
                    # knows the documents table can gate it. Same idiom as
                    # identifier_search — which hid retired documents while
                    # this method, feeding the same rescues, did not.
                    vis = ""
                    if self._docs_visibility_ready(session):
                        vis = self._hidden_doc_sql(self._table_name)
                    # Outside the readiness guard: a store that cannot compute the
                    # session scope reads no row (fail closed), not every row.
                    vis += self._session_hidden_sql(
                        session, project_id, f"{self._table_name}.doc_id", params,
                    )
                    sql = text(
                        "SELECT chunk_id, project_id, doc_id, chunk_index, text, "
                        "knowledge_layer, authority, page, ocr "
                        f"FROM {self._table_name} "
                        "WHERE project_id = :project_id "
                        f"AND {' AND '.join(clauses)} "
                        f"{extra_doc}"
                        f"{vis} "
                        "LIMIT :k"
                    )
                    rows = session.execute(sql, params).all()
        except Exception as exc:  # noqa: BLE001 — rescue must not break the turn
            logger.warning(
                "chunks_containing_all failed for project=%s needles %s: %s",
                project_id, _terms_ref(cleaned), _error_ref(exc),
            )
            return []
        return [
            Chunk(
                chunk_id=r.chunk_id,
                project_id=r.project_id,
                doc_id=r.doc_id,
                chunk_index=int(r.chunk_index),
                text=r.text or "",
                score=0.0,
                knowledge_layer=getattr(r, "knowledge_layer", None),
                authority=getattr(r, "authority", None),
                page=getattr(r, "page", None),
                ocr=bool(getattr(r, "ocr", False)),
            )
            for r in rows
        ]

    def chunks_following(
        self,
        project_id: str,
        anchors: List[Tuple[str, int]],
        *,
        n: int = 1,
    ) -> List[Chunk]:
        """Return the next ``n`` same-doc chunks after each (doc_id, index).

        Used by C1 spec-precedence list rescue: an intro that ends
        ``as follows`` under Sub-Clause 1.5.1(d) is chunk N; the
        bullet list is chunk N+1. ``chunks_for_docs`` cannot do this —
        it returns the first ``k_per_doc`` rows of a file, so a late
        split (or a long TOC) never yields the neighbor.

        Empty ``anchors`` or a store miss returns ``[]``; failures
        never raise into the answer path.
        """
        if not project_id or not anchors:
            return []
        step = max(1, int(n or 1))
        wanted: Dict[str, Set[int]] = {}
        for raw_id, raw_idx in anchors:
            did = (raw_id or "").strip()
            if not did:
                continue
            try:
                base = int(raw_idx)
            except (TypeError, ValueError):
                continue
            wanted.setdefault(did, set()).update(
                range(base + 1, base + 1 + step)
            )
        if not wanted:
            return []
        clauses = [
            (
                (self._rag_chunk_cls.doc_id == did)
                & self._rag_chunk_cls.chunk_index.in_(sorted(indexes))
            )
            for did, indexes in wanted.items()
        ]
        try:
            with self._lock:
                with self._session_factory()() as session:
                    stmt = (
                        select(*self._chunk_row_columns())
                        .where(
                            self._rag_chunk_cls.project_id == project_id,
                            or_(*clauses),
                        )
                        .order_by(
                            self._rag_chunk_cls.doc_id,
                            self._rag_chunk_cls.chunk_index,
                        )
                    )
                    hidden = self._hidden_doc_ids_among(session, project_id, list(wanted))
                    rows = [
                        r for r in session.execute(stmt).all()
                        if r.doc_id not in hidden
                    ]
        except Exception as exc:  # noqa: BLE001 — rescue must not break the turn
            logger.warning(
                "chunks_following failed for project=%s anchors=%s: %s",
                project_id, list(wanted), _error_ref(exc),
            )
            return []
        return [self._row_as_chunk(r) for r in rows]

    def search(
        self,
        project_id: str,
        query_vec: np.ndarray,
        k: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Chunk]:
        """Top-``k`` chunks for ``project_id``.

        Two modes:

        - **Semantic-only** (legacy): when ``RAG_HYBRID_SEARCH`` is
          falsy, ``query_text`` is None/empty, or no BM25 hits come
          back. Ranks by cosine similarity. Byte-for-byte the
          pre-hybrid behavior — existing callers untouched.

        - **Hybrid** (default when env flag is truthy AND ``query_text``
          is provided): pulls 50 semantic + 50 BM25 candidates, fuses
          with Reciprocal Rank Fusion (k=60), returns top ``k`` by RRF
          score. Each returned chunk's ``.score`` is its fused rank on
          the cosine scale (``_rrf_combine``); ``.rrf_score``,
          ``.vector_score`` and ``.lexical_score`` are set for debug.

        Returns an empty list when the project has no indexed chunks.

        The legacy positional signature ``search(project_id, query_vec,
        k=...)`` is preserved — ``query_text`` is a keyword-only-style
        opt-in. ``retriever.py`` does NOT pass it today, so the chat
        path stays on semantic-only until the operator wires it.
        """
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(
                f"query_vec must be 1-D of length {self.dim}; got shape {q.shape}"
            )

        hybrid = (
            _hybrid_enabled()
            and query_text is not None
            and query_text.strip() != ""
        )

        if not hybrid:
            return self._semantic_search(project_id, q, k)

        sem_results = self._semantic_search(project_id, q, HYBRID_FETCH_PER_LEG)
        bm25_results = self.bm25_search(project_id, query_text, HYBRID_FETCH_PER_LEG)

        if not bm25_results:
            # Graceful degrade to semantic-only; respect caller's k.
            #
            # Graceful, but no longer SILENT. With bag-of-words semantics on
            # both backends, an empty BM25 leg means no query term appears
            # anywhere in the project — rare enough to be worth a line, and the
            # single clearest signal that retrieval ran at half strength. While
            # this was silent (and PostgreSQL ANDed its terms, so it fired
            # constantly) the platform served cosine-only answers that were
            # indistinguishable, to the reader, from fully-grounded ones.
            if self.has_chunks(project_id):
                state = "no_lexeme_match"
                logger.info(
                    "lexical leg empty: no query lexeme in project=%s; "
                    "semantic-only for this layer. query %s",
                    project_id, _query_ref(query_text),
                )
            else:
                state = "no_chunks"
                logger.warning(
                    "lexical leg empty: project=%s has no chunks at all; "
                    "semantic-only for this layer. query %s",
                    project_id, _query_ref(query_text),
                )
            _memo_put(("lexical_leg", project_id), state)
            return sem_results[:k]

        _memo_put(("lexical_leg", project_id), "hit")
        return _rrf_combine(sem_results, bm25_results, k, project_id=project_id)

    def identifier_search(
        self,
        project_id: str,
        identifiers: List[str],
        k: int = 20,
    ) -> List[Chunk]:
        """Lexical exact-match search for construction reference identifiers.

        Searches the raw chunk text for any of the supplied identifier
        strings using case-insensitive token matching.  Intervening
        punctuation/label words (e.g. "VO Ref: 99" vs "VO 99") are ignored.
        No reindexing is required: this works against the existing
        ``chunks.text`` column.

        Scoring: each returned chunk gets a score equal to the fraction of
        identifiers that appear in its text. This lets callers boost chunks
        that match multiple reference tokens (e.g. a VO number + a date)
        above chunks that only match one generic token.

        Returns an empty list when ``identifiers`` is empty or no chunk
        contains any of them.
        """
        if not identifiers or not project_id:
            return []

        # Tokenise each identifier so punctuation between tokens is ignored.
        # CESMM codes are collapsed first so ``D549.2`` and ``D 549.2``
        # produce the same token list (``d549``, ``2``).
        ident_tokens: List[List[str]] = []
        for ident in identifiers:
            tokens = _identifier_tokens(ident)
            if tokens:
                ident_tokens.append(tokens)
        if not ident_tokens:
            return []

        # Build a pre-filter: every token of an identifier must appear as a
        # substring in LOWER(text).  Tokens are alphanumeric so no LIKE
        # wildcard escaping is required.
        #
        # Letter+digits tokens (``d549``) also match the OCR-spaced form
        # ``d 549`` — otherwise ``D 549.2`` in the stored chunk fails
        # ``ILIKE '%d549%'`` (live Neon: 0 rows). Both forms are plain
        # ``LOWER(text) LIKE`` so the trigram index on ``lower(text)`` serves
        # them; ``LOWER(REPLACE(text, ' ', ''))`` had no index and read every
        # row of the project.
        ident_clauses: List[str] = []
        params: Dict[str, Any] = {"project_id": project_id, "k": k}
        param_idx = 0
        for tokens in ident_tokens:
            token_clauses: List[str] = []
            for tok in tokens:
                if _CESMM_COMPACT_TOKEN_RE.fullmatch(tok):
                    token_clauses.append(
                        f"(LOWER(text) LIKE :p{param_idx} "
                        f"OR LOWER(text) LIKE :p{param_idx}s)"
                    )
                    params[f"p{param_idx}s"] = f"%{tok[0]} {tok[1:]}%"
                else:
                    token_clauses.append(f"LOWER(text) LIKE :p{param_idx}")
                params[f"p{param_idx}"] = f"%{tok}%"
                param_idx += 1
            ident_clauses.append("(" + " AND ".join(token_clauses) + ")")

        like_clauses = " OR ".join(ident_clauses)
        vis = ""
        try:
            with self._lock:
                with self._session_factory()() as session:
                    if self._docs_visibility_ready(session):
                        vis = self._hidden_doc_sql(self._table_name)
                    # Outside the readiness guard: a store that cannot compute the
                    # session scope reads no row (fail closed), not every row.
                    vis += self._session_hidden_sql(
                        session, project_id, f"{self._table_name}.doc_id", params,
                    )
                    sql = text(
                        "SELECT chunk_id, project_id, doc_id, chunk_index, text, "
                        "knowledge_layer, authority, page, ocr "
                        f"FROM {self._table_name} "
                        "WHERE project_id = :project_id "
                        f"AND ({like_clauses}) "
                        f"{vis} "
                        "LIMIT :k"
                    )
                    rows = session.execute(sql, params).all()
        except OperationalError as e:
            logger.warning(
                "identifier_search failed for project=%s: %s; identifiers %s",
                project_id, _error_ref(e), _terms_ref(identifiers),
            )
            return []

        def _tokens(text: str) -> Set[str]:
            return set(_identifier_tokens(text or ""))

        text_tokens_by_row: Dict[Any, Set[str]] = {}
        out: List[Chunk] = []
        for r in rows:
            if r not in text_tokens_by_row:
                text_tokens_by_row[r] = _tokens(r.text)
            row_tokens = text_tokens_by_row[r]
            matches = sum(
                1 for tokens in ident_tokens if tokens and all(t in row_tokens for t in tokens)
            )
            score = matches / len(ident_tokens) if ident_tokens else 0.0
            out.append(
                Chunk(
                    chunk_id=r.chunk_id,
                    project_id=r.project_id,
                    doc_id=r.doc_id,
                    chunk_index=int(r.chunk_index),
                    text=r.text,
                    score=round(score, 4),
                    knowledge_layer=r.knowledge_layer,
                    authority=r.authority,
                    page=getattr(r, "page", None),
                    ocr=bool(getattr(r, "ocr", False)),
                )
            )
        # Discard SQL pre-filter false positives: a chunk that passed the
        # LIKE clauses but has no identifier token in its word-set gets
        # score 0.0 and must not be returned.
        out = [c for c in out if c.score > 0]
        # Higher match fraction first; preserve stable order on ties.
        out.sort(key=lambda c: -c.score)
        return out

    def _semantic_search(
        self,
        project_id: str,
        q: np.ndarray,
        k: int,
    ) -> List[Chunk]:
        """Semantic leg — dispatches on backend. Pulled out of search()
        so the hybrid path can reuse without re-validating the query."""
        if self._use_pgvector:
            return self._search_pgvector(project_id, q, k)
        return self._search_numpy(project_id, q, k)

    def _project_is_small(self, session: Session, project_id: str) -> bool:
        """About ``_EXACT_SCAN_MAX_ROWS`` chunks or fewer? Decided from the
        planner's row estimate for the project filter, memoised for the turn.

        Any real count had a cost the answer path cannot pay. Bounded by
        LIMIT, the planner may scan the table hoping to meet enough of the
        project's rows early (CI caught that sequential scan); forced onto
        the index, each row is a heap visit wherever pages are not yet marked
        all-visible (CI measured 43 MB of an 82 MB table). The estimate is
        planning only -- it reads statistics, not the table -- and choosing
        between an exact scan and the HNSW index needs only the size class."""
        key = ("project-rows", self._table_name, project_id)
        n = _memo_get(key)
        if n is _MISS:
            plan = session.execute(text(
                _PROJECT_ROWS_ESTIMATE_SQL.format(table=self._table_name)
            ), {"p": project_id}).scalar()
            if isinstance(plan, str):
                plan = json_lib.loads(plan)
            n = _memo_put(key, int(plan[0]["Plan"]["Plan Rows"]))
        return n <= _EXACT_SCAN_MAX_ROWS

    def _search_pgvector(
        self, project_id: str, query_vec: np.ndarray, k: int
    ) -> List[Chunk]:
        """Nearest chunks for ONE project, via the HNSW index.

        The project filter is the whole problem here. HNSW searches the entire
        table and Postgres applies ``WHERE project_id = ...`` to whatever the
        index hands back, so a project's rows can be filtered out to nothing
        even when it has thousands of chunks. Measured on live 2026-08-23,
        project 184921da (13,922 chunks of a 172,809-chunk table)::

            Index Scan using chunks_v2_embedding_hnsw  (actual rows=0)
              Filter: ((project_id)::text = '184921da'::text)
              Rows Removed by Filter: 39

        Zero rows returned. The same query with the index disabled returns 10.
        And it is not binary -- 'effective date of the agreement' returned 3 of
        10, so recall degrades silently on ordinary queries, which is worse
        than failing outright because nothing looks wrong.

        ``_enable_iterative_scan`` is what fixes it: pgvector keeps scanning
        until enough rows survive the filter instead of giving up after the
        first ef_search candidates.

        A SMALL project is searched exactly instead (``_EXACT_SCAN_MAX_ROWS``).
        For a project holding a few percent of the table, the iterative scan
        walks much of the graph before enough of its rows turn up -- measured
        12-29 MB per search, varying with the query -- while ranking the
        project's own rows reads about one heap page per four chunks (the
        embedding is inline, alembic 0023) and is exact.
        """
        q_list = query_vec.tolist()
        # EmbeddingVector is a TypeDecorator; cast to Vector for pgvector ops.
        vec_col = cast(self._rag_chunk_cls.embedding, Vector(self.dim))
        distance = vec_col.cosine_distance(q_list)
        score_expr = (1 - distance).label("score")
        stmt = (
            select(
                self._rag_chunk_cls.chunk_id,
                self._rag_chunk_cls.project_id,
                self._rag_chunk_cls.doc_id,
                self._rag_chunk_cls.chunk_index,
                self._rag_chunk_cls.text,
                self._rag_chunk_cls.knowledge_layer,
                self._rag_chunk_cls.authority,
                self._rag_chunk_cls.page,
                self._rag_chunk_cls.ocr,
                score_expr,
            )
            .where(self._rag_chunk_cls.project_id == project_id)
        )
        with self._lock:
            with self._session_factory()() as session:
                if self._docs_visibility_ready(session):
                    hidden = (
                        select(Document.id)
                        .where(
                            Document.id == self._rag_chunk_cls.doc_id,
                            Document.retrieval_visible.is_(False),
                        )
                        .exists()
                    )
                    stmt = stmt.where(~hidden)
                stmt = self._session_visible(stmt, session, project_id)
                # ``distance + 0`` is not an expression HNSW can serve, so a
                # small project is ranked exactly from its own rows.
                order = distance + 0 if self._project_is_small(session, project_id) else distance
                stmt = stmt.order_by(order).limit(k)
                key = ("vector", self._table_name, project_id, k,
                       _projects_scope_key(),
                       hashlib.sha1(np.asarray(query_vec, dtype=np.float32).tobytes()).hexdigest())
                rows = _memo_get(key)
                if rows is _MISS:
                    _enable_iterative_scan(session)
                    rows = _memo_put(key, session.execute(stmt).all())
        return [
            Chunk(
                chunk_id=row.chunk_id,
                project_id=row.project_id,
                doc_id=row.doc_id,
                chunk_index=int(row.chunk_index),
                text=row.text,
                score=float(row.score),
                knowledge_layer=row.knowledge_layer,
                authority=row.authority,
                page=getattr(row, "page", None),
                ocr=bool(getattr(row, "ocr", False)),
            )
            for row in rows
        ]

    def _search_numpy(
        self, project_id: str, query_vec: np.ndarray, k: int
    ) -> List[Chunk]:
        stmt = select(self._rag_chunk_cls).where(self._rag_chunk_cls.project_id == project_id)
        with self._lock:
            with self._session_factory()() as session:
                hidden = self._hidden_doc_ids(session, project_id)
                if hidden is None:
                    return []
                rows = [
                    r for r in session.scalars(stmt).all()
                    if r.doc_id not in hidden
                ]

        if not rows:
            return []

        embs = np.stack([np.asarray(r.embedding, dtype=np.float32) for r in rows])
        sims = embs @ query_vec
        order = np.argsort(-sims)[:k]
        out: List[Chunk] = []
        for idx in order:
            r = rows[int(idx)]
            out.append(
                Chunk(
                    chunk_id=r.chunk_id,
                    project_id=r.project_id,
                    doc_id=r.doc_id,
                    chunk_index=int(r.chunk_index),
                    text=r.text,
                    score=float(sims[int(idx)]),
                    knowledge_layer=getattr(r, "knowledge_layer", None),
                    authority=getattr(r, "authority", None),
                    page=getattr(r, "page", None),
                    ocr=bool(getattr(r, "ocr", False)),
                )
            )
        return out

    # ── BM25 leg ─────────────────────────────────────────────────────────

    def bm25_search(
        self,
        project_id: str,
        query: str,
        k: int = 50,
        doc_ids: Optional[Iterable[str]] = None,
    ) -> List[Chunk]:
        """Top-``k`` chunks for ``project_id`` ranked by BM25.

        Dispatches on backend:

        - PostgreSQL: an idf-weighted, length-normed score (see
          ``_LEXICAL_SCORE_SCALE``) over ``text_search`` (the tsvector
          column added by Alembic 0003) with a ``@@`` tsquery predicate. GIN index ``chunks_fts_gin`` keeps it fast.
        - SQLite: FTS5 ``chunks_fts`` external-content virtual table,
          joined back to ``chunks`` by rowid.

        Empty query (after sanitization) → empty list. Malformed
        backend errors → empty list (logged); caller treats as
        "semantic only" and continues.

        Returned chunks carry ``score`` set to the BM25 rank value for
        debug; the RRF fuser uses positional rank, not ``.score``.

        ``doc_ids`` (optional) restricts the ranking to those documents, so a
        caller can rank one document set without the rest of the project
        crowding it out of the top-``k``. ``None`` means no restriction; an
        empty collection matches nothing and returns ``[]`` without a query.
        """
        if not query or not query.strip():
            return []
        scope_sql, scope_params = _doc_scope(doc_ids)
        if scope_sql is None:
            return []

        if self._use_pgvector:
            return self._bm25_postgres(project_id, query, k, scope_sql, scope_params)
        return self._bm25_sqlite(project_id, query, k, scope_sql, scope_params)

    def _bm25_postgres(
        self, project_id: str, query: str, k: int,
        scope_sql: str = "", scope_params: Optional[Dict[str, str]] = None,
    ) -> List[Chunk]:
        """Rarity-weighted score + GIN over a BAG-OF-WORDS tsquery.

        Bag-of-words (OR), not plainto_tsquery's AND — see
        ``_sanitize_websearch_query`` for the production failure that came from
        requiring every query term to appear in a chunk. The score still ranks
        a chunk matching more of the terms above one matching fewer, so relaxing
        the predicate widens recall without flattening precision.
        """
        safe_query = _sanitize_websearch_query(query)
        if not safe_query:
            return []
        table = self._table_name
        try:
            with self._lock:
                with self._session_factory()() as session:
                    vis = ""
                    scope_params = dict(scope_params or {})
                    if self._docs_visibility_ready(session):
                        vis = self._hidden_doc_sql("c")
                    # Outside the readiness guard: a store that cannot compute the
                    # session scope reads no row (fail closed), not every row.
                    vis += self._session_hidden_sql(
                        session, project_id, "c.doc_id", scope_params,
                    )
                    rows = self._bm25_postgres_rows(
                        session, project_id, safe_query, int(k), vis + scope_sql,
                        scope_params,
                    )
        except SQLAlchemyError as e:
            # Deliberately broader than OperationalError. A missing
            # ``text_search`` column raises ProgrammingError (UndefinedColumn),
            # which this used to let escape — turning a degraded lexical leg
            # into a failed request. Either way the leg is unavailable, so
            # degrade to semantic-only; log at ERROR because a silently
            # half-working retriever is what let this class of bug live in
            # production while every test stayed green.
            logger.error(
                "bm25_search (postgres) unavailable on %s: %s; query %s — "
                "retrieval is running SEMANTIC-ONLY for this request",
                table, _error_ref(e), _query_ref(query),
            )
            return []
        return [
            Chunk(
                chunk_id=r.chunk_id,
                project_id=r.project_id,
                doc_id=r.doc_id,
                chunk_index=int(r.chunk_index),
                text=r.text,
                score=float(r.rank),
                knowledge_layer=r.knowledge_layer,
                authority=r.authority,
                page=getattr(r, "page", None),
                ocr=bool(getattr(r, "ocr", False)),
            )
            for r in rows
        ]

    def _bm25_select(self, where: str, vis: str) -> str:
        """The BM25 statement over ``where``; scores by the FULL query's lexemes.

        The score is the one described above ``_LEXICAL_SCORE_SCALE``:
        ``:lx`` / ``:lw`` are the query's lexemes and their idf weights,
        ``:scale`` the per-query constant. The chunk's lexemes are read once
        per row (``OFFSET 0`` keeps the planner from inlining the read into
        both places it is used). ``chunk_id`` breaks score ties so the top-k
        is one fixed set rather than whichever tied rows a scan happened to
        reach first.
        """
        return (
            "SELECT c.chunk_id, c.project_id, c.doc_id, c.chunk_index, "
            "c.text, c.knowledge_layer, c.authority, c.page, c.ocr, "
            "(SELECT coalesce(sum(u.w), 0) "
            "FROM unnest(CAST(:lx AS text[]), CAST(:lw AS float8[])) AS u(lx, w) "
            "WHERE u.lx = ANY(a.lexemes)) * :scale "
            "/ (1 + ln(greatest(cardinality(a.lexemes), 1))) AS rank "
            f"FROM {self._table_name} c "
            "CROSS JOIN LATERAL (SELECT tsvector_to_array(c.text_search) AS lexemes "
            "OFFSET 0) a, "
            "websearch_to_tsquery('english', :q) AS q "
            f"WHERE {where} AND c.project_id = :project_id {vis} "
            "ORDER BY rank DESC, c.chunk_id LIMIT :k"
        )

    def _bm25_postgres_rows(
        self, session: Session, project_id: str, safe_query: str, k: int, vis: str,
        scope_params: Optional[Dict[str, str]] = None,
    ) -> list:
        scope_params = dict(scope_params or {})
        tsq = session.execute(
            text("SELECT websearch_to_tsquery('english', :q)::text"), {"q": safe_query},
        ).scalar()
        scored = tsquery_lexemes(tsq)
        if not scored:
            # Every word is a stopword: nothing can match. Asking anyway reads
            # every tsvector in the project to learn that.
            return []
        weights = lexeme_weights(scored, self._lexeme_frequencies(session))
        params = {
            "q": safe_query, "project_id": project_id, "k": k,
            "lx": scored, "lw": [weights[x] for x in scored],
            "scale": lexical_score_scale(weights),
            **scope_params,
        }
        # Keyed on the normalised tsquery and the weights it was scored with:
        # query wordings that reduce to the same tsquery rank the same rows.
        # ``vis`` carries the document scope's placeholders, not its values,
        # so the bound document ids are part of the key too.
        key = (
            "bm25", self._table_name, project_id, tsq, k, vis,
            tuple(params["lw"]), tuple(sorted(scope_params.items())),
        )
        hit = _memo_get(key)
        if hit is not _MISS:
            return hit
        lexemes = tsquery_or_lexemes(tsq)
        if lexemes and len(lexemes) > 1 and k > 0:
            rows = self._bm25_bounded(session, params, lexemes, weights, vis)
            if rows is not None:
                return _memo_put(key, rows)
        return _memo_put(key, session.execute(
            text(self._bm25_select("c.text_search @@ q", vis)), params,
        ).all())

    def _bm25_bounded(
        self, session: Session, params: Dict[str, Any], lexemes: List[str],
        weights: Dict[str, float], vis: str,
    ) -> Optional[list]:
        """Exact top-k scored over GIN-admitted candidates only, or ``None``.

        See the module notes above ``_LEXICAL_SCORE_SCALE``. A round at
        ``(rare, t)`` scores every chunk holding one of the ``rare`` rarest
        lexemes or ``t`` of the others (and possibly more); it is the answer
        when ``k`` rows came back and the k-th score reaches
        ``bm25_excluded_bound``. Otherwise ``t`` drops by one and ``rare``
        grows to the fewest rare lexemes whose bound the best k-th score seen
        so far already clears (at least one more). ``None`` sends the caller
        to the full scan.
        """
        n = len(lexemes)
        k = params["k"]
        sql = text(self._bm25_select("c.text_search @@ CAST(:cand AS tsquery)", vis))
        t = max(2, min(_BM25_START_THRESHOLD, (n + 1) // 2))
        rare = 1
        best_kth = 0.0
        rounds = short = 0
        while t >= 2 and rare < n and rounds < _BM25_MAX_ROUNDS:
            cand = bm25_candidate_tsquery(lexemes, t, weights, rare)
            if cand is None:
                t -= 1
                continue
            rounds += 1
            rows = session.execute(sql, {**params, "cand": cand}).all()
            if len(rows) >= k:
                kth = float(rows[k - 1].rank)
                if bm25_rank_bound_holds(kth, bm25_excluded_bound(lexemes, weights, t, rare)):
                    return rows
                best_kth = max(best_kth, kth)
            else:
                short += 1
                if short >= _BM25_MAX_SHORT_ROUNDS:
                    return None
            t -= 1
            rare += 1
            while best_kth > 0 and rare < n - 1 and not bm25_rank_bound_holds(
                best_kth, bm25_excluded_bound(lexemes, weights, t, rare),
            ):
                rare += 1
        return None

    _LEXEME_FREQ_TTL_S = 3600.0

    def _lexeme_frequencies(self, session: Session) -> Dict[str, float]:
        """Planner statistics' lexeme frequencies for ``text_search``.

        The lexical score's idf weights and the bounded leg's rarest-first
        order come from these. Any failure yields ``{}``: every lexeme then
        weighs the same, and the bounded leg stays exact.
        Cached per store for an hour -- statistics move with ANALYZE, not
        per request.
        """
        import time as _time

        cached = getattr(self, "_lexeme_freq_cache", None)
        now = _time.monotonic()
        if cached is not None and now - cached[0] < self._LEXEME_FREQ_TTL_S:
            return cached[1]
        freq: Dict[str, float] = {}
        try:
            row = session.execute(
                text(
                    "SELECT most_common_elems::text::text[], most_common_elem_freqs "
                    "FROM pg_stats WHERE schemaname = current_schema() "
                    "AND tablename = :t AND attname = 'text_search'"
                ),
                {"t": self._table_name},
            ).first()
            if row and row[0] and row[1]:
                freq = {str(lx): float(f) for lx, f in zip(row[0], row[1])}
        except SQLAlchemyError:
            session.rollback()
            logger.debug("lexeme statistics unavailable for %s", self._table_name, exc_info=True)
        self._lexeme_freq_cache = (now, freq)
        return freq

    _TERM_COUNT_TTL_S = 600.0
    # One entry per (project, session scope): bounded, oldest written first out.
    _TERM_COUNT_MAX_ENTRIES = 256

    def term_chunk_counts(
        self, project_id: str, terms: Iterable[str],
    ) -> Dict[str, int]:
        """How many of ``project_id``'s visible chunks hold each term.

        Read from the lexical index the BM25 leg already uses (the
        ``text_search`` tsvector on PostgreSQL, ``chunks_fts`` on SQLite),
        so "the index contains this word" means what a lexical re-query
        could find. A term the index cannot match (a stopword, punctuation
        only) counts 0. Chat-session files out of the turn's session scope
        are not counted, as the other search legs do not read them. Counts
        are cached per project AND session scope for ``_TERM_COUNT_TTL_S``,
        so one session's counts are never served to another. Each write drops
        the expired entries and keeps at most ``_TERM_COUNT_MAX_ENTRIES``
        (the oldest written go first). Backend errors propagate: the caller
        decides what an unreadable index means.
        """
        import time as _time

        wanted = [t for t in dict.fromkeys(str(t) for t in terms) if t]
        now = _time.monotonic()
        cache = getattr(self, "_term_count_cache", None)
        if cache is None:
            cache = self._term_count_cache = {}
        cache_key = (project_id, _projects_scope_key())
        stamp, counts = cache.get(cache_key, (now, {}))
        if now - stamp >= self._TERM_COUNT_TTL_S:
            stamp, counts = now, {}
        todo = [t for t in wanted if t not in counts]
        if todo:
            with self._lock:
                with self._session_factory()() as session:
                    vis = ""
                    params: Dict[str, Any] = {}
                    if self._docs_visibility_ready(session):
                        vis = self._hidden_doc_sql("c")
                    session_vis = self._session_hidden_sql(
                        session, project_id, "c.doc_id", params,
                    )
                    if session_vis == _NO_ROW_SQL:
                        # The scope cannot be computed: no chunk counts, and
                        # nothing remembered -- the next readable call counts.
                        return {t: 0 for t in wanted}
                    vis += session_vis
                    counts = {**counts, **{
                        term: self._term_chunk_count(
                            session, project_id, term, vis, params,
                        )
                        for term in todo
                    }}
            self._term_count_cache_put(cache, cache_key, stamp, counts, now)
        return {t: counts[t] for t in wanted}

    def _term_count_cache_put(
        self, cache: Dict[Any, Any], key: Any, stamp: float,
        counts: Dict[str, int], now: float,
    ) -> None:
        """Write one entry; evict expired ones, then the oldest over the cap.

        Under a lock: on PostgreSQL the store lock is a no-op and turns write
        concurrently."""
        with _TERM_COUNT_CACHE_LOCK:
            for k in [k for k, (t, _c) in list(cache.items())
                      if now - t >= self._TERM_COUNT_TTL_S]:
                cache.pop(k, None)
            cache.pop(key, None)  # re-inserted last: insertion order = write order
            cache[key] = (stamp, counts)
            while len(cache) > self._TERM_COUNT_MAX_ENTRIES:
                cache.pop(next(iter(cache)), None)

    def _term_chunk_count(
        self, session: Session, project_id: str, term: str, vis: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> int:
        table = self._table_name
        params = dict(params or {})
        if self._use_pgvector:
            sql = (
                f"SELECT count(*) FROM {table} c "
                "WHERE c.project_id = :project_id "
                "AND c.text_search @@ plainto_tsquery('english', :q)" + vis
            )
            return int(session.execute(
                text(sql), {**params, "project_id": project_id, "q": term},
            ).scalar() or 0)
        tokens = _FTS5_SAFE_RE.sub(" ", term).split()
        if not tokens:
            return 0
        fts_table = f"{table}_fts"
        sql = (
            f"SELECT count(*) FROM {fts_table} "
            f"JOIN {table} c ON c.rowid = {fts_table}.rowid "
            f"WHERE {fts_table} MATCH :q AND c.project_id = :project_id" + vis
        )
        phrase = '"' + " ".join(tokens) + '"'
        return int(session.execute(
            text(sql), {**params, "project_id": project_id, "q": phrase},
        ).scalar() or 0)

    def _bm25_sqlite(
        self, project_id: str, query: str, k: int,
        scope_sql: str = "", scope_params: Optional[Dict[str, str]] = None,
    ) -> List[Chunk]:
        """FTS5 MATCH joined to chunks by rowid (external-content shape).
        FTS5's ``rank`` is a negated BM25 — lower = better — so ASC."""
        safe_query = _sanitize_fts5_query(query)
        if not safe_query:
            return []
        table = self._table_name
        fts_table = f"{table}_fts"
        vis = ""
        try:
            with self._lock:
                with self._session_factory()() as session:
                    scope_params = dict(scope_params or {})
                    if self._docs_visibility_ready(session):
                        vis = self._hidden_doc_sql("c")
                    # Outside the readiness guard: a store that cannot compute the
                    # session scope reads no row (fail closed), not every row.
                    vis += self._session_hidden_sql(
                        session, project_id, "c.doc_id", scope_params,
                    )
                    sql = text(
                        f"""
                        SELECT c.chunk_id, c.project_id, c.doc_id, c.chunk_index,
                               c.text, c.knowledge_layer, c.authority, c.page, c.ocr,
                               {fts_table}.rank AS bm25_rank
                        FROM {fts_table}
                        JOIN {table} c ON c.rowid = {fts_table}.rowid
                        WHERE {fts_table} MATCH :q
                          AND c.project_id = :project_id
                          {vis}{scope_sql}
                        ORDER BY {fts_table}.rank
                        LIMIT :k
                        """
                    )
                    rows = session.execute(
                        sql,
                        {
                            "q": safe_query, "project_id": project_id, "k": k,
                            **(scope_params or {}),
                        },
                    ).all()
        except OperationalError as e:
            # FTS5 raises on malformed MATCH input. Treat as "no matches."
            logger.warning(
                "bm25_search (sqlite) FTS5 MATCH failed: %s; query %s",
                _error_ref(e),
                _query_ref(safe_query),
            )
            return []
        return [
            Chunk(
                chunk_id=r.chunk_id,
                project_id=r.project_id,
                doc_id=r.doc_id,
                chunk_index=int(r.chunk_index),
                text=r.text,
                score=float(r.bm25_rank),
                knowledge_layer=r.knowledge_layer,
                authority=r.authority,
                page=getattr(r, "page", None),
                ocr=bool(getattr(r, "ocr", False)),
            )
            for r in rows
        ]

    def bm25_search_photos(
        self,
        query: str,
        k: int = 5,
        project_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Search the ``photo_chunks`` table by caption keywords.

        Returns photo chunks ordered by caption relevance. When ``project_id``
        is supplied, global (NULL-project) photos are included plus photos
        owned by that project; other projects' photos are excluded. If the
        table does not exist (older deployments) an empty list is returned.
        """
        if not query or not query.strip():
            return []

        # Tokenise into SQL LIKE clauses (OR). Keeps the method usable on
        # a plain SQLite table without requiring an FTS5 mirror.
        tokens = [t for t in _FTS5_SAFE_RE.sub(" ", query).strip().split() if t]
        if not tokens:
            return []

        project_filter = ""
        params: Dict[str, Any] = {"k": k}
        if project_id is not None:
            project_filter = " AND (project_id IS NULL OR project_id = :project_id)"
            params["project_id"] = project_id

        like_clauses = " OR ".join(
            f"caption LIKE :like_{i}" for i in range(len(tokens))
        )
        for i, tok in enumerate(tokens):
            params[f"like_{i}"] = f"%{tok}%"

        sql = text(
            f"SELECT chunk_id, project_id, sha256, caption, photo_metadata "
            f"FROM photo_chunks WHERE ({like_clauses}){project_filter} "
            f"ORDER BY created_at DESC LIMIT :k"
        )

        try:
            with self._lock:
                with self._session_factory()() as session:
                    rows = session.execute(sql, params).all()
        except OperationalError:
            # photo_chunks table may not exist on older deployments.
            return []

        out: List[Chunk] = []
        for r in rows:
            meta = None
            photo_url = None
            if r.photo_metadata:
                try:
                    meta = json_lib.loads(r.photo_metadata)
                    photo_url = meta.get("source_url") if isinstance(meta, dict) else None
                except Exception:
                    logger.warning(
                        "swallowed %s in bm25_search_photos() — continuing",
                        "Exception", exc_info=True,
                    )
            out.append(
                Chunk(
                    chunk_id=r.chunk_id,
                    project_id=r.project_id or "",
                    doc_id=r.chunk_id,
                    chunk_index=0,
                    text=r.caption,
                    score=1.0,
                    kind="photo",
                    sha256=r.sha256,
                    photo_url=photo_url,
                    photo_metadata=meta,
                )
            )
        return out

# ── Helpers ───────────────────────────────────────────────────────────────


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rrf_combine(
    semantic: List[Chunk],
    bm25: List[Chunk],
    top_k: int,
    *,
    project_id: Optional[str] = None,
) -> List[Chunk]:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    Combines two ranked lists by summing 1/(RRF_K + rank) contributions.
    Chunks present in only one list still get a score from that list
    (the other term is 0). When a chunk appears in both lists the
    semantic instance is kept. Position in the returned list = RRF score
    desc (stable on ties).

    ``.score`` carries the fused rank on the cosine scale: the chunk at
    fused position i gets the i-th highest cosine of the vector leg
    (past the vector leg's length, the last cosine repeats). Every later
    step ranks, thresholds and adds its bonuses on ``.score``, so a chunk
    the lexical leg ranks first must not keep its raw lexical rank there:
    that rank is on another scale (``ts_rank`` < 0.1) and would sort it
    under every vector hit whatever its fused position. The remap keeps
    the score distribution of the vector leg, so the pool's top score is
    still the top cosine, and when both legs agree it is the identity.
    With an empty vector leg there is no cosine scale and ``.score`` is
    left as the leg gave it. The raw leg scores are kept on
    ``.vector_score`` / ``.lexical_score`` and the fused position on
    ``.fused_rank``, debug-only like ``.rrf_score``.

    Before the ``top_k`` cut, ``guarantee_leg_share`` places each leg's best
    single-leg candidates at their own leg rank (``RAG_FUSION_LEG_SHARE``
    per leg); ``.fused_rank`` is the position after that step.

    With ``project_id``, that full order (before the ``top_k`` cut) is
    summarised by leg into the turn memo (``record_fusion``) for the
    retrieval trace. Counts and ranks only; the trace changes no ranking.
    With the trace off (``RAG_RETRIEVAL_TRACE=0``) the summary is not
    computed at all: nothing would read it.
    """
    sem_rank = {c.chunk_id: r for r, c in enumerate(semantic, 1)}
    bm_rank = {c.chunk_id: r for r, c in enumerate(bm25, 1)}
    sem_score = {c.chunk_id: c.score for c in semantic}
    bm_score = {c.chunk_id: c.score for c in bm25}
    # Semantic instances take precedence on tie.
    by_id: dict[str, Chunk] = {}
    for c in bm25:
        by_id[c.chunk_id] = c
    for c in semantic:
        by_id[c.chunk_id] = c
    scored: List[Tuple[float, Chunk]] = []
    for chunk_id, c in by_id.items():
        s = sem_rank.get(chunk_id)
        b = bm_rank.get(chunk_id)
        rrf = (
            (1.0 / (RRF_K + s) if s is not None else 0.0)
            + (1.0 / (RRF_K + b) if b is not None else 0.0)
        )
        try:
            c.rrf_score = rrf
            c.vector_score = sem_score.get(chunk_id)
            c.lexical_score = bm_score.get(chunk_id)
        except Exception:
            logger.warning(
                "swallowed %s in _rrf_combine() — continuing",
                "Exception", exc_info=True,
            )
        scored.append((rrf, c))
    scored.sort(key=lambda x: -x[0])
    order = guarantee_leg_share(
        [c for _, c in scored], sem_rank, bm_rank, fusion_leg_share(),
    )
    if project_id is not None and _retrieval_trace_on():
        from app.core.rag.fusion_trace import leg_summary

        record_fusion(project_id, leg_summary(order, sem_rank, bm_rank, top_k))
    fused = order[:top_k]
    cosines = sorted(
        (float(v) for v in sem_score.values() if v is not None), reverse=True,
    )
    for i, c in enumerate(fused):
        c.fused_rank = i + 1
        if cosines:
            c.score = cosines[min(i, len(cosines) - 1)]
    return fused


# ── Boot-time serving gate (S1) ──────────────────────────────────────────────
# A dim/model mismatch between the live index and the resolved embedder makes
# retrieval silently wrong, so the service refuses to serve until it is fixed
# (the readiness gate reads this). Set once at startup, names/dims only.
_IDENTITY_SERVING_BLOCKED = False


def identity_serving_blocked() -> bool:
    return _IDENTITY_SERVING_BLOCKED


def run_boot_identity_check() -> Optional[Dict[str, Any]]:
    """At startup, compare the live corpus identity to the resolved embedder.
    Logs an ERROR (model/dim only — no secrets) and blocks serving on a
    mismatch. Best-effort: an empty/new store or an unreadable probe is NOT a
    mismatch. Returns the mismatch dict or None."""
    global _IDENTITY_SERVING_BLOCKED
    try:
        mismatch = get_store().check_identity()
    except Exception:  # noqa: BLE001 — a boot probe never crashes startup
        logger.info("boot identity check: store probe skipped", exc_info=True)
        return None
    if mismatch is not None:
        _IDENTITY_SERVING_BLOCKED = True
        logger.error(
            "EMBEDDER IDENTITY MISMATCH — refusing to serve: index is "
            "model=%s dim=%s but the resolved embedder is model=%s dim=%s. "
            "Set RAG_EMBEDDING_MODEL to the index's model (or re-embed).",
            mismatch["stored"]["model"], mismatch["stored"]["dim"],
            mismatch["expected"]["model"], mismatch["expected"]["dim"],
        )
    return mismatch
