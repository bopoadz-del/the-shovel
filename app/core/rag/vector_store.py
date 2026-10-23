"""Per-project chunk store backed by SQLAlchemy.

Storage shape: one row per chunk in the ``chunks`` table. On PostgreSQL
embeddings use ``pgvector`` ``vector(256)`` with cosine-distance ANN search
(``ORDER BY embedding <=> :q`` scoped by ``project_id``). On SQLite the
same table stores float32 BLOBs and search falls back to numpy cosine
similarity over the project's rows — slower but works everywhere.

Hybrid retrieval: when ``RAG_HYBRID_SEARCH`` is truthy and the caller
supplies ``query_text``, the store also runs a BM25 leg and fuses with
Reciprocal Rank Fusion. On PostgreSQL the BM25 leg is ``ts_rank`` over
the ``text_search`` tsvector column + GIN index (added by Alembic 0003).
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
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, delete, func, or_, select, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

import json as json_lib

from app.core.db import _engine_for_url, _session_factory_for_url, get_database_url
from app.core.models import EMBEDDING_DIM, Document, Project
from app.core.models import make_rag_chunk_class, rag_chunk_table_name
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
    return _CESMM_OCR_SPACE_RE.sub(r"\1\2", text)


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
    cleaned = _FTS5_SAFE_RE.sub(" ", query)
    tokens = cleaned.split()
    if not tokens:
        return ""
    return " OR ".join(tokens)


# Words websearch_to_tsquery reads as OPERATORS. Passing them through as if
# they were search terms changes the meaning of the query.
_WEBSEARCH_OPERATORS = frozenset({"or", "and", "not"})


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
    cleaned = _FTS5_SAFE_RE.sub(" ", query)
    tokens = [t for t in cleaned.split() if t.lower() not in _WEBSEARCH_OPERATORS]
    if not tokens:
        return ""
    return " or ".join(tokens)


# ── Bounded BM25 (exact top-k without ranking every matching row) ─────────
#
# The bag-of-words tsquery above matches most of a large project: measured on
# live, "commencement date stated agreement pump station works" matched 57,069
# of the master corpus's 118,665 chunks. ``ORDER BY ts_rank LIMIT k`` then has
# to read every one of those tsvectors (most are TOASTed), twice -- once for
# ``@@`` and once for ``ts_rank`` -- which was ~155k buffers (1.2 GB) per call
# and ~80% of all the database reading one question did.
#
# The top-k does not need every match ranked. ``text_search`` is
# ``to_tsvector('english', text)``: every position carries weight D (0.1), and
# for an OR query ``ts_rank`` sums, per matched query lexeme, a term that is
# below 0.1 however often the lexeme occurs (0.1 * sum(1/j^2) / (pi^2/6)), then
# divides by the number of distinct query lexemes n. So a chunk matching m of
# the n lexemes ranks strictly below m * 0.1 / n. Once k chunks with rank >= tau
# are in hand, a chunk can only enter the top-k if it matches more than
# tau * n / 0.1 lexemes -- and "matches at least t of n lexemes" is something
# the GIN index can answer without touching the heap for the rest.
#
# ``_bm25_bounded`` ranks exactly the chunks the GIN index admits for a
# threshold t, and accepts the result only when the k-th rank it found proves
# no excluded chunk could beat it. Otherwise it widens t, and in the end falls
# back to the full scan. The result is identical to ranking every match.

# Upper bound of one matched lexeme's contribution to ``ts_rank`` for an
# unweighted tsvector (weight D = 0.1), with a margin for float4 rounding.
_TS_RANK_TERM_CAP = 0.1 * (1 + 1e-4)
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


def tsquery_or_lexemes(tsquery_text: Optional[str]) -> Optional[List[str]]:
    """Distinct lexemes of a tsquery that is a plain OR of lexemes.

    ``"'commenc' | 'date'"`` -> ``['commenc', 'date']``; an empty query ->
    ``[]``; anything else (a phrase from a compound token, ``&``, ``!``, a
    weight or prefix marker) -> ``None``, which the caller answers with the
    full scan. Distinct, because ``ts_rank`` divides by the number of
    distinct operands.
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


def _tsquery_literal(lexeme: str) -> str:
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


def bm25_candidate_tsquery(
    lexemes: List[str], t: int, frequency: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """A tsquery matched by every chunk holding at least ``t`` of ``lexemes``.

    A chunk with ``t`` of the ``n`` lexemes holds at least ``j`` of any
    ``n - t + j`` of them, so OR-ing the ``j``-subsets of the ``n - t + j``
    rarest lexemes admits all such chunks while the common lexemes stay out
    of the index lookup. The largest ``j`` (up to 3) whose subset count fits
    ``_BM25_MAX_CONJUNCTIONS`` is used. ``frequency`` only orders the lexemes
    (rare first); it never changes which chunks qualify.
    """
    from itertools import combinations
    from math import comb

    n = len(lexemes)
    if n == 0 or t < 1 or t > n:
        return None
    freq = frequency or {}
    order = sorted(lexemes, key=lambda x: (freq.get(x, 0.0), x))
    for j in (3, 2, 1):
        if j > t:
            continue
        r = n - t + j
        if comb(r, j) > _BM25_MAX_CONJUNCTIONS:
            continue
        groups = (
            " & ".join(_tsquery_literal(x) for x in combo)
            for combo in combinations(order[:r], j)
        )
        return " | ".join(f"({g})" for g in groups)
    return None


def bm25_rank_bound_holds(kth_rank: float, n_lexemes: int, t: int) -> bool:
    """True when no chunk with fewer than ``t`` lexemes can reach ``kth_rank``.

    Such a chunk ranks below ``(t - 1) * cap / n``; when that is at most the
    k-th rank already found, it cannot displace any of the k.
    """
    if n_lexemes <= 0:
        return False
    return (t - 1) * _TS_RANK_TERM_CAP / n_lexemes <= kth_rank


# ── Public types ──────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """One indexed chunk. ``embedding`` is omitted from the public
    serializer to keep response payloads small.

    ``rrf_score`` is set on results from the hybrid path (debug-only,
    not serialized). ``score`` remains the primary ranking signal —
    cosine for semantic-only, semantic cosine for hybrid results."""

    chunk_id: str
    project_id: str
    doc_id: str
    chunk_index: int
    text: str
    score: Optional[float] = None  # set on search results, None when raw
    rrf_score: Optional[float] = field(default=None, repr=False, compare=False)
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
        # rrf_score is debug-only; never expose it via the wire
        d.pop("rrf_score", None)
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
        dim = EMBEDDING_DIM

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


def _memo_put(key: tuple, value):
    memo = _TURN_MEMO.get()
    if memo is not None:
        memo[key] = value
    return value


def _memoised(key: tuple, compute):
    hit = _memo_get(key)
    return _memo_put(key, compute()) if hit is _MISS else hit


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
    """True when a CREATE TABLE lost a race rather than genuinely failing.

    SQLite says "table X already exists" (OperationalError); PostgreSQL says
    "relation X already exists" with SQLSTATE 42P07 (ProgrammingError). Both
    mean the table we wanted is now there, which is the outcome we asked for.
    """
    code = getattr(getattr(exc, "orig", None), "pgcode", None)
    if code == "42P07":
        return True
    msg = str(exc).lower()
    return "already exists" in msg and ("table" in msg or "relation" in msg)


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


def _ensure_schema(url: str, rag_chunk_cls: type) -> None:
    global _INITIALIZED_NAMESPACES
    table_name = rag_chunk_cls.__tablename__
    init_key = (url, table_name)
    if init_key in _INITIALIZED_NAMESPACES:
        return
    with _INIT_LOCK:
        if init_key in _INITIALIZED_NAMESPACES:
            return
        _ensure_sqlite_parent_dir(url)
        eng = _engine_for_url(url)
        ensure_table(rag_chunk_cls.__table__, bind=eng)
        _ensure_page_column(eng, table_name)
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
        dim: int = EMBEDDING_DIM,
        namespace: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.db_path = db_path
        self.dim = dim
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

    def _docs_visibility_ready(self, session: Optional[Session] = None) -> bool:
        """True when ``documents.retrieval_visible`` can gate this store.

        Isolated hybrid-test DBs create only the chunks table. A missing
        documents table must not empty the result set — those chunks stay
        visible. Cached per store after the first probe.
        """
        if self._visibility_ready is not None:
            return self._visibility_ready
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

    def _hidden_doc_ids(self, session: Session, project_id: str) -> Set[str]:
        if not self._docs_visibility_ready(session):
            return set()
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
            self._visibility_ready = False
            return set()
        return {str(r) for r in rows}

    def _verify_embedding_identity(self) -> None:
        """Fail loud if the namespace contains chunks from a different model.

        Checks a representative row. If any row has a different model/dim/
        normalized flag, the store refuses to operate. This is the structural
        guard that makes mixed-model contamination impossible.

        Skipped for the legacy namespace because the original ``chunks`` table
        predates the identity columns; it is retired in place and never used
        for new embeddings.
        """
        cls = self._rag_chunk_cls
        # Legacy table has no identity columns — nothing to verify.
        if not hasattr(cls, "embedding_model"):
            return
        expected = {
            "model": self.model_name,
            "dim": self.dim,
            "normalized": True,
        }
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
            return
        stored = {
            "model": row.embedding_model,
            "dim": row.embedding_dim,
            "normalized": row.embedding_normalized,
        }
        if stored != expected:
            raise RuntimeError(
                f"Embedding identity mismatch in namespace {self.namespace!r}: "
                f"expected {expected}, found {stored}. "
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

    def count(self, project_id: Optional[str] = None) -> int:
        with self._lock:
            with self._session_factory()() as session:
                stmt = select(func.count()).select_from(self._rag_chunk_cls)
                if project_id is not None:
                    stmt = stmt.where(self._rag_chunk_cls.project_id == project_id)
                return int(session.scalar(stmt) or 0)

    def has_chunks(self, project_id: str) -> bool:
        """Whether the project has any chunk: one index probe, not a count."""
        with self._lock:
            with self._session_factory()() as session:
                return session.execute(
                    select(self._rag_chunk_cls.chunk_id)
                    .where(self._rag_chunk_cls.project_id == project_id)
                    .limit(1)
                ).first() is not None

    def count_by_doc(self, project_id: str) -> Dict[str, int]:
        """Return a ``{doc_id: chunk_count}`` map for ``project_id``.

        Uses the ``idx_chunks_doc`` index; one small GROUP BY regardless of
        corpus size. This is the source of truth for per-document chunk
        counts, including chunks written by ``/v1/admin/corpus/bulk-insert``
        (which bypasses the legacy ``doc_index`` JSON blob).
        """
        with self._lock:
            with self._session_factory()() as session:
                stmt = (
                    select(self._rag_chunk_cls.doc_id, func.count())
                    .where(self._rag_chunk_cls.project_id == project_id)
                    .group_by(self._rag_chunk_cls.doc_id)
                )
                return {row[0]: int(row[1]) for row in session.execute(stmt).all()}

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
            cls.knowledge_layer, cls.authority, cls.page,
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
        )

    def _hidden_doc_ids_among(
        self, session: Session, project_id: str, doc_ids: List[str],
    ) -> Set[str]:
        """``_hidden_doc_ids`` restricted to ``doc_ids`` -- primary-key probes
        instead of reading every document row of the project."""
        if not doc_ids or not self._docs_visibility_ready(session):
            return set()
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
            self._visibility_ready = False
            return set()
        return {str(r) for r in rows}

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
                    sql = text(
                        "SELECT chunk_id, project_id, doc_id, chunk_index, text, "
                        "knowledge_layer, authority, page "
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
                "chunks_containing_all failed for project=%s needles=%r: %s",
                project_id, cleaned, exc,
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
                project_id, list(wanted), exc,
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
          score. Each returned chunk keeps its semantic cosine in
          ``.score`` and gets ``.rrf_score`` set for debug.

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
            logger.info(
                "bm25 leg empty for project=%s; retrieval degraded to "
                "semantic-only. query=%r", project_id, (query_text or "")[:120],
            )
            return sem_results[:k]

        return _rrf_combine(sem_results, bm25_results, k)

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
                    sql = text(
                        "SELECT chunk_id, project_id, doc_id, chunk_index, text, "
                        "knowledge_layer, authority, page "
                        f"FROM {self._table_name} "
                        "WHERE project_id = :project_id "
                        f"AND ({like_clauses}) "
                        f"{vis} "
                        "LIMIT :k"
                    )
                    rows = session.execute(sql, params).all()
        except OperationalError as e:
            logger.warning(
                "identifier_search failed for project=%s: %s; identifiers=%r",
                project_id, e, identifiers,
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
                # ``distance + 0`` is not an expression HNSW can serve, so a
                # small project is ranked exactly from its own rows.
                order = distance + 0 if self._project_is_small(session, project_id) else distance
                stmt = stmt.order_by(order).limit(k)
                key = ("vector", self._table_name, project_id, k,
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
                )
            )
        return out

    # ── BM25 leg ─────────────────────────────────────────────────────────

    def bm25_search(
        self,
        project_id: str,
        query: str,
        k: int = 50,
    ) -> List[Chunk]:
        """Top-``k`` chunks for ``project_id`` ranked by BM25.

        Dispatches on backend:

        - PostgreSQL: ``ts_rank`` over ``text_search`` (the tsvector
          column added by Alembic 0003) with a ``@@ plainto_tsquery``
          predicate. GIN index ``chunks_fts_gin`` keeps it fast.
        - SQLite: FTS5 ``chunks_fts`` external-content virtual table,
          joined back to ``chunks`` by rowid.

        Empty query (after sanitization) → empty list. Malformed
        backend errors → empty list (logged); caller treats as
        "semantic only" and continues.

        Returned chunks carry ``score`` set to the BM25 rank value for
        debug; the RRF fuser uses positional rank, not ``.score``.
        """
        if not query or not query.strip():
            return []

        if self._use_pgvector:
            return self._bm25_postgres(project_id, query, k)
        return self._bm25_sqlite(project_id, query, k)

    def _bm25_postgres(
        self, project_id: str, query: str, k: int
    ) -> List[Chunk]:
        """ts_rank + GIN over a BAG-OF-WORDS tsquery.

        Bag-of-words (OR), not plainto_tsquery's AND — see
        ``_sanitize_websearch_query`` for the production failure that came from
        requiring every query term to appear in a chunk. ``ts_rank`` still ranks
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
                    if self._docs_visibility_ready(session):
                        vis = self._hidden_doc_sql("c")
                    rows = self._bm25_postgres_rows(
                        session, project_id, safe_query, int(k), vis,
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
                "bm25_search (postgres) unavailable on %s: %s; query=%r — "
                "retrieval is running SEMANTIC-ONLY for this request",
                table, e, query, exc_info=True,
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
            )
            for r in rows
        ]

    def _bm25_select(self, where: str, vis: str) -> str:
        """The BM25 statement over ``where``; ranks by the FULL query ``q``.

        ``chunk_id`` breaks rank ties so the top-k is one fixed set rather
        than whichever tied rows a scan happened to reach first.
        """
        return (
            "SELECT c.chunk_id, c.project_id, c.doc_id, c.chunk_index, "
            "c.text, c.knowledge_layer, c.authority, c.page, "
            "ts_rank(c.text_search, q) AS rank "
            f"FROM {self._table_name} c, websearch_to_tsquery('english', :q) AS q "
            f"WHERE {where} AND c.project_id = :project_id {vis} "
            "ORDER BY rank DESC, c.chunk_id LIMIT :k"
        )

    def _bm25_postgres_rows(
        self, session: Session, project_id: str, safe_query: str, k: int, vis: str,
    ) -> list:
        params = {"q": safe_query, "project_id": project_id, "k": k}
        tsq = session.execute(
            text("SELECT websearch_to_tsquery('english', :q)::text"), {"q": safe_query},
        ).scalar()
        lexemes = tsquery_or_lexemes(tsq)
        if lexemes == []:
            # Every word is a stopword: nothing can match. Asking anyway reads
            # every tsvector in the project to learn that.
            return []
        # Keyed on the normalised tsquery: query wordings that reduce to the
        # same tsquery rank the same rows.
        key = ("bm25", self._table_name, project_id, tsq, k, vis)
        hit = _memo_get(key)
        if hit is not _MISS:
            return hit
        if lexemes and len(lexemes) > 1 and k > 0:
            rows = self._bm25_bounded(session, params, lexemes, vis)
            if rows is not None:
                return _memo_put(key, rows)
        return _memo_put(key, session.execute(
            text(self._bm25_select("c.text_search @@ q", vis)), params,
        ).all())

    def _bm25_bounded(
        self, session: Session, params: Dict[str, Any], lexemes: List[str], vis: str,
    ) -> Optional[list]:
        """Exact top-k ranked over GIN-admitted candidates only, or ``None``.

        See the module notes above ``_TS_RANK_TERM_CAP``. A round at threshold
        ``t`` ranks every chunk holding at least ``t`` lexemes (and possibly
        others); it is the answer when ``k`` rows came back and the k-th rank
        proves nothing outside the candidates could beat it. Otherwise ``t``
        drops by one. ``None`` sends the caller to the full scan.
        """
        n = len(lexemes)
        k = params["k"]
        freq = self._lexeme_frequencies(session)
        sql = text(self._bm25_select("c.text_search @@ CAST(:cand AS tsquery)", vis))
        t = max(2, min(_BM25_START_THRESHOLD, (n + 1) // 2))
        rounds = short = 0
        while t >= 2 and rounds < _BM25_MAX_ROUNDS:
            cand = bm25_candidate_tsquery(lexemes, t, freq)
            if cand is None:
                t -= 1
                continue
            rounds += 1
            rows = session.execute(sql, {**params, "cand": cand}).all()
            if len(rows) >= k:
                if bm25_rank_bound_holds(float(rows[k - 1].rank), n, t):
                    return rows
            else:
                short += 1
                if short >= _BM25_MAX_SHORT_ROUNDS:
                    return None
            t -= 1
        return None

    _LEXEME_FREQ_TTL_S = 3600.0

    def _lexeme_frequencies(self, session: Session) -> Dict[str, float]:
        """Planner statistics' lexeme frequencies for ``text_search``.

        Only orders candidate lexemes rarest-first (cheaper index lookups);
        correctness never depends on it, so any failure yields ``{}``.
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

    def _bm25_sqlite(
        self, project_id: str, query: str, k: int
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
                    if self._docs_visibility_ready(session):
                        vis = self._hidden_doc_sql("c")
                    sql = text(
                        f"""
                        SELECT c.chunk_id, c.project_id, c.doc_id, c.chunk_index,
                               c.text, c.knowledge_layer, c.authority, c.page,
                               {fts_table}.rank AS bm25_rank
                        FROM {fts_table}
                        JOIN {table} c ON c.rowid = {fts_table}.rowid
                        WHERE {fts_table} MATCH :q
                          AND c.project_id = :project_id
                          {vis}
                        ORDER BY {fts_table}.rank
                        LIMIT :k
                        """
                    )
                    rows = session.execute(
                        sql,
                        {"q": safe_query, "project_id": project_id, "k": k},
                    ).all()
        except OperationalError as e:
            # FTS5 raises on malformed MATCH input. Treat as "no matches."
            logger.warning(
                "bm25_search (sqlite) FTS5 MATCH failed: %s; query=%r",
                e,
                safe_query,
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
) -> List[Chunk]:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    Combines two ranked lists by summing 1/(RRF_K + rank) contributions.
    Chunks present in only one list still get a score from that list
    (the other term is 0). When a chunk appears in both lists the
    semantic instance is kept (its ``.score`` cosine attaches), so
    callers reading ``.score`` see semantic relevance. ``.rrf_score``
    is set for debug. Position in the returned list = RRF score desc.
    """
    sem_rank = {c.chunk_id: r for r, c in enumerate(semantic, 1)}
    bm_rank = {c.chunk_id: r for r, c in enumerate(bm25, 1)}
    # Semantic instances take precedence on tie (they carry .score).
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
        except Exception:
            logger.warning(
                "swallowed %s in _rrf_combine() — continuing",
                "Exception", exc_info=True,
            )
        scored.append((rrf, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]
