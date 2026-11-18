"""Per-turn trace of the fused candidate list (S-R7b). Counts and ranks only.

Reciprocal-rank fusion (``vector_store._rrf_combine``) orders the union of
the vector leg and the lexical leg, then keeps the top ``fused_k``. The
pre_injection trace line used to show only the retriever's FINAL output, so
it could not say how many candidates each leg found alone, nor where the
best candidate that only one leg found ended up.

This module builds that summary and follows the best lexical-only and the
best vector-only candidate through the retriever: its fused rank, its
position after the scored sort, its position in the returned pool, or the
step that dropped it. It is pure: no store, no memo, no logging. Reading it
changes no ranking.

The summary keeps the two followed chunk ids privately (``chunk_id``) so the
trace can find them in the handed set; :func:`public_view` strips them.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional

#: The two single-leg classes the trace follows.
LEGS = ("lexical_only", "vector_only")

#: Steps that can drop a followed candidate.
OVERFETCH_CUT = "overfetch_cut"
ABSENT_AFTER_SCORING = "absent_after_scoring"
BELOW_TARGET = "below_target"
RERANK_CUT = "rerank_cut"
NOT_HANDED = "not_handed"

#: How deep each leg's own ranking is traced (``leg_top``). A measurement
#: depth only: reading it changes no ranking. It covers any leg share the
#: fusion guarantee can be set to without a second trace change.
LEG_TOP_TRACED = 10

#: How many documents per leg ``leg_docs`` lists, best leg rank first. The
#: trace line stays small however many chunks a leg ranked; the leg's full
#: document count is ``leg_docs_total``, so a shortened list is visible.
LEG_DOCS_CAP = 50

#: Hex characters of the id digest printed in place of a document id that
#: is also the document's stored name (see :func:`public_view`).
DOC_DIGEST_HEX = 12

#: How many of each leg's chunks ``leg_chunk_digests`` lists, by leg rank.
#: A measurement depth only, set to the leg's own retrieval depth so every
#: chunk a leg returned is placed; a longer leg is cut at this depth (its
#: chunk count is still in ``n_candidates`` and ``leg_docs``).
LEG_CHUNKS_TRACED = 50

#: The two legs by their rank maps' roles in ``leg_summary``.
LEG_NAMES = ("vector", "lexical")


def leg_summary(
    order: Iterable[Any],
    sem_rank: Mapping[str, int],
    bm_rank: Mapping[str, int],
    fused_k: int,
) -> Dict[str, Any]:
    """Counts of the full fused order (before the ``fused_k`` cut) by leg,
    and the best-ranked candidate found by one leg only, per leg."""
    n_candidates = n_both = n_lex = n_vec = 0
    best: Dict[str, Optional[Dict[str, Any]]] = {leg: None for leg in LEGS}
    for pos, c in enumerate(order, 1):
        n_candidates += 1
        cid = getattr(c, "chunk_id", None)
        in_vec = cid in sem_rank
        in_lex = cid in bm_rank
        if in_vec and in_lex:
            n_both += 1
            continue
        if in_lex:
            n_lex += 1
            leg, leg_rank = "lexical_only", bm_rank[cid]
        elif in_vec:
            n_vec += 1
            leg, leg_rank = "vector_only", sem_rank[cid]
        else:
            continue
        if best[leg] is None:
            best[leg] = {
                "chunk_id": cid,
                "fused_rank": pos,
                "leg_rank": leg_rank,
                "in_fused_list": pos <= fused_k,
            }
    return {
        "n_candidates": n_candidates,
        "n_both_legs": n_both,
        "n_lexical_only": n_lex,
        "n_vector_only": n_vec,
        "fused_k": fused_k,
        **{f"best_{leg}": best[leg] for leg in LEGS},
        "leg_top": leg_top(order, sem_rank, bm_rank),
        **_leg_docs_and_totals(order, sem_rank, bm_rank),
        "leg_ranked": leg_ranked(order, sem_rank, bm_rank),
    }


def leg_ranked(
    order: Iterable[Any],
    sem_rank: Mapping[str, int],
    bm_rank: Mapping[str, int],
    depth: int = LEG_CHUNKS_TRACED,
) -> Dict[str, List[Optional[str]]]:
    """Each leg's chunk ids by leg rank, position ``i`` holding rank
    ``i + 1`` (``None`` for a rank no fused candidate carries), cut at
    ``depth``. Private: :func:`public_view` prints digests only.

    ``leg_top`` stops at ``LEG_TOP_TRACED`` and ``leg_docs`` folds to
    documents, so a wanted chunk of a document that holds many leg slots
    cannot be placed inside its own leg. With this list it can: digest the
    wanted id offline (:func:`id_digest`) and find its position.
    Measurement only: reading it changes no ranking."""
    present = {getattr(c, "chunk_id", None) for c in order}
    out: Dict[str, List[Optional[str]]] = {}
    for name, mine in (("vector", sem_rank), ("lexical", bm_rank)):
        by_rank = {r: cid for cid, r in mine.items() if cid in present}
        top = min(depth, max(by_rank, default=0))
        out[name] = [by_rank.get(r) for r in range(1, top + 1)]
    return out


def _leg_docs_and_totals(
    order: Iterable[Any],
    sem_rank: Mapping[str, int],
    bm_rank: Mapping[str, int],
) -> Dict[str, Any]:
    """``leg_docs`` cut to ``LEG_DOCS_CAP`` per leg, and each leg's full
    document count as ``leg_docs_total`` (the fold runs once)."""
    full = leg_docs(order, sem_rank, bm_rank, cap=None)
    return {
        "leg_docs": {name: docs[:LEG_DOCS_CAP] for name, docs in full.items()},
        "leg_docs_total": {name: len(docs) for name, docs in full.items()},
    }


def leg_docs(
    order: Iterable[Any],
    sem_rank: Mapping[str, int],
    bm_rank: Mapping[str, int],
    cap: Optional[int] = LEG_DOCS_CAP,
) -> Dict[str, List[Dict[str, Any]]]:
    """Each leg's whole candidate list folded to DOCUMENTS: per document,
    its best rank in that leg and how many of the leg's chunks it holds,
    best rank first, at most ``cap`` documents per leg (``None``: all).

    ``leg_top`` stops at ``LEG_TOP_TRACED``, so a document the answer needed
    but nobody handed cannot be placed past that depth: inside a leg at rank
    30, or in neither leg at all (recall).
    This answers that from the document id the trace's ``chunks`` already
    publish. It also shows one document filling a leg's head (``n_chunks``).
    Measurement only: reading it changes no ranking."""
    doc_of = {
        getattr(c, "chunk_id", None): getattr(c, "doc_id", None) for c in order
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, mine in (("vector", sem_rank), ("lexical", bm_rank)):
        per_doc: Dict[str, Dict[str, int]] = {}
        for cid, r in mine.items():
            doc = doc_of.get(cid)
            if doc is None:
                continue
            entry = per_doc.setdefault(doc, {"best_leg_rank": r, "n_chunks": 0})
            entry["n_chunks"] += 1
            entry["best_leg_rank"] = min(entry["best_leg_rank"], r)
        ranked = sorted(
            per_doc.items(), key=lambda kv: (kv[1]["best_leg_rank"], kv[0]),
        )
        if cap is not None:
            ranked = ranked[:cap]
        out[name] = [{"doc_id": doc, **entry} for doc, entry in ranked]
    return out


def doc_digest(doc_id: str) -> str:
    """A short stable digest of a document id, for a trace line that must
    not print the id itself."""
    return hashlib.sha256(str(doc_id).encode("utf-8")).hexdigest()[:DOC_DIGEST_HEX]


#: The same digest for a chunk id: one function, so an offline reader
#: digests a wanted chunk id exactly as the trace did.
id_digest = doc_digest


def _public_doc(entry: Dict[str, Any], names: Mapping[str, str]) -> Dict[str, Any]:
    """One ``leg_docs`` entry for the trace. A document whose id IS its
    stored name (the vector store's parent-row fallback stores
    ``original_name = id``) would put a file name on the log line, so that
    id is replaced by ``doc_digest``; any other id is printed as is."""
    doc = entry.get("doc_id")
    if doc is None or names.get(doc) != doc:
        return dict(entry)
    view = {k: v for k, v in entry.items() if k != "doc_id"}
    view["doc_digest"] = doc_digest(doc)
    return view


def _public_leg_top_id(entry: Dict[str, Any], names: Mapping[str, str]) -> Dict[str, Any]:
    """The chunk id of one ``leg_top`` entry for the trace, so a chunk the
    answer needed can be placed in a leg by id (the ``chunks`` list of the
    same line already prints the handed ids). A chunk of a document whose id
    IS its stored name would put a file name on the line, so that chunk id is
    replaced by ``chunk_digest``, as ``_public_doc`` does for ``leg_docs``."""
    cid = entry.get("chunk_id")
    doc = entry.get("doc_id")
    if doc is not None and names.get(doc) == doc:
        return {"chunk_digest": doc_digest(str(cid))}
    return {"chunk_id": cid}


def leg_top(
    order: Iterable[Any],
    sem_rank: Mapping[str, int],
    bm_rank: Mapping[str, int],
    depth: int = LEG_TOP_TRACED,
) -> Dict[str, List[Dict[str, Any]]]:
    """Each leg's own top ``depth`` chunks, by leg rank: the other leg's
    rank for the same chunk (None when it did not find it), the chunk's
    place in the fused order, and its length in characters.

    This is what an offline replay of a fusion rule needs. It shows which
    chunks a leg-rank guarantee would lift, and how much of the handed
    set's character budget they take. The single best chunk per class in
    ``best_*`` cannot show either."""
    chunks = list(order)
    position = {getattr(c, "chunk_id", None): i for i, c in enumerate(chunks, 1)}
    chars = {
        getattr(c, "chunk_id", None): len(getattr(c, "text", "") or "")
        for c in chunks
    }
    doc_of = {getattr(c, "chunk_id", None): getattr(c, "doc_id", None) for c in chunks}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, mine, other in (
        ("vector", sem_rank, bm_rank), ("lexical", bm_rank, sem_rank),
    ):
        top = sorted(
            (r, cid) for cid, r in mine.items() if r <= depth and cid in position
        )
        out[name] = [
            {
                "chunk_id": cid,
                "doc_id": doc_of.get(cid),
                "leg_rank": r,
                "other_leg_rank": other.get(cid),
                "fused_rank": position[cid],
                "chars": chars[cid],
            }
            for r, cid in top
        ]
    return out


class Follow:
    """Follows the two best single-leg candidates of one fusion through the
    retriever's scored sort and keep loop. A ``None`` summary (no fusion ran:
    vector-only degrade, lexical-only path) makes every method a no-op."""

    def __init__(self, summary: Optional[Dict[str, Any]]):
        self.summary = summary
        self.ids = {
            entry["chunk_id"]
            for entry in (
                (summary or {}).get(f"best_{leg}") for leg in LEGS
            )
            if entry
        }
        self.reason: Dict[str, str] = {}

    def drop(self, chunk: Any, reason: str) -> None:
        """The keep loop skipped ``chunk`` for ``reason``. Kept for every
        skipped chunk (one entry each, at most the scored list); only the
        followed ids are ever read."""
        self.reason.setdefault(getattr(chunk, "chunk_id", None), reason)

    def finish(
        self,
        scored: List[Any],
        kept_before_rerank: List[Any],
        kept: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """The summary with each followed candidate's fate filled in."""
        if self.summary is None:
            return None
        scored_ids = [getattr(c, "chunk_id", None) for c in scored]
        loop_ids = {getattr(c, "chunk_id", None) for c in kept_before_rerank}
        kept_ids = [getattr(c, "chunk_id", None) for c in kept]
        out = dict(self.summary)
        for leg in LEGS:
            entry = out.get(f"best_{leg}")
            if not entry:
                continue
            entry = dict(entry)
            cid = entry["chunk_id"]
            entry["scored_position"] = (
                scored_ids.index(cid) + 1 if cid in scored_ids else None
            )
            entry["pool_position"] = (
                kept_ids.index(cid) + 1 if cid in kept_ids else None
            )
            if entry["pool_position"] is not None:
                entry["dropped_by"] = None
            elif not entry["in_fused_list"]:
                entry["dropped_by"] = OVERFETCH_CUT
            elif entry["scored_position"] is None:
                entry["dropped_by"] = ABSENT_AFTER_SCORING
            elif cid in self.reason:
                entry["dropped_by"] = self.reason[cid]
            elif cid in loop_ids:
                entry["dropped_by"] = RERANK_CUT
            else:
                entry["dropped_by"] = BELOW_TARGET
            out[f"best_{leg}"] = entry
        out["leg_top"] = {
            name: [
                {**e, "pool_position": (
                    kept_ids.index(e["chunk_id"]) + 1
                    if e["chunk_id"] in kept_ids else None
                )}
                for e in entries
            ]
            for name, entries in (self.summary.get("leg_top") or {}).items()
        }
        return out


def public_view(
    record: Optional[Dict[str, Any]],
    handed: Iterable[Any],
    names: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """The trace form: counts and ranks. Chunk ids only in ``leg_top`` (each
    leg's own head, so a miss can be placed); ``best_*`` carry none. Adds each followed
    candidate's position in the handed set; a pooled candidate that was not
    handed is marked ``not_handed``. ``names`` (document id -> stored name)
    lets ``leg_docs`` print a digest for an id that is also a name."""
    if not record:
        return None
    handed_ids = [getattr(c, "chunk_id", None) for c in handed]
    out = {
        k: v for k, v in record.items()
        if not k.startswith("best_")
        and k not in ("leg_top", "leg_docs", "leg_ranked")
    }
    if "leg_ranked" in record:
        ranked = record.get("leg_ranked") or {}
        out["leg_chunk_digests"] = {
            name: [id_digest(c) if c is not None else None for c in ids]
            for name, ids in ranked.items()
        }
    if "leg_docs" in record:
        out["leg_docs"] = {
            name: [_public_doc(e, names or {}) for e in entries]
            for name, entries in (record.get("leg_docs") or {}).items()
        }
    out["leg_top"] = {
        name: [
            {
                **_public_leg_top_id(e, names or {}),
                **{k: v for k, v in e.items() if k not in ("chunk_id", "doc_id")},
                "handed_position": (
                    handed_ids.index(e.get("chunk_id")) + 1
                    if e.get("chunk_id") in handed_ids else None
                ),
            }
            for e in entries
        ]
        for name, entries in (record.get("leg_top") or {}).items()
    }
    for leg in LEGS:
        entry = record.get(f"best_{leg}")
        if not entry:
            out[f"best_{leg}"] = None
            continue
        view = {k: v for k, v in entry.items() if k != "chunk_id"}
        cid = entry.get("chunk_id")
        view["handed_position"] = (
            handed_ids.index(cid) + 1 if cid in handed_ids else None
        )
        if view.get("pool_position") is not None and view["handed_position"] is None:
            view["dropped_by"] = NOT_HANDED
        out[f"best_{leg}"] = view
    return out
