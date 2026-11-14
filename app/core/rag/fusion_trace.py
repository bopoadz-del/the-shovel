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

from typing import Any, Dict, Iterable, List, Mapping, Optional

#: The two single-leg classes the trace follows.
LEGS = ("lexical_only", "vector_only")

#: Steps that can drop a followed candidate.
OVERFETCH_CUT = "overfetch_cut"
ABSENT_AFTER_SCORING = "absent_after_scoring"
BELOW_TARGET = "below_target"
RERANK_CUT = "rerank_cut"
NOT_HANDED = "not_handed"


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
    }


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
        return out


def public_view(
    record: Optional[Dict[str, Any]], handed: Iterable[Any],
) -> Optional[Dict[str, Any]]:
    """The trace form: counts and ranks, no chunk id. Adds each followed
    candidate's position in the handed set; a pooled candidate that was not
    handed is marked ``not_handed``."""
    if not record:
        return None
    handed_ids = [getattr(c, "chunk_id", None) for c in handed]
    out = {k: v for k, v in record.items() if not k.startswith("best_")}
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
