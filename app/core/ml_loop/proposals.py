"""Read-only proposal pass (ML-3).

Reads a window of stored rows and returns aggregate rates plus PROPOSALS.
It never writes and never changes an answer. A proposal carries only
labels the system itself produced (route and action names, verdict
classes) and the ids of the evidence rows: never question text, message
text or a document name, so every proposal is a general mechanism a
reviewer can test against unseen questions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from app.core.ml_loop.settings import MLLoopSettings
from app.core.ml_loop.store import MLLoopRepository, window_start

_UNROUTED = "(unrouted)"


def verdict_rates_by_route(turns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for turn in turns:
        grouped[turn.get("route") or _UNROUTED][turn["verdict"]] += 1
    out: dict[str, dict[str, Any]] = {}
    for route, verdicts in sorted(grouped.items()):
        total = sum(verdicts.values())
        out[route] = {
            "total": total,
            "verdicts": dict(sorted(verdicts.items())),
            "refused_rate": round(verdicts.get("refused", 0) / total, 4),
        }
    return out


def route_correction_pairs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for row in rows:
        original = row.get("original_action") or _UNROUTED
        key = f"{original}->{row['correct_action']}"
        entry = pairs.setdefault(
            key,
            {"from_action": original, "to_action": row["correct_action"], "ids": []},
        )
        entry["ids"].append(row["id"])
    return dict(sorted(pairs.items()))


def build_proposals(
    rates: dict[str, dict[str, Any]],
    turns: list[dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    settings: MLLoopSettings,
) -> list[dict[str, Any]]:
    support = settings.min_proposal_support
    proposals: list[dict[str, Any]] = []
    for entry in pairs.values():
        if entry["from_action"] == entry["to_action"] or len(entry["ids"]) < support:
            continue
        proposals.append({
            "kind": "route_rule",
            "from_action": entry["from_action"],
            "to_action": entry["to_action"],
            "support": len(entry["ids"]),
            "evidence": {"table": "ml_route_corrections", "ids": list(entry["ids"])},
        })
    for route, stats in rates.items():
        if stats["total"] < support or stats["refused_rate"] < settings.refusal_review_rate:
            continue
        ids = [
            t["id"] for t in turns
            if (t.get("route") or _UNROUTED) == route and t["verdict"] == "refused"
        ]
        proposals.append({
            "kind": "retrieval_review",
            "route": route,
            "refused_rate": stats["refused_rate"],
            "support": stats["total"],
            "evidence": {"table": "ml_turns", "ids": ids},
        })
    return proposals


def propose(
    repo: MLLoopRepository,
    settings: MLLoopSettings,
    *,
    project_id: str | None = None,
    window_days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    days = window_days or settings.default_window_days
    since = window_start(days, now)
    turns = repo.rows("ml_turns", project_id=project_id, since=since)
    feedback = repo.rows("ml_feedback", project_id=project_id, since=since)
    route_rows = repo.rows("ml_route_corrections", project_id=project_id, since=since)

    rates = verdict_rates_by_route(turns)
    pairs = route_correction_pairs(route_rows)
    wrong_cited = Counter(
        cid
        for row in feedback
        if row["kind"] in ("wrong_figure", "wrong_document")
        for cid in (row.get("cited_ids") or [])
    )
    return {
        "window_days": days,
        "since": since.isoformat(),
        "turns": len(turns),
        "feedback_by_kind": dict(sorted(Counter(r["kind"] for r in feedback).items())),
        "verdict_rates_by_route": rates,
        "route_corrections": {k: len(v["ids"]) for k, v in pairs.items()},
        "most_cited_in_wrong_feedback": [
            {"cited_id": cid, "count": n} for cid, n in wrong_cited.most_common(20)
        ],
        "proposals": build_proposals(rates, turns, pairs, settings),
    }
