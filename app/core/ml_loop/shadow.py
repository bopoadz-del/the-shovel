"""Shadow comparison (ML-4).

A candidate predictor's decision is logged next to the live decision and
is never acted on. The report gives, per predictor, how often the shadow
agreed with the live decision and, where a routing correction exists for
the same message, which of the two matched the corrected label. That is
the evidence the advisor needs before any candidate may replace a live
rule; acceptance itself is the owner-sealed set, not this report.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from app.core.ml_loop.settings import MLLoopSettings
from app.core.ml_loop.store import MLLoopRepository, window_start


def _corrected_label_by_message(rows: list[dict[str, Any]]) -> dict[tuple, str]:
    """Latest corrected action per (project, conversation, message)."""
    out: dict[tuple, str] = {}
    for row in rows:
        if not row.get("message_id"):
            continue
        key = (row["project_id"], row.get("conversation_id"), row["message_id"])
        out[key] = row["correct_action"]
    return out


def shadow_report(
    repo: MLLoopRepository,
    settings: MLLoopSettings,
    *,
    predictor: str | None = None,
    project_id: str | None = None,
    window_days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    days = window_days or settings.default_window_days
    since = window_start(days, now)
    where = {"predictor": predictor} if predictor else None
    preds = repo.rows("ml_shadow_predictions", project_id=project_id, since=since, where=where)
    corrected = _corrected_label_by_message(
        repo.rows("ml_route_corrections", project_id=project_id, since=since)
    )

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "agree": 0, "judged": 0, "live_right": 0, "shadow_right": 0}
    )
    for row in preds:
        s = stats[row["predictor"]]
        s["total"] += 1
        s["agree"] += int(row["live_label"] == row["shadow_label"])
        truth = corrected.get((row["project_id"], row.get("conversation_id"), row.get("message_id")))
        if truth is not None:
            s["judged"] += 1
            s["live_right"] += int(row["live_label"] == truth)
            s["shadow_right"] += int(row["shadow_label"] == truth)

    report = {}
    for name, s in sorted(stats.items()):
        report[name] = {
            **s,
            "agreement_rate": round(s["agree"] / s["total"], 4),
        }
    return {"window_days": days, "since": since.isoformat(), "predictors": report}
