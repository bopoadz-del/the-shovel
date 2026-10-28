"""Shared INPUT-DERIVED fake for the single LLM egress (llm_client.complete_json).

PR#3 RC1 (ORDERS_28OCT_1400Z): the prose-brief -> WBS path structures the brief
through ``app.core.llm_client.complete_json`` (schedule.py:_structure_activities_from_text).
In CI there is no LLM key, so tests of that path install this fake instead of
hitting a provider.

It is INPUT-DERIVED, not a canned list: the activities are built from the WORDS
of the brief text it receives, so two different briefs yield two different
schedules. Every ``source_span`` is a verbatim (lower-cased) substring of the
brief, so each activity survives the anti-hallucination gate
(schedule.py: a span absent from the brief is dropped). ``duration_days`` is a
positive, varied integer so every activity is "stated" and schedulable.
"""
from __future__ import annotations

import re
from typing import Any, Dict


def structured_from_brief(text: str, *, min_activities: int = 1) -> Dict[str, Any]:
    """Return ``{"activities": [...]}`` derived from ``text``.

    One activity per brief word (>=3 letters), cycled until at least
    ``min_activities`` exist; names are unique (an index suffix) so none is
    de-duped, and the activities form a simple predecessor chain.
    """
    blob = (text or "").strip()
    if not blob:
        return {"activities": []}
    low = blob.lower()
    spans = re.findall(r"[a-z][a-z\-]{2,}", low) or [low]
    activities = []
    i = 0
    needed = max(min_activities, len(spans))
    while len(activities) < needed:
        span = spans[i % len(spans)]
        name = f"{span.capitalize()} task {i + 1}"
        activities.append(
            {
                "name": name,
                "source_span": span,
                "duration_days": 2 + (i % 5),  # >0, derived from position
                "predecessors": [activities[-1]["name"]] if activities else [],
            }
        )
        i += 1
    return {"activities": activities}


def install(monkeypatch, *, min_activities: int = 1) -> None:
    """Patch the single egress with the input-derived structurer."""
    from app.core import llm_client

    async def fake(system, user, **kwargs):  # noqa: ANN001 - test double
        return structured_from_brief(user, min_activities=min_activities)

    monkeypatch.setattr(llm_client, "complete_json", fake)
