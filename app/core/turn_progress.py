"""Progress events: a user never stares at a blank answer.

Live 2026-10-07 (15 users): the first token arrived after ~27 s median, and
until then the answer bubble showed nothing. Each turn now streams a
``status`` event as soon as it is accepted and at each stage --

    {"type": "status", "stage": "searching", "label": "Searching the documents…"}

-- which the chat page shows in the bubble until the first token. The stages
are the same for every turn; the label is fixed per stage.
"""
from __future__ import annotations

import json
from typing import Dict

LABELS: Dict[str, str] = {
    "accepted": "Working on it…",
    "searching": "Searching the documents…",
    "calculating": "Calculating…",
    "writing": "Writing the answer…",
}


def event(stage: str) -> dict:
    return {"type": "status", "stage": stage, "label": LABELS[stage]}


def sse(stage: str) -> str:
    return f"data: {json.dumps(event(stage))}\n\n"


async def opened(stream):
    """``stream`` preceded by the ``accepted`` status: the first thing the
    user receives once the turn has passed the turn gate."""
    yield sse("accepted")
    async for item in stream:
        yield item
