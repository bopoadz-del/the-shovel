"""Progress events: a user never stares at a blank answer.

Live 2026-10-07 (15 users): the first token arrived after ~27 s median, and
until then the answer bubble showed nothing. Each turn now streams a
``status`` event as soon as it is accepted and at each stage --

    {"type": "status", "stage": "searching", "label": "Searching the documents…"}

-- which the chat page shows in the bubble until the first token. The stages
are the same for every turn; the label is fixed per stage.

A tool the turn runs is shown the same way (W4: no process talk). Every
``tool_call`` frame carries the stage of the KIND of result the tool
declares (``app.agents.core.tool_registry``: ``"search"`` passages,
``"computed"`` figures) and that stage's label -- never the tool's name, its
arguments or the model's query. A tool that declares no kind is shown as
plain work. The page reads only ``label``.
"""
from __future__ import annotations

import json
from typing import Dict

LABELS: Dict[str, str] = {
    "accepted": "Working on it…",
    "searching": "Searching the documents…",
    "calculating": "Calculating…",
    "writing": "Writing the answer…",
    "working": "Working through your request…",
}

#: The stage a running tool is shown as, by the kind of result it declares.
_STAGE_OF_KIND: Dict[str, str] = {"search": "searching", "computed": "calculating"}


def stage_for_kind(kind: str) -> str:
    return _STAGE_OF_KIND.get(kind or "", "working")


def _declared_kind(name: str) -> str:
    from app.agents.core import tool_registry

    spec = tool_registry.result_spec(name)
    return spec.kind if spec is not None else ""


def event(stage: str) -> dict:
    return {"type": "status", "stage": stage, "label": LABELS[stage]}


def tool_call(frame: dict) -> dict:
    """``frame`` (a ``tool_call`` SSE frame naming its tool under ``tool``)
    with the stage and label the user is shown while that tool runs."""
    stage = stage_for_kind(_declared_kind(str(frame.get("tool") or "")))
    return {**frame, "stage": stage, "label": LABELS[stage]}


def sse(stage: str) -> str:
    return f"data: {json.dumps(event(stage))}\n\n"


async def opened(stream):
    """``stream`` preceded by the ``accepted`` status: the first thing the
    user receives once the turn has passed the turn gate."""
    yield sse("accepted")
    async for item in stream:
        yield item
