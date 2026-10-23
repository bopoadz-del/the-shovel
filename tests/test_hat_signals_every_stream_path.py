"""hat_signals must be on the stream whichever path answered the turn.

Live on 51b8e70, project_id=master_corpus, after #604 had deployed:

    "What is the critical path for this project?"   hat_signals 7
    "Check the scaffold load and trench shoring..."  hat_signals 7
    "Draft a variation order under FIDIC clause"     hat_signals 0   <-- this

The zero turn's ``start`` carried ``"mode": "predefined"``. It was answered by
``_stream_from_predefined`` in the router, which builds its own ``start`` and
``end`` and never enters ``Agent.chat_stream`` -- the only place #604 emitted
from. The frames below are that turn's real shape.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List

import pytest

from app.routers import hat_frames
from app.routers.hat_frames import with_hat_signals

HAT_EVENT = {
    "type": "hat_signals",
    "hat_signals": [
        {"id": "fork.hat.contracts", "discipline": "contracts", "score": 0.7, "selected": True},
        {"id": "fork.hat.safety", "discipline": "safety", "score": 0.0, "selected": False},
    ],
    "selected": "fork.hat.contracts",
    "enabled": True,
}


def _frame(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _source(frames: List[str]) -> AsyncIterator[str]:
    for frame in frames:
        yield frame


def _run(frames: List[str], message: str = "Draft a variation order") -> List[Any]:
    async def collect():
        return [f async for f in with_hat_signals(_source(frames), message)]

    out = asyncio.run(collect())
    decoded = []
    for frame in out:
        try:
            decoded.append(json.loads(frame[len("data:"):].strip()))
        except (ValueError, TypeError):
            decoded.append(frame)
    return decoded


@pytest.fixture
def hats_on(monkeypatch):
    monkeypatch.setattr(hat_frames, "_hat_event", lambda _m: dict(HAT_EVENT))


@pytest.fixture
def hats_off(monkeypatch):
    monkeypatch.setattr(hat_frames, "_hat_event", lambda _m: None)


# The exact event shape of the live zero turn.
PREDEFINED_TURN = [
    _frame({"type": "start", "session_id": "ws-x-1", "mode": "predefined"}),
    _frame({"type": "token", "content": "Variation order drafted."}),
    _frame({"type": "end", "complete": True, "mode": "predefined",
            "workflow": "change_order_impact", "tools": ["change_order_impact"]}),
]


def test_the_predefined_path_now_reports_hat_activation(hats_on):
    events = _run(PREDEFINED_TURN)
    types = [e["type"] for e in events]

    assert types == ["start", "hat_signals", "token", "end"], types
    assert events[1]["hat_signals"], "the event is there but empty"
    assert events[-1]["hat_signals"] == HAT_EVENT["hat_signals"]
    assert events[-1]["hat_selected"] == "fork.hat.contracts"
    # Nothing the predefined path put on its end event was lost.
    assert events[-1]["workflow"] == "change_order_impact"
    assert events[-1]["tools"] == ["change_order_impact"]


def test_the_heavy_reasoning_path_is_covered_too(hats_on):
    """The third path. Nobody measured a zero here yet; that is luck."""
    events = _run([
        _frame({"type": "start", "mode": "heavy_reasoning"}),
        _frame({"type": "tool_call", "name": "generate_wbs"}),
        _frame({"type": "end", "complete": True, "mode": "heavy_reasoning"}),
    ])

    assert [e["type"] for e in events] == ["start", "hat_signals", "tool_call", "end"]
    assert events[-1]["hat_signals"]


def test_an_agent_turn_is_never_double_counted(hats_on):
    """The agent already emits its own, after ITS start -- which on the wire
    is the second one: start, route, start, hat_signals. Injecting eagerly
    after the first start would hand the floor scorer two events for one turn.
    """
    agent_turn = [
        _frame({"type": "start", "session_id": "s"}),
        _frame({"type": "route", "final": "project-assistant"}),
        _frame({"type": "start", "agent": "project-assistant"}),
        _frame(dict(HAT_EVENT)),
        _frame({"type": "token", "content": "ok"}),
        _frame({"type": "end", "hat_signals": HAT_EVENT["hat_signals"],
                "hat_selected": "fork.hat.contracts"}),
    ]
    events = _run(agent_turn)

    assert [e["type"] for e in events].count("hat_signals") == 1
    assert [e["type"] for e in events] == [
        "start", "route", "start", "hat_signals", "token", "end",
    ]


def test_with_hats_off_the_stream_is_byte_identical(hats_off):
    """Hats are off by default. Off must mean untouched, not 'mostly'."""
    async def collect():
        return [f async for f in with_hat_signals(_source(PREDEFINED_TURN), "q")]

    assert asyncio.run(collect()) == PREDEFINED_TURN


def test_a_frame_that_is_not_json_passes_through_untouched(hats_on):
    odd = [": keep-alive comment\n\n", "data: not json at all\n\n"]
    events = _run([PREDEFINED_TURN[0], *odd, PREDEFINED_TURN[-1]])

    assert odd[0] in events and odd[1] in events
    assert events[-1]["type"] == "end"


def test_a_crash_in_hat_scoring_costs_the_scores_not_the_answer(monkeypatch):
    def boom(_message):
        raise RuntimeError("planted: catalog blew up")

    monkeypatch.setattr("app.agents.runtime.hat_scores_sse", boom)

    events = _run(PREDEFINED_TURN)

    assert [e["type"] for e in events] == ["start", "token", "end"]
    assert events[1]["content"] == "Variation order drafted."


def test_an_error_turn_still_gets_its_hat_event_before_the_error(hats_on):
    """A turn that dies is still a turn the scorer counts."""
    events = _run([
        _frame({"type": "start"}),
        _frame({"type": "error", "message": "temporarily unavailable"}),
    ])
    assert [e["type"] for e in events] == ["start", "hat_signals", "error"]
