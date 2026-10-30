"""Post-hoc kills for #34 (G2 report on 1e01582): stage deltas count from
the last stage, not from the arrival stamp, and only a "data: " frame is
rewritten by the stream timings wrapper."""
from __future__ import annotations

import asyncio
import json

import pytest

import app.core.turn_timing as tt


@pytest.mark.parametrize("auth,gap,wait", [(0.25, 0.5, 1.5), (1.0, 2.0, 0.75)])
def test_stage_deltas_count_from_the_last_stage(auth, gap, wait, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(tt.time, "monotonic", lambda: clock[0])
    tt.mark_arrival(at=clock[0] - auth)
    clock[0] += gap
    tt.mark("route")
    clock[0] += wait
    tt.first_token()
    snap = tt.snapshot()
    assert snap["route"] == gap
    assert snap["llm_first_token"] == wait


@pytest.mark.parametrize("frame", ['data:\t{"type": "start"}\n\n', 'data:{"type": "end"}\n\n'])
def test_only_data_space_frames_are_rewritten(frame):
    from app.routers.chat import _with_stage_timings

    async def frames():
        yield frame

    async def collect():
        return [f async for f in _with_stage_timings(frames())]
    assert asyncio.run(collect()) == [frame]
    assert json.loads(frame[5:])["type"] in ("start", "end")
