"""POST /v1/chat/stream reports where the time before the first token went.

Every pre-first-token stage (auth, route, hat classification, retrieval,
model request to first token) is a key in the stream's ``start`` and ``end``
events, each >= 0, and the end event's figures follow the work actually done.
"""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from app.core import turn_timing
from tests._fixture_values import PW_06

STAGES = turn_timing.PRE_FIRST_TOKEN_STAGES


class _FakeAgent:
    def __init__(self, retrieval_s: float, model_s: float):
        self.retrieval_s, self.model_s = retrieval_s, model_s

    async def chat_stream(self, prompt, **kw):
        from app.agents.activation import select_hat

        select_hat(prompt)
        with turn_timing.stage("retrieval"):
            await asyncio.sleep(self.retrieval_s)
        await asyncio.sleep(self.model_s)
        yield {"type": "token", "content": "ok"}
        yield {"type": "end", "complete": True, "sources": []}


def _events(monkeypatch, prompt, retrieval_s, model_s):
    from fastapi.testclient import TestClient

    import app.agents as agents
    import app.agents.runtime as runtime
    from app.main import app
    from app.routers import chat

    agent = _FakeAgent(retrieval_s, model_s)

    async def _select(message, current):
        return current, {"agent": "fake", "reason": "test"}

    monkeypatch.setattr(agents, "get_agent", lambda name: agent)
    monkeypatch.setattr(runtime, "select_agent_for_message", _select)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: False)
    with TestClient(app) as c:
        email = f"s9a-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        r = c.post("/v1/chat/stream", headers={"Authorization": f"Bearer {token}"},
                   json={"message": prompt, "history": []})
    out = []
    for line in r.text.splitlines():
        if line.startswith("data:"):
            try:
                out.append(json.loads(line[5:]))
            except ValueError:
                pass
    return out


@pytest.mark.parametrize("prompt,retrieval_s,model_s", [
    ("What is the curing time for C40 concrete?", 0.0, 0.0),
    ("Summarise the delay damages clause.", 0.12, 0.08),
])
def test_every_stage_is_reported_and_follows_the_work(monkeypatch, prompt, retrieval_s, model_s):
    events = _events(monkeypatch, prompt, retrieval_s, model_s)
    start = next(e for e in events if e.get("type") == "start")
    end = [e for e in events if e.get("type") == "end"][-1]
    for ev in (start, end):
        t = ev["timings"]
        for k in STAGES:
            assert k in t, (k, t)
            assert isinstance(t[k], (int, float)) and t[k] >= 0, (k, t)
    t = end["timings"]
    assert t["auth"] > 0
    assert t["retrieval"] >= retrieval_s
    assert t["llm_first_token"] >= model_s
    assert t["since_arrival"] >= retrieval_s + model_s


@pytest.mark.parametrize("names", [("retrieval",), ("route", "retrieval", "route")])
def test_snapshot_accumulates_each_stage(names):
    turn_timing.mark_arrival()
    for n in names:
        with turn_timing.stage(n):
            pass
    snap = turn_timing.snapshot()
    assert set(STAGES) <= set(snap)
    assert all(snap[k] >= 0 for k in STAGES)
    assert snap["hat_classification"] == 0.0
