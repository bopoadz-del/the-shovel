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

import app.core.turn_timing as turn_timing
from tests._fixture_values import PW_06

STAGES = turn_timing.PRE_FIRST_TOKEN_STAGES


class _FakeAgent:
    name = "pinned-one"

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


def _events(monkeypatch, prompt, retrieval_s, model_s, select_s=0.0, predefined=False, **extra):
    from fastapi.testclient import TestClient

    import app.agents as agents
    import app.agents.runtime as runtime
    from app.main import app
    import app.routers.chat as chat

    agent = _FakeAgent(retrieval_s, model_s)

    async def _select(message, current):
        await asyncio.sleep(select_s)
        return current, {"agent": "fake", "reason": "test"}

    monkeypatch.setattr(agents, "get_agent", lambda name: agent)
    monkeypatch.setattr(runtime, "select_agent_for_message", _select)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: predefined)
    with TestClient(app) as c:
        email = f"s9a-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        r = c.post("/v1/chat/stream", headers={"Authorization": f"Bearer {token}"},
                   json={"message": prompt, "history": [], **extra})
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


# --- the stage clock itself (app.core.turn_timing) --------------------------
import contextvars  # noqa: E402
import time  # noqa: E402


def _fresh(fn, *a):
    """Run ``fn`` in a context with no turn started."""
    return contextvars.Context().run(fn, *a)


@pytest.mark.parametrize("ago", [0.5, 2.0])
def test_mark_arrival_counts_from_the_earlier_reading(ago):
    def go():
        turn_timing.mark_arrival(at=time.monotonic() - ago)
        return turn_timing.since_arrival(), turn_timing.snapshot()["since_arrival"]
    live, snap = _fresh(go)
    assert live >= ago and snap >= ago


@pytest.mark.parametrize("ahead", [5.0, 60.0])
def test_mark_arrival_never_starts_in_the_future(ahead):
    def go():
        turn_timing.mark_arrival(at=time.monotonic() + ahead)
        return turn_timing.since_arrival()
    assert 0 <= _fresh(go) < 1


@pytest.mark.parametrize("durs,want", [((0.25,), 0.25), ((0.25, 0.5, -1.0), 0.75)])
def test_record_adds_up_and_ignores_negative(durs, want):
    def go():
        turn_timing.mark_arrival()
        for d in durs:
            turn_timing.record("retrieval", d)
        return turn_timing.snapshot()
    snap = _fresh(go)
    assert snap["retrieval"] == want
    assert snap["route"] == 0.0


@pytest.mark.parametrize("name", ["retrieval", "custom_stage"])
def test_no_turn_means_nothing_recorded(name):
    def go():
        turn_timing.record(name, 1.0)
        turn_timing.first_token()
        turn_timing.mark(name)
        return turn_timing.snapshot(), turn_timing.since_arrival(), turn_timing.arrived_at()
    snap, since, at = _fresh(go)
    assert snap.get(name, 0.0) == 0.0 and snap["since_arrival"] == 0.0
    assert since is None and at is None


@pytest.mark.parametrize("extra", ["queue", "postprocess"])
def test_snapshot_keeps_other_stages_beside_the_fixed_ones(extra):
    def go():
        turn_timing.mark_arrival()
        turn_timing.record(extra, 0.125)
        return turn_timing.snapshot()
    snap = _fresh(go)
    assert snap[extra] == 0.125
    assert set(STAGES) <= set(snap)


@pytest.mark.parametrize("on,calls", [("1", 1), ("0", 0)])
def test_log_stage_writes_only_when_timing_log_is_on(monkeypatch, on, calls):
    seen = []
    monkeypatch.setenv("AGENT_TIMING_LOG", on)
    monkeypatch.setattr(turn_timing, "_WRITER", lambda fmt, *a: seen.append(a))

    def go():
        turn_timing.mark_arrival()
        turn_timing.log_stage("retrieval", 0.5)
        return turn_timing.snapshot()["retrieval"]
    assert _fresh(go) == 0.5
    assert len(seen) == calls
    if calls:
        assert seen[0][0] == "retrieval"


@pytest.mark.parametrize("on", ["true", "yes"])
def test_log_stage_without_a_writer_logs_a_warning(monkeypatch, caplog, on):
    monkeypatch.setenv("AGENT_TIMING_LOG", on)
    monkeypatch.setattr(turn_timing, "_WRITER", None)

    def go():
        turn_timing.mark_arrival()
        turn_timing.log_stage("route", 0.25)
    with caplog.at_level("WARNING", logger="app.agents.runtime"):
        _fresh(go)
    assert any("stage=route" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("value", [3, "x"])
def test_recorded_times_sync_calls_and_keeps_their_result(value):
    @turn_timing.recorded("hat_classification")
    def f(v):
        time.sleep(0.01)
        return v

    def go():
        turn_timing.mark_arrival()
        return f(value), turn_timing.snapshot()["hat_classification"]
    out, t = _fresh(go)
    assert out == value and t > 0
    assert f.__name__ == "f"


@pytest.mark.parametrize("value", [7, "y"])
def test_recorded_times_async_calls_and_keeps_their_result(value):
    @turn_timing.recorded("retrieval")
    async def g(v):
        await asyncio.sleep(0.01)
        return v

    def go():
        turn_timing.mark_arrival()
        return asyncio.run(g(value)), turn_timing.snapshot()["retrieval"]
    out, t = _fresh(go)
    assert out == value and t > 0
    assert g.__name__ == "g"


@pytest.mark.parametrize("value", [1, "z"])
def test_timed_times_calls_as_a_stage_and_keeps_their_result(value):
    @turn_timing.timed("route")
    def h(v):
        time.sleep(0.01)
        return v

    def go():
        turn_timing.mark_arrival()
        return h(value), turn_timing.snapshot()["route"]
    out, t = _fresh(go)
    assert out == value and t > 0
    assert h.__name__ == "h"


@pytest.mark.parametrize("wait", [0.02, 0.05])
def test_first_token_is_recorded_once(wait):
    def go():
        turn_timing.mark_arrival()
        time.sleep(wait)
        turn_timing.first_token()
        once = turn_timing.snapshot()["llm_first_token"]
        time.sleep(wait)
        turn_timing.first_token()
        return once, turn_timing.snapshot()["llm_first_token"]
    once, twice = _fresh(go)
    assert once >= wait
    assert twice == once


# --- app.routers.chat wiring -------------------------------------------------
@pytest.mark.parametrize("prompt", ["Curing time for C40?", "Delay damages clause?"])
def test_a_pinned_agent_skips_the_auto_router(monkeypatch, prompt):
    import app.core.privileges as privileges

    monkeypatch.setattr(privileges, "caller_may_use_agent", lambda agent, role: True)
    events = _events(monkeypatch, prompt, 0.0, 0.0, agent="pinned-one")
    route = next(e for e in events if e.get("type") == "route")
    assert route["reason"] == "user_pinned"
    assert route["final"] == "pinned-one"


@pytest.mark.parametrize("prompt", ["Curing time for C40?", "Delay damages clause?"])
def test_stream_without_the_chat_block_is_500(monkeypatch, prompt):
    from fastapi.testclient import TestClient

    import app.routers.chat as chat
    from app.main import app

    monkeypatch.delitem(chat.BLOCK_REGISTRY, "chat", raising=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        email = f"s9a-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        r = c.post("/v1/chat/stream", headers={"Authorization": f"Bearer {token}"}, json={"message": prompt})
    assert r.status_code == 500
    assert "Chat block not available" in r.text


@pytest.mark.parametrize("kind", ["start", "end"])
def test_stage_timings_ride_only_on_start_and_end(kind):
    import app.routers.chat as chat

    frames = [f"data: {json.dumps({'type': kind})}\n\n",
              f"data: {json.dumps({'type': 'token', 'content': 'a'})}\n\n",
              "data: not-json\n\n", ": keepalive\n\n"]

    async def src():
        for f in frames:
            yield f

    async def collect():
        return [x async for x in chat._with_stage_timings(src())]

    def go():
        turn_timing.mark_arrival()
        return asyncio.run(collect())
    out = _fresh(go)
    first = json.loads(out[0][6:])
    assert first["type"] == kind and set(STAGES) <= set(first["timings"])
    assert out[1:] == frames[1:]


# --- G6 BOX survivors (a6d2625) ----------------------------------------------
import app.agents.activation as activation  # noqa: E402


def _timings(monkeypatch, *, select_s=0.0, auth_s=0.0, predefined_s=None, hat_s=0.0):
    """End-event timings of one streamed turn with one stage slowed on purpose."""
    import time as _t

    import app.routers.chat as chat
    from app.dependencies import require_user
    from app.main import app

    real_hats_enabled = activation.hats_enabled

    def _slow_hats_enabled():
        _t.sleep(hat_s)
        return real_hats_enabled()

    monkeypatch.setattr(activation, "hats_enabled", _slow_hats_enabled)
    if predefined_s is not None:
        async def _intent(prompt, has_documents=False):
            await asyncio.sleep(predefined_s)
            return {"action": None}
        monkeypatch.setattr(chat, "_message_vetoes_predefined", lambda p: False)
        monkeypatch.setattr(chat, "understand_intent", _intent)
    if auth_s:
        async def _slow_user():
            await asyncio.sleep(auth_s)
            return {"user_id": "s9a-slow-auth", "role": "user"}
        app.dependency_overrides[require_user] = _slow_user
    try:
        events = _events(monkeypatch, "Curing time for C40?", 0.0, 0.0,
                         select_s=select_s, predefined=predefined_s is not None)
    finally:
        app.dependency_overrides.pop(require_user, None)
    return [e for e in events if e.get("type") == "end"][-1]["timings"]


# activation.py:81 hat classification is timed as its own stage.
@pytest.mark.parametrize("hat_s", [0.05, 0.12])
def test_hat_classification_is_timed(monkeypatch, hat_s):
    assert _timings(monkeypatch, hat_s=hat_s)["hat_classification"] >= hat_s


# chat.py select_agent_for_message is timed as the route stage.
@pytest.mark.parametrize("select_s", [0.05, 0.12])
def test_agent_selection_is_timed_as_route(monkeypatch, select_s):
    assert _timings(monkeypatch, select_s=select_s)["route"] >= select_s


# chat.py the predefined intent check is also timed as route.
@pytest.mark.parametrize("predefined_s", [0.05, 0.12])
def test_predefined_intent_check_is_timed_as_route(monkeypatch, predefined_s):
    assert _timings(monkeypatch, predefined_s=predefined_s)["route"] >= predefined_s


# chat.py the turn clock starts when the request arrived, before auth ran.
@pytest.mark.parametrize("auth_s", [0.08, 0.15])
def test_the_clock_starts_before_auth(monkeypatch, auth_s):
    t = _timings(monkeypatch, auth_s=auth_s)
    assert t["auth"] >= auth_s
    assert t["since_arrival"] >= t["auth"]


# chat.py the end event, and only it, writes the timings log line.
@pytest.mark.parametrize("kind,logged", [("end", True), ("start", False)])
def test_only_the_end_event_logs_the_timings(caplog, kind, logged):
    import logging

    import app.routers.chat as chat

    async def src():
        yield f"data: {json.dumps({'type': kind})}\n\n"

    async def collect():
        return [x async for x in chat._with_stage_timings(src())]

    def go():
        turn_timing.mark_arrival()
        return asyncio.run(collect())

    with caplog.at_level(logging.INFO, logger=chat.logger.name):
        _fresh(go)
    assert any("chat.stream stage timings" in r.getMessage() for r in caplog.records) is logged


# turn_timing.py:35 -- retrieval is a fixed pre-first-token stage: reported as
# 0.0 when it did not run, and with its time when it did.
@pytest.mark.parametrize("ran,want", [(None, 0.0), (0.375, 0.375)])
def test_retrieval_is_always_reported(ran, want):
    def go():
        turn_timing.mark_arrival()
        if ran is not None:
            turn_timing.record("retrieval", ran)
        return turn_timing.snapshot()
    snap = _fresh(go)
    assert "retrieval" in turn_timing.PRE_FIRST_TOKEN_STAGES
    assert snap["retrieval"] == want


# turn_timing.py:50 -- with no turn, record() stores nothing and says so;
# within a turn it returns the stage's running total.
@pytest.mark.parametrize("durs,want", [((), None), ((0.25, 0.5), 0.75)])
def test_record_returns_the_running_total_only_within_a_turn(durs, want):
    def go():
        if durs:
            turn_timing.mark_arrival()
            for d in durs:
                got = turn_timing.record("route", d)
            return got
        return turn_timing.record("route", 1.0)
    assert _fresh(go) == want
