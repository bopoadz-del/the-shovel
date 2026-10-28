"""The model-call meter counts a provider round-trip from EVERY route, because
recording sits at the shared HTTP egress (llm_client), not at one caller.

Regression for ORDERS_1750Z #2: the meter had undercounted (57 chat turns moved
the counter by 1) because usage_tracker.record() lived only at the agent path's
two call sites in app/agents/core/llm.py, while the reasoning/block routes made
their own provider POSTs. Both shared egresses — `complete` (predefined/dynamic
reasoning) and `simple_complete` (project_reasoner, formula_executor) — now
report through `record_provider_usage`. This test drives N calls through TWO
different routes and asserts the counter rises by exactly N each.

No real provider is called: httpx.AsyncClient is replaced with a fake that
returns a 200 with a usage block.
"""
from __future__ import annotations

import pytest
from sqlalchemy import delete

from app.core import llm_client, usage_tracker
from app.core.db import SessionLocal
from app.core.models import UsageRun


class _FakeResp:
    status_code = 200
    request = None

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


class _FakeStreamResp:
    status_code = 200

    async def aread(self):
        return b""

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield 'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}'
        yield "data: [DONE]"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    """Stands in for httpx.AsyncClient: async ctx manager with .post()/.stream()."""

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return _FakeResp()

    def stream(self, *a, **k):
        return _FakeStreamResp()


@pytest.fixture(autouse=True)
def _empty_runs_and_fake_http(monkeypatch):
    usage_tracker.init_db()
    with SessionLocal() as s:
        s.execute(delete(UsageRun))
        s.commit()
    # A funded key so simple_complete/complete don't short-circuit on an unset
    # key, and a fake HTTP layer so nothing leaves the box.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("httpx.AsyncClient", _FakeClient)
    yield


@pytest.mark.asyncio
async def test_complete_route_increments_the_meter_per_call():
    before = usage_tracker.totals()["calls"]
    n = 3
    for _ in range(n):
        out = await llm_client.complete([{"role": "user", "content": "hi"}])
        assert out == "ok"
    assert usage_tracker.totals()["calls"] == before + n


@pytest.mark.asyncio
async def test_simple_complete_route_increments_the_meter_per_call():
    before = usage_tracker.totals()["calls"]
    n = 4
    for _ in range(n):
        out = await llm_client.simple_complete("hi", agent_name="project_reasoner")
        assert out == "ok"
    assert usage_tracker.totals()["calls"] == before + n


@pytest.mark.asyncio
async def test_two_different_routes_sum_on_the_same_meter():
    before = usage_tracker.totals()["calls"]
    await llm_client.complete([{"role": "user", "content": "a"}])
    await llm_client.simple_complete("b", agent_name="formula_executor_v2")
    # One call from each distinct route -> the single meter sees both.
    assert usage_tracker.totals()["calls"] == before + 2


@pytest.mark.asyncio
async def test_streaming_route_increments_the_meter_from_the_usage_chunk():
    before = usage_tracker.totals()["calls"]
    got = []
    async for line in llm_client.provider_stream(
        "https://provider.example/chat", {"stream": True}, {},
        provider="deepseek", model="deepseek-flash", agent_name="stream_route",
    ):
        got.append(line)
    # The content + usage + [DONE] lines were yielded raw to the caller...
    assert any("hi" in line for line in got)
    # ...and the final usage chunk was recorded exactly once at the egress.
    assert usage_tracker.totals()["calls"] == before + 1


@pytest.mark.asyncio
async def test_streaming_records_even_when_caller_breaks_on_done():
    # Regression (ORDERS_1920Z/1950Z): real callers break on [DONE], abandoning
    # this generator. The egress records BEFORE yielding [DONE], and consumers
    # wrap it in contextlib.aclosing — so the meter is hit without the caller
    # (or this test) ever calling aclose itself.
    import contextlib
    before = usage_tracker.totals()["calls"]
    async with contextlib.aclosing(llm_client.provider_stream(
        "https://provider.example/chat", {"stream": True}, {},
        provider="deepseek", model="deepseek-flash", agent_name="break_route",
    )) as agen:
        async for line in agen:
            if "[DONE]" in line:
                break
    assert usage_tracker.totals()["calls"] == before + 1


class _FakeStreamNoUsage(_FakeStreamResp):
    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"


class _FakeClientNoUsage(_FakeClient):
    def stream(self, *a, **k):
        return _FakeStreamNoUsage()


@pytest.mark.asyncio
async def test_streaming_without_a_usage_chunk_still_counts_the_call(monkeypatch):
    # No usage chunk -> record the call with tokens unknown (ORDERS_1920Z #2b):
    # the meter counts calls, so a usage-less stream must not be invisible.
    import contextlib
    monkeypatch.setattr("httpx.AsyncClient", _FakeClientNoUsage)
    before = usage_tracker.totals()["calls"]
    before_tokens = usage_tracker.totals()["tokens"]
    async with contextlib.aclosing(llm_client.provider_stream(
        "https://provider.example/chat", {"stream": True}, {},
        provider="deepseek", model="deepseek-flash", agent_name="no_usage",
    )) as agen:
        async for line in agen:
            if "[DONE]" in line:
                break
    assert usage_tracker.totals()["calls"] == before + 1
    assert usage_tracker.totals()["tokens"] == before_tokens  # tokens unknown = 0


@pytest.mark.asyncio
async def test_real_chatblock_public_streaming_turn_increments_meter():
    # Drive ChatBlock through its PUBLIC turn entry (process) on the STREAMING
    # path, not the private _call_cloud helper -> +N (ORDERS_2050Z #2d). The
    # returned stream must be consumed for the egress to record.
    from app.blocks.chat import ChatBlock

    blk = ChatBlock()
    before = usage_tracker.totals()["calls"]
    for _ in range(2):
        res = await blk.process({"text": "hi"}, {"stream": True, "max_tokens": 16})
        assert res.get("status") == "success", res
        async for _chunk in res["stream"]:
            pass
    assert usage_tracker.totals()["calls"] == before + 2


@pytest.mark.asyncio
async def test_real_driver_turn_increments_meter():
    # A real driver agent turn -> +N on the same meter (ORDERS_1920Z #2d).
    from app.agents.driver import llm as driver_llm

    class _Agent:
        name = "driver"
        model = ""
        temperature = 0.0
        max_tokens = 32

    before = usage_tracker.totals()["calls"]
    for _ in range(2):
        res = await driver_llm.call(_Agent(), [{"role": "user", "content": "hi"}], None)
        assert res["status"] == "success", res
    assert usage_tracker.totals()["calls"] == before + 2
