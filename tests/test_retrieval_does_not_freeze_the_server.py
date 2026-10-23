"""One user's retrieval must not freeze the server for everyone else.

Live, 2026-09-20, build 72fcfd2: ONE chat turn stalled /livez -- an I/O-free
handler -- for 4.2 s (idle: 0.37 s). `rag_inject` (embedding + SQL + rerank) is
synchronous and was called directly inside the async chat path, so on the
single worker nothing else was served while it ran: no health check, no other
user's tokens. With six testers the stalls stacked and the proxy answered 502
with no deploy in progress.
"""
import asyncio
import time

import pytest

from app.agents import runtime as rt
from tests._fixture_values import STUB_TOKEN_06

# Measures the event loop under its own simulated load: CI runs it alone, not
# beside xdist workers competing for the runner's cores (pytest.ini: serial).
pytestmark = [pytest.mark.serial, pytest.mark.timing]

BLOCK_S = 0.6


def _slow_rag_inject(**kwargs):
    time.sleep(BLOCK_S)          # synchronous work, like the real one
    return None, {}


async def _longest_gap_while(coro_factory):
    """Run the coroutine while a heartbeat ticks every 20 ms; return the
    longest time the event loop went without ticking.

    The scenario runs once unmeasured first: a cold process pays one-time
    costs on its first turn (module imports, caches) that hold the loop for
    a moment and are not what this measures. The limit is unchanged."""
    await coro_factory()
    gaps, stop = [], False

    async def heartbeat():
        last = time.perf_counter()
        while not stop:
            await asyncio.sleep(0.02)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    try:
        await coro_factory()
    finally:
        stop = True
        await hb
    return max(gaps)


@pytest.fixture
def agent(monkeypatch):
    rt.load_agents()
    monkeypatch.setattr(rt, "rag_inject", _slow_rag_inject)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_06)
    # Past the "project has no documents" guard, so the turn reaches retrieval.
    monkeypatch.setattr(rt, "project_is_rag_ready", lambda pid: True)
    a = rt.AGENT_REGISTRY["project-assistant"]

    async def no_llm(self, *args, **kw):
        return {"status": "error", "error": "stop here"}

    monkeypatch.setattr(rt.Agent, "_call_llm", no_llm)
    return a


def test_the_stream_path_keeps_the_loop_alive_during_retrieval(agent):
    async def turn():
        async for _ in agent.chat_stream("What is the retention?", project_id="p1",
                                         user_id="u1"):
            pass

    gap = asyncio.run(_longest_gap_while(turn))
    assert gap < BLOCK_S / 2, f"event loop frozen for {gap:.2f}s during retrieval"


def test_the_non_stream_path_keeps_the_loop_alive_during_retrieval(agent):
    async def turn():
        await agent.chat("What is the retention?", project_id="p1", user_id="u1")

    gap = asyncio.run(_longest_gap_while(turn))
    assert gap < BLOCK_S / 2, f"event loop frozen for {gap:.2f}s during retrieval"


def test_the_caller_role_still_reaches_retrieval_in_the_thread(agent, monkeypatch):
    """asyncio.to_thread copies contextvars: the admin/user gate must survive."""
    from app.core.privileges import caller_role, set_caller_role

    seen = []

    def spy(**kwargs):
        seen.append(caller_role())
        return None, {}

    monkeypatch.setattr(rt, "rag_inject", spy)
    set_caller_role("user")
    try:
        asyncio.run(agent.chat("x", project_id="p1", user_id="u1"))
    finally:
        set_caller_role(None)
    assert seen and set(seen) == {"user"}
