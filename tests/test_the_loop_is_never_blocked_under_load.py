"""Fifteen users at once must not freeze the server's event loop.

Live 2026-10-07, 15 users holding 3-turn conversations: the I/O-free /livez
waited up to 10.9 s. Every turn here runs the real path -- retrieval over a
seeded corpus, a tool call, the answer's post-processing -- with only the
model replaced by a script, and app.core.loop_watchdog measures how late the
event loop serves a heartbeat. More than 250 ms late, at any moment, fails.

Synthetic corpus: invented documents and figures.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.core.loop_watchdog import LagMonitor
from tests._fixture_values import STUB_TOKEN_06

# Measures the event loop under its own simulated load: CI runs it alone, not
# beside xdist workers competing for the runner's cores (pytest.ini: serial).
pytestmark = [pytest.mark.serial, pytest.mark.timing]

USERS = 15
TURNS = 3
LIMIT_MS = 250
PID = "synthetic-load-project"

_QUESTIONS = (
    "What is the defects notification period?",
    "What compaction is required under the pavement?",
    "What is the rate for the excavation item?",
)


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_06)
    from app.core.rag import embeddings as emb, vector_store as vs
    emb.reset_embedder_cache()
    vs.reset_store_cache()
    e = emb.Embedder(model_name="fake")
    store = vs.get_store(dim=e.dim)
    texts = []
    for d in range(12):
        for i in range(40):
            texts.append((f"doc{d}", f"Section {d}.{i}. The contractor shall complete the drainage works, "
                          f"compaction to {90 + i % 9}% of maximum dry density, and the rate for item "
                          f"E{d}{i:02d} is {12 + i} per cubic metre. Defects notification period "
                          f"{300 + d} days. " * 3))
    for doc in sorted({d for d, _ in texts}):
        chunk_texts = [t for d, t in texts if d == doc]
        store.upsert_chunks(PID, doc, chunk_texts, e.encode(chunk_texts))
    yield
    emb.reset_embedder_cache()
    vs.reset_store_cache()


def _scripted_llm():
    """First call: search the project documents. Second: answer with a figure."""
    async def call(self, messages, *args, **kw):
        used_tool = any(m.get("role") == "tool" for m in messages)
        if not used_tool:
            q = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            return {"status": "success", "raw": {"model": "scripted"}, "choice": {"message": {
                "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {
                    "name": "search_project_documents", "arguments": json.dumps({"query": q[:200]})}}]}}}
        return {"status": "success", "raw": {"model": "scripted"}, "choice": {"message": {
            "content": "The defects notification period is 305 days [Source: doc5].", "tool_calls": []}}}
    return call


def test_fifteen_users_never_block_the_loop_for_more_than_250_ms(seeded, monkeypatch):
    from app.agents import runtime as rt

    rt.load_agents()
    monkeypatch.setattr(rt.Agent, "_call_llm", _scripted_llm())
    monkeypatch.setattr(rt, "project_is_rag_ready", lambda pid: True)
    agent = rt.AGENT_REGISTRY["project-assistant"]

    async def user(n: int):
        history = []
        for t in range(TURNS):
            q = _QUESTIONS[(n + t) % len(_QUESTIONS)]
            async for _ in agent.chat_stream(q, project_id=PID, user_id=f"load-user-{n}",
                                             conversation_id=f"load-conv-{n}", history=history):
                pass
            history.append({"role": "user", "content": q})

    async def run():
        # One unmeasured turn first: a cold process pays one-time costs
        # (imports, caches) on its first turn that are not the property
        # measured here. The limit is unchanged.
        async for _ in agent.chat_stream(_QUESTIONS[0], project_id=PID, user_id="load-warm",
                                         conversation_id="load-warm", history=[]):
            pass
        mon = LagMonitor(asyncio.get_running_loop(), threshold_ms=LIMIT_MS, tick_ms=20, log=False).start()
        try:
            await asyncio.gather(*(user(n) for n in range(USERS)))
        finally:
            mon.stop()
        return mon

    mon = asyncio.run(run())
    worst = max(mon.stalls, default=(0, ""))
    assert mon.max_lag_ms < LIMIT_MS, (
        f"event loop {mon.max_lag_ms:.0f} ms late under {USERS} concurrent users; "
        f"at the worst stall: {worst[1][:1500]}")


def test_the_monitor_sees_a_blocked_loop():
    """Control: a synchronous sleep on the loop is caught, with its stack."""
    import time

    async def run():
        mon = LagMonitor(asyncio.get_running_loop(), threshold_ms=100, tick_ms=10, log=False).start()
        await asyncio.sleep(0.05)
        time.sleep(0.4)
        await asyncio.sleep(0.1)
        mon.stop()
        return mon

    mon = asyncio.run(run())
    assert mon.max_lag_ms >= 300
    assert mon.stalls and "test_the_loop_is_never_blocked_under_load" in mon.stalls[0][1]
