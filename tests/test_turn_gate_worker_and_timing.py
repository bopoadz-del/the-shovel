"""The guard: bounded concurrent turns with a clear 503, retrieval CPU in a
worker process, and turn timing from arrival. Synthetic data."""
from __future__ import annotations

import asyncio
import os

import pytest
from fastapi import HTTPException

from app.core import file_mentions, turn_gate, turn_timing
from app.core.rag import retrieval_worker


# ── turn gate ────────────────────────────────────────────────────────────────

def test_turns_beyond_the_limit_wait_then_get_a_clear_503(monkeypatch):
    monkeypatch.setenv("CHAT_MAX_CONCURRENT_TURNS", "2")
    monkeypatch.setenv("CHAT_TURN_QUEUE_WAIT_SECONDS", "0.2")

    async def run():
        r1 = await turn_gate.acquire()
        r2 = await turn_gate.acquire()
        with pytest.raises(HTTPException) as exc:
            await turn_gate.acquire()
        assert exc.value.status_code == 503
        assert exc.value.detail == turn_gate.BUSY_MESSAGE
        assert exc.value.headers["Retry-After"]
        r1()
        r3 = await turn_gate.acquire()  # a freed slot is taken without waiting
        r1()  # releasing twice is harmless
        r2()
        r3()

    asyncio.run(run())


def test_a_waiting_turn_gets_the_slot_when_one_frees(monkeypatch):
    monkeypatch.setenv("CHAT_MAX_CONCURRENT_TURNS", "1")
    monkeypatch.setenv("CHAT_TURN_QUEUE_WAIT_SECONDS", "5")

    async def run():
        r1 = await turn_gate.acquire()
        waiter = asyncio.create_task(turn_gate.acquire())
        await asyncio.sleep(0.05)
        assert not waiter.done()
        r1()
        r2 = await asyncio.wait_for(waiter, 1)
        r2()

    asyncio.run(run())


def test_the_slot_is_released_when_the_stream_ends_or_fails(monkeypatch):
    monkeypatch.setenv("CHAT_MAX_CONCURRENT_TURNS", "1")
    monkeypatch.setenv("CHAT_TURN_QUEUE_WAIT_SECONDS", "0.2")

    async def ok():
        yield "a"
        yield "b"

    async def boom():
        yield "a"
        raise RuntimeError("stream died")

    async def run():
        release = await turn_gate.acquire()
        assert [x async for x in turn_gate.hold_until_done(ok(), release)] == ["a", "b"]
        release = await turn_gate.acquire()  # free again
        with pytest.raises(RuntimeError):
            async for _ in turn_gate.hold_until_done(boom(), release):
                pass
        (await turn_gate.acquire())()  # free again after a failure

    asyncio.run(run())


# ── retrieval worker process ─────────────────────────────────────────────────

# Opt-in since F-SPEED step 1 (2026-10-08): threads measured faster live.
@pytest.mark.parametrize("value,env,expected", [
    ("1", "testing", True), ("on", "production", True), ("0", "production", False),
    ("auto", "production", False), ("", "production", False), ("", "testing", False),
])
def test_the_worker_switch(monkeypatch, value, env, expected):
    monkeypatch.setenv("RAG_RETRIEVAL_PROCESS", value)
    monkeypatch.setenv("ENV", env)
    assert retrieval_worker.enabled() is expected


def _pid_job():
    return os.getpid()


def test_jobs_run_in_another_process():
    async def run():
        return await retrieval_worker.call(retrieval_worker.names_mentioned_job,
                                           "open fenwick_reach_drainage_spec.pdf please",
                                           ["fenwick_reach_drainage_spec.pdf", "other.pdf", ""])

    assert asyncio.run(run()) == ["fenwick_reach_drainage_spec.pdf"]
    assert retrieval_worker.call_sync(_pid_job) != os.getpid()


def test_retrieval_runs_end_to_end_in_the_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag import embeddings as emb, vector_store as vs
    emb.reset_embedder_cache()
    vs.reset_store_cache()
    e = emb.Embedder(model_name="fake")
    st = vs.get_store(dim=e.dim)
    texts = [f"Clause {i}: the contractor shall give notice of suspension in writing." for i in range(10)]
    st.upsert_chunks("synthetic-worker-project", "doc-w", texts, e.encode(texts))
    kwargs = {"user_message": "What notice of suspension is required?",
              "project_id": "synthetic-worker-project", "conversation_id": None,
              "user_id": "u-worker", "agent_name": "project-assistant", "history": []}
    from app.core.privileges import set_caller_role
    from app.core.rag.inject import rag_inject

    set_caller_role("user")
    here_msg, here_audit = rag_inject(**kwargs)
    # A fresh pool, so the worker starts with this test's environment.
    monkeypatch.setattr(retrieval_worker, "_POOL", None)
    try:
        ready = retrieval_worker.call_sync(retrieval_worker.project_is_rag_ready_job, "synthetic-worker-project")
        msg, audit = retrieval_worker.call_sync(retrieval_worker.rag_inject_job, kwargs, "user")
    finally:
        pool = retrieval_worker._POOL
        if pool is not None:
            pool.shutdown()
        monkeypatch.setattr(retrieval_worker, "_POOL", None)
        set_caller_role(None)
        emb.reset_embedder_cache()
        vs.reset_store_cache()
    assert ready is True
    # The worker answers exactly as retrieval in this process does.
    assert msg == here_msg
    assert audit["project_id"] == here_audit["project_id"] == "synthetic-worker-project"
    assert audit["top_score"] == here_audit["top_score"]
    assert audit["requested_k"] == here_audit["requested_k"]


# ── file mentions ────────────────────────────────────────────────────────────

def test_names_mentioned_keeps_order_and_needs_a_distinct_name():
    names = ["Morrow_Varga_drainage_spec.pdf", "a.pdf", "Fenwick Reach schedule.xlsx", ""]
    low = "compare morrow_varga_drainage_spec with fenwick reach schedule.xlsx"
    assert file_mentions.names_mentioned(low, names) == [
        "Morrow_Varga_drainage_spec.pdf", "Fenwick Reach schedule.xlsx"]
    assert file_mentions.names_mentioned("", names) == []


# ── turn timing ──────────────────────────────────────────────────────────────

def test_stages_count_from_arrival_and_first_token_is_logged_once(monkeypatch):
    monkeypatch.setenv("AGENT_TIMING_LOG", "1")
    lines = []
    monkeypatch.setattr(turn_timing._LOG, "warning", lambda fmt, *a: lines.append(fmt % a))
    turn_timing.mark_arrival()
    with turn_timing.stage("retrieval"):
        pass
    turn_timing.mark("predispatch")
    turn_timing.first_token()
    turn_timing.first_token()
    assert [line.split()[2] for line in lines] == ["stage=retrieval", "stage=predispatch", "stage=first_token"]
    assert all(" at=" in line and "n/a" not in line for line in lines)
    assert turn_timing.arrived_at() is not None and turn_timing.since_arrival() >= 0


def test_timing_is_silent_when_off_or_without_an_arrival(monkeypatch):
    monkeypatch.setenv("AGENT_TIMING_LOG", "0")
    lines = []
    monkeypatch.setattr(turn_timing._LOG, "warning", lambda fmt, *a: lines.append(fmt % a))
    turn_timing.mark_arrival()
    with turn_timing.stage("x"):
        pass
    assert lines == []
    turn_timing._STATE.set(None)
    turn_timing.mark("y")
    turn_timing.first_token()
    assert turn_timing.since_arrival() is None
