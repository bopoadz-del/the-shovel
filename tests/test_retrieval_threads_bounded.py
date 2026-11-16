"""Document retrieval runs on a pool bounded by the task's vCPUs, not the
loop's default executor (min(32, CPUs + 4) threads), so Python retrieval work
never outnumbers the CPUs and the event loop is not queued behind it for the
GIL. Every test builds its own inputs."""
from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import pytest

from app.core import offload


@pytest.fixture
def fresh_retrieval_pool(monkeypatch):
    old = offload._NAMED_POOLS.pop("retrieval", None)
    yield
    pool = offload._NAMED_POOLS.pop("retrieval", None)
    if pool is not None:
        pool.shutdown(wait=True)
    if old is not None:
        offload._NAMED_POOLS["retrieval"] = old


def _vcpus(monkeypatch, n: float) -> None:
    from app.core.rag import retrieval_worker

    monkeypatch.setattr(retrieval_worker, "available_vcpus", lambda: n)


@pytest.mark.parametrize("vcpus,expected", [(0.5, 2), (1.0, 2), (2.0, 2), (3.2, 4), (8.0, 8)])
def test_size_is_one_thread_per_vcpu_never_below_two(monkeypatch, vcpus, expected):
    monkeypatch.delenv("RETRIEVAL_THREADS", raising=False)
    _vcpus(monkeypatch, vcpus)
    assert offload._vcpu_pool_size("RETRIEVAL_THREADS") == expected


@pytest.mark.parametrize("raw,expected", [("3", 3), ("7", 7), ("0", 1), ("-4", 1)])
def test_an_explicit_size_wins_over_vcpus(monkeypatch, raw, expected):
    monkeypatch.setenv("RETRIEVAL_THREADS", raw)
    _vcpus(monkeypatch, 16.0)
    assert offload._vcpu_pool_size("RETRIEVAL_THREADS") == expected


@pytest.mark.parametrize("raw,vcpus,expected", [("many", 3.0, 3), ("", 6.0, 6), ("  ", 1.0, 2)])
def test_an_unusable_size_falls_back_to_vcpus(monkeypatch, raw, vcpus, expected):
    monkeypatch.setenv("RETRIEVAL_THREADS", raw)
    _vcpus(monkeypatch, vcpus)
    assert offload._vcpu_pool_size("RETRIEVAL_THREADS") == expected


@pytest.mark.parametrize("size,jobs", [(2, 7), (3, 9)])
def test_no_more_searches_run_at_once_than_the_pool_holds(monkeypatch, fresh_retrieval_pool, size, jobs):
    monkeypatch.setenv("RETRIEVAL_THREADS", str(size))
    lock = threading.Lock()
    state = {"now": 0, "peak": 0, "names": set()}

    def search(i: int) -> int:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            state["names"].add(threading.current_thread().name)
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return i * 10

    async def run():
        return await asyncio.gather(*(offload.to_thread_in("retrieval", search, i) for i in range(jobs)))

    assert asyncio.run(run()) == [i * 10 for i in range(jobs)]
    assert state["peak"] == size
    assert all(n.startswith("retrieval-work") for n in state["names"])
    assert offload.named_pool("retrieval")._max_workers == size


@pytest.mark.parametrize("caller_value,inner_value", [("caller-a", "inner-a"), ("caller-b", "inner-b")])
def test_to_thread_in_keeps_to_thread_context_semantics(fresh_retrieval_pool, caller_value, inner_value):
    var: contextvars.ContextVar[str] = contextvars.ContextVar("probe_ctx", default="unset")

    def fn() -> str:
        seen = var.get()
        var.set(inner_value)
        return seen

    async def run():
        var.set(caller_value)
        seen = await offload.to_thread_in("retrieval", fn)
        after_bounded = var.get()
        await offload.off_loop_in("background", fn)
        after_write_back = var.get()
        return seen, after_bounded, after_write_back

    seen, after_bounded, after_write_back = asyncio.run(run())
    assert seen == caller_value              # fn sees the caller's context
    assert after_bounded == caller_value     # nothing set inside comes back
    assert after_write_back == inner_value   # unlike off_loop_in, which writes back


@pytest.mark.parametrize("query,top_k", [("drainage compaction", 3), ("defects period", 5)])
def test_document_search_runs_on_the_retrieval_pool(monkeypatch, fresh_retrieval_pool, query, top_k):
    from app.core import doc_index
    from app.core.rag import retrieval_worker

    monkeypatch.setattr(retrieval_worker, "enabled", lambda: False)
    monkeypatch.setenv("RETRIEVAL_THREADS", "2")
    seen = {}

    def traced(project_id, q, k, exclude):
        seen.update(thread=threading.current_thread().name, q=q, k=k, pid=project_id)
        return [{"chunk_id": f"{q}-1"}], {"chunks": []}

    monkeypatch.setattr(doc_index, "_search_project_documents_traced", traced)
    results = asyncio.run(doc_index.search_project_documents("p-test", query, top_k=top_k))
    assert seen["thread"].startswith("retrieval-work")
    assert (seen["q"], seen["k"], seen["pid"]) == (query, top_k, "p-test")
    assert results == [{"chunk_id": f"{query}-1"}]
