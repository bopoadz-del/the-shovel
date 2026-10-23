"""Retrieval must not run on the event loop.

Live 21 Sep 2026: the agent's search tool and POST /v1/rag/search ran the
hybrid retriever (embedding + BM25/vector over Postgres + rerank) inline in
their coroutines. The single uvicorn worker's loop froze for the whole
search -- /livez stalled 4-6 s during ordinary chat turns, and at 18:32 UTC
one search passed Render's 5 s health-check timeout and the instance was
restarted. Retrieval is replaced here by a synthetic slow stand-in.
"""
import asyncio
import time
from types import SimpleNamespace

from app.core import doc_index
from app.core.rag import retriever
from app.routers import rag as rag_router

SLOW_S = 0.5


async def _worst_gap_while(coro_factory):
    gaps, done = [], False

    async def heartbeat():
        last = time.perf_counter()
        while not done:
            await asyncio.sleep(0.02)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    result = await coro_factory()
    done = True
    await hb
    return max(gaps), result


def _chunk():
    return SimpleNamespace(doc_id="d1", text="synthetic excerpt one two three", score=0.9,
                           chunk_id="c1", chunk_index=0, layer="own")


def test_the_search_tool_keeps_the_event_loop_running(monkeypatch):
    def slow_retrieve(query, project_id, k=5, **kw):
        time.sleep(SLOW_S)
        return [_chunk()], []

    monkeypatch.setattr(retriever, "retrieve_with_filter", slow_retrieve)
    monkeypatch.setattr(retriever, "_doc_name_for_id", lambda doc_id: "synthetic.pdf")
    monkeypatch.setattr(doc_index, "_load_index", lambda project_id: None)

    worst, results = asyncio.run(_worst_gap_while(
        lambda: doc_index.search_project_documents("p-synthetic", "synthetic query", 3)))
    assert [r["document_id"] for r in results] == ["d1"]
    assert worst < 0.2, f"event loop frozen {worst:.2f}s by the search tool"


def test_the_rag_search_endpoint_keeps_the_event_loop_running(monkeypatch):
    def slow_retrieve(query, project_id, k=5, intent=None):
        time.sleep(SLOW_S)
        return [_chunk()]

    monkeypatch.setattr(retriever, "available", lambda: True)
    monkeypatch.setattr(retriever, "retrieve", slow_retrieve)
    monkeypatch.setattr(rag_router, "_searchable_project_or_404", lambda project_id, auth: project_id)
    import app.core.rag.vector_store as vs
    monkeypatch.setattr(vs, "get_store", lambda dim=None: SimpleNamespace(fast_search=False))

    req = rag_router.RagSearchRequest(query="synthetic query", project_id="p-synthetic")
    worst, resp = asyncio.run(_worst_gap_while(lambda: rag_router.rag_search(req, auth={})))
    assert resp.count == 1 and resp.chunks[0].doc_id == "d1"
    assert worst < 0.2, f"event loop frozen {worst:.2f}s by /v1/rag/search"
