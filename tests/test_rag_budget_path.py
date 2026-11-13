"""S-R5c: the trace says which path spent the day's RAG token counter.

Every trace line carries ``budget_path`` and ``budget_tokens`` (what this
call added to the counter), so the day's spend sums by path from the trace
lines alone. Every chunk and question here is synthetic and built by the
test. Each rule is pinned on two inputs.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging

import pytest

from app.core.rag import inject
from app.core.rag.vector_store import Chunk
from app.infra.monitoring import request_id_ctx
from tests._fixture_values import STUB_TOKEN_02

TRACE_EVENT = "rag_retrieval_trace"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name in ("RAG_K", "RAG_RETRIEVAL_TRACE", "RAG_HANDED_CHAR_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    monkeypatch.setenv("MAX_RAG_TOKENS", "100000")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


@pytest.fixture
def trace(monkeypatch):
    lg = logging.getLogger("app.rag.retrieval_trace")
    records: list = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = H(level=logging.DEBUG)
    monkeypatch.setattr(lg, "disabled", False)
    monkeypatch.setattr(lg, "level", logging.INFO)
    lg.addHandler(h)
    yield records
    lg.removeHandler(h)


def _lines(records):
    out = []
    for r in records:
        msg = r.getMessage()
        if msg.startswith(TRACE_EVENT):
            out.append(json.loads(msg[len(TRACE_EVENT):].strip()))
    return out


def _pool(n, score, size=400):
    return [
        Chunk(chunk_id=f"c{i}", project_id="p", doc_id=f"d{i}", chunk_index=i,
              text=f"excerpt {i} " + "w" * size, score=score - i * 0.01)
        for i in range(n)
    ]


def _run(monkeypatch, pool, **kw):
    monkeypatch.setattr(inject, "retrieve_with_filter", lambda q, pid, k=5, **_: (list(pool)[:k], 0))
    token = request_id_ctx.set("rid-sr5c")
    try:
        return inject.rag_inject(
            user_message="what does the method statement say about piling?",
            project_id="p", conversation_id="ws-p-1", user_id="u",
            agent_name="project-assistant", **kw,
        )
    finally:
        request_id_ctx.reset(token)


def _consumed():
    from app.core.rag import budget
    return budget.snapshot(day=_dt.datetime.utcnow().strftime("%Y-%m-%d"))["consumed"]


@pytest.mark.parametrize("kw,path", [
    ({}, inject.BUDGET_PATH_FIRST_PASS),
    ({"budget_path": inject.BUDGET_PATH_MISSING_INPUT}, inject.BUDGET_PATH_MISSING_INPUT),
])
def test_trace_names_the_path_and_the_tokens_it_added(kw, path, monkeypatch, trace):
    before = _consumed()
    _msg, audit = _run(monkeypatch, _pool(3, 0.9), **kw)
    (line,) = _lines(trace)
    assert line["budget_path"] == path and audit["budget_path"] == path
    assert line["budget_tokens"] == audit["injected_tokens"] > 0
    # the trace's tokens are exactly what the counter grew by
    assert _consumed() - before == line["budget_tokens"]


@pytest.mark.parametrize("kw,path", [
    ({}, inject.BUDGET_PATH_FIRST_PASS),
    ({"budget_path": inject.BUDGET_PATH_MISSING_INPUT}, inject.BUDGET_PATH_MISSING_INPUT),
])
def test_a_below_threshold_call_spends_nothing(kw, path, monkeypatch, trace):
    before = _consumed()
    msg, _audit = _run(monkeypatch, _pool(3, 0.2), **kw)
    assert msg is None
    (line,) = _lines(trace)
    assert line["budget_path"] == path and line["budget_tokens"] == 0
    assert _consumed() == before


@pytest.mark.parametrize("sizes", [(400, 400), (200, 1200)])
def test_trace_tokens_sum_to_the_counter_across_paths(sizes, monkeypatch, trace):
    before = _consumed()
    _run(monkeypatch, _pool(3, 0.9, size=sizes[0]))
    _run(monkeypatch, _pool(2, 0.9, size=sizes[1]),
         budget_path=inject.BUDGET_PATH_MISSING_INPUT)
    lines = _lines(trace)
    by_path: dict = {}
    for ln in lines:
        by_path[ln["budget_path"]] = by_path.get(ln["budget_path"], 0) + ln["budget_tokens"]
    assert set(by_path) == {inject.BUDGET_PATH_FIRST_PASS, inject.BUDGET_PATH_MISSING_INPUT}
    assert sum(by_path.values()) == _consumed() - before
