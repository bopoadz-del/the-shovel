"""The agent's document-search tool writes a retrieval trace line (F0).

Every source a cross-source answer cites can come from the agent's own
``search_project_documents`` calls, and until this change those calls left
no trace: only pre-injection wrote ``rag_retrieval_trace``. This file holds
the tool path to the same line with ``path="tool"``. That line carries:
- the chunks handed in the result rows;
- the chunks the per-document collapse dropped, and why;
- ids, scores and ranks only, never chunk text, a file name or the query's words.

Every test builds its own chunks and runs on two inputs; no corpus, project
or document name is used.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core import doc_index


class _Chunk:
    def __init__(self, chunk_id, doc_id, score, text):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.score = score
        self.text = text
        self.layer = "own"


SECRET_TEXT = "zz-chunk-body-must-not-be-logged"
SECRET_NAME = "zz-file-name-must-not-be-logged.pdf"


def _chunks():
    # doc "a": three chunks; doc "b": one; doc "c": one (lowest)
    return [
        _Chunk("a1", "a", 0.90, SECRET_TEXT),
        _Chunk("b1", "b", 0.80, SECRET_TEXT),
        _Chunk("a2", "a", 0.70, SECRET_TEXT),
        _Chunk("a3", "a", 0.60, SECRET_TEXT),
        _Chunk("c1", "c", 0.50, SECRET_TEXT),
    ]


@pytest.fixture
def corpus(monkeypatch):
    import app.core.rag.retriever as retr

    box = {"chunks": _chunks()}
    monkeypatch.setattr(
        retr, "retrieve_with_filter",
        lambda query, project_id, k=5, **_kw: (list(box["chunks"]), 0),
    )
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda _d: SECRET_NAME)
    monkeypatch.setattr(doc_index, "_load_index", lambda _pid: {"documents": []})
    monkeypatch.setattr("app.core.rag.retrieval_worker.enabled", lambda: False)
    monkeypatch.delenv("RAG_RETRIEVAL_TRACE", raising=False)
    return box


def _trace_lines(caplog):
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        if rec.name == "app.rag.retrieval_trace" and msg.startswith("rag_retrieval_trace "):
            out.append((msg, json.loads(msg.split(" ", 1)[1])))
    return out


async def _tool(query, top_k=5):
    from app.agents.base.tools import handle_search_project_documents

    class _Call:
        name = "search_project_documents"
        project_id = "p-test"
        args = {"query": query, "top_k": top_k}

    return await handle_search_project_documents(_Call())


# -- one line per call, results or none ---------------------------------------

@pytest.mark.parametrize("has_results, query", [
    (True, "zz-query-one"),
    (False, "zz-query-one"),
    (True, "zz-query-two"),
    (False, "   "),
])
async def test_a_tool_call_writes_exactly_one_tool_line_results_or_none(
    corpus, caplog, has_results, query,
):
    if not has_results:
        corpus["chunks"] = []
    caplog.set_level(logging.INFO, logger="app.rag.retrieval_trace")
    out = await _tool(query)
    assert bool(out["result"]["results"]) == (has_results and bool(query.strip()))
    lines = _trace_lines(caplog)
    assert len(lines) == 1
    payload = lines[0][1]
    assert payload["path"] == "tool"
    assert (payload["handed_k"] > 0) == bool(out["result"]["results"])
    if not out["result"]["results"]:
        assert payload["retrieved_k"] == 0
        assert payload["chunks"] == [] and payload["not_handed"] == []


@pytest.mark.parametrize("trace_path, lines_expected", [
    ("tool", 1),
    (None, 0),
])
async def test_only_a_search_that_names_a_trace_path_writes_a_line(
    corpus, caplog, trace_path, lines_expected,
):
    caplog.set_level(logging.INFO, logger="app.rag.retrieval_trace")
    await doc_index.search_project_documents("p-test", "zz-query", 5, trace_path=trace_path)
    assert len(_trace_lines(caplog)) == lines_expected


@pytest.mark.parametrize("switch, lines_expected", [
    ("0", 0),
    ("1", 1),
])
async def test_the_kill_switch_governs_the_tool_line(
    corpus, caplog, monkeypatch, switch, lines_expected,
):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", switch)
    caplog.set_level(logging.INFO, logger="app.rag.retrieval_trace")
    out = await _tool("zz-query")
    assert out["result"]["results"]
    assert len(_trace_lines(caplog)) == lines_expected


# -- handed vs dropped ---------------------------------------------------------

@pytest.mark.parametrize("per_doc, top_k, handed_expected, dropped_expected", [
    # a3 is past a 2-per-document cap: dropped, not handed
    ("2", 5, {"a1", "a2", "b1", "c1"}, {"a3": "per_doc_cap"}),
    # the same chunk is handed when the cap admits it
    ("3", 5, {"a1", "a2", "a3", "b1", "c1"}, {}),
    # document c is past top_k: its chunk is dropped with that reason
    ("3", 2, {"a1", "a2", "a3", "b1"}, {"c1": "beyond_top_k"}),
    # both reasons at once
    ("1", 2, {"a1", "b1"}, {"a2": "per_doc_cap", "a3": "per_doc_cap",
                            "c1": "beyond_top_k"}),
])
async def test_a_dropped_row_is_in_the_dropped_set_and_not_the_handed_set(
    corpus, caplog, monkeypatch, per_doc, top_k, handed_expected, dropped_expected,
):
    monkeypatch.setenv("SEARCH_CHUNKS_PER_DOC", per_doc)
    caplog.set_level(logging.INFO, logger="app.rag.retrieval_trace")
    await _tool("zz-query", top_k=top_k)
    payload = _trace_lines(caplog)[0][1]
    handed = {e["chunk_id"] for e in payload["chunks"]}
    dropped = {e["chunk_id"]: e["dropped"] for e in payload["not_handed"]}
    assert handed == handed_expected
    assert dropped == dropped_expected
    assert not handed & set(dropped)
    assert payload["retrieved_k"] == len(_chunks())
    assert payload["handed_k"] == len(handed_expected)
    assert payload["handed_docs"] == len({c[0] for c in handed_expected})


@pytest.mark.parametrize("top_k", [5, 2])
async def test_ranks_are_the_retrievers_order_and_rows_are_the_result_order(
    corpus, caplog, top_k,
):
    caplog.set_level(logging.INFO, logger="app.rag.retrieval_trace")
    out = await _tool("zz-query", top_k=top_k)
    payload = _trace_lines(caplog)[0][1]
    by_id = {e["chunk_id"]: e for e in payload["chunks"]}
    assert by_id["a1"]["rank"] == 1 and by_id["b1"]["rank"] == 2
    assert by_id["a1"]["score"] == 0.9
    row_docs = [r["document_id"] for r in out["result"]["results"]]
    for e in payload["chunks"]:
        assert row_docs[e["result_rank"] - 1] == e["doc_id"]


# -- what the line never carries ---------------------------------------------

@pytest.mark.parametrize("query", [
    "zz-query-words-must-not-be-logged",
    "zz-other-words-also-kept-out",
])
async def test_the_line_carries_no_text_no_file_name_and_no_query_words(
    corpus, caplog, query,
):
    caplog.set_level(logging.INFO, logger="app.rag.retrieval_trace")
    out = await _tool(query, top_k=2)
    # the result rows the model sees do carry them; the trace line must not
    assert SECRET_NAME in json.dumps(out)
    raw = _trace_lines(caplog)[0][0]
    assert SECRET_TEXT not in raw
    assert SECRET_NAME not in raw
    assert query not in raw
    assert "doc_name" not in raw


@pytest.mark.parametrize("top_k", [5, 2])
async def test_the_result_rows_are_unchanged_by_tracing(corpus, monkeypatch, top_k):
    on = await _tool("zz-query", top_k=top_k)
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "0")
    off = await _tool("zz-query", top_k=top_k)
    assert on == off
    rows = doc_index._search_project_documents_sync("p-test", "zz-query", top_k)
    assert rows == on["result"]["results"]
