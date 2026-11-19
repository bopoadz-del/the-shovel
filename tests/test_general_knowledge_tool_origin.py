"""The general-knowledge search tool's hits must say they are general knowledge.

The tool searches each configured library project AS the project, so the
retriever names every hit "own" (the chunk belongs to the project searched).
Every reader of a hit (the driver sources panel, the answer tier) checks
``origin`` first, so a library passage was shown as the user's own project
document and a general-only answer was tiered "project", skipping the cap
for general-only answers (knowledge audit 15 Nov, headline 4).

These tests run the REAL tool handler through the REAL
``search_project_documents``; only the retriever underneath is stubbed, and it
returns what the real retriever returns when the searched project owns the
chunk: ``layer="own"``. Nothing here hands the reader the right value.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core import doc_index


def _chunk(project_id: str, doc_id: str, score: float) -> SimpleNamespace:
    # What retrieve_with_filter returns for a chunk of the searched project.
    return SimpleNamespace(
        chunk_id=f"c_{doc_id}", project_id=project_id, doc_id=doc_id,
        chunk_index=0, text=f"passage of {doc_id}", score=score, layer="own",
    )


_BY_PROJECT = {
    "lib-alpha": [("a_note", 0.7)],
    "lib-beta": [("b_note", 0.9)],
    "users-project": [("u_spec", 0.8)],
    "other-project": [("o_drawing", 0.6)],
}


@pytest.fixture
def library(monkeypatch):
    """Install a stubbed retriever; returns ``install(libraries)``."""
    import app.core.rag.retriever as retr

    from app.core import offload
    from app.core.rag import retrieval_worker

    def install(libraries: list[str]) -> list[str]:
        monkeypatch.setattr(retr, "_general_knowledge_project_ids", lambda: list(libraries))
        monkeypatch.setattr(
            retr, "retrieve_with_filter",
            lambda q, pid, k=5, **kw: (
                [_chunk(pid, d, sc) for d, sc in _BY_PROJECT.get(pid, [])], 0),
        )
        monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: f"{d}.md")
        monkeypatch.setattr(doc_index, "_load_index", lambda pid: {"documents": []})
        monkeypatch.setattr(doc_index, "_row_document_fields", lambda doc_id, kept: {})

        async def _direct(_pool, fn, *args, **kw):
            return fn(*args, **kw)

        monkeypatch.setattr(retrieval_worker, "enabled", lambda: False)
        monkeypatch.setattr(offload, "to_thread_in", _direct)
        return libraries

    return install


def _run_tool(query: str, project_id: str) -> dict:
    from app.agents.core import tool_registry

    spec = tool_registry.get("search_general_knowledge")
    call = tool_registry.ToolCall(agent=None, name="search_general_knowledge",
                                  args={"query": query, "top_k": 5}, project_id=project_id)
    return asyncio.run(spec.handler(call))


# Two configurations of the library set: both libraries, and one library
# only. The user's own project is never a library.
_LIBRARY_SETS = [
    (["lib-alpha", "lib-beta"], "users-project", {"a_note", "b_note"}),
    (["lib-beta"], "other-project", {"b_note"}),
]


@pytest.mark.parametrize("libraries,asking_project,expected_docs", _LIBRARY_SETS)
def test_real_tool_hits_carry_the_general_knowledge_origin(
        library, libraries, asking_project, expected_docs):
    library(libraries)
    out = _run_tool("curing period", asking_project)
    hits = out["result"]["results"]
    assert {h["document_id"] for h in hits} == expected_docs
    assert {h["origin"] for h in hits} == {"general_knowledge"}


@pytest.mark.parametrize("project_id,expected", [
    ("users-project", [("u_spec", "own")]),
    ("other-project", [("o_drawing", "own")]),
])
def test_a_user_project_search_still_reports_own(library, project_id, expected):
    library(["lib-alpha", "lib-beta"])
    rows = asyncio.run(doc_index.search_project_documents(project_id, "curing period", 5))
    assert [(r["document_id"], r["origin"]) for r in rows] == expected


@pytest.mark.parametrize("libraries,asking_project,expected_docs", _LIBRARY_SETS)
def test_sources_panel_and_answer_tier_follow_the_real_tool_result(
        library, libraries, asking_project, expected_docs):
    from app.agents.driver import sources
    from app.agents.runtime import _answer_tier

    library(libraries)
    out = _run_tool("formwork striking time", asking_project)
    trail = [{"role": "tool", "name": "search_general_knowledge", "content": json.dumps(out)}]
    panel = sources.panel(trail, [])
    assert len(panel) == len(expected_docs)
    assert {p["layer"] for p in panel} == {"general_knowledge"}
    assert all(p["layer_label"] != "Project document" for p in panel)
    assert _answer_tier(panel) == "general"


# A merged library chunk or a disclosed fallback keeps its own layer,
# whichever corpus was searched; only "own" (or no layer) takes the
# searched corpus.
@pytest.mark.parametrize("layer,searched,expected", [
    ("master_corpus", "own", "master_corpus"),
    ("general_knowledge", "own", "general_knowledge"),
    ("own", "general_knowledge", "general_knowledge"),
    (None, "general_knowledge", "general_knowledge"),
    ("own", "own", "own"),
    (None, "own", "own"),
])
def test_a_layer_set_by_the_retriever_is_kept(layer, searched, expected):
    assert doc_index._hit_origin(layer, searched) == expected
