"""The search tool reads its turn (S-R9, part 2).

1. The declared per-turn search cap holds per CALL. A batch of parallel
   searches past the cap is answered without searching; before this change
   the loops counted the calls and ran them all.
2. Chunks already handed this turn (the first-pass excerpts, then each
   search's rows) are not handed again by a later search.
3. A document the answer cites from the search tool's rows is recorded in
   Sources, as a cited pre-injected chunk is.

Every test builds its own chunks, rows and answers from invented words and
runs on two inputs. No corpus, project, document or value is named.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import Agent, _build_sources_from_audit
from app.core import doc_index
from app.core.rag.handed_set import SearchTurn
from tests._fixture_values import STUB_TOKEN_02


@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


class _Chunk:
    def __init__(self, chunk_id, doc_id, score, text, page=None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.score = score
        self.text = text
        self.page = page
        self.layer = "own"


@pytest.fixture
def corpus(monkeypatch):
    import app.core.rag.retriever as retr

    box = {"chunks": [], "asked_k": []}

    def _retrieve(query, project_id, k=5, **_kw):
        box["asked_k"].append(k)
        return list(box["chunks"]), 0

    monkeypatch.setattr(retr, "retrieve_with_filter", _retrieve)
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: f"{d}.pdf")
    monkeypatch.setattr(doc_index, "_load_index", lambda _pid: {"documents": []})
    monkeypatch.setattr(doc_index._projects, "get_document", lambda d: None)
    monkeypatch.setattr("app.core.rag.retrieval_worker.enabled", lambda: False)
    monkeypatch.setenv("SEARCH_CHUNKS_PER_DOC", "1")
    return box


# ── 0. the turn object ───────────────────────────────────────────────────


@pytest.mark.parametrize("cap", [1, 3])
def test_a_turn_admits_exactly_its_cap(cap):
    turn = SearchTurn(cap)
    admitted = [turn.admit() for _ in range(cap + 2)]
    assert admitted == [True] * cap + [False, False]
    assert turn.calls == cap and turn.spent


@pytest.mark.parametrize("ids", [["k1"], ["k1", "k2", "k3"]])
def test_a_turn_starts_with_the_pre_injected_chunks(ids):
    audit = {"chunks": [{"chunk_id": i, "doc_id": "d"} for i in ids] + [{"doc_id": "d"}]}
    turn = SearchTurn.from_audit(2, audit)
    assert turn.handed == set(ids)
    assert SearchTurn.from_audit(2, None).handed == set()


# ── 2. chunks already handed are not handed again ────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("handed", [["c0"], ["c0", "c1"]])
async def test_a_search_skips_chunks_already_handed(corpus, handed):
    corpus["chunks"] = [
        _Chunk(f"c{i}", f"d{i}", 0.9 - i / 10, f"qzword passage{i} body") for i in range(4)
    ]
    turn = SearchTurn(5, handed)
    rows = await doc_index.search_project_documents("p-t", "qzword", 5, turn=turn)
    got = [r["document_id"] for r in rows]
    assert got == [f"d{i}" for i in range(len(handed), 4)]
    assert turn.last_skipped == len(handed)
    # what this search handed joins the turn, so a repeat hands nothing new
    assert turn.handed == {f"c{i}" for i in range(4)}
    again = await doc_index.search_project_documents("p-t", "qzword", 5, turn=turn)
    assert again == [] and turn.last_skipped == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("n_handed", [0, 3])
async def test_skipping_reaches_that_many_candidates_deeper(corpus, n_handed):
    corpus["chunks"] = [_Chunk("c0", "d0", 0.9, "qzword")]
    base = SearchTurn(5)
    await doc_index.search_project_documents("p-t", "qzword", 5, turn=base)
    turn = SearchTurn(5, [f"x{i}" for i in range(n_handed)])
    await doc_index.search_project_documents("p-t", "qzword", 5, turn=turn)
    assert corpus["asked_k"][1] - corpus["asked_k"][0] == n_handed


@pytest.mark.asyncio
@pytest.mark.parametrize("handed", [[], ["c0"]])
async def test_the_trace_names_chunks_skipped_as_already_handed(corpus, handed):
    from app.core.doc_index import _search_project_documents_traced

    corpus["chunks"] = [_Chunk("c0", "d0", 0.9, "qzword a"), _Chunk("c1", "d1", 0.8, "qzword b")]
    _rows, trace = _search_project_documents_traced("p-t", "qzword", 5, frozenset(handed))
    dropped = [e["chunk_id"] for e in trace["not_handed"] if e["dropped"] == "already_handed"]
    assert dropped == handed
    assert [e["chunk_id"] for e in trace["chunks"]] == [c for c in ("c0", "c1") if c not in handed]


@pytest.mark.asyncio
@pytest.mark.parametrize("all_handed", [False, True])
async def test_the_tool_says_when_its_passages_were_all_handed_before(corpus, monkeypatch, all_handed):
    from app.agents.base.tools import handle_search_project_documents

    monkeypatch.setattr("app.agents.runtime._effective_tool_result_max_chars", lambda: 4000)
    corpus["chunks"] = [_Chunk("c0", "d0", 0.9, "qzword a"), _Chunk("c1", "d1", 0.8, "qzword b")]
    turn = SearchTurn(5, ["c0", "c1"] if all_handed else ["c0"])

    class _Call:
        name = "search_project_documents"
        project_id = "p-t"
        args = {"query": "qzword", "top_k": 5}
        search_turn = turn

    payload = (await handle_search_project_documents(_Call()))["result"]
    assert payload["already_handed"] == (2 if all_handed else 1)
    assert bool(payload["results"]) is (not all_handed)
    assert ("already handed" in payload.get("note", "")) is all_handed


# ── 1 + 2. both chat loops: cap per call, turn handed set ────────────────


def _agent() -> Agent:
    return Agent(
        name="project-assistant", description="pa",
        system_prompt="You answer from project documents.", allowed_blocks=[],
    )


def _inject(monkeypatch, chunk_ids):
    async def _is_ready(_pid):
        return True
    monkeypatch.setattr("app.agents.runtime._project_is_rag_ready_off_process", _is_ready)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.agents.runtime._project_has_non_rag_context", lambda pid, msg: False)

    def fake_rag_inject(**kwargs):
        return (
            {"role": "system", "content": "Relevant project context.\nqzexcerpt"},
            {"project_id": "proj_t", "chunks": [
                {"doc_id": "doc_pre", "chunk_index": i, "chunk_id": cid, "score": 0.5}
                for i, cid in enumerate(chunk_ids)
            ]},
        )
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr("app.core.projects.get_document", lambda did: {"original_name": "Pre.pdf"})


def _batch_then_answer(n_calls: int, answer: str, seen: list):
    """First model call asks for ``n_calls`` searches at once; the next one
    answers. ``seen`` records the tool messages the second call was given."""
    async def fake(messages, api_key, **kwargs):
        seen.append(list(messages))
        if len(seen) == 1:
            calls = [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "search_project_documents",
                             "arguments": json.dumps({"query": f"qzterm{i}"})},
            } for i in range(n_calls)]
            return {"status": "success", "choice": {
                "message": {"role": "assistant", "content": "", "tool_calls": calls},
                "finish_reason": "tool_calls"}}
        return {"status": "success", "choice": {
            "message": {"role": "assistant", "content": answer, "tool_calls": []},
            "finish_reason": "stop"}}
    return fake


def _counting_search(monkeypatch, ran: list, turns: list):
    async def fake(project_id, query, top_k=5, *, trace_path=None, turn=None):
        ran.append(query)
        turns.append(turn)
        return [{"document_id": f"doc_{len(ran)}", "filename": f"Row{len(ran)}.pdf",
                 "snippet": "qzrow text", "score": 0.6}]
    monkeypatch.setattr("app.core.doc_index.search_project_documents", fake)


async def _run(agent, path, question):
    if path == "stream":
        return [e async for e in agent.chat_stream(question, project_id="proj_t")]
    return await agent.chat(question, project_id="proj_t")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stream", "chat"])
@pytest.mark.parametrize("cap,batch", [(2, 3), (1, 2)])
async def test_a_parallel_batch_runs_no_more_searches_than_the_cap(monkeypatch, path, cap, batch):
    monkeypatch.setenv("AGENT_SEARCH_TOOL_CAP", str(cap))
    monkeypatch.setenv("MISSING_INPUT_FETCH", "0")
    _inject(monkeypatch, ["k_pre"])
    ran: list = []
    turns: list = []
    seen: list = []
    _counting_search(monkeypatch, ran, turns)
    agent = _agent()
    monkeypatch.setattr(agent, "_call_llm", _batch_then_answer(batch, "qzanswer stands.", seen))

    await _run(agent, path, "Find qzthing in the documents.")

    assert len(ran) == cap
    # every call the model made is answered; the ones past the cap say so
    tool_msgs = [m for m in seen[1] if m.get("role") == "tool"]
    assert len(tool_msgs) == batch
    capped = [m for m in tool_msgs if "search limit reached" in str(m.get("content"))]
    assert len(capped) == batch - cap
    # the searches that ran saw the turn, seeded with the pre-injected chunk
    assert all(t is turns[0] and "k_pre" in t.handed for t in turns)


# ── 3. a document cited from the search rows reaches Sources ─────────────


ROW_CASES = [
    ("Qzalpha-Schedule.pdf", "qzalpha bearing pads are type qzb with qzmm thickness"),
    ("Qzbeta_Notes_Rev2.pdf", "qzbeta sealant joints are spaced at qzn centres"),
]


def _search_result(rows):
    return [{"name": "search_project_documents", "ok": True, "result": {"results": rows}}]


def _row(doc_id, name, snippet, pages=None):
    row = {"document_id": doc_id, "filename": name, "snippet": snippet,
           "score": 0.62, "origin": "own"}
    if pages:
        row["pages"] = pages
    return row


@pytest.mark.parametrize("name,snippet", ROW_CASES)
def test_a_cited_search_row_is_a_source(name, snippet):
    other = _row("d_other", "Unrelated-Register.pdf", "qzother words entirely elsewhere here")
    tool_results = _search_result([_row("d_cited", name, snippet, pages=[7]), other])
    answer = f"The answer is stated in the documents [source: {name}]."
    sources = _build_sources_from_audit({"project_id": "p", "chunks": []}, answer, tool_results)
    assert [s["doc_id"] for s in sources] == ["d_cited"]
    assert sources[0]["page_or_section"] == "p. 7"
    assert sources[0]["layer"] == "own"


@pytest.mark.parametrize("name,snippet", ROW_CASES)
def test_a_quoted_search_row_is_a_source_and_an_uncited_one_is_not(name, snippet):
    tool_results = _search_result([_row("d_q", name, snippet)])
    quoted = f"The documents say: {snippet}."
    unrelated = "The excerpts cover a different subject altogether."
    audit = {"project_id": "p", "chunks": []}
    assert [s["doc_id"] for s in _build_sources_from_audit(audit, quoted, tool_results)] == ["d_q"]
    assert _build_sources_from_audit(audit, unrelated, tool_results) == []


@pytest.mark.parametrize("name,snippet", ROW_CASES)
def test_a_decline_naming_a_search_row_lists_no_source(name, snippet):
    tool_results = _search_result([_row("d_x", name, snippet)])
    decline = (f"I searched {name} but the documents do not state what was asked, "
               "so I cannot confirm it.")
    assert _build_sources_from_audit({"project_id": "p", "chunks": []}, decline, tool_results) == []


@pytest.mark.parametrize("name,snippet", ROW_CASES)
def test_a_cited_search_row_stands_beside_cited_injected_chunks(name, snippet):
    audit = {"project_id": "p", "chunks": [
        {"doc_id": "d_pre", "chunk_index": 2, "chunk_id": "p:d_pre:2", "score": 0.8,
         "text": "qzpre text", "layer": "own"},
    ]}
    tool_results = _search_result([_row("d_row", name, snippet)])
    answer = f"One part [source: d_pre, chunk 2] and another [source: {name}]."
    got = {s["doc_id"] for s in _build_sources_from_audit(audit, answer, tool_results)}
    assert got == {"d_pre", "d_row"}
