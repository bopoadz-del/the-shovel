"""Every retrieved chunk that was NOT handed carries the rule that dropped it.

The retrieval trace listed the chunks left out, but not why, so a failing row
needed a re-run of the selection by hand to tell a duplicate from a budget
cut from a score tie. Each ``not_handed`` entry now carries ``dropped_by``:
``duplicate``, ``family_share``, ``char_budget`` (the handed set),
``noise_filter``, ``kind_tie_yield``, ``score_tie_order``, ``token_budget``
(the token cap), ``below_threshold`` (nothing handed), else ``unexplained``.

Every test builds its own project, documents and chunks. No real document,
project, question or value is named.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core.rag import handed_set
from app.core.rag.inject import (
    _emit_retrieval_trace,
    apply_token_cap,
    not_handed_reasons,
)
from app.core.rag.vector_store import Chunk


def _chunk(doc: str, idx: int, score: float, text: str = "") -> Chunk:
    c = Chunk(chunk_id=f"p:{doc}:{idx}", project_id="p", doc_id=doc,
              chunk_index=idx, text=text or f"invented clause {doc} {idx} " * 6,
              score=score)
    c.source_name = f"invented-{doc}.pdf"
    return c


def _reasons(pool, *, k: int, query: str = "what grade is the kerb backing"):
    """The same three steps rag_inject runs, in its order."""
    handed_pool, _ = handed_set.select_handed(pool, k=k)
    cap: dict = {}
    kept, _ = apply_token_cap(handed_pool, query=query, reasons=cap)
    return kept, not_handed_reasons(pool, handed_pool, kept, cap, k=k)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    from app.core import projects, users
    users.init_db()
    projects.init_db()
    users.ensure_user_exists("u-drop")
    return projects.create_project("Synthetic Drop Works", user_id="u-drop")["id"]


def _stored(project_id: str, name: str) -> str:
    from app.core import projects
    return projects.add_document(project_id, original_name=name, stored_as=name,
                                 file_path=f"/tmp/{name}", size=1)["id"]


# --- the handed set ------------------------------------------------------------

@pytest.mark.parametrize("doc,body", [("alpha", "invented lintel clause " * 5),
                                      ("beta", "invented sealant clause " * 7)])
def test_the_same_text_twice_in_one_document_is_a_duplicate(project, doc, body):
    first, again = _chunk(doc, 0, 0.8, body), _chunk(doc, 1, 0.7, body)
    kept, why = _reasons([first, again], k=4)
    assert [c.chunk_id for c in kept] == [first.chunk_id]
    assert why == {again.chunk_id: "duplicate"}


@pytest.mark.parametrize("top,low", [(0.9, 0.5), (0.7, 0.3)])
def test_a_chunk_past_the_character_budget_is_char_budget(
        project, monkeypatch, top, low):
    a, b = _chunk("alpha", 0, top), _chunk("beta", 0, low)
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a.text) + 5))
    kept, why = _reasons([a, b], k=4)
    assert [c.chunk_id for c in kept] == [a.chunk_id]
    assert why == {b.chunk_id: "char_budget"}


@pytest.mark.parametrize("scores", [(0.90, 0.85, 0.80), (0.60, 0.58, 0.50)])
def test_a_chunk_deferred_behind_another_document_is_family_share(
        project, monkeypatch, scores):
    a0, a1, b0 = (_chunk("alpha", 0, scores[0]), _chunk("alpha", 1, scores[1]),
                  _chunk("beta", 0, scores[2]))
    monkeypatch.setenv("RAG_HANDED_CHAR_BUDGET", str(len(a0.text) * 2 + 5))
    kept, why = _reasons([a0, a1, b0], k=2)
    assert [c.chunk_id for c in kept] == [a0.chunk_id, b0.chunk_id]
    assert why == {a1.chunk_id: "family_share"}


# --- the token cap -------------------------------------------------------------

@pytest.mark.parametrize("top,low", [(0.9, 0.5), (0.62, 0.61)])
def test_a_lower_score_past_the_token_cap_is_token_budget(
        project, monkeypatch, top, low):
    a, b = _chunk("alpha", 0, top), _chunk("beta", 0, low)
    monkeypatch.setenv("MAX_RAG_TOKENS", str(len(a.text) // 4 + 2))
    cap: dict = {}
    kept, _ = apply_token_cap([a, b], query="what grade", reasons=cap)
    assert [c.chunk_id for c in kept] == [a.chunk_id]
    assert cap == {b.chunk_id: "token_budget"}


@pytest.mark.parametrize("first_id,second_id", [("no-row-a", "no-row-b"),
                                                ("gone-1", "gone-2")])
def test_a_tie_decided_by_retrieval_order_is_score_tie_order(
        project, monkeypatch, first_id, second_id):
    a, b = _chunk(first_id, 0, 0.5), _chunk(second_id, 0, 0.5)
    monkeypatch.setenv("MAX_RAG_TOKENS", str(len(a.text) // 4 + 2))
    cap: dict = {}
    kept, _ = apply_token_cap([a, b], query="what grade", reasons=cap)
    assert [c.chunk_id for c in kept] == [a.chunk_id]
    assert cap == {b.chunk_id: "score_tie_order"}


@pytest.mark.parametrize("progress_name,other_name", [
    ("pour sequence programme rev 2.xer", "kerb detail drawing sheet 4.pdf"),
    ("site weekly report 12.pdf", "finishes specification section 9.pdf"),
])
def test_a_tie_the_kind_rule_gave_away_is_kind_tie_yield(
        project, monkeypatch, progress_name, other_name):
    p = _chunk(_stored(project, progress_name), 0, 0.61)
    o = _chunk(_stored(project, other_name), 3, 0.61)
    monkeypatch.setenv("MAX_RAG_TOKENS", str(len(p.text) // 4 + 2))
    cap: dict = {}
    kept, _ = apply_token_cap([p, o], query="what grade is the kerb backing",
                              reasons=cap)
    assert [c.chunk_id for c in kept] == [o.chunk_id]
    assert cap == {p.chunk_id: "kind_tie_yield"}


@pytest.mark.parametrize("top,low", [(0.9, 0.5), (0.7, 0.3)])
def test_a_chunk_that_was_handed_has_no_reason(project, top, low):
    a, b = _chunk("alpha", 0, top), _chunk("beta", 0, low)
    kept, why = _reasons([a, b], k=4)
    assert {c.chunk_id for c in kept} == {a.chunk_id, b.chunk_id}
    assert why == {}


# --- the trace line ------------------------------------------------------------

def _trace(caplog, *, retrieved, handed, threshold_fired, drop_reasons):
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        _emit_retrieval_trace(
            query_label="user_message", project_id=None, conversation_id=None,
            agent_name="t", retrieved=retrieved, handed=handed,
            threshold_fired=threshold_fired, identifier_miss=False,
            top_score=0.5, drop_reasons=drop_reasons,
        )
    line = next(r.getMessage() for r in caplog.records
                if "rag_retrieval_trace" in r.getMessage())
    return json.loads(line.split(" ", 1)[1])


@pytest.mark.parametrize("code", ["char_budget", "kind_tie_yield"])
def test_the_trace_names_the_rule_for_each_chunk_left_out(
        project, caplog, code):
    a, b = _chunk("alpha", 0, 0.9), _chunk("beta", 0, 0.5)
    line = _trace(caplog, retrieved=[a, b], handed=[a], threshold_fired=False,
                  drop_reasons={b.chunk_id: code})
    assert [(e["chunk_id"], e["dropped_by"]) for e in line["not_handed"]] == [
        (b.chunk_id, code)]
    assert "dropped_by" not in line["chunks"][0]


@pytest.mark.parametrize("fired,expected", [(True, "below_threshold"),
                                            (False, "unexplained")])
def test_a_chunk_with_no_recorded_rule_is_never_left_blank(
        project, caplog, fired, expected):
    a, b = _chunk("alpha", 0, 0.9), _chunk("beta", 0, 0.5)
    line = _trace(caplog, retrieved=[a, b], handed=[], threshold_fired=fired,
                  drop_reasons=None)
    assert [e["dropped_by"] for e in line["not_handed"]] == [expected, expected]
