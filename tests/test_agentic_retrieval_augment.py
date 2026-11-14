"""Tests for the runtime seam that folds the #163 evidence loop into a turn.

The three entry points are exercised directly so a mutant of any one of them
(a constant return, a dropped guard) fails a test that observes the real
behaviour: a missing/empty first pass is a no-op, a priced figure already in
the first pass skips augmentation, and an uncovered figure merges the loop's
evidence back into the RAG context.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agents import runtime
from app.agents.runtime import (
    _agentic_figure_retrieval_augment,
    _agentic_figure_retrieval_excluded,
    _agentic_figure_retrieval_plan_caller,
    _stash_dataset_outcome,
)
from app.core.rag.plan_generation import EVIDENCE_PLAN_PROMPT, evidence_plan_request


class _FakeAgent:
    def __init__(self, plan_text="bulk excavation, blinding"):
        self.plan_text = plan_text
        self.calls = 0

    async def _call_llm(self, msgs, api_key, project_id=None, with_tools=None, user_id=None):
        self.calls += 1
        return {"choice": {"message": {"content": self.plan_text}}}


@pytest.mark.parametrize("rag", [None, {"content": ""}])
async def test_augment_noop_when_rag_missing_or_empty(rag):
    agent = _FakeAgent()
    result = await _agentic_figure_retrieval_augment(
        agent, "what is the cover", "p", "k", "u", rag,
    )
    assert result == rag
    assert agent.calls == 0


@pytest.mark.parametrize("ask", [
    "what is the quantity and amount for D599.5",
    "what is the amount for D549.2",
])
async def test_augment_skips_when_priced_row_already_in_first_pass(monkeypatch, ask):
    import app.core.rag.retriever as retriever

    monkeypatch.setattr(
        retriever, "compose_priced_boq_row",
        lambda q, e: {"qty": "1", "rate": "2.0", "amount": "2"},
    )
    rag = {"content": "D599.5 breaking out 340904 m2 31.00 10568024"}
    agent = _FakeAgent()
    result = await _agentic_figure_retrieval_augment(
        agent, ask, "p", "k", "u", rag,
    )
    assert result == rag
    assert agent.calls == 0


@pytest.mark.parametrize("ask", [
    "what is the part summary total for page x/1/1 of the roads bill",
    "what does page y/2/4 of the drainage bill total",
])
async def test_augment_skips_when_part_summary_total_already_in_first_pass(
    monkeypatch, ask,
):
    from app.core.rag import retriever

    monkeypatch.setattr(retriever, "compose_priced_boq_row", lambda q, e: None)
    monkeypatch.setattr(retriever, "query_asks_for_part_summary_total", lambda q: True)
    monkeypatch.setattr(
        retriever, "compose_part_summary_total",
        lambda q, e: {"page": "p", "total": "1,234.00"},
    )
    rag = {"content": "PART SUMMARY page total 1,234.00"}
    agent = _FakeAgent()
    result = await _agentic_figure_retrieval_augment(
        agent, ask, "p", "k", "u", rag,
    )
    assert result == rag
    assert agent.calls == 0


@pytest.mark.parametrize("ask", [
    "what is the combined total of pages x/1/1 and x/1/2",
    "what do pages y/2/1 and y/2/2 total together",
])
async def test_augment_skips_when_combined_part_summary_already_in_first_pass(
    monkeypatch, ask,
):
    from app.core.rag import retriever

    monkeypatch.setattr(retriever, "compose_priced_boq_row", lambda q, e: None)
    monkeypatch.setattr(retriever, "query_asks_for_part_summary_total", lambda q: False)
    monkeypatch.setattr(retriever, "query_asks_combined_part_summary", lambda q: True)
    monkeypatch.setattr(
        retriever, "compose_combined_part_summary_total",
        lambda q, e: {"pages": ["a", "b"], "total": "2,000.00"},
    )
    rag = {"content": "PART SUMMARY 1,000.00 ... PART SUMMARY 1,000.00"}
    agent = _FakeAgent()
    result = await _agentic_figure_retrieval_augment(
        agent, ask, "p", "k", "u", rag,
    )
    assert result == rag
    assert agent.calls == 0


@pytest.mark.parametrize("asks_page,asks_combined", [(True, False), (False, True)])
def test_composer_is_consulted_only_for_its_ask_shape(
    monkeypatch, asks_page, asks_combined,
):
    # The ask shape gates each composer: an ask of neither shape does not
    # skip the loop even if a composer would parse something.
    from app.core.rag import retriever

    monkeypatch.setattr(retriever, "compose_priced_boq_row", lambda q, e: None)
    monkeypatch.setattr(retriever, "query_asks_for_part_summary_total", lambda q: asks_page)
    monkeypatch.setattr(retriever, "query_asks_combined_part_summary", lambda q: asks_combined)
    monkeypatch.setattr(retriever, "compose_part_summary_total", lambda q, e: {"t": "1"})
    monkeypatch.setattr(
        retriever, "compose_combined_part_summary_total", lambda q, e: {"t": "2"},
    )
    assert runtime._first_pass_composes_figure("what is the total", "x") is True
    monkeypatch.setattr(retriever, "query_asks_for_part_summary_total", lambda q: False)
    monkeypatch.setattr(retriever, "query_asks_combined_part_summary", lambda q: False)
    assert runtime._first_pass_composes_figure("what is the total", "x") is False


@pytest.mark.parametrize("ask", [
    "what is the standard rate for blockwork walls",
    "how much productivity should I allow for wall plaster",
])
async def test_augment_merges_recovered_evidence(monkeypatch, ask):
    from app.core.rag import route as rag_route
    import app.core.rag.retriever as retriever
    import app.core.rag.vector_store as vector_store

    monkeypatch.setattr(retriever, "compose_priced_boq_row", lambda q, e: None)

    plan_calls = []

    async def _spy_plan_caller(agent, question, api_key, project_id, user_id):
        plan_calls.append(question)
        return {"choice": {"message": {"content": "bulk excavation, blinding"}}}

    monkeypatch.setattr(runtime, "_agentic_figure_retrieval_plan_caller", _spy_plan_caller)

    async def _fake_augment(**kwargs):
        plan = await kwargs["plan_caller"](kwargs["question"])
        return kwargs["first_pass_evidence"] + "\n" + plan["choice"]["message"]["content"]

    monkeypatch.setattr(rag_route, "augment_figure_retrieval", _fake_augment)
    monkeypatch.setattr(
        vector_store, "get_store",
        lambda: SimpleNamespace(bm25_search=lambda pid, q, k=50: []),
    )

    rag = {"content": "bulk excavation only"}
    agent = _FakeAgent()
    result = await _agentic_figure_retrieval_augment(
        agent, ask, "p", "k", "u", rag,
    )
    assert result["content"] != rag["content"]
    assert "bulk excavation" in result["content"]
    assert plan_calls == [ask]


@pytest.mark.parametrize("flags,expected", [
    ((True, False, False, False), True),
    ((False, False, False, False), False),
])
def test_excluded_predicate(monkeypatch, flags, expected):
    f, e, c, s = flags
    monkeypatch.setattr(runtime, "_message_is_formula_style_ask", lambda _t: f)
    monkeypatch.setattr(runtime, "_asks_for_export", lambda _t: e)
    monkeypatch.setattr(runtime, "_looks_like_self_contained_calculation", lambda _t: c)
    monkeypatch.setattr(runtime, "_asks_self_coding", lambda _t: s)
    assert _agentic_figure_retrieval_excluded("anything") == expected


@pytest.mark.parametrize("question,plan_items", [
    ("what is the cover to reinforcement", ("cover row", "concrete row")),
    ("what is the compaction density", ("compaction row",)),
])
def test_stash_dataset_outcome_records_question_and_plan(question, plan_items):
    agent = SimpleNamespace()
    outcome = SimpleNamespace(plan_items=plan_items)
    _stash_dataset_outcome(agent, question, outcome)
    assert agent._dataset_turn == {
        "question": question,
        "evidence_plan": plan_items,
    }


@pytest.mark.parametrize("question", [
    "what is the cover to reinforcement",
    "what is the compaction density",
])
async def test_plan_caller_calls_llm_tool_free(question):
    seen = {}

    class _Agent:
        async def _call_llm(self, msgs, api_key, project_id=None, with_tools=None, user_id=None):
            seen["msgs"] = msgs
            seen["api_key"] = api_key
            seen["project_id"] = project_id
            seen["with_tools"] = with_tools
            seen["user_id"] = user_id
            return {"choice": {"message": {"content": "plan"}}}

    agent = _Agent()
    result = await _agentic_figure_retrieval_plan_caller(agent, question, "k", "p", "u")
    assert result == {"choice": {"message": {"content": "plan"}}}
    assert seen["with_tools"] is False
    # The plan call carries the plan instruction and the question, so the
    # model returns what to fetch, not an unretrieved answer.
    assert seen["msgs"] == [{"role": "user", "content": evidence_plan_request(question)}]
    assert question in seen["msgs"][0]["content"]
    assert EVIDENCE_PLAN_PROMPT in seen["msgs"][0]["content"]
