"""A question carrying its own numbers is arithmetic, not a document lookup.

LIVE 2026-08-03, project-assistant on the master corpus:

    "Calculate the concrete volume for a raft 25m x 18m x 1.2m thick and
     the number of 6m3 truck loads"

    -> route None (below_routing_gate), construction_calc never invoked:
       "I cannot find that specific information in the provided reference
        context ... they do not contain a raft measuring 25 m x 18 m x 1.2 m
        thick, nor any reference to 6 m3 truck loads. Consequently, I cannot
        calculate the requested concrete volume or number of truck loads."

25 x 18 x 1.2 = 540 m3; 540 / 6 = 90 loads. Nothing needed retrieving. The
grounding discipline overrode arithmetic and refused a question whose every
input was supplied in the question itself.

`concrete_volume` and `guardrail_top_rail_height` both EXIST (76 calculators
are registered) — _INTENT_TOOL_MAP's ~22 phrases simply never reach them.

The guard detects the SHAPE instead of chasing 76 keyword spellings: a calc
verb plus >=2 measurement-bearing numbers. Document lookups have neither.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _forced_specific_tool,
    _looks_like_self_contained_calculation,
)

AVAILABLE = {"construction_calc", "search_project_documents"}


def _tail(text: str):
    return [{"role": "user", "content": text}]


# ── the live failures ────────────────────────────────────────────────────────

def test_the_live_concrete_volume_question_forces_the_calculator():
    q = ("calculate the concrete volume for a raft 25m x 18m x 1.2m thick "
         "and the number of 6m3 truck loads")
    assert _looks_like_self_contained_calculation(q)
    assert _forced_specific_tool(_tail(q), AVAILABLE) == "construction_calc"


@pytest.mark.parametrize(
    "q",
    [
        "calculate the excavation volume for a pit 12m x 8m x 3.5m deep",
        "how many 25 kg bags of cement for 40 m3 of concrete",
        "compute the rebar weight for 120 m of 16mm bar",
        "how much backfill for a trench 45 m long 0.8 m wide 1.2 m deep",
    ],
)
def test_other_self_contained_calculations_are_caught(q):
    assert _forced_specific_tool(_tail(q), AVAILABLE) == "construction_calc"


# ── document lookups must be UNAFFECTED ──────────────────────────────────────

@pytest.mark.parametrize(
    "q",
    [
        "what is the concrete volume in the BOQ",          # lookup, no verb/dims
        "what is the backfilling specification",
        "list the documents in this project",
        "what is the rebar specification",
        # a calc verb but NO self-supplied dimensions -> still a lookup
        "calculate the total value of the bill of quantities",
        # dimensions but NO calc verb -> prose about a drawing
        "the raft is 25m x 18m x 1.2m thick per the drawing",
    ],
)
def test_document_questions_are_not_hijacked(q):
    assert _forced_specific_tool(_tail(q), AVAILABLE) is None


def test_leftover_l6_bank_volume_is_self_contained():
    """Live leftover L6 had no 'calculate' verb — 'bank volume' + L×W×D."""
    q = (
        "Stay on smart-orchestrator. Bank volume of a rectangular trench "
        "14.5 m long by 3.2 m wide by 1.75 m deep. Start with the "
        "smart_orchestrator tool then a calculator."
    )
    assert _looks_like_self_contained_calculation(q)
    assert _forced_specific_tool(_tail(q), AVAILABLE) == "construction_calc"


@pytest.mark.asyncio
async def test_leftover_l6_predispatch_returns_none_before_listing_docs(monkeypatch):
    """Line 562: self-contained volume skips file predispatch entirely."""
    from app.agents import runtime

    called = []

    def _boom(pid):
        called.append(pid)
        raise AssertionError("list_documents must not run for leftover L6")

    monkeypatch.setattr("app.core.projects.list_documents", _boom, raising=False)
    agent = runtime.Agent(
        name="qa", description="", system_prompt="x",
        allowed_blocks=["drawing_qto", "construction"],
    )
    q = (
        "Stay on smart-orchestrator. Bank volume of a rectangular trench "
        "14.5 m long by 3.2 m wide by 1.75 m deep. Start with the "
        "smart_orchestrator tool then a calculator."
    )
    rec = await runtime._predispatch_file_tool(
        agent, [{"role": "user", "content": q}], "p1",
    )
    assert rec is None
    assert called == []


def test_one_dimension_is_not_enough():
    """A single number is usually a reference, not a calculation input."""
    assert not _looks_like_self_contained_calculation("calculate the cost of 5 m3")


# ── mechanics ────────────────────────────────────────────────────────────────

def test_kill_switch_disables_it(monkeypatch):
    monkeypatch.setenv("FORCE_CALC_ON_DIMENSIONS", "0")
    q = "calculate the concrete volume for a raft 25m x 18m x 1.2m thick"
    assert not _looks_like_self_contained_calculation(q)
    assert _forced_specific_tool(_tail(q), AVAILABLE) is None


def test_not_forced_when_the_calculator_is_unavailable():
    """Never force a tool the turn does not actually have."""
    q = "calculate the concrete volume for a raft 25m x 18m x 1.2m thick"
    assert _forced_specific_tool(_tail(q), {"search_project_documents"}) is None


def test_explicit_keyword_phrases_still_win():
    """The existing map must keep priority over the shape heuristic."""
    q = "calculate the interim payment for work valued at 900,000 with 10 percent retention"
    assert _forced_specific_tool(_tail(q), AVAILABLE) == "construction_calc"


def test_only_the_user_tail_is_inspected():
    msgs = [
        {"role": "user", "content": "calculate 25m x 18m x 1.2m volume"},
        {"role": "assistant", "content": "540 m3"},
    ]
    assert _forced_specific_tool(msgs, AVAILABLE) is None
