"""Arithmetic must not be forbidden by the strict-grounding directive.

LIVE 2026-08-03, verbatim from prod:

    "Calculate the concrete volume for a raft 25m x 18m x 1.2m thick and
     the number of 6m3 truck loads"

    -> "I cannot find that specific information in the provided reference
        context ... they do not contain a raft measuring 25 m x 18 m x 1.2 m
        thick, nor any reference to 6 m3 truck loads. Consequently, I cannot
        calculate the requested concrete volume or number of truck loads."

25 x 18 x 1.2 = 540 m3; 540 / 6 = 90 loads.

THE MODEL WAS OBEYING, NOT FAILING. `_apply_rag_context` classified the turn
as a lookup and appended:

    "Answer the question below using ONLY the reference context above ...
     If the context does not contain the answer, say you don't have it
     rather than inventing one."

The dimensions came from the USER, so they were never going to be in the
context — and the directive made refusing the correct behaviour.

tool_choice cannot rescue this: `_tool_choice_for` returns "auto" for kimi
(K2 rejects a specified tool outright), so the 76 deterministic calculators
can never be forced. Fixing the DIRECTIVE is provider-independent.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _apply_rag_context

CONTEXT = {"content": "[source: spec.pdf] The main tower raft is 4.0 m thick."}

LIVE_QUESTION = ("Calculate the concrete volume for a raft 25m x 18m x 1.2m "
                 "thick and the number of 6m3 truck loads")


def _apply(question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    assert _apply_rag_context(msgs, dict(CONTEXT)) is True
    return msgs[-1]["content"]


# ── the live failure ─────────────────────────────────────────────────────────

def test_calculation_does_not_get_the_only_use_context_directive():
    """The load-bearing assertion — this instruction caused the refusal."""
    out = _apply(LIVE_QUESTION)

    assert "using ONLY the reference context" not in out
    assert "say you don't have it" not in out


def test_calculation_is_told_not_to_refuse_on_absent_dimensions():
    out = _apply(LIVE_QUESTION)

    assert "CALCULATION" in out
    assert "Do NOT refuse" in out
    assert "they came from the user" in out


def test_the_users_question_survives_intact():
    """The directive must not swallow or paraphrase the request."""
    out = _apply(LIVE_QUESTION)
    assert LIVE_QUESTION in out


def test_project_facts_are_still_protected():
    """Permission to compute is NOT permission to invent project data."""
    out = _apply(LIVE_QUESTION)
    assert "never invent" in out
    assert "project-specific facts" in out


# ── the other two intents must be UNCHANGED ──────────────────────────────────

def test_plain_lookup_keeps_strict_grounding():
    out = _apply("what is the backfilling specification for soft ground")

    assert "using ONLY the reference context" in out
    assert "CALCULATION REQUEST" not in out


def test_generative_request_keeps_its_own_directive():
    out = _apply("produce a commissioning checklist for the HVAC systems")

    assert "REQUEST TO GENERATE" in out
    assert "CALCULATION REQUEST" not in out


@pytest.mark.parametrize(
    "q",
    [
        # calc verb but NO self-supplied dimensions -> still a lookup
        "calculate the total value of the bill of quantities",
        # dimensions but NO calc verb -> prose about a drawing
        "the raft is 25m x 18m x 1.2m thick per the drawing",
    ],
)
def test_near_miss_questions_stay_strictly_grounded(q):
    """The boundary matters: these must keep the cost-grounding path."""
    out = _apply(q)
    assert "using ONLY the reference context" in out


def test_kill_switch_restores_strict_grounding(monkeypatch):
    monkeypatch.setenv("FORCE_CALC_ON_DIMENSIONS", "0")
    out = _apply(LIVE_QUESTION)
    assert "using ONLY the reference context" in out


def test_no_context_is_still_a_noop():
    msgs = [{"role": "user", "content": LIVE_QUESTION}]
    assert _apply_rag_context(msgs, {"content": ""}) is False
    assert msgs[-1]["content"] == LIVE_QUESTION
