"""Evidence-plan generation (item 2): the prompt and its parse.

The answer agent writes a short evidence plan before answering a
project-figure question: which documents / sections / rows would answer it.
``EVIDENCE_PLAN_PROMPT`` is the instruction; ``parse_generated_plan`` turns
the model's reply into an ``EvidencePlan`` the fetch loop can consume. The
prompt and the parse are pure and unit-testable; the model call itself is a
thin runtime wrapper.
"""
from __future__ import annotations

from typing import Any

from app.core.rag.evidence_plan import EvidencePlan, parse_evidence_plan

#: Instruction the answer agent gets to write its evidence plan. It asks for
#: concrete items (documents, sections, rows), never figures: the loop uses
#: the plan only to decide WHAT to fetch — computation and citing stay typed.
EVIDENCE_PLAN_PROMPT = (
    "Before answering, write a short evidence plan: one line per document, "
    "section or row that would answer this question, in the order you would "
    "look them up. Name the work items or clauses (e.g. the priced BOQ items "
    "for the quantities asked, the Contract Data section for a rate), not the "
    "figures. Do not compute anything. List each item on its own line."
)


def evidence_plan_request(question: str) -> str:
    """The plan call's user message: the instruction, then the question."""
    return EVIDENCE_PLAN_PROMPT + "\n\nQuestion: " + (question or "").strip()


def extract_plan_text(response: dict[str, Any]) -> str:
    """The model's plan text out of a ``_call_llm``-style response dict.

    Accepts the ``{"choice": {"message": {"content": ...}}}`` shape the agent
    returns, and returns the content string (empty when it is absent).
    """
    if not isinstance(response, dict):
        return ""
    choice = response.get("choice") or response.get("choices")
    if isinstance(choice, list) and choice:
        choice = choice[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def parse_generated_plan(response: dict[str, Any]) -> EvidencePlan:
    """Parse a generated plan out of a model response."""
    return parse_evidence_plan(extract_plan_text(response))
