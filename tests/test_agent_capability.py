"""Agent capability / self-introspection short-circuit.

Questions ABOUT the agent (its tools, abilities, accessible documents) must be
answered from the REAL tool roster + REAL document list, NOT from retrieved
document text. Regression guard for the live bug where "what tools do you have?"
returned construction power-tool specs (abrasive saw, diamond-core drill) from
the RAG corpus instead of the agent's actual software tools.
"""

from __future__ import annotations

import pytest

from app.agents.runtime import _is_capability_request, _build_capability_answer


class _StubAgent:
    name = "project-assistant"

    def tool_definitions(self, project_id=None):
        return [
            {"type": "function", "function": {
                "name": "boq_processor",
                "description": "Extract a Bill of Quantities from a file. Returns line items.",
            }},
            {"name": "drawing_qto", "description": "Quantity takeoff from a drawing (DXF/PDF)."},
        ]


@pytest.mark.parametrize("q", [
    "what tools do you have?",
    "What tools can you use?",
    "what tools you can use on the project documents?",
    "what can you do",
    "who are you",
    "what are your capabilities?",
    "what documents can you access",
    "list your tools",
    "what can you work on in this project",
])
def test_capability_questions_detected(q):
    assert _is_capability_request(q) is True


@pytest.mark.parametrize("q", [
    "what is the OSHA guardrail top rail height?",
    "what does the spec say about tools",
    "what tools are required per the method statement",
    "generate a BOQ from the demolition file",
    "summarize the construction schedule",
    "as per the drawing, what tools install the pipe",
])
def test_document_questions_not_misclassified(q):
    assert _is_capability_request(q) is False


def test_answer_lists_real_tools_not_document_text():
    ans = _build_capability_answer(_StubAgent(), project_id=None)
    assert "boq_processor" in ans
    assert "drawing_qto" in ans
    assert "project-assistant" in ans
    # The bug's signature: physical construction tools pulled from documents.
    assert "abrasive saw" not in ans.lower()
    assert "diamond-core" not in ans.lower()
