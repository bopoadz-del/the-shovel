"""Tests for the project reasoner's document-excerpt injection (Fix #2).

The reasoner now calls ``doc_index.search_project_documents(project_id, request)``
before its UNDERSTAND+PLAN call and folds the top-k snippets into the prompt.
These tests verify:

1. The prompt builder includes/excludes the excerpts section correctly.
2. The reasoner calls search_project_documents with the right args.
3. Absent project_id → no document lookup, no excerpts in prompt.
4. doc_index errors degrade silently — the reasoner still produces a plan.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.blocks.project_reasoner import ProjectReasonerBlock
from app.prompts.reasoner_system import build_reasoner_prompt
from app.schemas.project_session import ProjectSession


# ── prompt builder ────────────────────────────────────────────────────────


def test_prompt_omits_excerpts_section_when_none_passed():
    session = ProjectSession.new("s1")
    out = build_reasoner_prompt(session, "What is the total cost?")
    assert "RELEVANT DOCUMENT EXCERPTS" not in out


def test_prompt_omits_excerpts_section_when_empty_list():
    session = ProjectSession.new("s1")
    out = build_reasoner_prompt(session, "What is the total cost?", [])
    assert "RELEVANT DOCUMENT EXCERPTS" not in out


def test_prompt_includes_excerpts_section_when_provided():
    session = ProjectSession.new("s1")
    excerpts = [
        {"document_id": "d1", "filename": "boq.xlsx",
         "snippet": "Concrete: 1200 m3 at 150 USD/m3", "score": 0.91},
        {"document_id": "d2", "filename": "spec.pdf",
         "snippet": "All concrete shall meet ACI 318.", "score": 0.65},
    ]
    out = build_reasoner_prompt(session, "What is the total cost?", excerpts)
    assert "RELEVANT DOCUMENT EXCERPTS" in out
    assert "boq.xlsx" in out
    assert "Concrete: 1200 m3" in out
    assert "spec.pdf" in out


def test_prompt_truncates_oversized_snippet():
    session = ProjectSession.new("s1")
    huge = "x" * 5000
    out = build_reasoner_prompt(
        session, "anything",
        [{"filename": "huge.txt", "snippet": huge}],
    )
    # Should be truncated to keep the prompt bounded; far less than the raw
    # 5000-char snippet.
    assert "..." in out
    snippet_section = out.split("RELEVANT DOCUMENT EXCERPTS")[1]
    assert len(snippet_section) < 2000


def test_prompt_skips_excerpts_with_empty_snippet():
    session = ProjectSession.new("s1")
    out = build_reasoner_prompt(
        session, "anything",
        [{"filename": "blank.pdf", "snippet": "   "}],
    )
    # All excerpts blank → no section at all.
    assert "RELEVANT DOCUMENT EXCERPTS" not in out


# ── reasoner.process() integration ────────────────────────────────────────


@pytest.mark.asyncio
async def test_reasoner_looks_up_project_docs_when_project_id_given(monkeypatch):
    """When project_id is provided, the reasoner must consult doc_index."""
    block = ProjectReasonerBlock()
    session = ProjectSession.new("session-abc", user_id="u1")

    captured: dict = {}

    async def fake_search(project_id, query, top_k=5):
        captured["project_id"] = project_id
        captured["query"] = query
        captured["top_k"] = top_k
        return [
            {"document_id": "d1", "filename": "boq.xlsx",
             "snippet": "Total: USD 457,000,000", "score": 0.95},
        ]

    captured_prompts: list[str] = []

    async def fake_call_llm(self, prompt):
        captured_prompts.append(prompt)
        if len(captured_prompts) == 1:
            return '{"understanding": "user asks total cost",' \
                   ' "steps": []}'
        return "The total is USD 457M based on the BOQ."

    monkeypatch.setattr(
        "app.core.doc_index.search_project_documents", fake_search
    )
    monkeypatch.setattr(
        ProjectReasonerBlock, "_call_llm", fake_call_llm
    )

    result = await block.process({
        "request": "what is the total cost?",
        "session": session,
        "project_id": "session-abc",
    })

    assert result["status"] in ("success", "partial")
    assert captured["project_id"] == "session-abc"
    assert "total cost" in captured["query"].lower()
    # The PLAN prompt (first LLM call) should now contain the excerpt section.
    assert "RELEVANT DOCUMENT EXCERPTS" in captured_prompts[0]
    assert "boq.xlsx" in captured_prompts[0]


@pytest.mark.asyncio
async def test_reasoner_skips_doc_lookup_when_no_project_id(monkeypatch):
    block = ProjectReasonerBlock()
    session = ProjectSession.new("session-xyz", user_id="u1")

    search_called = AsyncMock()
    monkeypatch.setattr(
        "app.core.doc_index.search_project_documents", search_called
    )

    captured_prompts: list[str] = []

    async def fake_call_llm(self, prompt):
        captured_prompts.append(prompt)
        if len(captured_prompts) == 1:
            return '{"understanding": "no docs", "steps": []}'
        return "answer"

    monkeypatch.setattr(
        ProjectReasonerBlock, "_call_llm", fake_call_llm
    )

    await block.process({
        "request": "hi there",
        "session": session,
    })

    search_called.assert_not_called()
    assert "RELEVANT DOCUMENT EXCERPTS" not in captured_prompts[0]


@pytest.mark.asyncio
async def test_reasoner_degrades_silently_when_doc_lookup_errors(monkeypatch):
    """A doc_index failure must not break the reasoner — fall back to
    structured-state-only planning."""
    block = ProjectReasonerBlock()
    session = ProjectSession.new("session-err", user_id="u1")

    async def boom(*args, **kwargs):
        raise RuntimeError("doc_index unavailable")

    captured_prompts: list[str] = []

    async def fake_call_llm(self, prompt):
        captured_prompts.append(prompt)
        if len(captured_prompts) == 1:
            return '{"understanding": "still works", "steps": []}'
        return "answer"

    monkeypatch.setattr(
        "app.core.doc_index.search_project_documents", boom
    )
    monkeypatch.setattr(
        ProjectReasonerBlock, "_call_llm", fake_call_llm
    )

    result = await block.process({
        "request": "anything",
        "session": session,
        "project_id": "session-err",
    })

    # No crash, plan still built, just no excerpts injected.
    assert result["status"] in ("success", "partial")
    assert "RELEVANT DOCUMENT EXCERPTS" not in captured_prompts[0]
