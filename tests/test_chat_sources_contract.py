"""Source contract tests for streaming and non-streaming chat.

Every successful RAG-backed chat answer must expose a structured ``sources``
array (doc_id, doc_name, page_or_section, score, confidence) in addition to
any inline citations.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import Agent
from tests._fixture_values import STUB_TOKEN_02



@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    """The name matchers read ``list_document_names``; these tests stub
    ``list_documents``. Route the former through whatever the latter is."""
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))

@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    """Agent.chat() guards on the provider api key *before* _call_llm is
    reached. These tests monkeypatch _call_llm so no network call ever
    happens, but the guard still fires when the key is unset (e.g. in CI,
    where conftest's load_dotenv finds no .env key). Pin the provider to
    deepseek so the guard checks DEEPSEEK_API_KEY regardless of a developer's
    LLM_PROVIDER, then satisfy it with a placeholder.

    Deliberately a per-file fixture, not in conftest.py: a global key would
    un-skip the live LLM acceptance tests (their skipif is evaluated at
    collection time) and make them run against the real API with a fake key.
    """
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


def _make_agent() -> Agent:
    return Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Cite sources.",
        allowed_blocks=[],
    )


def _patch_guardrail(monkeypatch):
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True
    )


def _fake_rag_audit():
    return {
        "project_id": "proj_a",
        "chunks": [
            {
                "doc_id": "doc_1",
                "chunk_index": 1,
                "chunk_id": "proj_a:doc_1:1",
                "score": 0.82,
            }
        ],
    }


@pytest.mark.asyncio
async def test_chat_returns_structured_sources(monkeypatch):
    agent = _make_agent()

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": "Answer from source [source: SomeDoc.pdf, chunk 1].",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return (
            {"role": "system", "content": "Relevant project context."},
            _fake_rag_audit(),
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr(
        "app.core.projects.get_document", lambda did: {"original_name": "SomeDoc.pdf"}
    )
    _patch_guardrail(monkeypatch)

    result = await agent.chat("What does the spec say?", project_id="proj_a")

    assert result["status"] == "success"
    assert "sources" in result
    assert len(result["sources"]) == 1
    src = result["sources"][0]
    assert src["doc_id"] == "doc_1"
    assert src["doc_name"] == "SomeDoc.pdf"
    assert src["page_or_section"] == ""
    assert src["score"] == 0.82
    assert src["confidence"] == "High"
    assert src["project_id"] == "proj_a"


@pytest.mark.asyncio
async def test_chat_stream_end_event_includes_sources(monkeypatch):
    agent = _make_agent()

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": "Answer from source [source: SomeDoc.pdf, chunk 1].",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return (
            {"role": "system", "content": "Relevant project context."},
            _fake_rag_audit(),
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr(
        "app.core.projects.get_document", lambda did: {"original_name": "SomeDoc.pdf"}
    )
    _patch_guardrail(monkeypatch)

    events = []
    async for event in agent.chat_stream("What does the spec say?", project_id="proj_a"):
        events.append(event)

    end_events = [e for e in events if e["type"] == "end"]
    assert len(end_events) == 1
    end = end_events[0]
    assert "sources" in end
    assert len(end["sources"]) == 1
    assert end["sources"][0]["doc_id"] == "doc_1"


@pytest.mark.asyncio
async def test_chat_source_labels_do_not_expose_raw_paths(monkeypatch):
    agent = _make_agent()

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": "See [source: X:\\Example Drive\\SomeDoc.pdf, chunk 1].",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return (
            {"role": "system", "content": "Relevant project context."},
            _fake_rag_audit(),
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr(
        "app.core.projects.get_document", lambda did: {"original_name": "SomeDoc.pdf"}
    )
    _patch_guardrail(monkeypatch)

    result = await agent.chat("What does the spec say?", project_id="proj_a")

    assert "X:\\Example Drive" not in result["answer"]
    assert "(Source: SomeDoc.pdf)" in result["answer"]
    assert "chunk" not in result["answer"]
    assert result["sources"][0]["doc_name"] == "SomeDoc.pdf"


@pytest.mark.asyncio
async def test_chat_returns_empty_sources_when_no_rag(monkeypatch):
    agent = _make_agent()

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {"content": "Plain answer.", "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return None, {"chunks": []}

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)

    result = await agent.chat("Hello?", project_id="proj_a")

    assert result["status"] == "success"
    assert result.get("sources") == []


# ── P0B source-contract hardening additions ──────────────────────────────────

from app.agents.runtime import (
    _answer_is_caveat,
    _sanitize_inline_paths,
)


def test_answer_is_caveat_detects_refusal_phrases():
    assert _answer_is_caveat("I could not locate any record for that.") is True
    assert _answer_is_caveat("I cannot confirm this from the sources.") is True
    assert _answer_is_caveat(
        "I searched the project's document repository for XKCD-99999, "
        "but none of the indexed files contain that identifier."
    ) is True
    assert _answer_is_caveat("The design review procedure is as follows.") is False


def test_sanitize_inline_paths_cleans_markdown_table_source_cell():
    raw = "| Source |\n| X:\\Example Drive\\XYZ-501.pdf |"
    cleaned = _sanitize_inline_paths(raw)
    assert "X:\\Example Drive" not in cleaned
    assert "XYZ-501.pdf" in cleaned


def test_build_sources_returns_empty_for_caveat(monkeypatch):
    audit = {
        "project_id": "proj_x",
        "chunks": [
            {"doc_id": "d1", "chunk_index": 0, "chunk_id": "c1", "score": 0.6},
        ],
    }
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "SomeDoc.pdf"},
    )
    from app.agents.runtime import _build_sources_from_audit
    out = _build_sources_from_audit(audit, "I could not locate any record for that.")
    assert out == []


@pytest.mark.parametrize("flag", ["identifier_miss", "threshold_fired"])
def test_build_sources_returns_empty_when_trust_gate_fires(flag, monkeypatch):
    audit = {
        "project_id": "proj_x",
        flag: True,
        "chunks": [
            {"doc_id": "d1", "chunk_index": 0, "chunk_id": "c1", "score": 0.6},
        ],
    }
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": "SomeDoc.pdf"},
    )
    from app.agents.runtime import _build_sources_from_audit
    out = _build_sources_from_audit(
        audit,
        "I searched but could not find anything specific.",
    )
    assert out == []


# ── Missing exact-reference fast-path tests ─────────────────────────────────

from app.agents.runtime import (
    _should_short_circuit_rag_miss,
    _build_missing_reference_answer,
)


def test_should_short_circuit_rag_miss_true_for_identifier_miss():
    audit = {"identifier_miss": True, "extracted_identifiers": ["vo ref 99"]}
    assert _should_short_circuit_rag_miss(audit, None) is True


def test_should_short_circuit_rag_miss_true_for_threshold_fired_with_identifiers():
    audit = {"threshold_fired": True, "extracted_identifiers": ["vo ref 99"]}
    assert _should_short_circuit_rag_miss(audit, None) is True


def test_should_short_circuit_rag_miss_false_for_generic_phrase_without_digit():
    # "contract value" matches a reference label but is not a specific lookup.
    audit = {"threshold_fired": True, "extracted_identifiers": ["contract value", "value"]}
    assert _should_short_circuit_rag_miss(audit, None) is False


def test_should_short_circuit_rag_miss_false_when_rag_context_exists():
    audit = {"identifier_miss": True, "extracted_identifiers": ["vo ref 99"]}
    assert _should_short_circuit_rag_miss(audit, {"role": "system", "content": "context"}) is False


def test_should_short_circuit_rag_miss_false_for_broad_question():
    audit = {"threshold_fired": True, "extracted_identifiers": []}
    assert _should_short_circuit_rag_miss(audit, None) is False


def test_should_short_circuit_rag_miss_false_for_self_coding_conversion():
    """Live 2026-08-20: pinned self-coding AED/m2 → USD/ft2 was refused as a
    missing document because `aed/m2` carries a digit."""
    audit = {"identifier_miss": True, "extracted_identifiers": ["aed/m2", "usd/ft2"]}
    msg = (
        "Stay on self-coding. No registered formula exists. Convert 1250 AED/m2 "
        "to USD/ft2 using 3.6725 AED = 1 USD and 1 m2 = 10.7639 ft2."
    )
    assert _should_short_circuit_rag_miss(audit, None, msg) is False
    # A real VO lookup still short-circuits when no calculation is asked.
    assert _should_short_circuit_rag_miss(
        {"identifier_miss": True, "extracted_identifiers": ["vo ref 99"]},
        None,
        "What is the status of VO Ref 99?",
    ) is True


@pytest.mark.asyncio
async def test_chat_short_circuits_missing_exact_reference(monkeypatch):
    """Absent exact reference skips model call and returns controlled not-found."""
    agent = _make_agent()

    call_count = {"n": 0}

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        call_count["n"] += 1
        return {
            "status": "success",
            "choice": {
                "message": {"content": "should not be called", "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return None, {
            "project_id": "proj_a",
            "identifier_miss": True,
            "threshold_fired": True,
            "extracted_identifiers": ["vo ref 999999"],
            "chunks": [],
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr(
        "app.core.projects.get_project",
        lambda pid, user_id=None, include_admin_approved=False: {"name": "Test Project"},
    )
    _patch_guardrail(monkeypatch)

    result = await agent.chat("What is the status of VO Ref 999999?", project_id="proj_a")

    assert call_count["n"] == 0
    assert result["status"] == "success"
    assert "could not confirm this reference" in result["answer"].lower()
    assert result["sources"] == []
    assert result["iterations"] == 0


@pytest.mark.asyncio
async def test_named_docx_is_not_rag_miss_short_circuited(monkeypatch):
    """Leftover L1: timestamped filename looks like an identifier, RAG
    misses, but fetch_document predispatch from disk must still run."""
    agent = _make_agent()
    call_count = {"n": 0}

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        call_count["n"] += 1
        blob = " ".join(str(m.get("content") or "") for m in messages)
        assert "WORD-SPEC-TOKEN-FENWICK" in blob
        return {
            "status": "success",
            "choice": {
                "message": {
                    "content": "Token: WORD-SPEC-TOKEN-FENWICK-20260819-081916",  # pragma: allowlist secret
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return None, {
            "project_id": "proj_a",
            "identifier_miss": True,
            "threshold_fired": True,
            "extracted_identifiers": ["fenwick_waterproofing_spec_20260819-081916.docx"],
            "chunks": [],
        }

    import app.core.projects as projects_mod
    monkeypatch.setattr(
        projects_mod, "list_documents",
        lambda pid: [{"original_name": "fenwick_waterproofing_spec_20260819-081916.docx"}],
        raising=False,
    )

    def fake_fetch(pid, doc_id, filename):
        return (
            {"text": "WORD-SPEC-TOKEN-FENWICK-20260819-081916", "truncated": False, "source": "extracted"},
            {"original_name": filename, "id": "d-docx"},
            None,
        )

    monkeypatch.setattr("app.agents.runtime._fetch_document_content", fake_fetch)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    _patch_guardrail(monkeypatch)

    result = await agent.chat(
        "Open fenwick_waterproofing_spec_20260819-081916.docx and quote the unique token.",
        project_id="proj_a",
    )
    assert call_count["n"] == 1
    assert "could not confirm this reference" not in result["answer"].lower()
    assert "WORD-SPEC-TOKEN-FENWICK-20260819-081916" in result["answer"]
    assert any(t.get("name") == "fetch_document" for t in (result.get("tool_calls") or []))


@pytest.mark.asyncio
async def test_chat_stream_short_circuits_missing_exact_reference(monkeypatch):
    """Streaming absent exact reference yields controlled not-found and empty sources."""
    agent = _make_agent()

    call_count = {"n": 0}

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        call_count["n"] += 1
        return {
            "status": "success",
            "choice": {
                "message": {"content": "should not be called", "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return None, {
            "project_id": "proj_a",
            "identifier_miss": True,
            "threshold_fired": True,
            "extracted_identifiers": ["rfi xkcd-99999"],
            "chunks": [],
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr(
        "app.core.projects.get_project",
        lambda pid, user_id=None, include_admin_approved=False: {"name": "Test Project"},
    )
    _patch_guardrail(monkeypatch)

    events = []
    async for event in agent.chat_stream("Find RFI XKCD-99999 details.", project_id="proj_a"):
        events.append(event)

    assert call_count["n"] == 0
    end_events = [e for e in events if e["type"] == "end"]
    assert len(end_events) == 1
    assert end_events[0].get("sources") == []
    assert end_events[0].get("iterations") == 0
    token_text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert "could not confirm this reference" in token_text.lower()


@pytest.mark.asyncio
async def test_chat_does_not_short_circuit_when_rag_context_exists(monkeypatch):
    """Exact reference with injected context still goes through the model."""
    agent = _make_agent()

    call_count = {"n": 0}

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        call_count["n"] += 1
        return {
            "status": "success",
            "choice": {
                "message": {"content": "Found it. [source: SomeDoc.pdf, chunk 1]", "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    def fake_rag_inject(**kwargs):
        return (
            {"role": "system", "content": "Relevant project context."},
            _fake_rag_audit(),
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr(
        "app.core.projects.get_document", lambda did: {"original_name": "SomeDoc.pdf"}
    )
    _patch_guardrail(monkeypatch)

    result = await agent.chat("What does the spec say?", project_id="proj_a")

    assert call_count["n"] == 1
    assert result["status"] == "success"
    assert len(result["sources"]) == 1


def test_build_missing_reference_answer_includes_project_name(monkeypatch):
    monkeypatch.setattr(
        "app.core.projects.get_project",
        lambda pid, user_id=None, include_admin_approved=False: {"name": "Test Project"},
    )
    answer = _build_missing_reference_answer("proj_a", "u1")
    assert "could not confirm this reference" in answer.lower()
    assert "for Test Project" in answer
    assert ".." not in answer


def test_build_missing_reference_answer_no_double_period_without_project_name():
    answer = _build_missing_reference_answer("unknown_proj", "u1")
    assert "could not confirm this reference" in answer.lower()
    assert ".." not in answer


def test_sanitizer_does_not_replace_controlled_missing_reference_answer():
    from app.agents.runtime import _sanitize_final_text
    text = "I could not confirm this reference in the indexed project sources."
    assert _sanitize_final_text(text) == text
