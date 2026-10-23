"""S4 — attachment reasoning wiring.

The composer's ``[attached: NAME]`` marker is now parsed server-side
(PILOT_READINESS T6b): the chat router resolves it to concrete project
documents, ships them as ``document_ids`` + an agent-visible system note, and
the agent can read the exact document via the new ``fetch_document`` tool.
Resolution is deterministic and follows the doc→slot honesty rules
(DECISIONS 2026-07-13): never guess across ambiguous matches.
"""
from __future__ import annotations

import asyncio

import pytest

import app.routers.chat as chat_mod
from app.agents import runtime as runtime_mod


# ── marker parsing ───────────────────────────────────────────────────────────

def test_parse_single_marker():
    names = chat_mod._parse_attached_markers("[attached: RFP_final.pdf] summarize it")
    assert names == ["RFP_final.pdf"]


def test_parse_photo_marker_with_observations():
    msg = "[attached: site.jpg | observations: missing guardrail; open edge] what do you see?"
    assert chat_mod._parse_attached_markers(msg) == ["site.jpg"]


def test_parse_multiple_markers_deduplicated_case_insensitive():
    msg = "[attached: a.pdf] [attached: B.xlsx] then [attached: A.PDF] again"
    assert chat_mod._parse_attached_markers(msg) == ["a.pdf", "B.xlsx"]


def test_parse_no_marker():
    assert chat_mod._parse_attached_markers("plain question, no attachments") == []
    assert chat_mod._parse_attached_markers("") == []


# ── deterministic resolution (honesty rules) ─────────────────────────────────

_DOCS = [
    {"id": "d1", "original_name": "RFP_final.pdf", "file_path": "/x/d1"},
    {"id": "d2", "original_name": "schedule.xer", "file_path": "/x/d2"},
    {"id": "d3", "original_name": "RFP_final.pdf", "file_path": "/x/d3"},  # re-upload
    {"id": "d4", "original_name": "BOQ_north.xlsx", "file_path": "/x/d4"},
    {"id": "d5", "original_name": "BOQ_south.xlsx", "file_path": "/x/d5"},
]


@pytest.fixture()
def project_docs(monkeypatch):
    monkeypatch.setattr("app.core.projects.list_documents", lambda pid: list(_DOCS))


def test_resolve_exact_match(project_docs):
    docs = chat_mod._resolve_attached_documents("p1", ["schedule.xer"])
    assert [d["id"] for d in docs] == ["d2"]


def test_resolve_duplicate_name_latest_upload_wins(project_docs):
    docs = chat_mod._resolve_attached_documents("p1", ["RFP_final.pdf"])
    assert [d["id"] for d in docs] == ["d3"]


def test_resolve_unique_substring(project_docs):
    docs = chat_mod._resolve_attached_documents("p1", ["BOQ_north"])
    assert [d["id"] for d in docs] == ["d4"]


def test_resolve_ambiguous_substring_skipped_never_guessed(project_docs):
    # "BOQ" matches two documents — the router must not pick one silently.
    assert chat_mod._resolve_attached_documents("p1", ["BOQ"]) == []


def test_resolve_unknown_name_skipped(project_docs):
    assert chat_mod._resolve_attached_documents("p1", ["nope.docx"]) == []


def test_resolve_without_project_is_noop():
    assert chat_mod._resolve_attached_documents(None, ["RFP_final.pdf"]) == []


def test_resolve_lookup_failure_is_best_effort(monkeypatch):
    def boom(pid):
        raise RuntimeError("db down")
    monkeypatch.setattr("app.core.projects.list_documents", boom)
    assert chat_mod._resolve_attached_documents("p1", ["RFP_final.pdf"]) == []


# ── agent-side system note ───────────────────────────────────────────────────

def test_attached_note_contains_name_id_and_tool_hint():
    note = runtime_mod._build_attached_documents_note(
        [{"id": "d3", "original_name": "RFP_final.pdf"}]
    )
    assert "RFP_final.pdf" in note
    assert "document_id: d3" in note
    assert "fetch_document" in note


def test_attached_note_empty_for_no_docs():
    assert runtime_mod._build_attached_documents_note([]) == ""
    assert runtime_mod._build_attached_documents_note([{}]) == ""


# ── fetch_document content helper ────────────────────────────────────────────

@pytest.fixture()
def fetch_env(monkeypatch):
    monkeypatch.setattr("app.core.projects.list_documents", lambda pid: list(_DOCS))
    # No RAG store in unit scope — force the raw-extraction fallback.
    monkeypatch.setattr("app.core.rag.retriever.available", lambda: False)
    monkeypatch.setattr(
        "app.core.doc_index.extract_document_text",
        lambda file_path, filename: f"TEXT OF {filename}",
    )


def test_fetch_by_id(fetch_env):
    content, doc, err = runtime_mod._fetch_document_content("p1", "d2", "")
    assert err is None
    assert doc["original_name"] == "schedule.xer"
    assert content["text"] == "TEXT OF schedule.xer"
    assert content["source"] == "extracted"
    assert content["truncated"] is False


def test_fetch_by_exact_name_duplicate_latest_wins(fetch_env):
    content, doc, err = runtime_mod._fetch_document_content("p1", "", "RFP_final.pdf")
    assert err is None
    assert doc["id"] == "d3"


def test_fetch_ambiguous_name_asks_which(fetch_env):
    content, doc, err = runtime_mod._fetch_document_content("p1", "", "BOQ")
    assert content is None
    assert err is not None and "ask the user which" in err


def test_fetch_unknown_id_honest_error(fetch_env):
    content, doc, err = runtime_mod._fetch_document_content("p1", "zzz", "")
    assert content is None
    assert "no document with id" in err


def test_fetch_truncates_large_documents(monkeypatch, fetch_env):
    cap = runtime_mod._effective_fetch_document_max_chars()
    monkeypatch.setattr(
        "app.core.doc_index.extract_document_text",
        lambda file_path, filename: "x" * (cap + 100),
    )
    content, doc, err = runtime_mod._fetch_document_content("p1", "d1", "")
    assert err is None
    assert content["truncated"] is True
    assert len(content["text"]) == cap


def test_fetch_no_extractable_text_honest_error(monkeypatch, fetch_env):
    monkeypatch.setattr(
        "app.core.doc_index.extract_document_text", lambda file_path, filename: ""
    )
    content, doc, err = runtime_mod._fetch_document_content("p1", "d1", "")
    assert content is None
    assert "no extractable text" in err


# ── fetch_document tool dispatch through the agent runtime ───────────────────

def _project_assistant():
    from app.agents.runtime import AGENT_REGISTRY, load_agents
    if "project-assistant" not in AGENT_REGISTRY:
        load_agents()
    agent = AGENT_REGISTRY.get("project-assistant")
    if agent is None:
        pytest.skip("project-assistant agent config not loaded")
    return agent


def test_fetch_document_tool_advertised_only_with_project():
    agent = _project_assistant()
    with_project = {t["function"]["name"] for t in agent.tool_definitions("p1")}
    without = {t["function"]["name"] for t in agent.tool_definitions(None)}
    assert "fetch_document" in with_project
    assert "fetch_document" not in without


def test_fetch_document_tool_call_roundtrip(fetch_env):
    agent = _project_assistant()
    call = {"function": {"name": "fetch_document", "arguments": {"document_id": "d2"}}}
    out = asyncio.run(agent._run_tool_call(call, project_id="p1"))
    assert out["ok"] is True
    assert out["result"]["document_id"] == "d2"
    assert out["result"]["filename"] == "schedule.xer"
    assert out["result"]["content"] == "TEXT OF schedule.xer"


def test_fetch_document_tool_requires_project():
    agent = _project_assistant()
    call = {"function": {"name": "fetch_document", "arguments": {"document_id": "d2"}}}
    out = asyncio.run(agent._run_tool_call(call, project_id=None))
    assert out["ok"] is False


def test_fetch_document_tool_requires_id_or_name():
    agent = _project_assistant()
    call = {"function": {"name": "fetch_document", "arguments": {}}}
    out = asyncio.run(agent._run_tool_call(call, project_id="p1"))
    assert out["ok"] is False
    assert "document_id or filename" in out["result"]["error"]
