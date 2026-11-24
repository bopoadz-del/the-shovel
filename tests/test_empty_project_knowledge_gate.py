"""An empty project does not refuse an ask the encoded knowledge answers.

A project with nothing indexed and no other context used to get the canned
unindexed-project refusal before any tool ran, so an ask the encoded
construction knowledge answers never reached the knowledge tool. The gate now
reads the same rule as the knowledge predispatch (``_knowledge_entry_answering``).

Every knowledge base and ask below is built in this file. Each decision has an
input that takes it and one that does not.
"""
from __future__ import annotations

import pytest

from app.agents import runtime
from app.agents.core import tool_registry
from app.agents.runtime import Agent, _UNINDEXED_PROJECT_MESSAGE
from app.blocks import _knowledge as kb
from tests._fixture_values import STUB_TOKEN_02


def _entries():
    # Made-up entries; "parapet" and "coping" name one entry only.
    return [
        {"id": "roofing.parapet_coping_overhang", "type": "rule",
         "title": "Parapet coping overhang",
         "statement": "A coping overhangs the parapet face by at least 40 mm."},
        {"id": "drainage.gully_spacing", "type": "rule",
         "title": "Road gully spacing",
         "statement": "Gullies are spaced by the catchment area they drain."},
    ]


COVERED = "How far should a parapet coping overhang?"
UNCOVERED = "Who signed the handover certificate for block C?"
COVERED_BUT_DOCUMENT = "What parapet coping overhang does the specification say?"
COVERED_OTHER_ENTRY = "What sets the spacing of road gullies?"


@pytest.fixture
def empty_project(monkeypatch):
    monkeypatch.setattr(kb, "load_knowledge", lambda domain=None: _entries())
    monkeypatch.setattr(
        "app.core.project_memory.build_project_context", lambda pid, msg: "",
    )


@pytest.fixture
def kb_tool(monkeypatch):
    real_get = tool_registry.get
    marker = object()
    monkeypatch.setattr(
        tool_registry, "get",
        lambda name: marker if name == runtime._KB_TOOL else real_get(name),
    )


# --- the shared rule ---------------------------------------------------------

def test_covered_ask_is_answered_by_an_entry(empty_project):
    entry = runtime._knowledge_entry_answering(COVERED)
    assert entry is not None and entry["id"] == "roofing.parapet_coping_overhang"


def test_uncovered_ask_has_no_answering_entry(empty_project):
    assert runtime._knowledge_entry_answering(UNCOVERED) is None


def test_ask_about_a_document_stays_with_the_documents(empty_project):
    assert runtime._knowledge_entry_answering(COVERED_BUT_DOCUMENT) is None
    assert runtime._knowledge_entry_answering(COVERED) is not None


def test_blank_ask_has_no_answering_entry(empty_project):
    assert runtime._knowledge_entry_answering("   ") is None


# --- the gate ----------------------------------------------------------------

def test_gate_opens_for_an_ask_the_knowledge_answers(empty_project, kb_tool):
    assert runtime._project_has_non_rag_context("empty-proj", COVERED) is True


def test_gate_stays_shut_for_an_ask_it_does_not(empty_project, kb_tool):
    assert runtime._project_has_non_rag_context("empty-proj", UNCOVERED) is False


def test_gate_stays_shut_for_a_document_question(empty_project, kb_tool):
    assert runtime._project_has_non_rag_context("empty-proj", COVERED_BUT_DOCUMENT) is False


def test_gate_stays_shut_when_no_knowledge_tool_is_registered(empty_project, monkeypatch):
    real_get = tool_registry.get
    monkeypatch.setattr(
        tool_registry, "get",
        lambda name: None if name == runtime._KB_TOOL else real_get(name),
    )
    assert runtime._project_has_non_rag_context("empty-proj", COVERED) is False


def test_knowledge_load_error_leaves_the_gate_to_project_context(monkeypatch, kb_tool):
    def broken(domain=None):
        raise OSError("knowledge file unreadable")
    monkeypatch.setattr(kb, "load_knowledge", broken)
    monkeypatch.setattr(
        "app.core.project_memory.build_project_context", lambda pid, msg: "Fact: crane on site.",
    )
    assert runtime._project_has_non_rag_context("empty-proj", COVERED) is True
    monkeypatch.setattr(
        "app.core.project_memory.build_project_context", lambda pid, msg: "",
    )
    assert runtime._project_has_non_rag_context("empty-proj", COVERED) is False


# --- the real stream: refusal vs a model turn ---------------------------------

def _scripted_llm(reply: str, seen: list):
    async def fake(messages, api_key, **kwargs):
        seen.append(list(messages))
        return {"status": "success", "choice": {
            "message": {"role": "assistant", "content": reply, "tool_calls": []},
            "finish_reason": "stop"}}
    return fake


def _not_ready(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)

    async def _is_ready(_pid):
        return False
    monkeypatch.setattr("app.agents.runtime._project_is_rag_ready_off_process", _is_ready)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: False)


async def _stream_text(monkeypatch, question):
    agent = Agent(
        name="project-assistant", description="pa",
        system_prompt="You answer construction questions.", allowed_blocks=[],
    )
    seen: list = []
    monkeypatch.setattr(
        agent, "_call_llm",
        _scripted_llm("A coping overhangs the parapet face by at least 40 mm.", seen),
    )
    events = [e async for e in agent.chat_stream(question, project_id="empty-proj")]
    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    return text, seen


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [COVERED, COVERED_OTHER_ENTRY])
async def test_stream_answers_a_covered_ask_in_an_empty_project(
        monkeypatch, empty_project, kb_tool, question):
    _not_ready(monkeypatch)
    text, seen = await _stream_text(monkeypatch, question)
    assert seen, "the model was never called"
    assert text != _UNINDEXED_PROJECT_MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [UNCOVERED, COVERED_BUT_DOCUMENT])
async def test_stream_refuses_an_uncovered_ask_in_an_empty_project(
        monkeypatch, empty_project, kb_tool, question):
    _not_ready(monkeypatch)
    text, seen = await _stream_text(monkeypatch, question)
    assert seen == []
    assert text == _UNINDEXED_PROJECT_MESSAGE
