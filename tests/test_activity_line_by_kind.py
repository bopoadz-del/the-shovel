"""W4: while a tool runs, the user reads the KIND of work, never the machinery.

Every ``tool_call`` frame the chat stream sends carries a ``stage`` and a
``label`` chosen by the kind of result the tool declares in the tool registry
(``"search"`` -> searching, ``"computed"`` -> calculating, anything else ->
plain work). The label never holds the tool's name, its arguments or the
model's query. Every case builds its own invented tool name and arguments.
"""
from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.agents.core import tool_registry
from app.agents.runtime import Agent
from app.core import turn_progress
from tests._fixture_values import STUB_TOKEN_02

_TOOL = "tally_widget_rows"
_QUERY = "orange gaskets on level nine"


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", text.lower()) if len(w) > 2}


def _kinds(mapping):
    """A registry whose only declarations are ``mapping`` (name -> kind)."""
    def spec(name):
        return SimpleNamespace(kind=mapping[name]) if name in mapping else None
    return spec


@pytest.mark.parametrize("kind, stage", [
    ("search", "searching"),
    ("computed", "calculating"),
    ("", "working"),
])
def test_a_tool_call_frame_carries_the_stage_of_its_declared_kind(monkeypatch, kind, stage):
    monkeypatch.setattr(tool_registry, "result_spec", _kinds({_TOOL: kind}))
    frame = {"type": "tool_call", "tool": _TOOL, "name": _TOOL,
             "args_preview": json.dumps({"query": _QUERY})}
    out = turn_progress.tool_call(frame)
    assert out["stage"] == stage
    assert out["label"] == turn_progress.LABELS[stage]
    # The frame is otherwise unchanged: the end event and the trace still
    # read the tool's name from it.
    assert {k: out[k] for k in frame} == frame
    assert not (_words(out["label"]) & (_words(_TOOL) | _words(_QUERY)))


def test_a_tool_the_registry_does_not_know_is_shown_as_plain_work(monkeypatch):
    monkeypatch.setattr(tool_registry, "result_spec", _kinds({}))
    out = turn_progress.tool_call({"type": "tool_call", "tool": _TOOL})
    assert out["stage"] == "working"
    assert out["label"] == turn_progress.LABELS["working"]


def test_each_kind_of_work_reads_differently():
    labels = {turn_progress.LABELS[turn_progress.stage_for_kind(k)]
              for k in tool_registry.RESULT_KINDS}
    assert len(labels) == len(tool_registry.RESULT_KINDS)


def test_no_registered_tool_is_shown_by_its_own_name():
    """Over the real registry: whatever tools exist, the line a user reads
    while one runs is one of the fixed stage labels, and neither the tool's
    id nor its id read as words appears in it."""
    tool_registry.load()
    names = sorted(tool_registry._TOOLS)
    assert names
    for name in names:
        out = turn_progress.tool_call({"type": "tool_call", "tool": name})
        assert out["label"] in turn_progress.LABELS.values()
        assert name not in out["label"]
        assert name.replace("_", " ") not in out["label"].lower(), (name, out["label"])


# ── through the real chat stream ───────────────────────────────────────────

@pytest.fixture
def _stub_llm_key(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


def _script(tool_name: str, arguments: str):
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": tool_name, "arguments": arguments}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": "There are 14 rows."}}}
    return fake


def _fake_tool(name: str):
    async def run(self, tool_call, **kwargs):
        return {"name": name, "ok": True, "result": {"status": "success", "value": 14}}
    return run


@pytest.mark.parametrize("kind, stage", [("computed", "calculating"), ("search", "searching")])
def test_the_streamed_tool_call_shows_its_kind_not_its_name(_stub_llm_key, monkeypatch, kind, stage):
    monkeypatch.setattr(tool_registry, "result_spec", _kinds({_TOOL: kind}))
    agent = Agent(name="project-assistant", description="pa",
                  system_prompt="x", allowed_blocks=["construction"])
    args = json.dumps({"query": _QUERY})

    async def drain():
        return [ev async for ev in agent.chat_stream(
            user_message="How many rows are there?",
            history=[], project_id=None, conversation_id=None, user_id=None,
        )]

    with patch.object(Agent, "_call_llm", _script(_TOOL, args)), \
         patch.object(Agent, "_run_tool_call", _fake_tool(_TOOL)):
        events = asyncio.run(drain())
    calls = [e for e in events if e.get("type") == "tool_call"]
    assert calls, "no tool_call frame was streamed"
    for call in calls:
        assert call["tool"] == _TOOL
        assert call["stage"] == stage
        assert call["label"] == turn_progress.LABELS[stage]
        assert not (_words(call["label"]) & (_words(_TOOL) | _words(_QUERY)))
