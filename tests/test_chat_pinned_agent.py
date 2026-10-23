"""The / agent-picker: a user-PINNED agent runs directly, bypassing the
predefined shortcut and the smart-orchestrator auto-override. Tests the
module-level generator so no full-app HTTP boot is needed (mirrors
test_chat_predefined_wiring)."""
from __future__ import annotations

import asyncio
import json

import app.routers.chat as chat_mod


class _FakeAgent:
    name = "quantity-surveyor"

    async def chat_stream(self, prompt, history=None, project_id=None,
                          conversation_id=None, user_id=None,
                          attached_documents=None):
        # Echo the project_id the pin path resolved, so the tenant gate is
        # observable in the test.
        yield {"type": "token", "content": f"QS on {project_id}"}
        yield {"type": "end", "complete": True}


def _run(gen):
    async def _collect():
        evs = []
        async for chunk in gen:
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    evs.append(json.loads(line[6:]))
        return evs
    return asyncio.run(_collect())


def test_pinned_agent_bypasses_router(monkeypatch):
    monkeypatch.setattr("app.agents.get_agent", lambda name: _FakeAgent())
    monkeypatch.setattr("app.core.projects.get_project",
                        lambda pid, user_id=None, **kw: {"id": pid, "name": "P"})

    evs = _run(chat_mod._stream_from_pinned_agent(
        agent_name="quantity-surveyor", prompt="take off the concrete",
        project_id="proj1", user_id="u1", history=[], conversation_id=None))

    route = next(e for e in evs if e["type"] == "route")
    assert route["reason"] == "user_pinned"
    assert route["final"] == "quantity-surveyor"
    assert route["requested"] == "quantity-surveyor"
    text = "".join(e.get("content", "") for e in evs if e["type"] == "token")
    assert "QS on proj1" in text


def test_pinned_unknown_agent_errors_cleanly(monkeypatch):
    monkeypatch.setattr("app.agents.get_agent", lambda name: None)

    evs = _run(chat_mod._stream_from_pinned_agent(
        agent_name="does-not-exist", prompt="hi",
        project_id=None, user_id="u1", history=[], conversation_id=None))

    assert any(e["type"] == "error" and "Unknown agent" in e["message"] for e in evs)
    # No route/token events when the agent is unknown.
    assert not any(e["type"] == "route" for e in evs)


def test_pinned_drops_unowned_project(monkeypatch):
    monkeypatch.setattr("app.agents.get_agent", lambda name: _FakeAgent())
    # Ownership check returns None -> project must be dropped (tenant gate).
    monkeypatch.setattr("app.core.projects.get_project",
                        lambda pid, user_id=None, **kw: None)

    evs = _run(chat_mod._stream_from_pinned_agent(
        agent_name="quantity-surveyor", prompt="x",
        project_id="someone-elses", user_id="u1", history=[], conversation_id=None))

    text = "".join(e.get("content", "") for e in evs if e["type"] == "token")
    assert "QS on None" in text  # project_id was dropped to None
