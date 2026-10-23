"""Heavy-reasoning route tenancy + conversation isolation.

Regression for the two Codex P1 findings on the merged PR #18:

  1. `_stream_from_heavy_reasoning` passes `project_id` to the runtime
     agent without an ownership check, so an authenticated user can
     pull another tenant's indexed docs via the agent's
     `search_project_documents` tool.

  2. The same function keyed the agent's `conversation_id` only on
     `session_id`, which defaults to "default" on the calling path.
     Two users on the default session both write to `hr-default` in
     `agent_memory` → user B sees user A's prior turns.

Both fixes live in `app/routers/chat.py::_stream_from_heavy_reasoning`:
- Drop `project_id` (set to None) when the caller doesn't own it
- Prefix `conversation_id` with the user_id so per-user state stays
  per-user even on the default session
"""

from __future__ import annotations

import asyncio
import json

import pytest


def _start_event(items):
    """Helper — first SSE chunk is always the 'start' event."""
    for chunk in items:
        text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
        if text.startswith("data: ") and '"type": "start"' in text:
            return json.loads(text[len("data: "):].split("\n\n")[0])
    return None


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import projects as projects_store
    from app.core import agent_memory as _am
    # Force module re-init on the relocated DATA_DIR
    if hasattr(_am, "_initialized"):
        _am._initialized = False
    if hasattr(projects_store, "_initialized"):
        projects_store._initialized = False
    projects_store.init_db()
    yield tmp_path


@pytest.mark.asyncio
async def test_heavy_reasoning_drops_project_id_when_caller_does_not_own_it(
    isolated_data_dir, monkeypatch
):
    """Codex P1 #1: an authenticated user supplying another tenant's
    project_id must not have it forwarded to the agent's
    search_project_documents tool. Match the fast path's behaviour:
    silently drop the unowned project_id (no 403 — that would create
    an oracle for probing valid ids)."""
    from app.core import projects as projects_store
    from app.core import users as users_mod
    from app.routers import chat as chat_router

    users_mod.ensure_user_exists("user-A")
    users_mod.ensure_user_exists("user-B")
    # Tenant A owns the project; user-B is the attacker.
    proj = projects_store.create_project("Tenant A project", user_id="user-A")
    pid = proj["id"]

    captured = {}

    class _FakeAgent:
        async def chat(self, **kwargs):
            captured.update(kwargs)
            return {"status": "success", "response": "ok"}

    def _fake_get_agent(name: str):
        return _FakeAgent() if name == "heavy-reasoning" else None

    monkeypatch.setattr(
        "app.agents.get_agent", _fake_get_agent, raising=False
    )

    # user-B asks with user-A's project_id — agent must see project_id=None
    gen = chat_router._stream_from_heavy_reasoning(
        user_message="show me everything in this project",
        project_id=pid,
        user_id="user-B",
        history=[],
        session_id="s1",
    )
    # Drain the async generator so run_agent executes
    async for _chunk in gen:
        pass

    assert captured.get("project_id") is None, (
        f"heavy-reasoning forwarded an unowned project_id={pid!r} from user-B "
        f"to the agent. Cross-tenant data leak. Fix: drop the project_id when "
        f"ownership check fails. (captured: {captured!r})"
    )


@pytest.mark.asyncio
async def test_heavy_reasoning_keeps_project_id_when_caller_owns_it(
    isolated_data_dir, monkeypatch
):
    """Counter-test for the fix above: legitimate ownership must still
    pass through. Without this, the security fix could over-redact and
    break the feature for the actual owner."""
    from app.core import projects as projects_store
    from app.core import users as users_mod
    from app.routers import chat as chat_router

    users_mod.ensure_user_exists("user-A")
    proj = projects_store.create_project("Owner's project", user_id="user-A")
    pid = proj["id"]

    captured = {}

    class _FakeAgent:
        async def chat(self, **kwargs):
            captured.update(kwargs)
            return {"status": "success", "response": "ok"}

    monkeypatch.setattr(
        "app.agents.get_agent",
        lambda name: _FakeAgent() if name == "heavy-reasoning" else None,
        raising=False,
    )

    gen = chat_router._stream_from_heavy_reasoning(
        user_message="anything",
        project_id=pid,
        user_id="user-A",  # the actual owner
        history=[],
        session_id="s1",
    )
    async for _chunk in gen:
        pass

    assert captured.get("project_id") == pid, (
        f"ownership check incorrectly dropped project_id for the owner "
        f"(captured: {captured!r})"
    )


@pytest.mark.asyncio
async def test_heavy_reasoning_conversation_id_is_user_scoped(
    isolated_data_dir, monkeypatch
):
    """Codex P1 #2: conversation_id was keyed only on session_id. Two
    different users on session_id='default' both wrote to and read from
    the same `hr-default` row in agent_memory.

    Fix: prefix with user_id so identical session_ids across users get
    distinct conversation_ids.
    """
    from app.routers import chat as chat_router

    captured_per_user = []

    class _FakeAgent:
        async def chat(self, **kwargs):
            captured_per_user.append(kwargs.get("conversation_id"))
            return {"status": "success", "response": "ok"}

    monkeypatch.setattr(
        "app.agents.get_agent",
        lambda name: _FakeAgent() if name == "heavy-reasoning" else None,
        raising=False,
    )

    # Two different users, SAME session_id (the dangerous case — both
    # are using the default).
    for uid in ("user-A", "user-B"):
        gen = chat_router._stream_from_heavy_reasoning(
            user_message="hi",
            project_id=None,
            user_id=uid,
            history=[],
            session_id="default",
        )
        async for _chunk in gen:
            pass

    assert len(captured_per_user) == 2, "expected two agent.chat calls"
    a_id, b_id = captured_per_user
    assert a_id != b_id, (
        f"two different users on the same session_id='default' got the "
        f"SAME conversation_id={a_id!r} — user B will see user A's prior "
        f"turns. conversation_id must include user_id."
    )
    # And both must contain their respective user id
    assert "user-A" in a_id, f"user-A's conversation_id={a_id!r} missing user_id"
    assert "user-B" in b_id, f"user-B's conversation_id={b_id!r} missing user_id"


@pytest.mark.asyncio
async def test_heavy_reasoning_anon_fallback_when_user_id_missing(
    isolated_data_dir, monkeypatch
):
    """The conversation_id prefix must not crash when user_id is None.
    Belt-and-suspenders — the route requires auth so this path is
    unreachable in production, but a programming-error caller (a future
    test, an internal helper) shouldn't trip a NoneType+str TypeError.
    """
    from app.routers import chat as chat_router

    captured = {}

    class _FakeAgent:
        async def chat(self, **kwargs):
            captured.update(kwargs)
            return {"status": "success", "response": "ok"}

    monkeypatch.setattr(
        "app.agents.get_agent",
        lambda name: _FakeAgent() if name == "heavy-reasoning" else None,
        raising=False,
    )

    gen = chat_router._stream_from_heavy_reasoning(
        user_message="hi",
        project_id=None,
        user_id=None,  # the edge case
        history=[],
        session_id="s1",
    )
    async for _chunk in gen:
        pass

    cid = captured.get("conversation_id") or ""
    assert "anon" in cid, (
        f"conversation_id={cid!r} should use 'anon' as user_id fallback"
    )
