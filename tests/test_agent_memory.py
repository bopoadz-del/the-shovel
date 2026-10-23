"""Tests for app/core/agent_memory.py — persistent agent memory.

Phase C4 · Stream C.

Tests are isolated with tmp_path + monkeypatch so they never touch the
production or development database.
"""

import importlib
import sys

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _reload(tmp_path, monkeypatch):
    """Set DATA_DIR, reload agent_memory, and return the fresh module."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.core.agent_memory as _mod
    importlib.reload(_mod)
    return _mod


# ── create + fetch conversation (idempotent) ─────────────────────────────────

def test_get_or_create_conversation_idempotent(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    cid = "conv-abc"
    c1 = am.get_or_create_conversation(cid, "scout", project_id="proj-1")
    assert c1["id"] == cid
    assert c1["agent_name"] == "scout"
    assert c1["project_id"] == "proj-1"

    # Calling again returns the same row, not a duplicate
    c2 = am.get_or_create_conversation(cid, "scout", project_id="proj-1")
    assert c2["id"] == cid
    assert c2["created_at"] == c1["created_at"]


# ── append + read messages (oldest-first) ────────────────────────────────────

def test_append_and_get_messages_oldest_first(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    cid = "conv-msg"
    am.get_or_create_conversation(cid, "scout")

    am.append_message(cid, "user", "Hello")
    am.append_message(cid, "assistant", "Hi there")
    am.append_message(cid, "user", "Goodbye")

    msgs = am.get_messages(cid)
    assert len(msgs) == 3
    assert msgs[0]["content"] == "Hello"
    assert msgs[1]["content"] == "Hi there"
    assert msgs[2]["content"] == "Goodbye"
    # oldest-first verified by position
    assert msgs[0]["created_at"] <= msgs[1]["created_at"] <= msgs[2]["created_at"]


# ── messages survive importlib.reload (persistence) ──────────────────────────

def test_messages_survive_reload(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    cid = "conv-persist"
    am.get_or_create_conversation(cid, "scout")
    am.append_message(cid, "user", "persistent message")

    # Reload the module — _initialized resets to False but DB file is unchanged
    am2 = _reload(tmp_path, monkeypatch)
    msgs = am2.get_messages(cid)
    assert any(m["content"] == "persistent message" for m in msgs)


# ── get_messages limit ────────────────────────────────────────────────────────

def test_get_messages_limit(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    cid = "conv-limit"
    am.get_or_create_conversation(cid, "scout")

    for i in range(10):
        am.append_message(cid, "user", f"message {i}")

    # limit=4 should return only the 4 most recent, still oldest-first
    msgs = am.get_messages(cid, limit=4)
    assert len(msgs) == 4
    # These should be messages 6, 7, 8, 9 (last 4 of 0-9)
    assert msgs[0]["content"] == "message 6"
    assert msgs[-1]["content"] == "message 9"


# ── set_agent_fact upsert ─────────────────────────────────────────────────────

def test_set_agent_fact_upsert(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)

    f1 = am.set_agent_fact("scout", "home_city", "Dubai")
    assert f1["value"] == "Dubai"

    # Second set with same (agent_name, key) overwrites value
    f2 = am.set_agent_fact("scout", "home_city", "Abu Dhabi")
    assert f2["value"] == "Abu Dhabi"

    # Only one row exists for this agent+key
    facts = am.list_agent_facts("scout")
    home_facts = [f for f in facts if f["key"] == "home_city"]
    assert len(home_facts) == 1
    assert home_facts[0]["value"] == "Abu Dhabi"


# ── delete_conversation cascades messages ─────────────────────────────────────

def test_delete_conversation_cascades_messages(tmp_path, monkeypatch):
    am = _reload(tmp_path, monkeypatch)
    cid = "conv-del"
    am.get_or_create_conversation(cid, "scout")
    am.append_message(cid, "user", "will be deleted")
    am.append_message(cid, "assistant", "also deleted")

    result = am.delete_conversation(cid)
    assert result is True

    # Messages should be gone (ON DELETE CASCADE)
    msgs = am.get_messages(cid)
    assert msgs == []

    # Conversation itself is gone
    convs = am.list_conversations(agent_name="scout")
    assert not any(c["id"] == cid for c in convs)

    # Deleting again returns False
    assert am.delete_conversation(cid) is False


# ── lifespan wires agent_memory init ─────────────────────────────────────────

def test_main_initializes_agent_memory():
    """Starting the app via TestClient runs lifespan → agent_memory._initialized is True."""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.core.agent_memory as agent_memory

    with TestClient(app):
        # Lifespan has run; _initialized should be True on the same module object
        assert agent_memory._initialized is True
