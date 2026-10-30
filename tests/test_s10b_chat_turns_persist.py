"""/v1/chat (JSON) and the block stream /chat/stream store each turn under conversation_id, and
the conversation export reads them. No conversation_id → nothing stored."""
from __future__ import annotations

import uuid

import pytest

from tests._fixture_values import PW_06


class _Block:
    async def execute(self, message, params):
        return {"result": {"text": f"answer to {message}", "model": "m", "provider": "p"}}


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app
    import app.routers.chat as chat

    async def _search(prompt, project_id, user_id, top_k=5):
        return prompt, []

    monkeypatch.setattr(chat, "_with_doc_search", _search)
    monkeypatch.setitem(chat.block_instances, "chat", _Block())
    with TestClient(app, raise_server_exceptions=False) as c:
        email = f"s10b-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        tok = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        pid = c.post("/v1/projects", json={"name": "s10b"}, headers=h).json()["id"]
        yield c, h, pid


@pytest.mark.parametrize("with_id", [True, False])
def test_two_turns_are_stored_and_exported(client, with_id):
    from app.core import agent_memory

    c, h, pid = client
    cid = f"ws-{pid}-{uuid.uuid4().hex[:6]}"
    extra = {"conversation_id": cid} if with_id else {}
    r1 = c.post("/v1/chat", headers=h, json={"message": "Q1", "project_id": pid, **extra})
    assert r1.status_code == 200, r1.text
    r2 = c.post("/chat/stream", headers=h, json={"message": "Q2", "project_id": pid, **extra})
    assert r2.status_code == 200
    msgs = agent_memory.get_messages(cid, limit=50)
    assistant = [m["content"] for m in msgs if m["role"] == "assistant"]
    if with_id:
        assert assistant == ["answer to Q1", "answer to Q2"]
        e = c.post(f"/v1/projects/{pid}/conversations/{cid}/export?format=docx", headers=h)
        assert e.status_code == 200, e.text
    else:
        assert msgs == []


def _user(c, tag):
    email = f"s10b-{tag}-{uuid.uuid4().hex[:8]}@x.com"
    c.post("/v1/users/register", json={"email": email, "password": PW_06})
    tok = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.mark.parametrize("path", ["/v1/chat", "/chat/stream"])
def test_another_users_conversation_is_refused(client, path):
    from app.core import agent_memory

    c, h, pid = client
    cid = f"ws-{pid}"
    other = _user(c, "intruder")
    r = c.post(path, headers=other, json={"message": "leak?", "conversation_id": cid})
    assert r.status_code == 404, r.text
    assert agent_memory.get_messages(cid, limit=50) == []


@pytest.mark.parametrize("path", ["/v1/chat", "/chat/stream"])
def test_a_store_failure_does_not_drop_the_answer(client, monkeypatch, path):
    from app.core import agent_memory

    c, h, pid = client

    def _down(*a, **k):
        raise RuntimeError("store down")

    monkeypatch.setattr(agent_memory, "append_message", _down)
    r = c.post(path, headers=h, json={"message": "Q9", "project_id": pid, "conversation_id": f"ws-{pid}"})
    assert r.status_code == 200, r.text
    assert "answer" in r.text and "Q9" in r.text


@pytest.mark.parametrize("cid", [None, ""])
def test_no_conversation_id_touches_no_store(monkeypatch, cid):
    import app.routers.chat as chat
    from app.core import agent_memory

    calls = []
    monkeypatch.setattr(agent_memory, "get_or_create_conversation", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(agent_memory, "append_message", lambda *a, **k: calls.append(a))
    chat._persist_chat_turn(chat.ChatRequest(message="q", conversation_id=cid), "u1", "a")
    assert calls == []


@pytest.mark.parametrize("path", ["/v1/chat", "/chat/stream"])
def test_missing_chat_block_is_500(client, monkeypatch, path):
    import app.routers.chat as chat

    c, h, pid = client
    monkeypatch.delitem(chat.BLOCK_REGISTRY, "chat", raising=False)
    r = c.post(path, headers=h, json={"message": "Q", "project_id": pid})
    assert r.status_code == 500
    assert "Chat block not available" in r.text


@pytest.mark.parametrize("message", ["Q7", "Q8"])
def test_stream_builds_the_block_when_none_is_cached(client, monkeypatch, message):
    import app.routers.chat as chat
    from app.core import agent_memory

    c, h, pid = client
    cid = f"ws-{pid}"
    monkeypatch.delitem(chat.block_instances, "chat", raising=False)
    monkeypatch.setitem(chat.BLOCK_REGISTRY, "chat", _Block)
    r = c.post("/chat/stream", headers=h, json={"message": message, "project_id": pid, "conversation_id": cid})
    assert r.status_code == 200
    assert [m["content"] for m in agent_memory.get_messages(cid, limit=50) if m["role"] == "assistant"] == [
        f"answer to {message}"]
