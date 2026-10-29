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
    from app.routers import chat

    async def _search(prompt, project_id, user_id, top_k=5):
        return prompt, []

    monkeypatch.setattr(chat, "_with_doc_search", _search)
    monkeypatch.setitem(chat.block_instances, "chat", _Block())
    with TestClient(app) as c:
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
