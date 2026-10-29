"""POST /v1/chat returns {text, model, provider, sources}; sources come from
the same builder the stream's end event uses, fed this turn's retrieval."""
from __future__ import annotations

import uuid

import pytest

from tests._fixture_values import PW_06

SNIPPET = {"document_id": "doc-1", "filename": "Spec.pdf", "snippet": "Concrete C40 to BS EN 206.",
           "score": 0.9, "chunk_index": 3}


class _Block:
    async def execute(self, message, params):
        return {"result": {"text": "C40 concrete per the spec.", "model": "m-1", "provider": "p-1"}}


def _post(monkeypatch, snippets):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.routers import chat

    async def _search(prompt, project_id, user_id, top_k=5):
        return prompt, list(snippets)

    monkeypatch.setattr(chat, "_with_doc_search", _search)
    monkeypatch.setitem(chat.block_instances, "chat", _Block())
    with TestClient(app) as c:
        email = f"s10a-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        return c.post("/v1/chat", headers={"Authorization": f"Bearer {token}"},
                      json={"message": "What concrete grade?"})


@pytest.mark.parametrize("snippets,expect_ids", [([SNIPPET], ["doc-1"]), ([], [])])
def test_sources_follow_the_turns_retrieval(monkeypatch, snippets, expect_ids):
    r = _post(monkeypatch, snippets)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"text", "model", "provider", "sources"}
    assert body["model"] == "m-1" and body["provider"] == "p-1"
    assert [s["doc_id"] for s in body["sources"]] == expect_ids
    for s in body["sources"]:
        assert {"doc_id", "doc_name", "score", "confidence", "layer"} <= set(s)
