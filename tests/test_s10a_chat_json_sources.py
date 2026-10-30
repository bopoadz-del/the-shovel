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
    import app.routers.chat as chat

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


@pytest.mark.parametrize("snippets,expect_ids", [
    ([SNIPPET, "junk", {"snippet": "no id"}], ["doc-1"]),
    ([{"document_id": "doc-2", "filename": "B.pdf", "snippet": "x", "score": 0.5}, 7], ["doc-2"]),
])
def test_sources_skip_entries_without_a_document(snippets, expect_ids):
    import app.routers.chat as chat

    out = chat._sources_from_snippets(snippets, "answer", "q", None)
    assert [s["doc_id"] for s in out] == expect_ids


@pytest.mark.parametrize("snippets", [None, []])
def test_no_retrieval_means_no_sources(snippets):
    import app.routers.chat as chat

    assert chat._sources_from_snippets(snippets, "answer", "q", None) == []


@pytest.mark.parametrize("snippets", [[SNIPPET], [dict(SNIPPET, document_id="doc-9")]])
def test_a_failing_builder_degrades_to_no_sources(monkeypatch, snippets):
    import app.agents.runtime as runtime
    import app.routers.chat as chat

    def _boom(audit, answer):
        raise RuntimeError("builder down")

    monkeypatch.setattr(runtime, "_build_sources_from_audit", _boom)
    assert chat._sources_from_snippets(snippets, "answer", "q", None) == []


@pytest.mark.parametrize("message", ["What concrete grade?", "Which cement?"])
def test_chat_without_the_chat_block_is_500(monkeypatch, message):
    from fastapi.testclient import TestClient

    import app.routers.chat as chat
    from app.main import app

    monkeypatch.delitem(chat.BLOCK_REGISTRY, "chat", raising=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        email = f"s10a-{uuid.uuid4().hex[:8]}@x.com"
        c.post("/v1/users/register", json={"email": email, "password": PW_06})
        token = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
        r = c.post("/v1/chat", headers={"Authorization": f"Bearer {token}"}, json={"message": message})
    assert r.status_code == 500
    assert "Chat block not available" in r.text
