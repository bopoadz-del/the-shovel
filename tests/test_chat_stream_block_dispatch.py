"""/chat/stream: the turn is refused when no chat block is registered, and a
registered chat block's own answer is what streams to the user.

Each test builds its own block and message; nothing reads a provider."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from app.routers import chat as chat_mod


class _TextBlock:
    """A chat block that answers with fixed text and records what it was asked."""

    def __init__(self, text: str):
        self.text = text
        self.asked: list[str] = []

    async def execute(self, message, options):
        self.asked.append(message)
        return {"result": {"text": self.text}}


def _drain(response) -> list[dict]:
    async def run():
        frames = []
        async for chunk in response.body_iterator:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            for line in text.splitlines():
                if line.startswith("data: "):
                    try:
                        frames.append(json.loads(line[6:]))
                    except ValueError:
                        frames.append({"raw": line[6:]})
        return frames
    return asyncio.run(run())


@pytest.mark.parametrize("registry", [{}, {"some_other_block": object}])
def test_no_chat_block_refuses_the_turn(monkeypatch, registry):
    monkeypatch.setattr(chat_mod, "BLOCK_REGISTRY", dict(registry))
    request = chat_mod.ChatRequest(message="what is the slab thickness?")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(chat_mod.chat_stream(request, auth={"user_id": "u-probe", "role": "user"}))
    assert caught.value.status_code == 500


@pytest.mark.parametrize("message, answer", [
    ("how deep is the trench?", "The trench is 1.2 m deep"),
    ("list the pour dates", "Pours on Monday and Thursday"),
])
def test_the_chat_blocks_own_answer_streams(monkeypatch, message, answer):
    block = _TextBlock(answer)
    monkeypatch.setattr(chat_mod, "BLOCK_REGISTRY", {"chat": lambda: block})
    monkeypatch.setitem(chat_mod.block_instances, "chat", block)
    monkeypatch.setattr(chat_mod, "_persist_chat_turn", lambda *a, **k: None)
    monkeypatch.setattr(chat_mod.asyncio, "sleep", _no_sleep)
    request = chat_mod.ChatRequest(message=message)

    response = asyncio.run(chat_mod.chat_stream(request, auth={"user_id": "u-probe", "role": "user"}))
    frames = _drain(response)

    assert block.asked == [message]
    streamed = "".join(f.get("content", "") for f in frames if f.get("type") == "token")
    assert streamed.strip() == answer
    assert not [f for f in frames if f.get("type") == "error"]


async def _no_sleep(_seconds):
    return None
