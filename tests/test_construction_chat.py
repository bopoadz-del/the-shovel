"""Tests for ConstructionContainer.chat — the conversational entry point
that delegates to ChatBlock with the EVM system prompt pre-injected.

The container owns the policy (which prompt file is the construction
project default); ChatBlock owns the mechanics. These tests stub the
ChatBlock at the _resolve_block boundary so they neither hit the LLM
provider nor depend on dependency-injection wiring.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app.containers.construction import ConstructionContainer


class _FakeChatBlock:
    """Captures the (input_data, params) passed to process() so tests can
    assert on what the container forwarded."""

    def __init__(self) -> None:
        self.called = False
        self.captured_input: Any = None
        self.captured_params: Dict = {}

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        self.called = True
        self.captured_input = input_data
        self.captured_params = dict(params or {})
        return {
            "status": "success",
            "text": "ok",
            "provider": "stub",
            "model": "stub-model",
        }


@pytest.fixture
def container_and_chat(monkeypatch):
    container = ConstructionContainer()
    fake_chat = _FakeChatBlock()
    monkeypatch.setattr(
        container,
        "_resolve_block",
        lambda name: fake_chat if name == "chat" else None,
    )
    return container, fake_chat


@pytest.mark.asyncio
async def test_construction_chat_injects_evm_prompt_file(container_and_chat):
    """When the caller passes no prompt override, the container must
    pre-inject system_prompt_file = construction_evm.md."""
    container, fake_chat = container_and_chat

    result = await container.chat({"text": "test"}, {})

    assert fake_chat.called is True
    assert fake_chat.captured_params.get("system_prompt_file") == "construction_evm.md"
    assert "system_prompt" not in fake_chat.captured_params
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_construction_chat_respects_caller_override(container_and_chat):
    """When the caller supplies a literal system_prompt, the container must
    forward it as-is and must NOT inject the EVM file."""
    container, fake_chat = container_and_chat

    await container.chat({"text": "test"}, {"system_prompt": "custom prompt"})

    assert fake_chat.called is True
    assert fake_chat.captured_params.get("system_prompt") == "custom prompt"
    assert "system_prompt_file" not in fake_chat.captured_params


@pytest.mark.asyncio
async def test_construction_route_chat_delegates(container_and_chat):
    """The route() entry point must dispatch action='chat' to chat()."""
    container, fake_chat = container_and_chat

    await container.process({"text": "x"}, {"action": "chat"})

    assert fake_chat.called is True
    # EVM injection still happens via the route path.
    assert fake_chat.captured_params.get("system_prompt_file") == "construction_evm.md"


@pytest.mark.asyncio
async def test_construction_container_defaults_use_rag_true(monkeypatch):
    """Caller doesn't pass use_rag; container must default it to True so
    chat block invokes retrieval."""
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    captured_params = {}

    class _FakeChatBlock:
        async def process(self, input_data, params):
            captured_params.update(params)
            return {"status": "success", "text": "ok"}

    monkeypatch.setattr(container, "_resolve_block",
                        lambda name: _FakeChatBlock() if name == "chat" else None)

    await container.chat({"text": "hi"}, {"project_id": "p1"})
    assert captured_params.get("use_rag") is True
