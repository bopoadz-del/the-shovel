"""Chat block — local-fallback guarantees (no cloud provider names).

Locks in two invariants we never want to regress:

1. The chat block must never go completely dark on the user. When no cloud
   provider is configured AND no local LLM backend is reachable, ``process()``
   must still return ``status: success`` with a graceful offline-template
   message — the UI must never see a raw error from the chat block.

2. The ChatBlock source must not reference Anthropic / OpenAI / Grok / Claude
   anywhere. Those provider names were deliberately removed from the chat API.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from app.blocks.chat import ChatBlock
from tests._fixture_values import STUB_KEY_08, STUB_KEY_09, STUB_KEY_12


@pytest.mark.asyncio
async def test_offline_template_when_no_provider_available(monkeypatch):
    """No cloud key configured → graceful offline template."""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    block = ChatBlock()
    result = await block.process("Hello, what is 2+2?", {"stream": False})

    assert result["status"] == "success", "chat must not go dark — offline path returns success"
    assert result["provider"] == "offline_template"
    text = result.get("text", "")
    assert "offline mode" in text.lower()
    # The user's message must be echoed back so they know the chat is alive.
    assert "Hello, what is 2+2?" in text
    # The template must surface the cloud restore path.
    assert "DEEPSEEK_API_KEY" in text or "OPENROUTER_API_KEY" in text


@pytest.mark.asyncio
async def test_deepseek_402_does_not_serve_offline_when_fallback_is_ready(monkeypatch):
    """Live tip 78bd9ca: POST /v1/chat returned the offline template while
    GET /v1/health said primary_ready + fallback_ready. ChatBlock treated a
    DeepSeek 402 as 'no language model is reachable' and never walked
    ``_llm_fallback_ladder``.
    """
    import httpx

    from app.agents.runtime import DEEPSEEK_API_URL, OPENROUTER_API_URL
    from app.core.health_probes import probe_llm

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_KEY_08)
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_KEY_09)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    assert probe_llm() == {"primary_ready": True, "fallback_ready": True}

    urls: list[str] = []

    class _Resp:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self._payload = payload
            self.text = text if text else (
                "" if payload is None else __import__("json").dumps(payload)
            )

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None, **kwargs):
            urls.append(url)
            if url.startswith(DEEPSEEK_API_URL.rsplit("/", 2)[0]):
                return _Resp(
                    402,
                    text='{"error":{"message":"Insufficient Balance"}}',
                )
            return _Resp(
                200,
                payload={"choices": [{"message": {"content": "openrouter recovered"}}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    block = ChatBlock()
    result = await block.process("Hello, what is 2+2?", {"stream": False})

    assert result["status"] == "success", result
    assert result.get("provider") != "offline_template", result
    assert "offline mode" not in (result.get("text") or "").lower()
    assert result.get("text") == "openrouter recovered"
    assert any(u.startswith(DEEPSEEK_API_URL.rsplit("/", 2)[0]) for u in urls)
    assert any(u.startswith(OPENROUTER_API_URL.rsplit("/", 2)[0]) for u in urls)


@pytest.mark.asyncio
async def test_cloud_path_routes_through_llm_config_provider(monkeypatch):
    """process() selects the cloud provider via _llm_config and forwards that
    provider's key, URL and default model to _call_cloud."""

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_KEY_08)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    calls: list[dict] = []

    async def fake_call(self, message, model, max_tokens, temperature,
                        stream, api_key, cfg, **kwargs):
        calls.append({"api_key": api_key, "url": cfg["url"], "model": model})
        return {"status": "success", "text": "hi from deepseek",
                "provider": "deepseek", "model": model}

    monkeypatch.setattr(ChatBlock, "_call_cloud", fake_call)

    block = ChatBlock()
    result = await block.process("hi", {"stream": False})

    assert result["status"] == "success"
    assert result["text"] == "hi from deepseek"
    assert len(calls) == 1
    assert calls[0]["api_key"] == STUB_KEY_08
    assert calls[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    # Unpinned model uses the active provider default.
    assert calls[0]["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_call_cloud_omits_authorization_header_when_api_key_empty(monkeypatch):
    """Regression guard for the Ollama-empty-Bearer bug.

    When LLM_PROVIDER=ollama, ``_llm_config()`` returns ``env_key=""`` and the
    chat block passes ``api_key=""`` to ``_call_cloud``. httpx raises
    ``ValueError: Illegal header value b'Bearer '`` if we then send
    ``Authorization: Bearer `` with an empty value — silently breaking the
    entire fast chat path under Ollama.

    The fix mirrors ``runtime.py`` line 1525: omit the Authorization header
    entirely when api_key is empty. This test pins that behaviour by
    intercepting the httpx POST and asserting the absent header.
    """

    import httpx

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None, **kwargs):
            captured["url"] = url
            captured["headers"] = dict(headers or {})
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    block = ChatBlock()
    result = await block._call_cloud(
        message="hi",
        model="qwen3-coder:480b-cloud",
        max_tokens=64,
        temperature=0.2,
        stream=False,
        api_key="",
        cfg={"provider": "ollama", "url": "http://my-pc.tunnel.cf/v1/chat/completions"},
    )

    assert result["status"] == "success"
    headers = captured["headers"]
    assert "Content-Type" in headers
    # The empty-Bearer regression: header MUST be absent when api_key is empty.
    assert "Authorization" not in headers, (
        f"Authorization header must be omitted when api_key is empty (Ollama path); "
        f"got headers={headers}"
    )


@pytest.mark.asyncio
async def test_call_cloud_includes_bearer_when_api_key_set(monkeypatch):
    """Counterpart: a keyed cloud provider must still send the Bearer token."""

    import httpx

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None, **kwargs):
            captured["headers"] = dict(headers or {})
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    block = ChatBlock()
    await block._call_cloud(
        message="hi",
        model="deepseek-chat",
        max_tokens=64,
        temperature=0.2,
        stream=False,
        api_key=STUB_KEY_12,
        cfg={"provider": "deepseek", "url": "https://api.deepseek.com/v1/chat/completions"},
    )

    assert captured["headers"].get("Authorization") == "Bearer " + STUB_KEY_12


def test_chat_block_source_has_no_forbidden_provider_names():
    """Removed cloud providers must not return. DeepSeek was restored
    2026-09-06 as a first-class native chat provider and is allowed."""

    src = inspect.getsource(ChatBlock)
    forbidden = ["anthropic", "openai", "grok", "claude"]
    for term in forbidden:
        assert term.lower() not in src.lower(), f"forbidden provider name '{term}' reappeared in ChatBlock"


def test_chat_block_metadata():
    """Surface fields the platform relies on (name/version/tags) — sanity check."""

    assert ChatBlock.name == "chat"
    assert ChatBlock.version.startswith("3.")
    assert "ai" in ChatBlock.tags
    assert "chat" in ChatBlock.tags


def test_shaped_cloud_payload_honours_a_provider_fixed_temperature():
    """A provider that declares ``fixed_temperature`` / ``reasoning_min_tokens``
    gets them applied; a provider without pins keeps the agent's own values.
    Neither live provider (DeepSeek/OpenRouter) pins today, but the generic
    hook must keep working for a future constrained provider."""
    from app.blocks.chat import _shaped_cloud_payload

    pinned = {
        "provider": "deepseek",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "fixed_temperature": 1,
        "reasoning_min_tokens": 4096,
    }
    payload = _shaped_cloud_payload(
        pinned, model="deepseek-chat", messages=[], max_tokens=128, temperature=0.7,
    )
    assert payload["temperature"] == 1
    assert payload["max_tokens"] >= 4096

    deepseek = {"provider": "deepseek", "url": "https://api.deepseek.com/v1/chat/completions"}
    ds_payload = _shaped_cloud_payload(
        deepseek, model="deepseek-chat", messages=[], max_tokens=128, temperature=0.7,
    )
    assert ds_payload["temperature"] == 0.7
    assert ds_payload["max_tokens"] == 128


@pytest.mark.asyncio
async def test_call_cloud_posts_a_provider_fixed_temperature(monkeypatch):
    """The payload chokepoint applies a provider's fixed_temperature on the wire."""
    import httpx

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None, **kwargs):
            captured["json"] = dict(json or {})
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    block = ChatBlock()
    result = await block._call_cloud(
        message="hi",
        model="deepseek-chat",
        max_tokens=64,
        temperature=0.7,
        stream=False,
        api_key=STUB_KEY_12,
        cfg={
            "provider": "deepseek",
            "url": "https://api.deepseek.com/v1/chat/completions",
            "fixed_temperature": 1,
        },
    )
    assert result["status"] == "success"
    assert captured["json"]["temperature"] == 1
