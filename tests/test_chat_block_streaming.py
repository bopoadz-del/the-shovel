"""The chat block's streaming generator must actually stream.

Caught by the dead-function gate on its first CI run: `_stream_generator` was
covered on a developer machine but had ZERO coverage in CI, so the streaming
fast-path was effectively untested on the only environment that ships. A
generator is lazy -- `_call_cloud` returns it without executing a single line,
so a test that merely calls `_call_cloud` and asserts `status == "success"`
covers none of the streaming body and still looks green.

These consume the generator, which is the only way its body runs at all.

Mocked at the httpx TRANSPORT boundary so every line of our own parsing runs:
the SSE framing, the `[DONE]` sentinel, the non-200 error path, and the
Ollama empty-Bearer guard the comments call out by name.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.blocks.chat import ChatBlock
from tests._fixture_values import API_KEY_A


class _FakeStream:
    """Async context manager standing in for httpx's streaming response."""

    def __init__(self, lines, status_code=200, body=b""):
        self._lines = lines
        self.status_code = status_code
        self._body = body
        self.captured_headers = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


def _sse(content: str) -> str:
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": content}}]}
    )


def _patch_stream(fake, recorder=None):
    """Patch httpx.AsyncClient.stream, recording the kwargs it was called with."""
    def _stream(self, method, url, **kwargs):
        if recorder is not None:
            recorder.update(kwargs)
        return fake
    return patch("httpx.AsyncClient.stream", _stream)


CFG = {"url": "https://api.example.test/v1/chat/completions",
       "provider": "test-provider"}


async def _drain(result):
    return [chunk async for chunk in result["stream"]]


@pytest.mark.asyncio
async def test_streamed_tokens_reach_the_consumer():
    fake = _FakeStream([_sse("Hello"), "", _sse(" world"), "data: [DONE]"])
    block = ChatBlock()
    with _patch_stream(fake):
        result = await block._call_cloud(
            "hi", "gpt-4o-mini", 100, 0.2, True, "sk-test", CFG, None,
        )
        assert result["status"] == "success"
        chunks = await _drain(result)
    body = "".join(chunks)
    assert "Hello" in body and "world" in body


@pytest.mark.asyncio
async def test_done_sentinel_is_not_emitted_as_content():
    """`[DONE]` is protocol, not text. Leaking it puts the literal string in
    the user's answer."""
    fake = _FakeStream([_sse("done-check"), "data: [DONE]"])
    block = ChatBlock()
    with _patch_stream(fake):
        chunks = await _drain(await block._call_cloud(
            "hi", "m", 10, 0.0, True, "sk", CFG, None))
    assert "[DONE]" not in "".join(chunks)


@pytest.mark.asyncio
async def test_non_sse_lines_are_ignored_not_parsed():
    """Keep-alive blanks and comment lines must not crash the parse."""
    fake = _FakeStream(["", ": keep-alive", "garbage", _sse("ok")])
    block = ChatBlock()
    with _patch_stream(fake):
        chunks = await _drain(await block._call_cloud(
            "hi", "m", 10, 0.0, True, "sk", CFG, None))
    assert "ok" in "".join(chunks)


@pytest.mark.asyncio
async def test_upstream_error_is_reported_as_an_error_event():
    """A non-200 must surface as a typed error chunk, not an empty stream that
    reads to the user as the model having nothing to say."""
    fake = _FakeStream([], status_code=500, body=b"upstream exploded")
    block = ChatBlock()
    with _patch_stream(fake):
        chunks = await _drain(await block._call_cloud(
            "hi", "m", 10, 0.0, True, "sk", CFG, None))
    assert chunks, "an upstream failure produced a silent empty stream"
    first = json.loads(chunks[0])
    assert first["type"] == "error"
    assert "500" in first["message"]


@pytest.mark.asyncio
async def test_empty_api_key_omits_the_authorization_header():
    """The documented empty-Bearer guard: an empty key must send no
    Authorization header, because ``Bearer `` makes httpx raise "Illegal
    header value" before the request leaves the client, silently breaking the
    whole fast chat path for a keyless configuration."""
    seen: dict = {}
    fake = _FakeStream([_sse("x"), "data: [DONE]"])
    block = ChatBlock()
    with _patch_stream(fake, recorder=seen):
        await _drain(await block._call_cloud(
            "hi", "m", 10, 0.0, True, "", CFG, None))
    assert "Authorization" not in (seen.get("headers") or {}), seen.get("headers")


@pytest.mark.asyncio
async def test_a_real_api_key_does_send_the_authorization_header():
    """The other direction, so the guard cannot degrade into never authing."""
    seen: dict = {}
    fake = _FakeStream([_sse("x"), "data: [DONE]"])
    block = ChatBlock()
    with _patch_stream(fake, recorder=seen):
        await _drain(await block._call_cloud(
            "hi", "m", 10, 0.0, True, API_KEY_A, CFG, None))
    assert (seen.get("headers") or {}).get("Authorization") == "Bearer " + API_KEY_A


@pytest.mark.asyncio
async def test_stream_payload_honours_a_provider_fixed_temperature():
    """Streaming ChatBlock must apply a provider's ``fixed_temperature`` pin,
    same as the non-stream path. Neither live provider (DeepSeek/OpenRouter)
    pins one today, but the generic hook must keep working for a future
    constrained provider."""
    seen: dict = {}
    fake = _FakeStream([_sse("x"), "data: [DONE]"])
    block = ChatBlock()
    pinned_cfg = {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "provider": "deepseek",
        "fixed_temperature": 1,
    }
    with _patch_stream(fake, recorder=seen):
        await _drain(await block._call_cloud(
            "hi", "deepseek-chat", 10, 0.7, True, "sk-live", pinned_cfg, None))
    assert (seen.get("json") or {}).get("temperature") == 1
    assert (seen.get("json") or {}).get("stream") is True
