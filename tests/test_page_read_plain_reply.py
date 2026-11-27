"""A page read asks the provider for a plain reply.

A provider's hidden reasoning is spent out of the same allowance as the
reply. Measured on two dense pages of a priced bill, it took 5,400 to 16,000
tokens and left the reply empty at the page's 8,000-token limit; with it
switched off the same pages read in full. The provider's config says how a
plain reply is asked for (``plain_reply``); ``llm_client.complete`` sends
those fields only when the caller asks, and only to a provider that declares
them; the page reader asks. Every provider reply below is scripted.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.agents import runtime
from app.core import bill_page_read as bpr
from app.core import llm_client


@pytest.mark.parametrize("provider, declared", [
    (None, {"thinking": {"type": "disabled"}}),
    ("deepseek", {"thinking": {"type": "disabled"}}),
    ("something-else", {"thinking": {"type": "disabled"}}),    # unrecognised resolves to the primary
    ("openrouter", None),
])
def test_llm_config_declares_how_a_provider_is_asked_for_a_plain_reply(monkeypatch, provider, declared):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    cfg = runtime._llm_config(provider)
    assert cfg.get("plain_reply") == declared
    assert cfg["provider"] == ("openrouter" if provider == "openrouter" else "deepseek")


def _provider(monkeypatch, cfg, reply="  the reply  "):
    """``llm_client.complete`` against one scripted provider: what it was sent."""
    sent: list[dict] = []

    async def post(url, payload, headers, timeout=None, provider="", model="", user_id=None, agent_name=""):
        sent.append({"url": url, "payload": dict(payload), "timeout": timeout, "agent_name": agent_name})
        request = httpx.Request("POST", url)
        return httpx.Response(200, request=request), {"choices": [{"message": {"content": reply}}]}

    monkeypatch.setattr(llm_client, "provider_post", post)
    monkeypatch.setattr(runtime, "_llm_config", lambda provider=None: dict(cfg))
    monkeypatch.setattr(runtime, "_llm_fallback_config", lambda primary: None)
    return sent


_PLAIN = {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("declared, asked, extra", [
    (_PLAIN, True, _PLAIN),                                  # declared and asked for: sent
    ({"reasoning": "off", "effort": 0}, True, {"reasoning": "off", "effort": 0}),   # whatever the provider declares
    (_PLAIN, False, {}),                                     # not asked for: an ordinary completion
    (None, True, {}),                                        # asked for, the provider declares nothing: nothing sent
    ({}, True, {}),
])
def test_complete_sends_the_providers_plain_reply_fields_only_when_asked(monkeypatch, declared, asked, extra):
    cfg = {"provider": "scripted", "url": "http://provider.invalid/v1", "env_key": "", "default_model": "m"}
    if declared is not None:
        cfg["plain_reply"] = declared
    sent = _provider(monkeypatch, cfg)
    text = asyncio.run(llm_client.complete([{"role": "user", "content": "x"}], max_tokens=123,
                                           plain_reply=asked))
    assert text == "the reply"
    (call,) = sent
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.0,
            "max_tokens": 123, "stream": False}
    assert call["payload"] == {**base, **extra}
    assert call["timeout"] is not None


@pytest.mark.parametrize("max_tokens", [bpr.PAGE_READ_MAX_TOKENS, 1500])
def test_the_page_reader_asks_for_a_plain_reply(monkeypatch, max_tokens):
    cfg = {"provider": "scripted", "url": "http://provider.invalid/v1", "env_key": "", "default_model": "m",
           "plain_reply": _PLAIN}
    lines = [{"type": "item", "bill": "B", "amount": 5}]
    sent = _provider(monkeypatch, cfg, reply=json.dumps({"lines": lines}))
    messages = bpr.page_read_messages(bpr.PAGE_READ_PROMPT, b"png")
    reply = bpr.provider_ask(max_tokens)(messages)
    assert bpr.lines_of_reply(reply) == lines
    (call,) = sent
    assert call["payload"]["thinking"] == {"type": "disabled"}
    assert call["payload"]["max_tokens"] == max_tokens and call["payload"]["temperature"] == 0.0
    assert call["payload"]["messages"] == messages and call["agent_name"] == "bill_page_read"
