"""6d: llm_first_token + cache_hit_tokens on the LIVE /v1/chat/stream egress.

The project-assistant answer streams from _stream_synthesis (app/agents/core/
llm.py -> llm_client.provider_stream); that egress had no mark_llm_dispatch, so
on the live endpoint llm_first_token read ~1ms while the real wait hid in
since_arrival (the 6b stamp is on the DRIVER agent, a different path). This
drives the real _stream_synthesis with provider_stream mocked to wait `gap`
before the first delta and to send a usage chunk, calls first_token() on the
first delta as chat.py does, and asserts (1) llm_first_token tracks `gap` and
(2) the provider's prefix-cache hit tokens surface as cache_hit_tokens.
Dropping the mark_llm_dispatch stamp reverts llm_first_token to ~0; dropping
the usage capture drops cache_hit_tokens.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import app.core.turn_timing as tt
import app.core.llm_client as llm_client
import app.agents.runtime as runtime
from app.agents import get_agent, load_agents
from app.agents.core.llm import _cache_hit_tokens


def _wire(monkeypatch, gap: float, hits: int):
    monkeypatch.setattr(runtime, "_llm_config",
                        lambda: {"provider": "deepseek", "url": "https://x/v1/chat", "model": "m"})

    async def _fake_provider_stream(url, payload, headers, timeout=None, provider="",
                                    model="", user_id=None, agent_name=""):
        await asyncio.sleep(gap)
        # An intervening stage fires between dispatch and the first token on the
        # live path (predispatch/retrieval marks reset turn_timing's `last`).
        # Reproduce it so the fallback-to-`last` path would collapse
        # llm_first_token to ~0 -- only the explicit dispatch stamp survives it.
        tt.mark("retrieval")
        yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "A provisional sum "}}]})
        yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "is a contingency."},
                                                   "finish_reason": "stop"}]})
        yield 'data: ' + json.dumps({"choices": [], "usage": {"prompt_cache_hit_tokens": hits}})
        yield 'data: [DONE]'

    monkeypatch.setattr(llm_client, "provider_stream", _fake_provider_stream)


async def _run(agent, gap, hits):
    tt.mark_arrival()
    saw = False
    async for _delta in agent._stream_synthesis([{"role": "user", "content": "q"}], api_key="k",
                                                user_id="6d-tester"):
        if not saw and _delta:
            tt.first_token()      # the endpoint stamps the first token here
            saw = True
    assert saw, "no delta streamed"
    snap = tt.snapshot()
    return snap["llm_first_token"], snap.get("cache_hit_tokens")


@pytest.mark.parametrize("gap, hits", [(0.20, 512), (0.45, 0)])
def test_egress_stamps_real_first_token_and_surfaces_cache_hits(monkeypatch, gap, hits):
    monkeypatch.delenv("USAGE_DAILY_CAP_USD", raising=False)
    _wire(monkeypatch, gap, hits)
    load_agents()
    agent = get_agent("project-assistant")
    lft, cache_hits = asyncio.run(_run(agent, gap, hits))
    # (1) the real dispatch -> first-token wait is captured, not the ~0 of the
    # un-stamped path.
    assert lft >= gap * 0.7, (lft, gap)
    assert lft <= gap + 1.0, (lft, gap)
    # (2) the provider's prefix-cache hit tokens surface verbatim.
    assert cache_hits == hits, (cache_hits, hits)


@pytest.mark.parametrize("usage, expected", [
    ({"prompt_cache_hit_tokens": 300}, 300),                       # DeepSeek shape
    ({"prompt_tokens_details": {"cached_tokens": 128}}, 128),      # OpenAI/OpenRouter shape
])
def test_cache_hit_tokens_reads_both_provider_shapes(usage, expected):
    assert _cache_hit_tokens(usage) == expected
    assert _cache_hit_tokens({}) == 0
    assert _cache_hit_tokens(None) == 0
