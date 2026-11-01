"""6d-ctx: llm_first_token + cache_hit_tokens on the NON-streaming answer path.

#60 stamped _stream_synthesis, but that only runs on the force_synthesis branch
(app/agents/core/streaming.py: `if force_synthesis and _synth_stream_enabled`).
A short /v1/chat/stream ask has force_synthesis=False and answers from
`_call_llm` (the single non-streaming egress `llm_client.provider_post`), whose
response is then chunked into token events -- so on live, llm_first_token still
collapsed to ~1ms and cache_hit_tokens never appeared (measured: after-harness
n=30 on 2cfef3f). This drives the real _call_llm with provider_post mocked to
wait `gap` before replying and to carry a usage block, fires an intervening
turn_timing stage between the dispatch stamp and the reply (so the
fallback-to-`last` path would collapse the metric), then calls first_token() as
the router does, and asserts (1) llm_first_token tracks `gap` and (2) the
provider's prefix-cache hit tokens surface as cache_hit_tokens.

Mutants: drop the mark_llm_dispatch stamp before provider_post -> llm_first_token
reverts to ~0 (reddens test 1); drop the mark_value("cache_hit_tokens", ...)
before the success return -> cache_hit_tokens is absent (reddens test 2).

Imported full-dotted so CI-M selects this file for the llm.py / _call_llm
mutants (mutation_gate.tests_for matches `import app.agents.core.llm`).
"""
from __future__ import annotations

import asyncio

import pytest

import app.core.turn_timing as tt
import app.core.llm_client as llm_client
import app.agents.runtime as runtime
import app.agents.core.llm  # noqa: F401 -- full dotted path for CI-M selection
from app.agents import get_agent, load_agents


class _Resp:
    status_code = 200

    def __init__(self, hits: int):
        self._hits = hits

    def json(self):
        return {
            "choices": [{"message": {"content": "A provisional sum is a contingency.",
                                     "tool_calls": []}}],
            "usage": {"prompt_cache_hit_tokens": self._hits},
            "model": "m",
        }


def _wire(monkeypatch, gap: float, hits: int):
    monkeypatch.delenv("USAGE_DAILY_CAP_USD", raising=False)
    monkeypatch.setattr(runtime, "_llm_config",
                        lambda: {"provider": "deepseek", "url": "https://x/v1/chat",
                                 "model": "m", "env_key": ""})

    async def _fake_provider_post(url, payload, headers, timeout=None, provider="",
                                  model="", user_id=None, agent_name=""):
        await asyncio.sleep(gap)
        # An intervening stage fires between dispatch and the reply on the live
        # path (retrieval/predispatch marks reset turn_timing's `last`); reproduce
        # it so a fallback-to-`last` would collapse llm_first_token to ~0 -- only
        # the explicit dispatch stamp survives it.
        tt.mark("retrieval")
        resp = _Resp(hits)
        return resp, resp.json()

    monkeypatch.setattr(llm_client, "provider_post", _fake_provider_post)


async def _run(agent, api_key="k"):
    tt.mark_arrival()
    resp = await agent._call_llm([{"role": "user", "content": "q"}], api_key=api_key,
                                 with_tools=False, user_id="6dctx-tester")
    assert resp.get("status") == "success", resp
    # The router fires first_token() on the first chunk of the chunked answer.
    tt.first_token()
    snap = tt.snapshot()
    return snap["llm_first_token"], snap.get("cache_hit_tokens")


@pytest.mark.parametrize("gap, hits", [(0.20, 512), (0.45, 0)])
def test_call_llm_stamps_real_first_token_and_surfaces_cache_hits(monkeypatch, gap, hits):
    _wire(monkeypatch, gap, hits)
    load_agents()
    agent = get_agent("project-assistant")
    lft, cache_hits = asyncio.run(_run(agent))
    # (1) the real dispatch -> reply wait is captured, not the ~0 of the
    # un-stamped path (the intervening mark reset `last`).
    assert lft >= gap * 0.7, (lft, gap)
    assert lft <= gap + 1.0, (lft, gap)
    # (2) the provider's prefix-cache hit tokens surface verbatim.
    assert cache_hits == hits, (cache_hits, hits)
