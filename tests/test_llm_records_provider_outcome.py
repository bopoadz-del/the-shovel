"""The agent's _call_llm must record each real provider outcome so /health can
tell "serving" from "configured" (the 19 Oct outage: keys present, every call
402/429, /health still green).

Failing-first: on main (before PR-B added the record_provider_result hooks in
app/agents/core/llm.py) a 402 or a success left the provider snapshot empty, so
these assertions fail. The recorder is keyed by ROLE ('primary'/'fallback'),
never the provider name.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.core import health_probes as hp
from tests._fixture_values import ENV_SECRET_01


@pytest.fixture(autouse=True)
def _reset_provider_state():
    with hp._PROVIDER_LOCK:
        hp._PROVIDER_LAST.clear()
        hp._PROVIDER_COUNTS.update({"billing_402": 0, "rate_429": 0})
    hp._LAST_SERVED_AT = 0.0
    yield


@pytest.fixture
def _deepseek_no_fallback(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER", "OPENROUTER_API_KEY",
                 "USAGE_DAILY_CAP_USD", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", ENV_SECRET_01)


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or ("" if payload is None else json.dumps(payload))
        self.headers = {}
        self.request = None

    def json(self):
        return self._payload


def _agent():
    from app.agents.runtime import Agent

    return Agent(name="safety-officer", description="t", system_prompt="t",
                 allowed_blocks=[], model="deepseek-chat")


def test_call_llm_records_402_as_exhausted(monkeypatch, _deepseek_no_fallback):
    async def post(self, url, json=None, headers=None, **kw):
        return _Resp(402, text='{"error":{"message":"Insufficient Balance"}}')

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    asyncio.run(_agent()._call_llm([{"role": "user", "content": "q"}], "k",
                                   with_tools=False))

    snap = hp.provider_snapshot()
    assert snap["counts"]["billing_402"] >= 1
    assert snap["providers"]["primary"]["status"] == "exhausted"
    assert snap["providers"]["primary"]["http"] == 402


def test_call_llm_records_success_as_ok(monkeypatch, _deepseek_no_fallback):
    body = {"model": "m", "choices": [{"message": {"role": "assistant",
            "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 1}}

    async def post(self, url, json=None, headers=None, **kw):
        return _Resp(200, payload=body)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    resp = asyncio.run(_agent()._call_llm([{"role": "user", "content": "q"}], "k",
                                          with_tools=False))

    assert resp["status"] == "success"
    snap = hp.provider_snapshot()
    assert snap["serving"] is True
    assert snap["providers"]["primary"]["status"] == "ok"
    assert snap["last_served_age_s"] is not None
