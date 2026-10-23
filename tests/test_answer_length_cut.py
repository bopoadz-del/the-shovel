"""An answer the model stopped writing part-way never ends mid-sentence.

The output token limit (``finish_reason=length``) or a dropped stream can stop
the text anywhere. Whatever stopped it, what the user sees ends on its last
whole sentence and then says why it stops, on both the streamed and the
non-streamed path. Inputs are generated paragraphs cut at varied points.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from copy import deepcopy

import httpx
import pytest

from app.agents import answer_exit
from tests._fixture_values import ENV_SECRET_01

_SEEDS = range(20261009, 20261009 + 12)
_NOUNS = ("specification", "contract", "drainage", "tender", "survey", "method", "addendum", "schedule",
          "pile", "minutes", "programme", "drawing")


def _ax():
    return answer_exit


def _paragraphs(seed: int) -> str:
    rng = random.Random(seed)
    sentences = []
    for _ in range(rng.randint(3, 6)):
        words = rng.sample(_NOUNS, rng.randint(3, 6))
        figure = rng.choice(("", f" of {rng.randint(2, 900)}.{rng.randint(1, 9)} m", " e.g. the slab"))
        sentences.append(" ".join(words).capitalize() + figure + rng.choice((".", "!", "?")))
    lines = [" ".join(sentences[:2]), "", "Entries:", *[f"- {s}" for s in sentences[2:]]]
    return "\n".join(lines)


_CUT_CASES = [(s, frac) for s in _SEEDS for frac in (0.31, 0.47, 0.62, 0.79, 0.93)]


@pytest.mark.parametrize("seed,frac", _CUT_CASES)
def test_cut_answer_ends_on_its_last_whole_sentence(seed, frac):
    end_on = getattr(_ax(), "end_on_a_sentence", None)
    assert end_on is not None, "no sentence-end rule for a cut answer"
    full = _paragraphs(seed)
    cut = full[: max(1, int(len(full) * frac))]
    out = end_on(cut)
    assert full.startswith(out), (cut, out)
    assert not out or re.search(r"[.!?]$", out), (cut, out)
    assert not re.search(r"\b(?:e\.g|No)\.$", out), out
    assert end_on(full) == full


def _length_body(content: str, reason: str = "length") -> dict:
    return {"model": "m", "choices": [{"message": {"role": "assistant", "content": content},
                                       "finish_reason": reason}], "usage": {"total_tokens": 1}}


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = {}
        self.request = None

    def json(self):
        return deepcopy(self._payload)


def _agent():
    from app.agents.runtime import Agent

    return Agent(name="safety-officer", description="t", system_prompt="t", allowed_blocks=[],
                 model="deepseek-chat")


@pytest.fixture
def _deepseek(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_FALLBACK_PROVIDER", "OPENROUTER_API_KEY", "USAGE_DAILY_CAP_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", ENV_SECRET_01)


@pytest.mark.parametrize("seed,frac", _CUT_CASES[::3])
def test_non_streamed_length_cut_ends_on_a_sentence(monkeypatch, _deepseek, seed, frac):
    from app.agents.runtime import _LENGTH_CUT_NOTICE

    full = _paragraphs(seed)
    cut = full[: int(len(full) * frac)]

    async def post(self, url, json=None, headers=None, **kw):
        return _Resp(_length_body(cut))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    resp = asyncio.run(_agent()._call_llm([{"role": "user", "content": "q"}], "k", with_tools=False))
    content = resp["choice"]["message"]["content"]
    assert content.endswith(_LENGTH_CUT_NOTICE), content
    body = content[: -len(_LENGTH_CUT_NOTICE)].rstrip()
    assert full.startswith(body) and (not body or re.search(r"[.!?]$", body)), content


def test_non_streamed_complete_answer_is_untouched(monkeypatch, _deepseek):
    async def post(self, url, json=None, headers=None, **kw):
        return _Resp(_length_body("The bond is 10% of the contract price, and it", reason="stop"))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    resp = asyncio.run(_agent()._call_llm([{"role": "user", "content": "q"}], "k", with_tools=False))
    assert resp["choice"]["message"]["content"] == "The bond is 10% of the contract price, and it"


class _StreamResp:
    def __init__(self, lines):
        self.status_code = 200
        self.text = ""
        self._lines = lines

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _CM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


@pytest.mark.parametrize("seed", list(_SEEDS)[:4])
def test_streamed_length_cut_is_signalled_after_every_token(monkeypatch, _deepseek, seed):
    from app.agents.runtime import _SynthLengthCut

    full = _paragraphs(seed)
    deltas = [full[i:i + 7] for i in range(0, len(full) // 2, 7)]
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": d}}]}) for d in deltas]
    lines.append("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}))
    lines.append("data: [DONE]")
    monkeypatch.setattr(httpx.AsyncClient, "stream", lambda self, *a, **k: _CM(_StreamResp(lines)))
    got: list[str] = []

    async def run():
        async for d in _agent()._stream_synthesis([{"role": "user", "content": "q"}], "k"):
            got.append(d)

    with pytest.raises(_SynthLengthCut):
        asyncio.run(run())
    assert got == deltas


@pytest.mark.parametrize("seed,frac", _CUT_CASES[::4])
def test_streamed_turn_cut_part_way_ends_on_a_sentence(monkeypatch, seed, frac):
    """The streamed synthesis stops part-way (length limit or a dropped
    stream): the end event the client shows ends on a whole sentence."""
    from unittest.mock import patch

    from app.agents.runtime import (
        _LENGTH_CUT_NOTICE, _SYNTH_CUTOFF_NOTICE, Agent, _SynthLengthCut, _SynthStreamError,
    )

    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: {
        "provider": "deepseek", "url": "https://api.deepseek.com/v1/chat/completions",
        "env_key": "DEEPSEEK_API_KEY", "default_model": "deepseek-chat"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", ENV_SECRET_01)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")
    full = _paragraphs(seed)
    cut = full[: int(len(full) * frac)]

    async def call_llm(_self, messages, api_key, **kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            return {"status": "success", "choice": {"message": {"role": "assistant", "content": "",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "commissioning_checklist", "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {"role": "assistant", "content": "unused"}}}

    async def tool_ok(self, tool_call, **kwargs):
        return {"name": "commissioning_checklist", "ok": True,
                "result": {"status": "success", "checklists_by_system": {"electrical": []}}}

    for error, notice in ((_SynthLengthCut("limit"), _LENGTH_CUT_NOTICE),
                          (_SynthStreamError("drop"), _SYNTH_CUTOFF_NOTICE)):
        async def stream(self, messages, api_key, **kwargs):
            for i in range(0, len(cut), 9):
                yield cut[i:i + 9]
            raise error

        agent = Agent(name="project-assistant", description="pa", system_prompt="x",
                      allowed_blocks=["construction"])
        with patch.object(Agent, "_call_llm", call_llm), patch.object(Agent, "_run_tool_call", tool_ok), \
                patch.object(Agent, "_stream_synthesis", stream):
            async def drain():
                return [ev async for ev in agent.chat_stream(
                    user_message="Generate a commissioning checklist for electrical.", history=[],
                    project_id=None, conversation_id=None, user_id=None)]

            events = asyncio.run(drain())
        end = events[-1]
        assert end["type"] == "end", end
        content = end.get("content") or ""
        assert content.rstrip().endswith(notice), content
        body = content.rstrip()[: -len(notice)].rstrip()
        assert not body or re.search(r"[.!?]$", body), content
