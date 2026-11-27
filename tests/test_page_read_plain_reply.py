"""A page read asks the provider for a plain reply.

A provider's hidden reasoning is spent out of the same allowance as the
reply. Measured on two dense pages of a priced bill, it took 5,400 to 16,000
tokens and left the reply empty at the page's 8,000-token limit; with it
switched off the same pages read in full. The provider's config says how a
plain reply is asked for (``plain_reply``); ``llm_client.complete`` sends
those fields only when the caller asks, and only to a provider that declares
them. The page reader keeps the reasoning on its first call (it reads a
page's sum lines best) and asks for a plain reply only on the call after an
empty one. Every provider reply below is scripted.
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


@pytest.mark.parametrize("plain, max_tokens", [(True, bpr.PAGE_READ_MAX_TOKENS), (True, 1500),
                                               (False, bpr.PAGE_READ_MAX_TOKENS), (False, 1500)])
def test_the_page_readers_ask_asks_for_a_plain_reply_only_when_told(monkeypatch, plain, max_tokens):
    cfg = {"provider": "scripted", "url": "http://provider.invalid/v1", "env_key": "", "default_model": "m",
           "plain_reply": _PLAIN}
    lines = [{"type": "item", "bill": "B", "amount": 5}]
    sent = _provider(monkeypatch, cfg, reply=json.dumps({"lines": lines}))
    messages = bpr.page_read_messages(bpr.PAGE_READ_PROMPT, b"png")
    reply = bpr.provider_ask(max_tokens, plain_reply=plain)(messages)
    assert bpr.lines_of_reply(reply) == lines
    (call,) = sent
    assert ("thinking" in call["payload"]) is plain
    assert call["payload"]["max_tokens"] == max_tokens and call["payload"]["temperature"] == 0.0
    assert call["payload"]["messages"] == messages and call["agent_name"] == "bill_page_read"


class _Page:
    def __init__(self, index):
        self.index = index


_READ = json.dumps({"lines": [{"type": "item", "bill": "B", "amount": 5}]})


@pytest.mark.parametrize("first, attempts, want_calls, read", [
    ([_READ], 2, ["ask"], True),                               # read at once: the plain ask is never used
    (["", _READ], 2, ["ask", "again"], True),                  # empty, then read by the plain ask
    (["not json", "", _READ], 3, ["ask", "again", "again"], True),
    (["", ""], 2, ["ask", "again"], False),                    # both empty: the page is NOT READ
])
def test_only_the_calls_after_the_first_go_through_ask_again(first, attempts, want_calls, read):
    replies = list(first)
    calls: list[str] = []

    def ask(messages):
        calls.append("ask")
        return replies.pop(0)

    def again(messages):
        calls.append("again")
        return replies.pop(0)

    got = bpr.model_page_reader(ask, lambda i: b"png", attempts=attempts, ask_again=again)(_Page(0))
    assert calls == want_calls
    assert (bpr.read_lines(got) is not None) is read
    if not read:
        assert got[bpr.NOT_READ_KEY] is True and got["attempts"] == attempts


@pytest.mark.parametrize("error", [RuntimeError("down"), httpx.ConnectError("no route")])
def test_a_call_that_raises_is_followed_by_ask_again_too(error):
    calls: list[str] = []

    def ask(messages):
        calls.append("ask")
        raise error

    def again(messages):
        calls.append("again")
        return _READ

    got = bpr.model_page_reader(ask, lambda i: b"png", ask_again=again)(_Page(1))
    assert calls == ["ask", "again"] and bpr.read_lines(got) is not None


@pytest.mark.parametrize("attempts", [2, 3])
def test_without_ask_again_every_call_goes_through_ask(attempts):
    calls: list[int] = []

    def ask(messages):
        calls.append(len(calls))
        return ""

    got = bpr.model_page_reader(ask, lambda i: b"png", attempts=attempts)(_Page(0))
    assert calls == list(range(attempts)) and got[bpr.NOT_READ_KEY] is True


@pytest.mark.parametrize("tolerance", [0.5, 2.0])
def test_production_keeps_the_reasoning_first_and_asks_plain_on_the_retry(monkeypatch, tmp_path, tolerance):
    from scripts import bill_plan_once as bpo

    made: list[dict] = []
    built: list[dict] = []

    def fake_provider_ask(max_tokens=bpr.PAGE_READ_MAX_TOKENS, *, plain_reply=False):
        made.append({"plain_reply": plain_reply})
        return ("plain" if plain_reply else "reasoning")

    def fake_reader(ask, render, **kw):
        built.append({"ask": ask, **kw})
        return lambda page: None

    monkeypatch.setattr(bpr, "provider_ask", fake_provider_ask)
    monkeypatch.setattr(bpr, "model_page_reader", fake_reader)
    bpo.production_model_read("d1", str(tmp_path / "wd"), tolerance)
    assert made == [{"plain_reply": False}, {"plain_reply": True}]
    assert built == [{"ask": "reasoning", "ask_again": "plain"}]


# --- a completion that fails raises the failure it saw (one place raises it) ---

def _failing(monkeypatch, cfg, outcomes, fallback=None):
    """``complete`` against a provider that answers ``outcomes`` in turn: an
    int is an HTTP status, an exception is raised by the egress, a str is a
    reply. Returns the payloads sent, with the provider each went to."""
    sent: list[dict] = []
    queue = list(outcomes)

    async def post(url, payload, headers, timeout=None, provider="", model="", user_id=None, agent_name=""):
        sent.append({"provider": provider, "payload": dict(payload)})
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        request = httpx.Request("POST", url)
        if isinstance(outcome, int):
            return httpx.Response(outcome, request=request), None
        return httpx.Response(200, request=request), {"choices": [{"message": {"content": outcome}}]}

    monkeypatch.setattr(llm_client, "provider_post", post)
    monkeypatch.setattr(runtime, "_llm_config", lambda provider=None: dict(cfg))
    monkeypatch.setattr(runtime, "_llm_fallback_config", lambda primary: dict(fallback) if fallback else None)
    return sent


_PRIMARY = {"provider": "first", "url": "http://first.invalid/v1", "env_key": "", "default_model": "m1",
            "plain_reply": _PLAIN}
_SECOND = {"provider": "second", "url": "http://second.invalid/v1", "env_key": "", "default_model": "m2"}


@pytest.mark.parametrize("status", [401, 400, 503, 429])
def test_a_failed_completion_raises_the_failure_it_saw(monkeypatch, status):
    sent = _failing(monkeypatch, _PRIMARY, [status])
    with pytest.raises(httpx.HTTPStatusError) as err:
        asyncio.run(llm_client.complete([{"role": "user", "content": "x"}]))
    assert err.value.response.status_code == status and "first HTTP" in str(err.value)
    assert len(sent) == 1


@pytest.mark.parametrize("error", [httpx.ConnectError("no route"), httpx.ReadTimeout("slow"), ValueError("bad body")])
def test_a_network_failure_with_nothing_left_to_try_is_raised_as_it_is(monkeypatch, error):
    _failing(monkeypatch, _PRIMARY, [error])
    with pytest.raises(type(error)) as err:
        asyncio.run(llm_client.complete([{"role": "user", "content": "x"}]))
    assert err.value is error


@pytest.mark.parametrize("status", [401, 503])
def test_a_refusal_raised_by_the_egress_itself_is_never_retried(monkeypatch, status):
    request = httpx.Request("POST", "http://first.invalid/v1")
    refusal = httpx.HTTPStatusError("refused", request=request, response=httpx.Response(status, request=request))
    sent = _failing(monkeypatch, _PRIMARY, [refusal, "never asked"], fallback=_SECOND)
    with pytest.raises(httpx.HTTPStatusError) as err:
        asyncio.run(llm_client.complete([{"role": "user", "content": "x"}]))
    assert err.value is refusal and [c["provider"] for c in sent] == ["first"]


@pytest.mark.parametrize("first, reply", [(503, "from the second"), (429, "ok"), (httpx.ConnectError("down"), "ok")])
def test_the_plain_reply_fields_go_to_the_provider_that_declares_them_only(monkeypatch, first, reply):
    """The primary declares how a plain reply is asked for; the fallback does
    not. A retry on the fallback carries none of the primary's fields."""
    sent = _failing(monkeypatch, _PRIMARY, [first, reply], fallback=_SECOND)
    text = asyncio.run(llm_client.complete([{"role": "user", "content": "x"}], plain_reply=True))
    assert text == reply
    assert [(c["provider"], c["payload"].get("thinking")) for c in sent] == [
        ("first", {"type": "disabled"}), ("second", None)]
    assert "thinking" not in sent[1]["payload"] and sent[1]["payload"]["model"] == "m2"


@pytest.mark.parametrize("statuses", [[503, 503], [429, 500]])
def test_when_the_fallback_fails_too_its_failure_is_the_one_raised(monkeypatch, statuses):
    sent = _failing(monkeypatch, _PRIMARY, list(statuses), fallback=_SECOND)
    with pytest.raises(httpx.HTTPStatusError) as err:
        asyncio.run(llm_client.complete([{"role": "user", "content": "x"}]))
    assert err.value.response.status_code == statuses[1] and "second HTTP" in str(err.value)
    assert [c["provider"] for c in sent] == ["first", "second"]


@pytest.mark.parametrize("status", [401, 400])
def test_a_refusal_that_is_not_retryable_does_not_go_to_the_fallback(monkeypatch, status):
    sent = _failing(monkeypatch, _PRIMARY, [status, "never asked"], fallback=_SECOND)
    with pytest.raises(httpx.HTTPStatusError) as err:
        asyncio.run(llm_client.complete([{"role": "user", "content": "x"}]))
    assert err.value.response.status_code == status and [c["provider"] for c in sent] == ["first"]

