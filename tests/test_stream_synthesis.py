"""`_stream_synthesis` -- the path that decides whether the user watches an
answer appear or waits in silence.

Audit bar 3, 2026-08-12, agents subsystem. Replacing the whole body with a bare
`return` left all 150 agent tests green. One test file referenced it, and every
reference was a mock.

The reason a gutted body is invisible is that this function is DESIGNED to be
fallible: it raises `_SynthStreamError` on any pre-first-token failure and the
caller silently falls back to the non-streaming `_call_llm`. So a version that
never streams anything produces a working platform that is merely slower --
which no test notices, and which a user experiences as "it hangs for 40 seconds
then dumps the whole answer at once".

The invariant that carries real risk is the mid-stream one. The caller must
never restart a stream it has already begun, and it distinguishes the two cases
solely by whether tokens were emitted. If a mid-stream drop stopped raising, or
started raising before yielding what it had, the user would see a duplicated or
truncated answer. That is pinned explicitly below.

Mocked at `httpx.AsyncClient.stream`, which is a different shape from
`_call_llm`'s `post` -- a synchronous call returning an async context manager
whose object exposes `status_code`, `aread()` and `aiter_lines()`.
"""
from __future__ import annotations

import json
from unittest import mock

import httpx
import pytest

from app.agents import runtime
from app.agents.runtime import Agent, _SynthStreamError
from tests._fixture_values import STUB_KEY_05, STUB_KEY_06, STUB_KEY_10

_LLM_ENV = (
    "LLM_PROVIDER", "LLM_FALLBACK_PROVIDER",
    "KIMI_API_KEY", "KIMI_MODEL", "KIMI_FALLBACK_MODEL",
    "GROQ_API_KEY", "GROQ_MODEL",
    "OLLAMA_API_KEY", "OLLAMA_URL", "OLLAMA_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_ALLOW_PAID",
    "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
    "USAGE_DAILY_CAP_USD",
)

@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    """Same load-bearing isolation as the `_call_llm` suite: `_llm_config`
    reads every one of these live, and the provider gate at the top of this
    function turns a leaked LLM_PROVIDER into an immediate raise that would
    look like the behaviour under test."""
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)


def _sse(*deltas, usage=None, done=True):
    """An OpenAI-shaped SSE line sequence."""
    lines = []
    for d in deltas:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": d}}]}))
        lines.append("")            # blank keep-alive lines are normal and skipped
    if usage:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    if done:
        lines.append("data: [DONE]")
    return lines


def _reasoning_sse(thinking=(), answer=(), done=True):
    """A reasoning model's SSE sequence: the thinking phase streams first as
    `reasoning_content`, then the answer streams as `content`. Shape taken from
    a live kimi-k2.6 stream, not invented."""
    lines = ["data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]})]
    for t in thinking:
        lines.append("data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": t}}]}))
    for a in answer:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": a}}]}))
    if done:
        lines.append("data: [DONE]")
    return lines


class _StreamResponse:
    def __init__(self, lines, status_code=200, text="", mid_stream_error=None):
        self.status_code = status_code
        self.text = text
        self._lines = list(lines)
        self._mid_stream_error = mid_stream_error
        self.aread_called = False

    async def aread(self):
        self.aread_called = True
        return self.text.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        if self._mid_stream_error is not None:
            raise self._mid_stream_error


class _StreamCM:
    def __init__(self, response, recorder, args, kwargs):
        self._response = response
        self._recorder = recorder
        self._args = args
        self._kwargs = kwargs

    async def __aenter__(self):
        self._recorder.append({
            "method": self._args[0] if self._args else None,
            "url": self._args[1] if len(self._args) > 1 else self._kwargs.get("url"),
            "payload": self._kwargs.get("json"),
            "headers": self._kwargs.get("headers"),
        })
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def stream(monkeypatch):
    """Install a fake `client.stream`. Returns (calls, install)."""
    calls: list[dict] = []

    def _install(response):
        def _stream(self, *args, **kwargs):
            return _StreamCM(response, calls, args, kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "stream", _stream, raising=True)
        return calls

    return _install


def _agent(model="deepseek-chat", temperature=0.3):
    return Agent(
        name="project-assistant",
        description="test agent",
        system_prompt="you are a test",
        allowed_blocks=["construction"],
        model=model,
        temperature=temperature,
    )


def _deepseek(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_KEY_05)


async def _drain(agent, **kw):
    return [chunk async for chunk in agent._stream_synthesis(
        [{"role": "user", "content": "summarise the findings"}],
        kw.pop("api_key", STUB_KEY_05), **kw)]


# ── the provider gate ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openrouter_streams_like_groq(monkeypatch, stream):
    """OpenRouter is OpenAI-compatible; SYNTHESIS_STREAMING must not skip it."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_KEY_06)
    stream(_StreamResponse(_sse("ok")))

    assert await _drain(_agent(), api_key=STUB_KEY_06) == ["ok"]


@pytest.mark.asyncio
async def test_deepseek_streams_like_groq(monkeypatch, stream):
    """DeepSeek is OpenAI-compatible; SYNTHESIS_STREAMING must not skip it."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_KEY_05)
    stream(_StreamResponse(_sse("ok")))

    assert await _drain(_agent(), api_key=STUB_KEY_05) == ["ok"]


@pytest.mark.asyncio
async def test_an_unverified_provider_is_refused_before_the_call(monkeypatch, stream):
    """The allowlist is what keeps an unverified provider's stream shape from
    reaching the user as answer text. It must fire before any network call.

    `_llm_config` can only return deepseek/openrouter, so an unverified
    provider is synthesised here by patching `_llm_config` directly -- the gate
    must still refuse anything outside the allowlist.
    """
    monkeypatch.setattr(runtime, "_llm_config", lambda: {
        "provider": "someone-elses-cloud",
        "url": "http://test:11434/v1/chat/completions",
        "env_key": "",
        "default_model": "x",
    })
    calls = stream(_StreamResponse(_sse("never")))

    with pytest.raises(_SynthStreamError, match="only verified"):
        await _drain(_agent(), api_key=STUB_KEY_10)
    assert calls == [], "an unverified provider was called before the gate fired"


# ── reasoning models: the thinking phase is not the answer ───────────────

@pytest.mark.asyncio
async def test_reasoning_deltas_are_never_yielded_as_answer_text(monkeypatch, stream):
    """Measured against the live Moonshot API on 2026-08-24: kimi-k2.6 emits
    181 `reasoning_content` deltas starting at 1.62s, then 66 `content` deltas
    starting at 6.0s. Both arrive on the same `delta` object.

    `reasoning_content` is discarded intermediate work — it contains claims the
    model went on to reject. Yielding it would put that text in front of the
    user as though it were the answer, and it is not what gets persisted, so
    the streamed and stored answers would disagree.
    """
    _deepseek(monkeypatch)
    stream(_StreamResponse(_reasoning_sse(
        thinking=["The user asks about retention. ", "Maybe 20%? No, that is wrong. "],
        answer=["Retention is ", "typically 5% to 10%."],
    )))

    assert await _drain(_agent(), api_key=STUB_KEY_05) == [
        "Retention is ", "typically 5% to 10%.",
    ]


@pytest.mark.asyncio
async def test_a_thinking_only_response_yields_nothing_and_says_why(
    monkeypatch, stream,
):
    """A reasoning model can spend its whole shared token budget on thinking and
    return HTTP 200 with no content at all — that is the documented failure the
    `reasoning_min_tokens` floor exists to prevent, and the floor can be
    defeated by a low agent budget.

    Yielding nothing is correct: the caller falls back to non-streaming. But in
    the logs it is indistinguishable from a dead provider, so it has to be
    named.

    Asserted against the logger itself rather than `caplog`: the full suite
    reconfigures this logger, so a caplog assertion passes alone and fails in
    CI depending on what ran before it.
    """
    _deepseek(monkeypatch)
    stream(_StreamResponse(_reasoning_sse(thinking=["thinking ", "hard "], answer=[])))

    with mock.patch.object(runtime._LOG, "warning") as warn:
        assert await _drain(_agent(), api_key=STUB_KEY_05) == []
    logged = " ".join(str(c.args[0]) for c in warn.call_args_list if c.args)
    assert "reasoning deltas and no content" in logged, (
        "a thinking-only response was indistinguishable from a dead provider"
    )
    # The count is the diagnostic — "some reasoning" and "181 reasoning" are
    # different problems.
    assert 2 in warn.call_args_list[0].args, warn.call_args_list


# ── the happy path ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_content_deltas_are_yielded_in_order(monkeypatch, stream):
    _deepseek(monkeypatch)
    stream(_StreamResponse(_sse("The ", "notice ", "period is 28 days.")))

    assert await _drain(_agent()) == ["The ", "notice ", "period is 28 days."]


@pytest.mark.asyncio
async def test_the_stream_stops_at_done(monkeypatch, stream):
    """Anything after [DONE] is not part of the answer. Emitting it would
    append provider bookkeeping to the user's text."""
    _deepseek(monkeypatch)
    lines = _sse("answer") + [
        "data: " + json.dumps({"choices": [{"delta": {"content": "LEAKED"}}]})
    ]
    stream(_StreamResponse(lines))

    assert await _drain(_agent()) == ["answer"]


@pytest.mark.asyncio
async def test_blank_and_unparseable_lines_do_not_break_the_stream(monkeypatch, stream):
    """Keep-alives and partial frames are routine. Treating one as fatal would
    truncate an answer mid-sentence for a transport artefact."""
    _deepseek(monkeypatch)
    lines = ["", "data: {not json", ": keep-alive",
             "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
             "data: [DONE]"]
    stream(_StreamResponse(lines))

    assert await _drain(_agent()) == ["ok"]


@pytest.mark.asyncio
async def test_no_tools_are_offered_and_streaming_is_requested(monkeypatch, stream):
    """"Tool iterations stay non-streaming" is the documented contract. A tool
    definition here would let a tool_call delta appear mid-stream, which this
    path has no way to dispatch."""
    _deepseek(monkeypatch)
    calls = stream(_StreamResponse(_sse("x")))

    await _drain(_agent())

    payload = calls[0]["payload"]
    assert "tools" not in payload and "tool_choice" not in payload, payload
    assert payload["stream"] is True, payload
    assert payload["stream_options"] == {"include_usage": True}, payload


@pytest.mark.asyncio
async def test_it_mirrors_call_llm_on_model_and_temperature(monkeypatch, stream):
    """The docstring commits to mirroring `_call_llm`'s setup. Drift means the
    streamed answer and the fallback answer come from different models -- the
    user sees a different voice depending on transport."""
    _deepseek(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    calls = stream(_StreamResponse(_sse("x")))

    # A legacy/foreign hat pin is replaced by the provider default, exactly as
    # _call_llm does it, so the streamed model matches the non-streaming hop.
    await _drain(_agent(model="kimi-k2.6", temperature=0.3))

    assert calls[0]["payload"]["model"] == "deepseek-reasoner", (
        "a dead provider's model name was streamed instead of the provider default"
    )
    assert calls[0]["payload"]["temperature"] == 0.3


@pytest.mark.asyncio
async def test_non_standard_message_fields_never_reach_the_provider(monkeypatch, stream):
    """The same `reasoning`-field 400 that hung deliverable generation. The
    docstring calls this chokepoint critical, so it is asserted at the wire."""
    _deepseek(monkeypatch)
    calls = stream(_StreamResponse(_sse("x")))

    messages = [{"role": "assistant", "content": "t", "reasoning": "chain of thought"},
                {"role": "user", "content": "and now?"}]
    async for _ in _agent()._stream_synthesis(messages, STUB_KEY_05):
        pass

    sent = calls[0]["payload"]["messages"]
    assert not any("reasoning" in m for m in sent), sent


# ── the failures, and which side of the first token they fall on ─────────

@pytest.mark.asyncio
async def test_a_non_200_raises_before_any_token(monkeypatch, stream):
    """Pre-first-token, so the caller may safely fall back to `_call_llm`."""
    _deepseek(monkeypatch)
    response = _StreamResponse([], status_code=503, text="service unavailable")
    stream(response)

    with pytest.raises(_SynthStreamError, match="503"):
        await _drain(_agent())
    assert response.aread_called, (
        "the error body was never read, so the raised message cannot carry the "
        "provider's reason"
    )


@pytest.mark.asyncio
async def test_a_connect_error_raises_before_any_token(monkeypatch, stream):
    _deepseek(monkeypatch)
    stream(httpx.ConnectError("dns failure"))

    with pytest.raises(_SynthStreamError, match="stream error"):
        await _drain(_agent())


@pytest.mark.asyncio
async def test_a_mid_stream_drop_surfaces_after_the_tokens_already_emitted(
        monkeypatch, stream):
    """The invariant with real user-visible consequences.

    The caller distinguishes a pre-first-token failure from a mid-stream drop
    by whether it has emitted anything, and never restarts a stream it has
    begun. If the tokens were swallowed and the raise happened first, the
    caller would treat it as pre-first-token and re-run the whole turn through
    `_call_llm` -- the user sees the answer twice.
    """
    _deepseek(monkeypatch)
    stream(_StreamResponse(
        _sse("The notice ", "period is ", done=False),
        mid_stream_error=httpx.ReadError("connection reset"),
    ))

    emitted = []
    with pytest.raises(_SynthStreamError):
        async for chunk in _agent()._stream_synthesis(
                [{"role": "user", "content": "q"}], STUB_KEY_05):
            emitted.append(chunk)

    assert emitted == ["The notice ", "period is "], (
        f"tokens produced before the drop were lost: {emitted}"
    )


@pytest.mark.asyncio
async def test_the_synth_stream_error_is_not_rewrapped(monkeypatch, stream):
    """`except _SynthStreamError: raise` sits above the catch-all so the
    specific message survives. Wrapping it would bury the HTTP status the
    operator needs in a generic "stream error"."""
    _deepseek(monkeypatch)
    stream(_StreamResponse([], status_code=429, text="rate limited"))

    with pytest.raises(_SynthStreamError) as exc:
        await _drain(_agent())
    assert "429" in str(exc.value), str(exc.value)
    assert "stream error" not in str(exc.value), (
        f"the specific error was rewrapped by the catch-all: {exc.value}"
    )


@pytest.mark.asyncio
async def test_a_user_over_the_daily_cap_falls_back_instead_of_streaming(
        monkeypatch, stream):
    """Raising sends the turn to `_call_llm`, which emits the structured cap
    error the UI expects. Streaming past the cap would spend money the cap
    exists to stop."""
    _deepseek(monkeypatch)
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "5.00")
    from app.core import usage_tracker
    monkeypatch.setattr(usage_tracker, "is_over_cap", lambda user_id, cap: True)
    calls = stream(_StreamResponse(_sse("never")))

    with pytest.raises(_SynthStreamError, match="cap"):
        await _drain(_agent(), user_id="u1")
    assert calls == [], "the provider was called despite the cap being hit"


@pytest.mark.asyncio
async def test_a_broken_usage_tracker_does_not_block_the_stream(monkeypatch, stream):
    """Explicit in the code: a broken tracker must not block. The failure it
    prevents is every turn silently losing streaming because a bookkeeping
    table is unreachable."""
    _deepseek(monkeypatch)
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "5.00")
    from app.core import usage_tracker

    def _boom(*a, **kw):
        raise RuntimeError("usage table unreachable")

    monkeypatch.setattr(usage_tracker, "is_over_cap", _boom)
    stream(_StreamResponse(_sse("ok")))

    assert await _drain(_agent(), user_id="u1") == ["ok"]


# ── cost tracking ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_usage_is_recorded_after_a_completed_stream(monkeypatch, stream):
    """`stream_options.include_usage` is requested for exactly this. Without
    the record, streamed turns are free as far as the cap is concerned -- the
    cap above becomes unenforceable for the transport it mostly runs on."""
    _deepseek(monkeypatch)
    recorded = {}
    from app.core import usage_tracker
    monkeypatch.setattr(usage_tracker, "record",
                        lambda **kw: recorded.update(kw))
    stream(_StreamResponse(_sse("ok", usage={"total_tokens": 412})))

    await _drain(_agent(), user_id="u1")

    assert recorded.get("usage") == {"total_tokens": 412}, recorded
    assert recorded.get("provider") == "deepseek", recorded
    assert recorded.get("user_id") == "u1", recorded


@pytest.mark.asyncio
async def test_a_broken_recorder_does_not_fail_a_completed_stream(monkeypatch, stream):
    """Best-effort by design -- the answer has already reached the user, so
    raising here would turn a delivered turn into an error."""
    _deepseek(monkeypatch)
    from app.core import usage_tracker

    def _boom(**kw):
        raise RuntimeError("usage table unreachable")

    monkeypatch.setattr(usage_tracker, "record", _boom)
    stream(_StreamResponse(_sse("ok", usage={"total_tokens": 5})))

    assert await _drain(_agent(), user_id="u1") == ["ok"]
