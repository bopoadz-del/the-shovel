"""`_call_llm`'s own logic -- the lines every other test mocks past.

Audit bar 3, 2026-08-12, agents subsystem. Replacing the whole body with
`return {"status": "error", "error": "gutted"}` left all agent tests green:
`_call_llm` is the most-mocked function in the repository and, until this file,
the least tested. Universal mocking reads as heavy coverage and is its exact
opposite -- every one of those tests asserts what happens AFTER this function
returns.

The platform now supports two providers: DeepSeek (primary) and OpenRouter
(fallback). What lives in `_call_llm` and had no detection:

  * provider fallback on retryable failures, and the deliberate REFUSAL to
    fall back on the others
  * conversation-shape 400s (content_filter / tokenization / tool-pairing) are
    retryable even though a generic 400 is not
  * `tool_choice`, decided per-attempt -- both providers stay on "auto"
  * temperature, decided per-attempt -- neither provider pins it
  * recovery of Llama-native tool markup out of an HTTP 400 tool_use_failed
  * OpenRouter free-tier 402 / 429 same-hop retries
  * the soft daily cost cap

Mocked at `httpx.AsyncClient.post`, which is the real boundary: everything
above it is this function's own work.

ENV ISOLATION IS LOAD-BEARING. `_llm_config` reads LLM_PROVIDER and the
per-provider keys/model overrides live on every call, with no caching, and
`_llm_fallback_config` temporarily MUTATES os.environ. A developer machine with
a real key exported, or a CI leg that pins LLM_PROVIDER, would silently take a
different branch than the test intends. So the fixture clears every var first,
and `test_the_attempt_list_is_what_these_tests_assume` catches a leak that
collapses two attempts into one.
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from tests._fixture_values import STUB_TOKEN_01, STUB_TOKEN_03

from app.agents.runtime import (
    DEEPSEEK_API_URL,
    OPENROUTER_API_URL,
    Agent,
    _http_400_is_retryable,
    _http_status_is_retryable,
)

# Every env var that can change which branch runs. Cleared before each test.
_LLM_ENV = (
    "LLM_PROVIDER", "LLM_FALLBACK_PROVIDER",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_ALLOW_PAID",
    "OPENROUTER_MAX_TOKENS", "OPENROUTER_PROMPT_TOKEN_CEILING",
    "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
    "USAGE_DAILY_CAP_USD", "FORCE_CALC_ON_DIMENSIONS",
)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    """No inherited env may reach `_llm_config`."""
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)


class _Resp:
    """The parts of an httpx.Response `_call_llm` touches."""

    def __init__(self, status_code=200, payload=None, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else _ok_body()
        self.text = text if text is not None else json.dumps(self._payload)
        self.headers = headers or {}
        # llm_client wraps a >=400 response in HTTPStatusError(request=r.request,
        # response=r). Without this attribute that construction raises
        # AttributeError, which its generic except then misreads as a network
        # error and RETRIES -- inverting what the 401 test asserts.
        self.request = None

    def json(self):
        return self._payload


def _ok_body(content="an answer", model="some-model", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": model, "choices": [{"message": message}],
            "usage": {"total_tokens": 10}}


def _tool_use_failed_body(failed_generation):
    return {"error": {"code": "tool_use_failed",
                      "failed_generation": failed_generation}}


class _Http:
    """Records every outbound call and replays a scripted response per call.

    A response may be an exception instance, which is raised -- that is how the
    timeout and transport-error branches are reached.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def post(self, url, json=None, headers=None, **kw):
        # DEEP COPY, and not incidentally. `_call_llm` builds ONE payload dict
        # and rebinds `model`, `temperature` and `tool_choice` on it before
        # each attempt, so storing the reference makes every recorded call show
        # the LAST attempt's values. The per-attempt tests below exist
        # precisely to catch a fallback that carries the primary's values
        # across, and a by-reference recorder hides that bug.
        self.calls.append({"url": url, "payload": deepcopy(json), "headers": headers})
        nxt = self.script.pop(0) if self.script else _Resp()
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    # -- readability helpers used by the assertions ------------------------
    @property
    def urls(self):
        return [c["url"] for c in self.calls]

    @property
    def models(self):
        return [c["payload"].get("model") for c in self.calls]

    @property
    def tool_choices(self):
        return [c["payload"].get("tool_choice") for c in self.calls]

    @property
    def temperatures(self):
        return [c["payload"].get("temperature") for c in self.calls]


@pytest.fixture
def http(monkeypatch):
    def _install(*script):
        fake = _Http(script)
        monkeypatch.setattr(httpx.AsyncClient, "post", fake.post, raising=True)
        return fake

    return _install


def _agent(name="project-assistant", model="deepseek-chat", temperature=0.3,
           blocks=("construction",)):
    return Agent(
        name=name,
        description="test agent",
        system_prompt="you are a test",
        allowed_blocks=list(blocks),
        model=model,
        temperature=temperature,
    )


def _deepseek_primary(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)


def _openrouter_primary(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_03)
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)


def _openrouter_fallback(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_03)


def _deepseek_fallback(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)


USER = [{"role": "user", "content": "What is the notice period?"}]
DELIVERABLE = [{"role": "user", "content": "Generate a construction schedule with 200 activities"}]


# ── the guard ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_attempt_list_is_what_these_tests_assume(monkeypatch, http):
    """Without this, a leaked env that leaves no fallback configured makes
    every "did not fall back" test below pass for the wrong reason -- there was
    never a second attempt to take.

    Asserts the positive shape: two attempts, at two DIFFERENT providers'
    URLs, in the documented order.
    """
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(429, text="rate limited"), _Resp(200))

    await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert fake.urls == [DEEPSEEK_API_URL, OPENROUTER_API_URL], (
        f"the ladder is not deepseek -> openrouter: {fake.urls}"
    )


@pytest.mark.asyncio
async def test_with_no_fallback_configured_there_is_exactly_one_attempt(monkeypatch, http):
    """The other half of the guard. If this ever shows 2, an env var leaked in
    and the no-fallback tests are meaningless."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(500, text="server error"))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 1, f"an unconfigured fallback was attempted: {fake.urls}"
    assert result["status"] == "error"


# ── fallback: the retryable / non-retryable fence ────────────────────────

@pytest.mark.parametrize("status", [408, 413, 429, 500, 502, 503])
@pytest.mark.asyncio
async def test_a_retryable_status_falls_back_to_the_next_provider(monkeypatch, http, status):
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(status, text="upstream said no"), _Resp(200, _ok_body("recovered")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, f"HTTP {status} did not trigger the fallback"
    assert result["status"] == "success", result
    assert result["choice"]["message"]["content"] == "recovered"


@pytest.mark.parametrize("status", [401, 403, 404])
@pytest.mark.asyncio
async def test_a_non_retryable_status_does_not_fall_back(monkeypatch, http, status):
    """The half of the fence that is easy to lose.

    A different provider cannot fix a bad key or a malformed request, so
    retrying burns the fallback's quota and doubles the user's wait for the
    same error. Both directions are asserted because a one-sided check leaves
    the other open -- which is the failure this whole audit started from.
    """
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(status, text="bad request"), _Resp(200, _ok_body("should never run")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 1, (
        f"HTTP {status} is not retryable but a second provider was called anyway: "
        f"{fake.urls}"
    )
    assert result["status"] == "error", result
    assert str(status) in result["error"], result
    assert "deepseek" in result["error"], "the failing provider is not named in the error"


_PAIRING_400 = (
    "Invalid request: an assistant message with 'tool_calls' must be "
    "followed by tool messages responding to each 'tool_call_id'. "
    "Missing: fetch_document:3, fetch_document:4"
)


@pytest.mark.asyncio
async def test_tool_pairing_400_falls_back_to_the_next_provider(monkeypatch, http):
    """Live UI IPC_NEG: a provider 400'd orphaned fetch_document ids and the
    turn died because HTTP 400 was non-retryable. A conversation-shape 400 is
    a class the next provider CAN serve once history is repaired, so it MUST
    hop rather than end the turn."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(400, text=_PAIRING_400), _Resp(200, _ok_body("recovered")))

    broken = [
        {"role": "user", "content": "issue the payment certificate"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fetch_document:1", "type": "function",
             "function": {"name": "fetch_document", "arguments": "{}"}},
            {"id": "fetch_document:2", "type": "function",
             "function": {"name": "fetch_document", "arguments": "{}"}},
            {"id": "fetch_document:3", "type": "function",
             "function": {"name": "fetch_document", "arguments": "{}"}},
            {"id": "fetch_document:4", "type": "function",
             "function": {"name": "fetch_document", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "fetch_document:1", "name": "fetch_document", "content": "{}"},
        {"role": "tool", "tool_call_id": "fetch_document:2", "name": "fetch_document", "content": "{}"},
        {"role": "user", "content": "The router matched NO action."},
        {"role": "tool", "tool_call_id": "fetch_document:3", "name": "fetch_document", "content": "{}"},
        {"role": "tool", "tool_call_id": "fetch_document:4", "name": "fetch_document", "content": "{}"},
    ]
    result = await _agent()._call_llm(broken, STUB_TOKEN_01)

    assert len(fake.calls) == 2, (
        f"tool-pairing 400 did not fall back: {fake.urls}"
    )
    assert fake.urls == [DEEPSEEK_API_URL, OPENROUTER_API_URL], fake.urls
    assert result["status"] == "success", result
    assert result["choice"]["message"]["content"] == "recovered"


def test_http_400_content_filter_is_retryable():
    live_m12 = (
        '{"error":{"code":400,"message":"The request was rejected because it '
        'was considered high risk","param":"prompt","type":"content_filter"}}'
    )
    assert _http_400_is_retryable(live_m12)
    assert not _http_400_is_retryable('{"error":"bad request"}')


@pytest.mark.asyncio
async def test_http_400_content_filter_falls_back(monkeypatch, http):
    """A content_filter 400 is a conversation-shape failure the next provider
    can serve."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(
        _Resp(
            400,
            text=(
                '{"error":{"code":400,"message":"The request was rejected '
                'because it was considered high risk","param":"prompt",'
                '"type":"content_filter"}}'
            ),
        ),
        _Resp(200, _ok_body("as-built 40 m3 / 11.43%")),
    )

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, fake.urls
    assert fake.urls == [DEEPSEEK_API_URL, OPENROUTER_API_URL], fake.urls
    assert result["status"] == "success", result
    assert "40" in result["choice"]["message"]["content"]


@pytest.mark.asyncio
async def test_content_filter_200_falls_back(monkeypatch, http):
    """HTTP 200 + finish_reason=content_filter used to end empty; it must hop."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    filtered = {
        "model": "deepseek-chat",
        "choices": [{
            "finish_reason": "content_filter",
            "message": {"role": "assistant", "content": ""},
        }],
        "usage": {"total_tokens": 10},
    }
    fake = http(_Resp(200, filtered), _Resp(200, _ok_body("recovered")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, fake.urls
    assert result["status"] == "success", result
    assert result["choice"]["message"]["content"] == "recovered"


@pytest.mark.asyncio
async def test_empty_200_falls_back(monkeypatch, http):
    """Live M9: HTTP 200 with empty content and no tools must not end the turn."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    empty = {
        "model": "deepseek-chat",
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": ""},
        }],
        "usage": {"total_tokens": 10},
    }
    fake = http(_Resp(200, empty), _Resp(200, _ok_body("claim notice")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, fake.urls
    assert result["status"] == "success", result
    assert result["choice"]["message"]["content"] == "claim notice"


@pytest.mark.asyncio
async def test_tokenization_400_falls_back(monkeypatch, http):
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(
        _Resp(400, text='{"message": "Invalid request: tokenization failed"}'),
        _Resp(200, _ok_body("recovered")),
    )

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, fake.urls
    assert result["status"] == "success", result


@pytest.mark.asyncio
async def test_generic_400_still_does_not_fall_back(monkeypatch, http):
    """The fence still holds: a junk 400 is not a reason to burn the fallback."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(400, text="bad request"), _Resp(200, _ok_body("should never run")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 1, fake.urls
    assert result["status"] == "error", result


@pytest.mark.asyncio
async def test_a_timeout_falls_back(monkeypatch, http):
    """The primary failure mode with a name: a reasoning model on a large
    grounded turn times out. The documented reason a fallback exists at all."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(httpx.TimeoutException("read timeout"), _Resp(200, _ok_body("recovered")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, "a timeout did not fall back"
    assert result["status"] == "success", result


@pytest.mark.asyncio
async def test_a_transport_error_falls_back(monkeypatch, http):
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(httpx.ConnectError("dns failure"), _Resp(200, _ok_body("recovered")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, "a connect error did not fall back"
    assert result["status"] == "success", result


@pytest.mark.asyncio
async def test_when_every_provider_fails_the_last_error_is_returned(monkeypatch, http):
    """Not a generic "LLM call failed" -- the operator needs the provider and
    status that actually ended the turn."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(429, text="deepseek rate limited"), _Resp(503, text="openrouter unavailable"))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2
    assert result["status"] == "error"
    assert "openrouter" in result["error"] and "503" in result["error"], (
        f"the surviving error is not the last provider's: {result['error']}"
    )


# ── temperature, decided per attempt ─────────────────────────────────────

@pytest.mark.asyncio
async def test_each_attempt_gets_the_temperature_its_own_provider_accepts(monkeypatch, http):
    """Neither DeepSeek nor OpenRouter pins temperature, so both hops carry the
    agent's own value. Asserting only the first payload would miss a regression
    where a fallback silently changed it -- so both hops are checked."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(429, text="rate limited"), _Resp(200))

    await _agent(temperature=0.3)._call_llm(list(USER), STUB_TOKEN_01)

    assert fake.temperatures == [0.3, 0.3], (
        f"temperature is not being decided per attempt: {fake.temperatures}"
    )


# ── tool_choice, decided per attempt ─────────────────────────────────────

@pytest.mark.asyncio
async def test_deepseek_is_never_sent_a_forced_tool_choice(monkeypatch, http):
    """Forcing a specific tool 400s on DeepSeek's reasoner/thinking mode, and
    OpenRouter's routed free models must not be forced either. Both stay on
    "auto" even when the turn names a deliverable. Pinned deliberately: a
    future change must be a change to THIS assertion."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent()._call_llm(list(DELIVERABLE), STUB_TOKEN_01)

    assert fake.tool_choices == ["auto"], (
        f"a forced tool_choice was sent to deepseek: {fake.tool_choices}"
    )


@pytest.mark.asyncio
async def test_openrouter_is_never_sent_a_forced_tool_choice(monkeypatch, http):
    _openrouter_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent(model="openrouter/free")._call_llm(list(DELIVERABLE), STUB_TOKEN_03)

    assert fake.tool_choices == ["auto"], fake.tool_choices


@pytest.mark.asyncio
async def test_an_ordinary_question_is_not_forced_to_call_a_tool(monkeypatch, http):
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert fake.tool_choices == ["auto"], fake.tool_choices


@pytest.mark.asyncio
async def test_with_tools_disabled_no_tools_and_no_tool_choice_are_sent(monkeypatch, http):
    """The synthesis call. A tool_choice on a tool-free payload is a 400."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent()._call_llm(list(DELIVERABLE), STUB_TOKEN_01, with_tools=False)

    payload = fake.calls[0]["payload"]
    assert "tools" not in payload, "tools were sent with with_tools=False"
    assert "tool_choice" not in payload, payload.get("tool_choice")


@pytest.mark.asyncio
async def test_excluded_tools_are_withheld(monkeypatch, http):
    """Used to stop the loop re-offering a tool that already ran and failed."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent()._call_llm(list(USER), STUB_TOKEN_01,
                             exclude_tools={"construction_calc"})

    names = {t["function"]["name"] for t in fake.calls[0]["payload"]["tools"]}
    assert "construction_calc" not in names, "an excluded tool was offered anyway"
    assert names, "every tool was dropped, not just the excluded one"


# ── tool_use_failed recovery, both outcomes ──────────────────────────────

@pytest.mark.asyncio
async def test_llama_native_tool_markup_is_recovered_from_a_400(monkeypatch, http):
    """A validator that rejects Llama-native function markup returns HTTP 400
    and buries the markup in `error.failed_generation`. Recovering it turns a
    dead turn into a working tool call, so a regression here reads to the user
    as "the assistant randomly fails on some questions"."""
    _deepseek_primary(monkeypatch)
    markup = '<function=construction_calc{"formula": "concrete_volume"}></function>'
    fake = http(_Resp(400, _tool_use_failed_body(markup), text=json.dumps(
        _tool_use_failed_body(markup))))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert result["status"] == "success", (
        f"recoverable tool markup was returned as an error instead: {result}"
    )
    calls = result["choice"]["message"]["tool_calls"]
    assert calls and calls[0]["function"]["name"] == "construction_calc", calls
    assert len(fake.calls) == 1, "a recovered call should not also fall back"


@pytest.mark.asyncio
async def test_a_tool_call_emitted_as_prose_falls_back_instead_of_erroring(monkeypatch, http):
    """The subtle half. Nothing can be recovered from prose, but it is a
    MODEL-side failure another provider can handle -- so it is retryable even
    though HTTP 400 normally is not. Without this the turn dies on an error the
    fallback would have answered."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    prose = "I will now calculate the concrete volume for you."
    body = _tool_use_failed_body(prose)
    fake = http(_Resp(400, body, text=json.dumps(body)),
                _Resp(200, _ok_body("recovered")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, (
        "unrecoverable tool_use_failed did not fall back -- the turn dies on a "
        "400 the next provider could have answered"
    )
    assert result["status"] == "success", result


@pytest.mark.asyncio
async def test_prose_tool_use_failed_with_no_fallback_is_an_error(monkeypatch, http):
    """The pair of the prose-falls-back test: with no fallback configured the
    same 400 must surface as an error, not be swallowed."""
    _deepseek_primary(monkeypatch)
    body = _tool_use_failed_body("## prose checklist, not a tool call")
    fake = http(_Resp(400, body, text=json.dumps(body)))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert result["status"] == "error", result
    assert len(fake.calls) == 1


# ── model pinning ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_legacy_placeholder_model_is_replaced_by_the_provider_default(
        monkeypatch, http):
    """Agents still carry kimi-k2.6 / deepseek- / gpt-4 names from before those
    providers were removed. On DeepSeek only a `deepseek-*` pin survives; a
    foreign pin resolves to the provider default rather than 404ing."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent(model="kimi-k2.6")._call_llm(list(USER), STUB_TOKEN_01)

    assert fake.models == ["deepseek-chat"], (
        f"a dead provider's model name was sent upstream: {fake.models}"
    )


@pytest.mark.asyncio
async def test_an_agent_that_pinned_a_live_model_keeps_it(monkeypatch, http):
    """The other direction: an agent that pins a live deepseek model keeps it,
    so overriding it would silently downgrade that agent."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent(model="deepseek-reasoner")._call_llm(list(USER), STUB_TOKEN_01)

    assert fake.models == ["deepseek-reasoner"], fake.models


@pytest.mark.asyncio
async def test_openrouter_remaps_a_foreign_pin_but_keeps_a_free_slug(monkeypatch, http):
    _openrouter_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent(model="kimi-k2.6")._call_llm(list(USER), STUB_TOKEN_03)
    assert fake.models == ["openrouter/free"], fake.models

    fake2 = http(_Resp(200))
    await _agent(model="meta-llama/llama-3.3-70b-instruct:free")._call_llm(
        list(USER), STUB_TOKEN_03)
    assert fake2.models == ["meta-llama/llama-3.3-70b-instruct:free"], fake2.models


# ── the outbound sanitisation chokepoint ─────────────────────────────────

@pytest.mark.asyncio
async def test_non_standard_message_fields_never_reach_the_provider(monkeypatch, http):
    """A `reasoning` field on a tool-call message caused an HTTP 400 that hung
    deliverable generation. `_call_llm` is the single chokepoint that strips
    it, so it is asserted at the wire rather than on the helper."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    messages = [{"role": "assistant", "content": "thinking",
                 "reasoning": "internal chain of thought"},
                {"role": "user", "content": "and now?"}]
    await _agent()._call_llm(messages, STUB_TOKEN_01)

    sent = fake.calls[0]["payload"]["messages"]
    assert not any("reasoning" in m for m in sent), (
        f"a non-standard field reached the provider: {sent}"
    )
    assert len(sent) == 2, "sanitisation dropped whole messages"


# ── the soft daily cap ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_user_over_the_daily_cap_is_refused_before_any_spend(monkeypatch, http):
    """The point of a cost cap is that the request is never made. Returning the
    error AFTER calling the provider would cap nothing."""
    _deepseek_primary(monkeypatch)
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "5.00")
    from app.core import usage_tracker
    monkeypatch.setattr(usage_tracker, "is_over_cap", lambda user_id, cap: True)
    monkeypatch.setattr(usage_tracker, "daily_total", lambda user_id: {"cost_usd": 7.5})
    fake = http(_Resp(200))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01, user_id="u1")

    assert fake.calls == [], "the provider was called despite the cap being hit"
    assert result["status"] == "error"
    assert "cap" in result["error"].lower(), result["error"]


@pytest.mark.asyncio
async def test_an_internal_call_with_no_user_is_not_capped(monkeypatch, http):
    """Documented: calls without a user_id aren't billable, so capping them
    would break internal work for a spend it never caused."""
    _deepseek_primary(monkeypatch)
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "5.00")
    from app.core import usage_tracker
    monkeypatch.setattr(usage_tracker, "is_over_cap",
                        lambda user_id, cap: pytest.fail("cap checked with no user_id"))
    fake = http(_Resp(200))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 1 and result["status"] == "success", result


@pytest.mark.asyncio
async def test_a_broken_usage_tracker_does_not_block_the_call(monkeypatch, http):
    """Explicit in the code: a broken tracker must never block a real call.
    The failure mode it prevents is total -- every turn refused because a
    bookkeeping table is unreachable."""
    _deepseek_primary(monkeypatch)
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "5.00")
    from app.core import usage_tracker

    def _boom(*a, **kw):
        raise RuntimeError("usage table unreachable")

    monkeypatch.setattr(usage_tracker, "is_over_cap", _boom)
    fake = http(_Resp(200))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01, user_id="u1")

    assert result["status"] == "success", result
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_an_unparseable_cap_disables_the_check(monkeypatch, http):
    _deepseek_primary(monkeypatch)
    monkeypatch.setenv("USAGE_DAILY_CAP_USD", "not-a-number")
    from app.core import usage_tracker
    monkeypatch.setattr(usage_tracker, "is_over_cap",
                        lambda user_id, cap: pytest.fail("checked an unparseable cap"))
    fake = http(_Resp(200))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01, user_id="u1")
    assert result["status"] == "success" and len(fake.calls) == 1


# ── the success path ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_successful_call_returns_the_providers_choice_and_raw_body(
        monkeypatch, http):
    _deepseek_primary(monkeypatch)
    body = _ok_body("28 days under clause 20.1", model="deepseek-chat")
    http(_Resp(200, body))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert result["status"] == "success"
    assert result["choice"]["message"]["content"] == "28 days under clause 20.1"
    assert result["raw"] == body, "the raw body is not forwarded"


@pytest.mark.asyncio
async def test_a_malformed_success_body_falls_back(monkeypatch, http):
    """A 200 whose body cannot be parsed is still a failed turn, and another
    provider can serve it."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)

    class _Garbage(_Resp):
        def json(self):
            raise ValueError("not json")

    fake = http(_Garbage(200), _Resp(200, _ok_body("recovered")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert len(fake.calls) == 2, "an unparseable 200 did not fall back"
    assert result["status"] == "success", result


@pytest.mark.asyncio
async def test_the_api_key_is_sent_as_a_bearer_header(monkeypatch, http):
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert fake.calls[0]["headers"]["Authorization"] == "Bearer " + STUB_TOKEN_01


@pytest.mark.asyncio
async def test_an_empty_key_sends_no_authorization_header(monkeypatch, http):
    """An empty key must not send `Bearer ` -- some gateways reject it. The
    caller passes the key, so an unset one degrades to no header rather than a
    malformed one."""
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent()._call_llm(list(USER), "")

    assert "Authorization" not in fake.calls[0]["headers"], fake.calls[0]["headers"]


@pytest.mark.asyncio
async def test_the_fallback_attempt_uses_the_fallback_providers_own_key(
        monkeypatch, http):
    """The primary's key on the fallback's endpoint is a 401 -- a fallback that
    always fails, which looks like "the fallback provider is down"."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(429, text="rate limited"), _Resp(200))

    await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert fake.calls[0]["headers"]["Authorization"] == "Bearer " + STUB_TOKEN_01
    assert fake.calls[1]["headers"]["Authorization"] == "Bearer " + STUB_TOKEN_03, (
        "the fallback was called with the primary's key"
    )


@pytest.mark.asyncio
async def test_a_successful_primary_never_touches_the_fallback(monkeypatch, http):
    """Success must not leak a second request -- a fallback that fires on
    success doubles cost and latency invisibly."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(200, _ok_body("primary ok")),
                _Resp(200, _ok_body("SHOULD NOT REACH")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert result["choice"]["message"]["content"] == "primary ok"
    assert len(fake.calls) == 1, f"the fallback was called on success: {fake.urls}"


# ── _llm_fallback_config branch coverage ─────────────────────────────────

def test_fallback_config_is_none_when_unset(monkeypatch):
    from app.agents.runtime import _llm_fallback_config

    assert _llm_fallback_config({"provider": "deepseek"}) is None


def test_fallback_config_is_none_when_it_names_the_primary(monkeypatch):
    """Falling back to the provider that just failed is a retry loop wearing a
    fallback's name."""
    from app.agents.runtime import _llm_fallback_config

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    assert _llm_fallback_config({"provider": "deepseek"}) is None


def test_fallback_config_is_none_when_its_key_is_missing(monkeypatch):
    """A fallback with no key is an attempt that can only 401 -- worse than
    reporting the primary's real error."""
    from app.agents.runtime import _llm_fallback_config

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _llm_fallback_config({"provider": "deepseek"}) is None


def test_fallback_config_resolves_openrouter(monkeypatch):
    from app.agents.runtime import OPENROUTER_API_URL, _llm_fallback_config

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_03)
    cfg = _llm_fallback_config({"provider": "deepseek"})
    assert cfg is not None
    assert cfg["provider"] == "openrouter"
    assert cfg["url"] == OPENROUTER_API_URL
    assert cfg["env_key"] == "OPENROUTER_API_KEY"
    assert cfg["default_model"] == "openrouter/free"


def test_fallback_config_resolves_deepseek(monkeypatch):
    from app.agents.runtime import DEEPSEEK_API_URL, _llm_fallback_config

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    cfg = _llm_fallback_config({"provider": "openrouter"})
    assert cfg is not None
    assert cfg["provider"] == "deepseek"
    assert cfg["url"] == DEEPSEEK_API_URL
    assert cfg["env_key"] == "DEEPSEEK_API_KEY"
    assert cfg["default_model"] == "deepseek-chat"


def test_fallback_config_is_none_when_deepseek_key_is_missing(monkeypatch):
    from app.agents.runtime import _llm_fallback_config

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert _llm_fallback_config({"provider": "openrouter"}) is None


def test_fallback_ladder_is_the_single_cross_provider_target(monkeypatch):
    """With two providers the ladder is exactly the configured cross-provider
    degrade target, or empty."""
    from app.agents.runtime import _llm_fallback_ladder

    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", STUB_TOKEN_03)
    ladder = _llm_fallback_ladder({"provider": "deepseek", "default_model": "deepseek-chat"})
    assert [c["provider"] for c in ladder] == ["openrouter"], ladder

    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    assert _llm_fallback_ladder({"provider": "deepseek"}) == []


# ── llm_client.complete(): the orchestrator intent path ─────────────────
# A different module with its own fallback ladder. If it stopped degrading,
# smart routing would silently vanish for the turn on every provider blip.

@pytest.mark.asyncio
async def test_orchestrator_complete_falls_back_on_a_rate_limit(monkeypatch, http):
    from app.core import llm_client

    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(429, text="rate limited"), _Resp(200, _ok_body("intent json")))

    out = await llm_client.complete([{"role": "user", "content": "hi"}])

    assert out == "intent json"
    assert len(fake.calls) == 2, f"the intent path did not degrade: {fake.urls}"


@pytest.mark.asyncio
async def test_orchestrator_complete_does_not_fall_back_on_a_bad_key(monkeypatch, http):
    from app.core import llm_client

    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(401, text="invalid api key"),
                _Resp(200, _ok_body("SHOULD NOT REACH")))

    with pytest.raises(httpx.HTTPStatusError):
        await llm_client.complete([{"role": "user", "content": "hi"}])
    assert len(fake.calls) == 1, f"a 401 was retried on the intent path: {fake.urls}"


@pytest.mark.asyncio
async def test_orchestrator_complete_falls_back_on_openrouter_402(monkeypatch, http):
    """Intent routing used a private retryable set that omitted 402."""
    from app.core import llm_client

    _openrouter_primary(monkeypatch)
    _deepseek_fallback(monkeypatch)
    fake = http(
        _Resp(402, text="You requested up to 2048 tokens, but can only afford 996"),
        _Resp(200, _ok_body("intent json")),
    )

    out = await llm_client.complete([{"role": "user", "content": "hi"}])

    assert out == "intent json"
    assert fake.urls == [OPENROUTER_API_URL, DEEPSEEK_API_URL], fake.urls


def test_http_status_is_retryable_includes_402():
    assert _http_status_is_retryable(402)
    assert _http_status_is_retryable(429)
    assert not _http_status_is_retryable(401)


# ── DeepSeek primary → OpenRouter fallback (the documented recipe) ───────

@pytest.mark.asyncio
async def test_deepseek_402_insufficient_credit_falls_back_to_openrouter(
    monkeypatch, http,
):
    """Live tip 78bd9ca: DeepSeek HTTP 402 Insufficient Balance ended the
    stream in 0.8s with no OpenRouter hop, even though health said
    fallback_ready. ``_http_status_is_retryable(402)`` is True; the local
    ``_is_retryable`` inside ``_call_llm`` omitted 402 so the ladder never
    ran. OpenRouter generic 402 still must not burn DeepSeek — that
    direction stays covered by ``test_openrouter_402_does_not_fall_back_to_deepseek``.
    """
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(
        _Resp(
            402,
            text='{"error":{"message":"Insufficient Balance","type":"unknown_error","code":"invalid_request_error"}}',
        ),
        _Resp(200, _ok_body("recovered on openrouter")),
    )

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert result["status"] == "success", result
    assert fake.urls == [DEEPSEEK_API_URL, OPENROUTER_API_URL], fake.urls
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer " + STUB_TOKEN_01
    assert fake.calls[1]["headers"]["Authorization"] == "Bearer " + STUB_TOKEN_03
    content = result["choice"]["message"]["content"]
    assert content == "recovered on openrouter"


@pytest.mark.asyncio
async def test_deepseek_primary_falls_back_to_openrouter(monkeypatch, http):
    """Operator recipe: LLM_PROVIDER=deepseek, LLM_FALLBACK_PROVIDER=openrouter."""
    _deepseek_primary(monkeypatch)
    _openrouter_fallback(monkeypatch)
    fake = http(_Resp(429, text="rate limited"), _Resp(200, _ok_body("recovered")))

    result = await _agent()._call_llm(list(USER), STUB_TOKEN_01)

    assert result["status"] == "success", result
    assert fake.urls == [DEEPSEEK_API_URL, OPENROUTER_API_URL], fake.urls
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer " + STUB_TOKEN_01
    assert fake.calls[1]["headers"]["Authorization"] == "Bearer " + STUB_TOKEN_03
    assert fake.models[0] == "deepseek-chat"
    # Neither provider pins temperature.
    assert fake.temperatures == [0.3, 0.3], fake.temperatures


@pytest.mark.asyncio
async def test_deepseek_sends_tool_choice_auto(monkeypatch, http):
    _deepseek_primary(monkeypatch)
    fake = http(_Resp(200))

    await _agent()._call_llm(list(DELIVERABLE), STUB_TOKEN_01)

    assert fake.urls == [DEEPSEEK_API_URL]
    assert fake.tool_choices == ["auto"], fake.tool_choices


# ── OpenRouter free-tier 402 (same-hop retry, not a provider fallback) ──

@pytest.fixture
def no_sleep(monkeypatch):
    async def _instant(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)


@pytest.mark.asyncio
async def test_openrouter_402_in_flight_retries_same_hop(monkeypatch, http, no_sleep):
    """Live Wave1: in_flight_budget_exhausted after a tool-round balloon."""
    _openrouter_primary(monkeypatch)
    fake = http(
        _Resp(
            402,
            text=(
                '{"error":{"message":"This request requires more credits, '
                'or fewer max_tokens. in_flight_budget_exhausted"}}'
            ),
        ),
        _Resp(200, _ok_body("Time for Completion is 852 days")),
    )

    result = await _agent(model="openrouter/free")._call_llm(
        list(USER), STUB_TOKEN_03,
    )

    assert result["status"] == "success", result
    assert "852" in result["choice"]["message"]["content"]
    assert fake.urls == [OPENROUTER_API_URL, OPENROUTER_API_URL]
    assert fake.calls[0]["payload"]["max_tokens"] == fake.calls[1]["payload"]["max_tokens"]


@pytest.mark.asyncio
async def test_openrouter_402_afford_fewer_lowers_max_tokens(
    monkeypatch, http, no_sleep,
):
    """Live: 'You requested up to 2048 tokens, but can only afford 996'."""
    _openrouter_primary(monkeypatch)
    fake = http(
        _Resp(402, text="You requested up to 2048 tokens, but can only afford 996"),
        _Resp(200, _ok_body("ok")),
    )

    result = await _agent(model="openrouter/free")._call_llm(
        list(USER), STUB_TOKEN_03,
    )

    assert result["status"] == "success", result
    assert fake.urls == [OPENROUTER_API_URL, OPENROUTER_API_URL]
    assert fake.calls[0]["payload"]["max_tokens"] == 2048
    assert fake.calls[1]["payload"]["max_tokens"] == 996


@pytest.mark.asyncio
async def test_openrouter_402_generic_does_not_retry(monkeypatch, http, no_sleep):
    _openrouter_primary(monkeypatch)
    fake = http(
        _Resp(402, text="Insufficient credits"),
        _Resp(200, _ok_body("should never run")),
    )

    result = await _agent(model="openrouter/free")._call_llm(
        list(USER), STUB_TOKEN_03,
    )

    assert result["status"] == "error", result
    assert "402" in result["error"]
    assert "insufficient" in result["error"].lower()
    assert "credit" in result["error"].lower()
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_openrouter_hop_compacts_large_tool_payload(monkeypatch, http):
    _openrouter_primary(monkeypatch)
    huge = "x" * 40000
    messages = [
        {"role": "user", "content": "What is Time for Completion?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "fetch_document", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "name": "fetch_document",
         "content": huge},
    ]
    fake = http(_Resp(200, _ok_body("852 days")))

    result = await _agent(model="openrouter/free")._call_llm(
        messages, STUB_TOKEN_03,
    )

    assert result["status"] == "success", result
    tool = next(
        m for m in fake.calls[0]["payload"]["messages"] if m.get("role") == "tool"
    )
    assert len(str(tool.get("content") or "")) < len(huge)
    assert "OpenRouter" in str(tool.get("content") or "")


@pytest.mark.asyncio
async def test_openrouter_402_does_not_fall_back_to_deepseek(
    monkeypatch, http, no_sleep,
):
    """402 retry is same-provider; a generic 402 must not burn the fallback."""
    _openrouter_primary(monkeypatch)
    _deepseek_fallback(monkeypatch)
    fake = http(
        _Resp(402, text="Insufficient credits"),
        _Resp(200, _ok_body("should never run")),
    )

    result = await _agent(model="openrouter/free")._call_llm(
        list(USER), STUB_TOKEN_03,
    )

    assert result["status"] == "error", result
    assert fake.urls == [OPENROUTER_API_URL]


_OR_402_PROMPT_LIMIT = (
    '{"error":{"message":"Prompt tokens limit exceeded: 18398 > 10335. '
    'To increase, visit https://openrouter.ai/settings/credits and upgrade '
    'to a paid account","code":402}}'
)


@pytest.mark.asyncio
async def test_openrouter_402_prompt_limit_compacts_and_retries(
        monkeypatch, http, no_sleep):
    """#514's proactive ceiling can miss when remaining credit drops.

    Raise the proactive budget so the first hop stays large; the live
    prompt-limit 402 must still compact on the same hop.
    """
    _openrouter_primary(monkeypatch)
    monkeypatch.setenv("OPENROUTER_PROMPT_TOKEN_CEILING", "20000")
    # Between the raised proactive budget (~60k chars) and the live cap
    # (10335 * 3 ≈ 31k chars) so only the reactive hop shrinks it.
    huge = "x" * 45000
    messages = [
        {"role": "user", "content": "What is the Accepted Contract Amount including VAT?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "name": "search", "content": huge},
    ]
    fake = http(
        _Resp(402, text=_OR_402_PROMPT_LIMIT),
        _Resp(200, _ok_body("SAR 450,000,000")),
    )

    result = await _agent(model="openrouter/free")._call_llm(
        messages, STUB_TOKEN_03,
    )

    assert result["status"] == "success", result
    assert fake.urls == [OPENROUTER_API_URL, OPENROUTER_API_URL], fake.urls
    first_tool = next(
        m for m in fake.calls[0]["payload"]["messages"] if m.get("role") == "tool"
    )
    second_tool = next(
        m for m in fake.calls[1]["payload"]["messages"] if m.get("role") == "tool"
    )
    assert len(str(second_tool.get("content") or "")) < len(
        str(first_tool.get("content") or "")
    ), "prompt-limit 402 did not compact the retry payload"


@pytest.mark.asyncio
async def test_openrouter_429_retries_same_provider(monkeypatch, http):
    _openrouter_primary(monkeypatch)
    slept: list[float] = []

    async def _fake_sleep(seconds):
        slept.append(float(seconds))

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    fake = http(
        _Resp(429, text="Rate limit exceeded", headers={"retry-after": "1.5"}),
        _Resp(200, _ok_body("recovered")),
    )

    result = await _agent(model="openrouter/free")._call_llm(
        list(USER), STUB_TOKEN_03,
    )

    assert result["status"] == "success", result
    assert fake.urls == [OPENROUTER_API_URL, OPENROUTER_API_URL], fake.urls
    assert slept == [1.5], slept
