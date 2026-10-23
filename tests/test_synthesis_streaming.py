"""True token streaming for the forced synthesis call (SYNTHESIS_STREAMING=1).

After a deliverable tool returns, force_synthesis makes the next call tool-free.
When streaming is enabled (DeepSeek/OpenRouter + flag on), that synthesis call streams provider
deltas as token events instead of computing the whole answer then re-chunking it.
These tests cover: the happy path (deltas stream, line-buffered + sanitised),
pre-first-token fallback to the non-streaming path, citation sanitisation on
streamed lines, the empty-final forced retry, the default-off inertness, and the
_stream_synthesis SSE parser + sanitiser chokepoint.
"""
from __future__ import annotations

import asyncio
import time
import json
from unittest.mock import patch

import pytest

from app.agents.runtime import Agent, _SYNTH_CUTOFF_NOTICE, _SynthStreamError
from tests._fixture_values import STUB_TOKEN_04


# ── DeepSeek config so the streaming gate (provider in the allowlist) opens ──
_DEEPSEEK_CFG = {
    "provider": "deepseek",
    "url": "https://api.deepseek.com/v1/chat/completions",
    "env_key": "DEEPSEEK_API_KEY",
    "default_model": "deepseek-chat",
}


@pytest.fixture(autouse=True)
def _disable_commissioning_remaining(monkeypatch):
    """These tests need the model to emit the commissioning tool_call.

    Remaining predispatch would draft the checklist first and lock
    synthesis before iter-0, so _call_llm would never see a tool hop.
    """
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")


@pytest.fixture
def deepseek_streaming(monkeypatch):
    """Enable the streaming path deterministically: deepseek provider, flag on,
    an api key present."""
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")


def _pa_agent():
    return Agent(name="project-assistant", description="pa",
                 system_prompt="x", allowed_blocks=["construction"])


def _drain(agen):
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def _tool_then_final():
    """Fake _call_llm: call 1 -> commissioning tool_call; later calls -> a plain
    final answer. Counts invocations so tests can assert the synthesis call did
    NOT go through _call_llm when streamed."""
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": "NON-STREAMED fallback answer."}}}

    fake.state = state
    return fake


async def _tool_ok(self, tool_call, **kwargs):
    return {"name": "commissioning_checklist", "ok": True,
            "result": {"status": "success", "checklists_by_system": {"electrical": []}}}


def _mk_stream(deltas):
    async def fake_stream(self, messages, api_key, **kwargs):
        for d in deltas:
            yield d
    return fake_stream


def _mk_stream_raise():
    async def fake_stream(self, messages, api_key, **kwargs):
        raise _SynthStreamError("pre-first-token failure")
        yield ""  # pragma: no cover — marks this as an async generator
    return fake_stream


def _mk_stream_midway_drop(deltas):
    """Yield 1–2 deltas then raise — the A4 mid-stream cut-off shape."""
    async def fake_stream(self, messages, api_key, **kwargs):
        for d in deltas:
            yield d
        raise _SynthStreamError("mid-stream drop")
    return fake_stream


def _mk_stream_empty():
    async def fake_stream(self, messages, api_key, **kwargs):
        return
        yield ""  # pragma: no cover — async generator that yields nothing
    return fake_stream


def _run_turn(agent):
    return _drain(agent.chat_stream(
        user_message="Generate a commissioning checklist for electrical.",
        history=[], project_id=None, conversation_id=None, user_id=None,
    ))


def _tokens(events):
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


# ── happy path ────────────────────────────────────────────────────────────────

def test_streamed_synthesis_emits_progressive_tokens(deepseek_streaming):
    call_llm = _tool_then_final()
    deltas = ["- Verify ", "electrical ", "isolation\n", "- Check ", "earthing\n", "Done."]
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    # synthesis was STREAMED, not routed through _call_llm -> only the tool call hit it
    assert call_llm.state["n"] == 1, "synthesis should not call _call_llm when streamed"
    token_events = [e for e in events if e.get("type") == "token"]
    # more than one token event => progressive (line-buffered) streaming, not one burst
    assert len(token_events) >= 2, token_events
    text = _tokens(events)
    assert "Verify electrical isolation" in text
    assert "Check earthing" in text
    assert "Done." in text  # trailing partial line flushed at end
    assert events[-1]["type"] == "end"
    assert "NON-STREAMED" not in text


def test_streaming_is_on_by_default(monkeypatch):
    """ORDERS_0305Z: the built streaming path is ON unless SYNTHESIS_STREAMING=0.
    With the flag UNSET, the forced synthesis streams (does not fall to the
    non-streaming _call_llm synthesis)."""
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.delenv("SYNTHESIS_STREAMING", raising=False)  # default
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), patch.object(Agent, "_run_tool_call", _tool_ok), patch.object(Agent, "_stream_synthesis", _mk_stream(["One sentence.\n", "Two.\n"])):
        events = _run_turn(_pa_agent())
    assert call_llm.state["n"] == 1, "synthesis streamed, so no second _call_llm"
    assert "One sentence." in _tokens(events)


def _mk_delayed_stream(deltas, gap=0.05):
    """A provider that emits deltas with a real delay between them, and records
    when its LAST delta was produced, so a test can prove a token left the
    server before the completion finished."""
    marks = {"last_delta": None}

    async def fake_stream(self, messages, api_key, **kwargs):
        for i, d in enumerate(deltas):
            if i:
                await asyncio.sleep(gap)
            if i == len(deltas) - 1:
                marks["last_delta"] = time.monotonic()
            yield d
    return fake_stream, marks


def test_a_token_leaves_before_the_completion_finishes(deepseek_streaming):
    """ORDERS_0305Z acceptance shape: the first token frame is yielded BEFORE
    the provider has produced its last delta, and token frames span a real
    part of the turn -- not all at the end (the TTFT~=total signature)."""
    deltas = [f"Sentence number {i} of the streamed answer.\n" for i in range(6)]
    stream, marks = _mk_delayed_stream(deltas, gap=0.05)

    async def go():
        first_token_at = None
        last_token_at = None
        start = time.monotonic()
        with patch.object(Agent, "_call_llm", _tool_then_final()), patch.object(Agent, "_run_tool_call", _tool_ok), patch.object(Agent, "_stream_synthesis", stream):
            async for ev in _pa_agent().chat_stream(
                user_message="Generate a commissioning checklist for electrical.",
                history=[], project_id=None, conversation_id=None, user_id=None,
            ):
                if ev.get("type") == "token" and ev.get("content", "").strip():
                    now = time.monotonic()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
        return start, first_token_at, last_token_at, marks["last_delta"]

    start, first_token_at, last_token_at, last_delta_at = asyncio.run(go())
    assert first_token_at is not None, "no token was emitted"
    assert last_delta_at is not None
    assert first_token_at < last_delta_at, "first token only after the provider finished (TTFT~=total)"
    # Token frames span a real part of the turn, not a burst at the end.
    total = last_token_at - start
    assert total > 0 and (last_token_at - first_token_at) >= 0.5 * total


# ── pre-first-token fallback ──────────────────────────────────────────────────

def test_pretoken_stream_error_falls_back_to_non_streaming(deepseek_streaming):
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream_raise()):
        events = _run_turn(_pa_agent())

    # fell back -> synthesis went through _call_llm (2nd invocation)
    assert call_llm.state["n"] == 2, "should fall back to non-streaming _call_llm"
    assert "NON-STREAMED fallback answer." in _tokens(events)
    assert events[-1]["type"] == "end"


# ── A4: mid-stream drop is honest, not a finished answer ──────────────────────

def test_midstream_synth_error_appends_cutoff_notice(deepseek_streaming):
    """Tokens already left the provider. Finish with them — do not restart
    `_call_llm` (that would duplicate the body) — and mark the cut-off on
    persist/`end` so a partial is not saved as a complete answer.
    """
    call_llm = _tool_then_final()
    deltas = [
        "The raft volume is 30 × 20 × 1.5 = 900 m³.\n",
        "Including the documented ",
    ]
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream_midway_drop(deltas)):
        events = _run_turn(_pa_agent())

    assert call_llm.state["n"] == 1, (
        "mid-stream drop must not restart _call_llm (anti-duplicate)"
    )
    assert events[-1]["type"] == "end"
    end_content = events[-1].get("content") or ""
    assert _SYNTH_CUTOFF_NOTICE in end_content, end_content
    assert "900 m³" in end_content
    assert "NON-STREAMED" not in end_content
    assert "NON-STREAMED" not in _tokens(events)
    assert _SYNTH_CUTOFF_NOTICE in _tokens(events)


# ── citation sanitisation on streamed lines ───────────────────────────────────

def test_streamed_lines_are_citation_sanitised(deepseek_streaming):
    call_llm = _tool_then_final()
    deltas = ["See [source: C:\\\\Users\\\\example\\\\data\\\\XYZ-406_HSE.pdf, chunk 3]\n"]
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())
    text = _tokens(events)
    assert "XYZ-406_HSE.pdf" in text
    assert "C:\\" not in text and "/Users/" not in text  # raw path stripped


# ── empty-final forced retry preserved ────────────────────────────────────────

def test_empty_stream_triggers_forced_retry(deepseek_streaming):
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream_empty()):
        events = _run_turn(_pa_agent())
    # empty stream -> non-streamed forced retry -> _call_llm called a 2nd time
    assert call_llm.state["n"] == 2
    assert "NON-STREAMED fallback answer." in _tokens(events)
    assert events[-1]["type"] == "end"


# ── default off: streaming never engaged ──────────────────────────────────────

def test_flag_off_never_streams(monkeypatch):
    # ORDERS_0305Z: streaming is default ON now, so the OFF case sets the
    # kill-switch explicitly (SYNTHESIS_STREAMING=0) rather than relying on
    # the default. Same intent: with it off, the forced synthesis goes
    # through the non-streaming _call_llm path.
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "0")  # kill-switch OFF
    call_llm = _tool_then_final()

    async def _boom_stream(self, *a, **k):
        raise AssertionError("_stream_synthesis must not be called when flag is off")
        yield ""  # pragma: no cover

    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _boom_stream):
        events = _run_turn(_pa_agent())
    assert call_llm.state["n"] == 2  # synthesis via non-streaming _call_llm
    assert "NON-STREAMED fallback answer." in _tokens(events)


def test_an_unlisted_provider_never_streams(monkeypatch):
    """The outer gate carries its own provider list. OAI-shaped Ollama is not
    on it, and must not reach _stream_synthesis at all."""
    ollama_cfg = {"provider": "ollama", "url": "http://x/v1/chat/completions",
                  "env_key": "", "default_model": "glm-5.2:cloud"}
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(ollama_cfg))
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")
    call_llm = _tool_then_final()

    async def _boom_stream(self, *a, **k):
        raise AssertionError("must not stream on an unlisted provider")
        yield ""  # pragma: no cover

    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _boom_stream):
        events = _run_turn(_pa_agent())
    assert call_llm.state["n"] == 2


def test_deepseek_the_production_primary_reaches_the_streaming_path(monkeypatch):
    """The gate that matters in production. SYNTHESIS_STREAMING=1 and
    LLM_PROVIDER=deepseek are both set on the live service; the outer gate's
    provider allowlist must include the primary or the flag is on and does
    nothing — every live turn would silently take the non-streaming fallback.

    Asserted at the OUTER gate specifically: `_stream_synthesis` having a
    DeepSeek allowlist entry is not enough if this gate never calls it.
    """
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: {
        "provider": "deepseek",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    })
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")
    call_llm = _tool_then_final()

    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(["Isolate ", "the panel."])):
        events = _run_turn(_pa_agent())

    assert call_llm.state["n"] == 1, "synthesis went through _call_llm, not the stream"
    assert "Isolate the panel." in _tokens(events)


def test_streamed_xml_tool_leak_is_not_flushed(deepseek_streaming):
    """Leftover L4 class: SYNTHESIS_STREAMING must not emit XML tool calls."""
    from app.agents.runtime import _TOOL_FORMAT_FALLBACK

    call_llm = _tool_then_final()
    xml = (
        "<function_calls>\n"
        '<invoke name="formula_executor_v2">\n'
        "<parameter name=\"expression\">1+1</parameter>\n"
        "</invoke>\n"
        "</function_calls>\n"
    )
    deltas = [xml[i : i + 18] for i in range(0, len(xml), 18)]
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())
    text = _tokens(events)
    collapsed = text.lower().replace(" ", "")
    assert "<function_calls" not in collapsed
    assert "<invoke" not in collapsed
    assert _TOOL_FORMAT_FALLBACK[:40] in text
    assert events[-1]["type"] == "end"


@pytest.mark.parametrize("prose_first", [False, True])
def test_streamed_dsml_tool_leak_is_not_flushed(deepseek_streaming, prose_first):
    """Live d8d9573: the guard above knew XML and raw JSON, not DeepSeek's
    DSML, and ``<｜｜DSML｜｜ invoke name="construction_calc">`` reached the
    browser. Both live orders: markup first, and markup after a line of
    prose (which may legitimately have been shown already)."""
    from tests.test_a_failed_tool_is_not_a_deliverable import LIVE_LEAK

    text_in = ("The calculator returned a zeroed result.\n" if prose_first else "") + LIVE_LEAK
    deltas = [text_in[i : i + 11] for i in range(0, len(text_in), 11)]
    with patch.object(Agent, "_call_llm", _tool_then_final()), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())
    shown = _tokens(events)
    assert "DSML" not in shown
    assert "invoke" not in shown
    assert "contract_price" not in shown, "tool arguments leaked"
    assert events[-1]["type"] == "end"
    assert "DSML" not in (events[-1].get("content") or "")


def test_a_tool_that_failed_is_offered_again_not_disarmed(deepseek_streaming):
    """The cause behind the leak. Call 1 asks for a calculation that does not
    exist; the tool says so. The next model call must still HAVE tools --
    ``with_tools`` true -- so the model can use the list of valid names it was
    just handed, rather than being forced to answer empty-handed."""
    seen = []

    async def fake_llm(_self, messages, api_key, **kwargs):
        seen.append(kwargs.get("with_tools", True))
        if len(seen) == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "construction_calc",
                    "arguments": '{"calculation":"calculate_delay_damages"}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": "Recovered: re-ran it as delay_damages_daily."}}}

    async def tool_failed(self, tool_call, **kwargs):
        return {"name": "construction_calc", "ok": True, "result": {
            "status": "error", "error": "Unknown calculation 'calculate_delay_damages'.",
            "available": ["delay_damages_daily"]}}

    streamed = []

    async def spy_stream(self, messages, api_key, **kwargs):
        streamed.append(True)
        yield "should never be reached"

    with patch.object(Agent, "_call_llm", fake_llm), \
         patch.object(Agent, "_run_tool_call", tool_failed), \
         patch.object(Agent, "_stream_synthesis", spy_stream):
        events = _drain(_pa_agent().chat_stream(
            user_message="If Milestone 1 is 30 days late, what are the delay damages?",
            history=[], project_id=None, conversation_id=None, user_id=None,
        ))

    assert not streamed, "a failed tool was treated as the deliverable"
    assert len(seen) >= 2 and seen[1] is not False, (
        f"tools were disarmed after a FAILED call: with_tools={seen}"
    )
    from app.lib.source_labels import formula_display_name
    shown = formula_display_name("delay_damages_daily")
    tokens = _tokens(events)
    assert shown and f"re-ran it as {shown}" in tokens
    assert "delay_damages_daily" not in tokens


def test_streamed_json_tool_leak_is_not_flushed(deepseek_streaming):
    """UI-PHYS A5: SYNTHESIS_STREAMING must not flush raw tool-call JSON."""
    import json
    from app.agents.runtime import _TOOL_FORMAT_FALLBACK

    call_llm = _tool_then_final()
    leaked = json.dumps({
        "name": "search_project_documents",
        "arguments": {"query": "Delay Damages Contract Data"},
    })
    deltas = [leaked[i : i + 20] for i in range(0, len(leaked), 20)]
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())
    text = _tokens(events)
    assert leaked not in text
    assert "search_project_documents" not in text
    assert _TOOL_FORMAT_FALLBACK[:40] in text
    assert events[-1]["type"] == "end"
    assert leaked not in (events[-1].get("content") or "")


# ── _stream_synthesis unit: SSE parse + sanitiser chokepoint ──────────────────

class _FakeStreamResponse:
    def __init__(self, lines, status=200):
        self.status_code = status
        self._lines = lines
        self.text = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self):
        return b""


class _FakeAsyncClient:
    """Captures the posted payload and returns canned SSE lines."""
    captured = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method, url, json=None, headers=None):
        _FakeAsyncClient.captured["payload"] = json
        _FakeAsyncClient.captured["url"] = url
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello "}}]}',
            'data: {"choices":[{"delta":{"content":"world"}}]}',
            'data: {"choices":[{"delta":{}}],"usage":{"total_tokens":5}}',
            "data: [DONE]",
        ]
        return _FakeStreamResponse(lines)


def test_stream_synthesis_parses_sse_and_sanitises_outbound(monkeypatch):
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setattr("app.agents.runtime.httpx.AsyncClient", _FakeAsyncClient)
    agent = _pa_agent()
    # A contaminated message with a non-standard `reasoning` field must be
    # stripped by the SAME chokepoint the non-streaming path uses.
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "prior", "reasoning": "SECRET CHAIN OF THOUGHT",
         "tool_calls": [{"id": "x", "type": "function", "extra": "junk",
                         "function": {"name": "f", "arguments": "{}", "bogus": 1}}]},
        {"role": "user", "content": "go"},
    ]

    async def go():
        return [d async for d in agent._stream_synthesis(messages, "k")]

    out = asyncio.run(go())
    assert "".join(out) == "Hello world"
    posted = _FakeAsyncClient.captured["payload"]
    assert posted["stream"] is True
    assert "tools" not in posted  # synthesis offers no tools
    # sanitiser chokepoint applied to the streamed path too
    for m in posted["messages"]:
        assert "reasoning" not in m
        for tc in m.get("tool_calls", []) or []:
            assert set(tc.keys()) <= {"id", "type", "function"}
            assert set(tc.get("function", {}).keys()) <= {"name", "arguments"}


# ── tool-phase status line (long-ask stall fix) ─────────────────────────────

def _status_labels(events):
    return [e.get("label") for e in events if e.get("type") == "status"]


def test_tool_phase_status_precedes_first_token(deepseek_streaming):
    """The bubble must not sit blank through the pre-first-token tool phase:
    a `status` label is emitted at tool-phase entry, before any token."""
    from app.agents.core.streaming import _TOOL_PHASE_STATUS

    call_llm = _tool_then_final()
    deltas = ["Here ", "is ", "the ", "answer."]
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    types = [e.get("type") for e in events]
    assert "status" in types, "no status event emitted during the tool phase"
    assert _TOOL_PHASE_STATUS in _status_labels(events), "status carried no label"
    first_status = types.index("status")
    assert "token" in types, "turn produced no tokens"
    first_token = types.index("token")
    assert first_status < first_token, "status must precede the first token"


def test_tool_phase_status_also_on_nonstreamed_path(monkeypatch):
    """Even with streaming off (answer computed then chunked), the status still
    leads the turn -- the fix is on the tool phase, not the synthesis call."""
    from app.agents.core.streaming import _TOOL_PHASE_STATUS

    # Configure a provider so the turn reaches the tool phase, but keep
    # SYNTHESIS_STREAMING off so the answer is chunked, not delta-streamed.
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "0")

    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok):
        events = _run_turn(_pa_agent())

    assert _TOOL_PHASE_STATUS in _status_labels(events)
    types = [e.get("type") for e in events]
    assert types.index("status") < types.index("token")


# ── tool-phase budget / cap (long-ask sub-fix 2) ────────────────────────────

def test_tool_phase_budget_default_is_inert():
    """No env set -> every probe is allowed, nothing skipped."""
    from app.agents.core.streaming import _make_tool_phase_budget, _TOOL_PHASE_PRIORITY
    b = _make_tool_phase_budget(lambda: 0.0)
    assert all(b.allow(k) for k in _TOOL_PHASE_PRIORITY)
    assert b.skipped == []


def test_tool_phase_count_cap_is_deterministic(monkeypatch):
    """A count cap keeps the top-N priority probes and skips the rest, the same
    regardless of timing (clock never advances here)."""
    from app.agents.core.streaming import _make_tool_phase_budget, _TOOL_PHASE_PRIORITY
    monkeypatch.setenv("TOOL_PHASE_MAX_PROBES", "2")
    b = _make_tool_phase_budget(lambda: 0.0)
    verdicts = {k: b.allow(k) for k in _TOOL_PHASE_PRIORITY}
    allowed = [k for k, v in verdicts.items() if v]
    assert allowed == list(_TOOL_PHASE_PRIORITY[:2])
    assert b.skipped == list(_TOOL_PHASE_PRIORITY[2:])


def test_tool_phase_time_backstop_trips_when_deadline_passed(monkeypatch):
    """The wall-time arm refuses once the clock is past the deadline; it is a
    backstop, off by default (0 = disabled)."""
    from app.agents.core.streaming import _ToolPhaseBudget
    now = {"t": 0.0}
    b = _ToolPhaseBudget(max_probes=99, time_budget_ms=10.0, clock=lambda: now["t"])
    assert b.allow("file")           # within budget
    now["t"] = 1.0                   # 1s > 10ms deadline
    assert not b.allow("wbs_duration")
    assert "wbs_duration" in b.skipped


def test_tight_cap_skips_all_probes_but_turn_still_answers(deepseek_streaming, monkeypatch):
    """With the cap at 0, no predispatch probe runs, yet the turn still streams
    a status then an answer (the cap never breaks the turn)."""
    monkeypatch.setenv("TOOL_PHASE_MAX_PROBES", "0")
    from app.agents.core.streaming import _TOOL_PHASE_STATUS
    call_llm = _tool_then_final()
    deltas = ["All ", "good."]
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())
    types = [e.get("type") for e in events]
    assert _TOOL_PHASE_STATUS in _status_labels(events)
    assert "token" in types and types.index("status") < types.index("token")
