"""The forced no-tools retry names the rule that fired, never the text.

Live D1 (TESTER 10:38Z and 11:37Z, 18 Nov): a question whose honest answer is
"that is not in the project documents" ended as the generic fallback
sentence in 3 of 7 new chats. The trace shows the exit (``EMPTY-FINAL raw=Nc
-> forced retry``, then the fallback at 132 chars), but not WHICH rule made
the draft and the retry unusable: the drafts are not logged, and the
predicate returned a bare bool. ``forced_retry_reason`` is the one
implementation behind the retry decision, its nudge and the log line, so the
next live turn names its rule. No answer changes.

Every input here is invented by the test.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from tests._fixture_values import STUB_TOKEN_01

from app.agents import runtime as rt
from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK, Agent

# Two inputs per rule.
CASES = {
    "empty": ["", "   \n  "],
    "tool_format": [rt._TOOL_FORMAT_FALLBACK, rt._TOOL_FORMAT_FALLBACK],
    "search_promise": [
        "No sheet names a Quillon valve crate. Let me search the delivery notes next.",
        "I'll search the drawings register for the Brask manifold now.",
    ],
    "greeting_only": ["What can I help you with today?", "Hello! How can I assist you?"],
}
# Usable replies, including plain not-found statements: never retried.
USABLE = [
    "The Quillon valve crate is not in any project document, so there is no count or file to cite.",
    "No document mentions a Brask manifold; I cannot give a unit count for it.",
    "Grade B",
    "Yes",
]


@pytest.mark.parametrize("reason,texts", sorted(CASES.items()))
def test_each_rule_is_named(reason, texts):
    for text in texts:
        assert rt.forced_retry_reason(text) == reason, text
        assert rt._final_text_needs_forced_retry(text) is True


@pytest.mark.parametrize("text", USABLE)
def test_a_usable_reply_has_no_reason(text):
    assert rt.forced_retry_reason(text) is None
    assert rt._final_text_needs_forced_retry(text) is False
    assert rt._forced_retry_nudge(text) is None


@pytest.mark.parametrize("text", [
    "The fetch_document step returned no rows.",
    "The bill_rows operation returned zero chunks.",
])
def test_platform_only_is_named(text, monkeypatch):
    from app.agents import answer_exit
    monkeypatch.setattr(answer_exit, "check_text", lambda t: "")
    assert rt.forced_retry_reason(text) == "platform_only"
    assert rt._forced_retry_nudge(text) == rt._USER_TERMS_RETRY_NUDGE


@pytest.mark.parametrize("text,nudge", [
    (CASES["search_promise"][0], "_SEARCH_PREAMBLE_RETRY_NUDGE"),
    (CASES["search_promise"][1], "_SEARCH_PREAMBLE_RETRY_NUDGE"),
    (CASES["greeting_only"][0], "_GREETING_RETRY_NUDGE"),
    (CASES["greeting_only"][1], "_GREETING_RETRY_NUDGE"),
    (rt._TOOL_FORMAT_FALLBACK, "_TOOL_FORMAT_RETRY_NUDGE"),
    ("", "_EMPTY_RETRY_NUDGE"),
    ("  ", "_EMPTY_RETRY_NUDGE"),
])
def test_nudge_follows_the_named_rule(text, nudge):
    expected = getattr(rt, nudge) if nudge else None
    assert rt._forced_retry_nudge(text) == expected


# ── through both drivers, model stubbed at its last hop ─────────────────────

def _ok(content: str) -> Dict[str, Any]:
    return {"status": "success", "choice": {"message": {"content": content, "role": "assistant"}},
            "raw": {"model": "test-model"}}


def _run(mode: str, replies: List[Dict[str, Any]], monkeypatch) -> tuple[str, int]:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "20")
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "1")
    agent = Agent(name="reason-test", description="d", system_prompt="x", allowed_blocks=[])
    calls: List[int] = []

    async def _llm(messages, *_a, **_kw):
        calls.append(1)
        return replies[min(len(calls), len(replies)) - 1]

    async def drain():
        if mode == "chat":
            out = await agent.chat(user_message="how many units does the lot hold?", history=[],
                                   project_id=None, conversation_id=None, user_id=None)
            return (out or {}).get("content") or (out or {}).get("response") or str(out)
        events = [e async for e in agent.chat_stream(user_message="how many units does the lot hold?")]
        return "".join(e.get("content", "") for e in events if e["type"] == "token")

    with patch.object(agent, "_call_llm", _llm):
        text = asyncio.run(drain())
    return text, len(calls)


@pytest.mark.parametrize("mode", ["stream", "chat"])
@pytest.mark.parametrize("reason", ["search_promise", "greeting_only"])
def test_the_fallback_line_names_the_rule_and_not_the_text(mode, reason, monkeypatch, caplog):
    draft, retry = CASES[reason]
    caplog.set_level(logging.INFO)
    text, n = _run(mode, [_ok(draft), _ok(retry)], monkeypatch)
    assert n == 2
    # D1: an instructed retry that still gave nothing is stated plainly.
    assert _EMPTY_RESPONSE_FALLBACK not in text and "did not yield an answer" in text
    lines = [r.getMessage() for r in caplog.records]
    assert any(f"unusable final_text reason={reason}" in m for m in lines), lines
    assert any(f"forced retry unusable reason={reason}" in m for m in lines), lines
    assert not any(draft in m or retry in m for m in lines), "a draft reached the log"


@pytest.mark.parametrize("mode", ["stream", "chat"])
def test_a_provider_error_on_the_retry_is_named(mode, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    text, n = _run(mode, [_ok(CASES["search_promise"][1]), {"status": "error", "error": "x"}],
                   monkeypatch)
    assert n == 2
    assert _EMPTY_RESPONSE_FALLBACK not in text and "did not yield an answer" in text
    lines = [r.getMessage() for r in caplog.records]
    assert any("forced retry failed reason=provider_error" in m for m in lines), lines
    assert not any("forced retry unusable reason=" in m for m in lines), lines


@pytest.mark.parametrize("mode", ["stream", "chat"])
def test_a_good_retry_logs_no_fallback(mode, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    text, n = _run(mode, [_ok(CASES["search_promise"][0]), _ok(USABLE[0])], monkeypatch)
    assert n == 2
    assert "not in any project document" in text
    lines = [r.getMessage() for r in caplog.records]
    assert any("unusable final_text reason=search_promise" in m for m in lines), lines
    assert not any("fallback shown" in m for m in lines), lines


# The three trace helpers the turn paths call: each decides AND names the
# rule. Direct, two inputs per decision; the return value and the line both
# matter (a helper that returned the right bool but logged nothing, or
# logged but always returned False, would hide the rule or change the turn).

def _lines(caplog) -> List[str]:
    return [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("where,reason,text", [
    ("chat", "search_promise", CASES["search_promise"][0]),
    ("chat_stream", "greeting_only", CASES["greeting_only"][1]),
])
def test_the_start_line_names_the_path_and_the_rule(where, reason, text, caplog):
    caplog.set_level(logging.INFO)
    rt.log_forced_retry_start(where, text)
    assert f"{where}: unusable final_text reason={reason}, forcing no-tools retry" in _lines(caplog)
    assert not any(text.strip() and text in m for m in _lines(caplog))


@pytest.mark.parametrize("where,resp", [
    ("chat", {"status": "error", "error": "upstream 502"}),
    ("chat_stream", {"status": "error"}),
])
def test_an_errored_retry_is_failed_and_named(where, resp, caplog):
    caplog.set_level(logging.INFO)
    assert rt.forced_retry_failed(where, resp) is True
    assert f"{where}: forced retry failed reason=provider_error; fallback shown" in _lines(caplog)


@pytest.mark.parametrize("resp", [
    {"status": "success", "choice": {"message": {"content": "Grade B"}}},
    {"choice": {"message": {"content": "Yes"}}},
])
def test_a_retry_that_returned_is_not_failed(resp, caplog):
    caplog.set_level(logging.INFO)
    assert rt.forced_retry_failed("chat_stream", resp) is False
    assert not any("forced retry failed" in m for m in _lines(caplog))


@pytest.mark.parametrize("where,reason,text,raw_len,tail", [
    ("chat", "search_promise", CASES["search_promise"][1], None, "; fallback shown"),
    ("chat_stream", "empty", CASES["empty"][1], 37, " raw=37c; fallback shown"),
])
def test_an_unusable_retry_is_unusable_and_named(where, reason, text, raw_len, tail, caplog):
    caplog.set_level(logging.INFO)
    assert rt.forced_retry_still_unusable(where, text, raw_len=raw_len) is True
    assert f"{where}: forced retry unusable reason={reason}{tail}" in _lines(caplog)


@pytest.mark.parametrize("text", USABLE[:2])
def test_a_usable_retry_is_kept_and_not_traced(text, caplog):
    caplog.set_level(logging.INFO)
    assert rt.forced_retry_still_unusable("chat_stream", text, raw_len=len(text)) is False
    assert not any("forced retry unusable" in m for m in _lines(caplog))


@pytest.mark.parametrize("where", ["chat", "chat_stream"])
def test_our_own_fallback_is_never_traced_as_a_model_reply(where, caplog):
    caplog.set_level(logging.INFO)
    expected = rt.forced_retry_reason(_EMPTY_RESPONSE_FALLBACK) is not None
    assert rt.forced_retry_still_unusable(where, _EMPTY_RESPONSE_FALLBACK) is expected
    assert not any("forced retry unusable" in m for m in _lines(caplog))
