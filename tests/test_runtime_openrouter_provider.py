"""The agent runtime supports OpenRouter as a first-class provider.

When ``LLM_PROVIDER=openrouter`` is set, the runtime:
- routes to OpenRouter's OpenAI-compatible chat-completions endpoint
- authenticates with ``OPENROUTER_API_KEY``
- defaults to ``openrouter/free`` (override with ``OPENROUTER_MODEL``)
- refuses paid slugs unless ``OPENROUTER_ALLOW_PAID=1``
- caps outbound ``max_tokens`` (``OPENROUTER_MAX_TOKENS``, default 2048)
  so a $0-balance free account is not 402'd by agent YAML of 8192
- truncates tool / fetch payloads and compacts the prompt under
  ``OPENROUTER_PROMPT_TOKEN_CEILING`` (default 8000, 3 chars/token)
- retries HTTP 402 ``in_flight`` / ``can only afford N`` on the same hop
- retries HTTP 402 ``Prompt tokens limit exceeded`` by compacting to the cap
- retries HTTP 429 on the same hop (honors ``Retry-After``)

No live OpenRouter calls. Keys in these tests are placeholders.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.runtime import (
    OPENROUTER_API_URL,
    OPENROUTER_DEFAULT_MAX_TOKENS,
    OPENROUTER_DEFAULT_MODEL,
    OPENROUTER_DEFAULT_PROMPT_TOKEN_CEILING,
    _FETCH_DOCUMENT_MAX_CHARS,
    _OPENROUTER_FETCH_DOCUMENT_MAX_CHARS,
    _OPENROUTER_TOOL_RESULT_MAX_CHARS,
    _TOOL_RESULT_MAX_CHARS,
    _approx_prompt_tokens,
    _compact_messages_for_openrouter,
    _effective_fetch_document_max_chars,
    _effective_tool_result_max_chars,
    _llm_config,
    _http_status_is_retryable,
    _openrouter_402_afford_max_tokens,
    _openrouter_402_is_in_flight,
    _openrouter_402_should_retry,
    _parse_prompt_token_limit,
    _provider_max_tokens,
    _provider_retry_delay_seconds,
    _resolve_openrouter_model,
    _tool_result_content,
)


def test_llm_config_picks_openrouter_when_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _llm_config()
    assert cfg["provider"] == "openrouter"
    assert cfg["url"] == OPENROUTER_API_URL
    assert cfg["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert cfg["env_key"] == "OPENROUTER_API_KEY"
    assert cfg["default_model"] == OPENROUTER_DEFAULT_MODEL
    assert cfg["default_model"] == "openrouter/free"


def test_openrouter_model_override_free_slug(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    assert _llm_config()["default_model"] == "meta-llama/llama-3.3-70b-instruct:free"


def test_openrouter_missing_key_still_returns_config(monkeypatch):
    """Callers check env_key themselves; missing key must not drop the branch."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _llm_config()
    assert cfg["provider"] == "openrouter"
    assert cfg["env_key"] == "OPENROUTER_API_KEY"
    assert cfg["url"] == OPENROUTER_API_URL


def test_openrouter_refuses_paid_slug_unless_allowlisted(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ALLOW_PAID", raising=False)
    assert (
        _resolve_openrouter_model("anthropic/claude-3.5-sonnet")
        == OPENROUTER_DEFAULT_MODEL
    )
    monkeypatch.setenv("OPENROUTER_ALLOW_PAID", "1")
    assert (
        _resolve_openrouter_model("anthropic/claude-3.5-sonnet")
        == "anthropic/claude-3.5-sonnet"
    )


def test_openrouter_paid_slug_in_config_falls_back_to_free_router(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o")
    monkeypatch.delenv("OPENROUTER_ALLOW_PAID", raising=False)
    assert _llm_config()["default_model"] == "openrouter/free"


def test_openrouter_has_no_fixed_temperature(monkeypatch):
    """OpenRouter is OpenAI-shaped: the agent keeps its own temperature."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert "fixed_temperature" not in _llm_config()


def test_openrouter_caps_agent_max_tokens_to_default(monkeypatch):
    """Agent YAML of 8192 402s a $0 OpenRouter balance; clamp to the default."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)
    cfg = _llm_config()
    assert cfg["provider"] == "openrouter"
    assert _provider_max_tokens(cfg, 8192) == OPENROUTER_DEFAULT_MAX_TOKENS
    assert _provider_max_tokens(cfg, 8192) == 2048
    # A smaller agent budget is not lifted.
    assert _provider_max_tokens(cfg, 1024) == 1024


def test_openrouter_max_tokens_env_override(monkeypatch):
    """OPENROUTER_MAX_TOKENS wins over the default ceiling."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "2000")
    cfg = _llm_config()
    assert _provider_max_tokens(cfg, 8192) == 2000
    assert _provider_max_tokens({"provider": "openrouter"}, 8192) == 2000


def test_openrouter_max_tokens_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "not-a-number")
    assert _provider_max_tokens({"provider": "openrouter"}, 8192) == 2048
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "0")
    assert _provider_max_tokens({"provider": "openrouter"}, 8192) == 2048


def test_openrouter_ceiling_does_not_touch_other_providers(monkeypatch):
    """DeepSeek keeps its full budget; the ceiling is OpenRouter-only."""
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "2000")
    assert _provider_max_tokens({"provider": "deepseek"}, 8192) == 8192
    # The generic reasoning floor still lifts a starving budget, but never
    # applies the OpenRouter ceiling to a non-OpenRouter provider.
    assert _provider_max_tokens({"provider": "deepseek", "reasoning_min_tokens": 4096}, 8192) == 8192


def test_openrouter_tool_and_fetch_caps_are_tighter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert _effective_tool_result_max_chars() == _OPENROUTER_TOOL_RESULT_MAX_CHARS
    assert _effective_fetch_document_max_chars() == _OPENROUTER_FETCH_DOCUMENT_MAX_CHARS
    assert _OPENROUTER_TOOL_RESULT_MAX_CHARS < _TOOL_RESULT_MAX_CHARS
    assert _OPENROUTER_FETCH_DOCUMENT_MAX_CHARS < _FETCH_DOCUMENT_MAX_CHARS


def test_openrouter_caps_do_not_change_other_providers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    assert _effective_tool_result_max_chars() == _TOOL_RESULT_MAX_CHARS
    assert _effective_fetch_document_max_chars() == _FETCH_DOCUMENT_MAX_CHARS
    # Unset LLM_PROVIDER resolves to DeepSeek — still no OpenRouter caps.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert _effective_tool_result_max_chars() == _TOOL_RESULT_MAX_CHARS
    assert _effective_fetch_document_max_chars() == _FETCH_DOCUMENT_MAX_CHARS


def test_openrouter_tool_result_truncates_under_tighter_cap(monkeypatch):
    """A mid-size payload fits DeepSeek's budget but not OpenRouter's 4k cap."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    payload = {"text": "A" * 5000, "source": "extracted"}
    out = _tool_result_content(payload)
    assert len(out) <= _OPENROUTER_TOOL_RESULT_MAX_CHARS
    parsed = json.loads(out)
    assert parsed["truncated"] is True
    assert "A" * 80 in parsed["preview"]


def test_openrouter_compaction_keeps_prompt_under_ceiling(monkeypatch):
    """search + fetch dumps that 402'd live (18398 > 10335) stay under 8k."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_PROMPT_TOKEN_CEILING", raising=False)
    messages = [
        {"role": "system", "content": "You are a project assistant. " * 200},
        {"role": "user", "content": "What is the Time for Completion in Contract Data?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "s1", "type": "function",
             "function": {"name": "search_project_documents", "arguments": "{}"}},
        ]},
        {
            "role": "tool",
            "tool_call_id": "s1",
            "name": "search_project_documents",
            "content": "SEARCH " + ("hit " * 5000),
        },
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "f1", "type": "function",
             "function": {"name": "fetch_document", "arguments": "{}"}},
        ]},
        {
            "role": "tool",
            "tool_call_id": "f1",
            "name": "fetch_document",
            "content": (
                "CONTRACT DATA Time for Completion 852 days. "
                + ("x" * 24000)
            ),
        },
    ]
    assert _approx_prompt_tokens(messages) > OPENROUTER_DEFAULT_PROMPT_TOKEN_CEILING
    out = _compact_messages_for_openrouter(messages)
    assert _approx_prompt_tokens(out) <= OPENROUTER_DEFAULT_PROMPT_TOKEN_CEILING
    last_tool = next(
        m for m in reversed(out)
        if isinstance(m, dict) and m.get("role") == "tool"
    )
    content = str(last_tool.get("content") or "")
    assert "Time for Completion" in content
    assert "852" in content
    assert "OpenRouter" in content


def test_openrouter_402_in_flight_is_retryable():
    live = (
        '{"error":{"message":"This request requires more credits, '
        'or fewer max_tokens. in_flight_budget_exhausted"}}'
    )
    assert _openrouter_402_is_in_flight(live)
    assert _openrouter_402_should_retry(live)
    assert _openrouter_402_afford_max_tokens(live) is None


def test_openrouter_402_afford_fewer_is_parsed():
    live = "You requested up to 2048 tokens, but can only afford 996"
    assert _openrouter_402_afford_max_tokens(live) == 996
    assert _openrouter_402_should_retry(live)
    assert not _openrouter_402_is_in_flight(live)


def test_openrouter_402_generic_credits_is_not_retried():
    assert not _openrouter_402_should_retry("Insufficient credits")
    assert _openrouter_402_afford_max_tokens("bad") is None


def test_openrouter_402_prompt_limit_is_retryable():
    live = "Prompt tokens limit exceeded: 18398 > 10335."
    assert _parse_prompt_token_limit(live) == (18398, 10335)
    assert _openrouter_402_should_retry(live)
    assert not _openrouter_402_should_retry("Insufficient credits")


def test_http_status_is_retryable_includes_openrouter_402():
    assert _http_status_is_retryable(402)
    assert not _http_status_is_retryable(400)


def test_provider_retry_delay_honors_retry_after():
    class _H:
        headers = {"Retry-After": "2.5"}

    assert _provider_retry_delay_seconds(_H(), 1) == 2.5
    assert 0.4 <= _provider_retry_delay_seconds(object(), 1) <= 8.0


def test_frontend_maps_live_openrouter_402_wording():
    """The Wave1 banner was the fallback: live 402 bodies never said
    'insufficient'. The workspace mapper must match the live strings."""
    src = Path("frontend/src/pages/ProjectWorkspace.tsx").read_text(
        encoding="utf-8"
    )
    for needle in (
        "can only afford",
        "http 402",
        "in-flight",
        "prompt tokens limit",
        "fewer max_tokens",
    ):
        assert needle in src.lower(), f"friendlyErrorMessage lost {needle!r}"
