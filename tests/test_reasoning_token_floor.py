"""A reasoning model's max_tokens must cover its reasoning, or content is
structurally impossible.

A reasoning model spends the shared ``max_tokens`` budget on chain-of-thought
FIRST; a cap below that burn yields finish_reason=length with EMPTY content
every time, the empty-final forced retry runs with the same cap and fails
identically, and the user gets the canned message.

The mechanism is a generic ``reasoning_min_tokens`` hook applied at the same
outbound chokepoint as ``fixed_temperature`` (``_provider_max_tokens``), so a
constrained provider can declare its floor in ``_llm_config`` rather than
hardcoding a value globally. Neither supported provider (DeepSeek primary,
OpenRouter fallback) declares one today, so both pass the agent's own budget
through untouched (OpenRouter still applies its credit ceiling separately).
"""
from __future__ import annotations

import app.agents.runtime as rt


def test_deepseek_config_declares_no_reasoning_floor(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    cfg = rt._llm_config()
    assert cfg["provider"] == "deepseek"
    assert "reasoning_min_tokens" not in cfg, cfg


def test_floor_lifts_a_starving_budget():
    """The generic hook lifts a starving budget when a provider declares it."""
    cfg = {"provider": "deepseek", "reasoning_min_tokens": 4096}
    assert rt._provider_max_tokens(cfg, 1024) == 4096


def test_floor_does_not_shrink_a_generous_budget():
    cfg = {"provider": "deepseek", "reasoning_min_tokens": 4096}
    assert rt._provider_max_tokens(cfg, 8192) == 8192


def test_deepseek_keeps_the_agents_budget_with_no_floor(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    cfg = rt._llm_config()
    assert rt._provider_max_tokens(cfg, 1024) == 1024
    assert rt._provider_max_tokens(cfg, 8192) == 8192
