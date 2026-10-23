"""The agent runtime supports DeepSeek as a first-class cloud provider.

When ``LLM_PROVIDER=deepseek`` is set, the runtime:
- routes to DeepSeek's native OpenAI-compatible chat-completions endpoint
- authenticates with ``DEEPSEEK_API_KEY``
- defaults to ``deepseek-chat`` (override with ``DEEPSEEK_MODEL``)
- accepts ``deepseek-reasoner`` (and any other catalogue id) when set
- does NOT pin ``fixed_temperature`` (unlike Moonshot K2)

DeepSeek is the primary: an unset or unrecognised ``LLM_PROVIDER`` resolves
here. OpenRouter is the only other selectable provider.

No live DeepSeek calls. Keys in these tests are placeholders.
"""
from __future__ import annotations

from app.agents.runtime import (
    DEEPSEEK_API_URL,
    DEEPSEEK_DEFAULT_MODEL,
    OPENROUTER_DEFAULT_MODEL,
    _llm_config,
    _provider_temperature,
    _resolve_attempt_model,
)


def test_llm_config_picks_deepseek_when_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = _llm_config()
    assert cfg["provider"] == "deepseek"
    assert cfg["url"] == DEEPSEEK_API_URL
    assert cfg["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert cfg["env_key"] == "DEEPSEEK_API_KEY"
    assert cfg["default_model"] == DEEPSEEK_DEFAULT_MODEL
    assert cfg["default_model"] == "deepseek-chat"


def test_deepseek_model_override_reasoner(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    assert _llm_config()["default_model"] == "deepseek-reasoner"


def test_deepseek_model_override_catalogue_id(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    assert _llm_config()["default_model"] == "deepseek-v4-flash"


def test_deepseek_missing_key_still_returns_config(monkeypatch):
    """Callers check env_key themselves; missing key must not drop the branch."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = _llm_config()
    assert cfg["provider"] == "deepseek"
    assert cfg["env_key"] == "DEEPSEEK_API_KEY"
    assert cfg["url"] == DEEPSEEK_API_URL


def test_deepseek_has_no_fixed_temperature(monkeypatch):
    """DeepSeek accepts the agent's temperature; do not copy Kimi's pin."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    cfg = _llm_config()
    assert "fixed_temperature" not in cfg
    assert _provider_temperature(cfg, 0.3) == 0.3


def test_unset_provider_resolves_to_deepseek(monkeypatch):
    """DeepSeek is the primary: an unset LLM_PROVIDER resolves here."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    cfg = _llm_config()
    assert cfg["provider"] == "deepseek"
    assert cfg["url"] == DEEPSEEK_API_URL


def test_unrecognised_provider_falls_through_to_deepseek(monkeypatch):
    """``openrouter`` resolves; leftover ``openai`` / removed names / junk
    fall through to DeepSeek rather than reviving a dead provider."""
    for name in ("openai", "kimi", "groq", "ollama", "nonsense"):
        monkeypatch.setenv("LLM_PROVIDER", name)
        assert _llm_config()["provider"] == "deepseek", name


def test_deepseek_remaps_foreign_hat_pin(monkeypatch):
    """Every hat YAML pins kimi-k2.6; that id 404s on api.deepseek.com."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    cfg = _llm_config()
    assert _resolve_attempt_model(cfg, "kimi-k2.6") == "deepseek-chat"
    assert _resolve_attempt_model(cfg, "moonshot-v1-128k") == "deepseek-chat"
    assert _resolve_attempt_model(cfg, "") == "deepseek-chat"
    assert _resolve_attempt_model(cfg, "deepseek-chat") == "deepseek-chat"
    assert _resolve_attempt_model(cfg, "deepseek-reasoner") == "deepseek-reasoner"


def test_openrouter_remaps_foreign_hat_pin(monkeypatch):
    """A leftover kimi-k2.6 / moonshot pin must not be sent to OpenRouter."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    cfg = _llm_config()
    assert _resolve_attempt_model(cfg, "kimi-k2.6") == OPENROUTER_DEFAULT_MODEL
    assert _resolve_attempt_model(cfg, "moonshot-v1-128k") == OPENROUTER_DEFAULT_MODEL
    assert _resolve_attempt_model(cfg, "deepseek-chat") == OPENROUTER_DEFAULT_MODEL
    assert _resolve_attempt_model(cfg, "") == OPENROUTER_DEFAULT_MODEL
