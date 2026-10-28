"""The cross-provider fallback builds its config without mutating the
process-wide LLM_PROVIDER env (ORDERS_1440Z correctness item).

_cross_provider_fallback used to set os.environ["LLM_PROVIDER"] to the fallback
name, call _llm_config(), then restore it — a global mutation that raced under
concurrent chats and could flip another request's provider mid-flight. It now
passes the provider name to _llm_config() explicitly; these tests pin that the
env is untouched and the primary resolution is unaffected.
"""
import os

import app.agents.runtime as rt


def test_llm_config_accepts_an_explicit_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    assert rt._llm_config(provider="openrouter")["provider"] == "openrouter"
    # The env is the source of truth when no arg is given, and is unchanged.
    assert rt._llm_config()["provider"] == "deepseek"
    assert os.environ["LLM_PROVIDER"] == "deepseek"


def test_cross_provider_fallback_does_not_touch_os_environ(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    primary = rt._llm_config()
    assert primary["provider"] == "deepseek"

    before = os.environ.get("LLM_PROVIDER")
    cfg = rt._cross_provider_fallback(primary)
    assert cfg is not None and cfg["provider"] == "openrouter"
    # No global mutation: LLM_PROVIDER is exactly what it was.
    assert os.environ.get("LLM_PROVIDER") == before
    # And the primary still resolves to deepseek afterwards.
    assert rt._llm_config()["provider"] == "deepseek"


def test_cross_provider_fallback_none_when_no_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Fallback declares an env_key it cannot satisfy -> no usable fallback, and
    # still no env mutation.
    before = os.environ.get("LLM_PROVIDER")
    assert rt._cross_provider_fallback(rt._llm_config()) is None
    assert os.environ.get("LLM_PROVIDER") == before


class _RecordingEnv(dict):
    """A process-env stand-in that records every write to LLM_PROVIDER.

    before/after equality (the tests above) cannot see a value that is set and
    then restored within the call — exactly what the old mutate-then-restore
    code did. This records the write itself, so ANY touch of LLM_PROVIDER during
    the call is caught, restored or not.
    """

    def __init__(self, base):
        super().__init__(base)
        self.provider_writes = 0

    def __setitem__(self, key, value):
        if key == "LLM_PROVIDER":
            self.provider_writes += 1
        super().__setitem__(key, value)

    def __delitem__(self, key):
        if key == "LLM_PROVIDER":
            self.provider_writes += 1
        super().__delitem__(key)


def _resolve_without_touching_env(monkeypatch, primary_provider, fallback_provider, fallback_key):
    """Run a cross-provider fallback with the env spied; return (cfg, writes)."""
    rec = _RecordingEnv(os.environ)
    rec["LLM_PROVIDER"] = primary_provider
    rec["LLM_FALLBACK_PROVIDER"] = fallback_provider
    rec[fallback_key] = "test-key"
    # os.getenv / os.environ both read the module global, so rebinding it routes
    # runtime.py's reads and any write through the spy.
    monkeypatch.setattr(os, "environ", rec)
    primary = rt._llm_config()
    assert primary["provider"] == primary_provider
    rec.provider_writes = 0  # ignore the setup writes; count only the call's
    cfg = rt._cross_provider_fallback(primary)
    return cfg, rec.provider_writes


def test_fallback_never_writes_llm_provider_deepseek_to_openrouter(monkeypatch):
    cfg, writes = _resolve_without_touching_env(
        monkeypatch, "deepseek", "openrouter", "OPENROUTER_API_KEY"
    )
    assert cfg is not None and cfg["provider"] == "openrouter"
    # The regression (set-then-restore) would make this >= 1.
    assert writes == 0


def test_fallback_never_writes_llm_provider_openrouter_to_deepseek(monkeypatch):
    # The reverse pair: the mechanism must hold whichever provider is primary,
    # so the fix cannot be a one-direction special case.
    cfg, writes = _resolve_without_touching_env(
        monkeypatch, "openrouter", "deepseek", "DEEPSEEK_API_KEY"
    )
    assert cfg is not None and cfg["provider"] == "deepseek"
    assert writes == 0
