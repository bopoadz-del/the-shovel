"""The browser nightly's harness is checked on every PR, not once a night.

``browser-nightly`` was red for at least four consecutive nights with

    AssertionError: assistant error bubble: 'Something went wrong. Please try again.'

on its first question. Nothing in the product was broken. The harness had
gone stale in two places, one behind the other:

1. It passed the API-key gate with a dummy ``GROQ_API_KEY``. Groq was removed
   as a provider, so the gate answered "No DEEPSEEK_API_KEY configured."
   before the stub LLM was ever called.
2. Behind that, the stub's ``_call_llm`` lacked the ``deadline`` parameter the
   runtime now passes, so the turn crashed with a TypeError.

Regular CI runs ``--ignore=tests/browser``, so no PR that caused either drift
could see it. These tests are the missing tripwire: they need no browser, and
they fail in the PR that changes the runtime out from under the harness.
"""
from __future__ import annotations

import inspect
import json

import pytest

from app.agents import runtime
from tests.browser import _llm_stub


@pytest.fixture
def stub_installed(monkeypatch):
    """Install the nightly's stub for one test, and put the real method back."""
    real = runtime.Agent._call_llm
    monkeypatch.setattr(runtime.Agent, "_call_llm", real)  # registers the undo
    monkeypatch.setenv("CEREBRUM_UI_PHYS_STUB", "1")
    _llm_stub.install_stub()
    assert runtime.Agent._call_llm is not real, "stub did not install"
    return real


def test_the_stub_accepts_every_argument_the_runtime_passes(stub_installed):
    real = inspect.signature(stub_installed).parameters
    stub = inspect.signature(runtime.Agent._call_llm).parameters
    missing = [name for name in real if name not in stub]
    assert not missing, (
        f"tests/browser/_llm_stub.py::_stub is missing {missing}; the runtime "
        "passes them, so the nightly's first question will crash with a "
        "TypeError and the UI will show 'Something went wrong'."
    )


def test_the_harness_env_gets_past_the_api_key_gate(monkeypatch):
    for name in ("LLM_PROVIDER", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    env = _llm_stub.stub_provider_env()
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    needed = runtime._llm_config()["env_key"]

    assert needed in env, (
        f"the runtime's provider wants {needed}, but the nightly harness only "
        f"sets {sorted(env)}; chat_stream will answer 'No {needed} configured.'"
    )


def test_the_harness_never_carries_a_real_key():
    for name, value in _llm_stub.stub_provider_env().items():
        if name.endswith("_API_KEY"):
            assert "not-a-real-key" in value, name


@pytest.mark.asyncio
async def test_the_stub_answers_the_nightlys_first_question(stub_installed):
    """Called the way the runtime calls it, keyword arguments and all."""
    catalog = json.loads(_llm_stub._CATALOG.read_text(encoding="utf-8"))
    a1 = catalog["cases"]["A1"]
    real_params = inspect.signature(stub_installed).parameters
    kwargs = {
        name: None for name in real_params
        if name not in ("self", "messages", "api_key")
    }

    resp = await runtime.Agent._call_llm(
        object(), [{"role": "user", "content": a1["ask"]}], "stub", **kwargs,
    )

    assert resp["status"] == "success", resp
    content = resp["choice"]["message"]["content"]
    for token in a1["must"]:
        assert token in content
