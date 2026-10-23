"""Subprocesses must not inherit app secrets (audit §6.1).

Before this fix every subprocess (antiword/catdoc converters, the code/bash
sandbox, hardware probes) inherited the full parent environment — SECRET_KEY,
DATABASE_URL, the LLM API keys, the master key — and the bash sandbox could
``echo $SECRET_KEY`` straight to output. scrubbed_env() strips secret-bearing
vars while keeping functional system env; every subprocess call site now uses
it. The python-code-block test is end-to-end: a real subprocess proves the
planted secret does not reach it.
"""
from __future__ import annotations

import os

import pytest

from app.core.subprocess_env import is_secret_env_name, scrubbed_env
from tests._fixture_values import ENV_SECRET_11, ENV_SECRET_12, ENV_SECRET_13


# ── unit: the denylist ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "SECRET_KEY", "DATABASE_URL", "REDIS_URL", "DATA_ENCRYPTION_KEY",
    "CEREBRUM_MASTER_KEY", "SENTRY_DSN", "GDRIVE_SERVICE_ACCOUNT_JSON",
    "BOOTSTRAP_USER_PASSWORD", "GOOGLE_CLIENT_SECRET", "GOOGLE_ACCESS_TOKEN",
    "GROQ_API_KEY", "KIMI_API_KEY", "OLLAMA_API_KEY", "SERPER_API_KEY",
    "GITHUB_TOKEN", "RENDER_API_KEY", "CEREBRUM_API_KEY_CDEV",
    "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY",
    "SOME_REFRESH_TOKEN", "X_SIGNING_KEY",
])
def test_secret_names_are_stripped(name):
    assert is_secret_env_name(name) is True


@pytest.mark.parametrize("name", [
    # System/runtime env a subprocess actually NEEDS — must always be kept.
    "PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "QT_QPA_PLATFORM",
    "LD_LIBRARY_PATH", "TESSDATA_PREFIX", "PYTHONDONTWRITEBYTECODE",
    "NODE_PATH",
    # Non-secret app config that a subprocess does not read but also isn't a
    # secret — kept because it matches no secret pattern.
    "FRONTEND_URL", "OLLAMA_URL", "MASTER_CORPUS_NAME", "JWT_EXPIRY_SECONDS",
])
def test_functional_names_are_kept(name):
    assert is_secret_env_name(name) is False


def test_harmless_overstrip_of_nonsecret_config_is_acceptable():
    # MAX_RAG_TOKENS / RAG_DAILY_TOKEN_BUDGET are NOT secrets but match the
    # '_TOKEN' pattern and get stripped. This is by design and harmless: the
    # subprocesses that receive scrubbed_env (converters, code/bash sandbox,
    # hw probes) do not read the app's config from the environment. The
    # denylist deliberately errs toward stripping, never toward leaking.
    assert is_secret_env_name("MAX_RAG_TOKENS") is True


def test_scrubbed_env_strips_secrets_keeps_functional(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_11)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@h/db")
    monkeypatch.setenv("GROQ_API_KEY", ENV_SECRET_12)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("FRONTEND_URL", "https://theshovel.ai")
    env = scrubbed_env()
    assert "SECRET_KEY" not in env
    assert "DATABASE_URL" not in env
    assert "GROQ_API_KEY" not in env
    assert env.get("PATH") == "/usr/bin:/bin"
    assert env.get("FRONTEND_URL") == "https://theshovel.ai"


def test_scrubbed_env_extra_overrides_win(monkeypatch):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    env = scrubbed_env({"PYTHONDONTWRITEBYTECODE": "1"})
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


# ── end-to-end: a real subprocess cannot read the planted secret ─────────────

def test_python_code_block_subprocess_cannot_read_secret(monkeypatch):
    from app.blocks import code as code_mod

    monkeypatch.setenv("SECRET_KEY", ENV_SECRET_13)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:PLANTED_PW@h/db")
    snippet = (
        "import os\n"
        "print('LEAK' if os.environ.get('SECRET_KEY') "
        "or os.environ.get('DATABASE_URL') else 'CLEAN')\n"
        "print('HASPATH' if os.environ.get('PATH') else 'NOPATH')\n"
    )
    result = code_mod._run_python(snippet, timeout=15)
    assert result["status"] == "success", result
    out = result["output"]
    assert "CLEAN" in out and "LEAK" not in out
    # The functional env (PATH) must still be present so real code runs.
    assert "HASPATH" in out
