"""PR #95 — .env.example contract test.

Locks in the set of security-critical environment variables that MUST be
declared (key + descriptive header) in .env.example. Operators provision
from this file; missing a critical var means they ship without it.

The list below is the set the production startup guard or the security
layer actively reads. New critical vars added to the codebase should be
added to BOTH the consuming code and this contract.
"""
from __future__ import annotations

import re
from pathlib import Path


REQUIRED_KEYS = [
    # Security / auth — production guard hard-fails on the first one
    "SECRET_KEY",
    "DATA_ENCRYPTION_KEY",
    "CEREBRUM_MASTER_KEY",
    # Persistence — without these, prod silently falls back to SQLite
    # and per-process state.
    "DATABASE_URL",
    "DATA_DIR",
    "REDIS_URL",
    # Ingestion routing. Not a tuning knob: set true without a live worker and
    # every upload lands on a queue nobody drains (200 + chunk_count 0
    # forever). An operator provisioning from .env.example must see it.
    "INGEST_WORKER_ENABLED",
    # First-boot admin bootstrap
    "BOOTSTRAP_USER_EMAIL",
    "BOOTSTRAP_USER_PASSWORD",
    # Observability — dark without it past the in-memory /v1/metrics
    "SENTRY_DSN",
    # Block-registry profile (prod must explicitly opt out of "virgin")
    "CEREBRUM_VIRGIN",
    "CEREBRUM_DOMAIN_KITS",
    # Runtime profile
    "ENV",
]


def _declared_keys(env_example_text: str) -> set:
    """Return the set of variables declared (uncommented) in the file."""
    keys = set()
    for line in env_example_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Z][A-Z0-9_]*)=", stripped)
        if m:
            keys.add(m.group(1))
    return keys


def test_env_example_declares_every_required_key():
    """Every security-critical env var must be uncommented in .env.example.

    A commented-out line (e.g. ``# DATABASE_URL=...``) does NOT count —
    operators who copy the file as-is must see the variable in the
    "fill this in" position, not hidden in a comment they might miss.
    """
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / ".env.example").read_text(encoding="utf-8")
    declared = _declared_keys(text)
    missing = [k for k in REQUIRED_KEYS if k not in declared]
    assert not missing, (
        f".env.example is missing {len(missing)} REQUIRED security-critical "
        f"variable(s): {missing}. Operators copying this file as their .env "
        f"will deploy without these set. Add them with a descriptive header "
        f"and an empty value (placeholder, not a real secret)."
    )


def test_env_example_uses_cors_extra_origins():
    """The app reads CORS_EXTRA_ORIGINS, not CORS_ORIGINS."""
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / ".env.example").read_text(encoding="utf-8")
    declared = _declared_keys(text)
    assert "CORS_EXTRA_ORIGINS" in declared
    assert "CORS_ORIGINS" not in declared


def test_env_example_declares_deepseek():
    """DeepSeek is a first-class cloud primary; operators must see the
    key + model when copying .env.example. The 2026-07-25 purge banned
    these names; the 2026-09-06 restore requires them again."""
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / ".env.example").read_text(encoding="utf-8")
    declared = _declared_keys(text)
    assert "DEEPSEEK_API_KEY" in declared
    assert "DEEPSEEK_MODEL" in declared
    assert "DEEPSEEK_API_KEY" in text
    assert "deepseek-chat" in text.lower()


def test_env_example_does_not_default_to_production():
    """ENV=production as the literal default is a footgun: copying the
    file and running it crashes the startup guard (missing SECRET_KEY).
    Default must be 'development' so the file is safely runnable as-is.
    Operators explicitly flip ENV=production on the host.
    """
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ENV="):
            value = stripped.split("=", 1)[1].strip().strip('"\'')
            assert value != "production", (
                f".env.example sets ENV=production as the default. "
                f"This crashes new installs unless SECRET_KEY is also set. "
                f"Default should be 'development' — production is set "
                f"explicitly on the deploy target."
            )
            return
    raise AssertionError("No ENV= line found in .env.example")


def test_env_example_does_not_use_tmp_for_durable_state():
    """EVIDENCE_VAULT_PATH and LEARNING_ENGINE_STORAGE used to default
    to /tmp paths. /tmp is wiped on container restart, silently erasing
    the evidence trail and learning state. Must default under DATA_DIR.
    """
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / ".env.example").read_text(encoding="utf-8")
    for var in ("EVIDENCE_VAULT_PATH", "LEARNING_ENGINE_STORAGE"):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{var}="):
                value = stripped.split("=", 1)[1].strip()
                assert "/tmp" not in value, (
                    f"{var} defaults to a /tmp path ({value!r}). /tmp is "
                    f"wiped on container restart — point it under DATA_DIR."
                )
                break
        else:
            raise AssertionError(f"{var} not found in .env.example")
