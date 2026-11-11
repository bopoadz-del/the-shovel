"""The startup guard fails closed on a missing JWT signing key.

A deploy must never regenerate the JWT signing secret: without a durable
``SECRET_KEY`` the fallback writes a fresh per-container key on every
force-new-deployment and signs every signed-in user out. The guard therefore
requires ``SECRET_KEY`` in every profile except the test profile
(``ENV=test``/``ENV=testing``), where the suite issues its own short-lived
tokens and never serves signed-in users across a deploy.
"""
from __future__ import annotations

import pytest

from app.main import _validate_startup_env


@pytest.mark.parametrize("profile", ["development", "production"])
def test_non_test_profiles_require_a_secret_key(monkeypatch, profile):
    """Two non-test profiles refuse to boot without SECRET_KEY."""
    monkeypatch.setenv("ENV", profile)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        _validate_startup_env()


@pytest.mark.parametrize("profile", ["test", "testing"])
def test_test_profile_boots_without_a_secret_key(monkeypatch, profile):
    """The test profile may run with an ephemeral signing secret."""
    monkeypatch.setenv("ENV", profile)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    _validate_startup_env()  # must not raise
