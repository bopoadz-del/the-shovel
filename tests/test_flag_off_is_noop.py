"""
Test that when FORK_HATS_ENABLED is OFF, the adapter is a no-op.

This is the critical safety test: existing runtime.py behavior must be
completely unchanged when the feature flag is not set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Imported through the PACKAGE path, not via sys.path.insert(app/agents).
# The old form loaded these as top-level `catalog` / `models` / `activation`
# while application code imports `app.agents.*` — two distinct module
# objects for the same file, so a pydantic class from one would fail an
# isinstance check against the other. It also required the modules to use
# implicit relative imports (`from models import ...`), which are Python 2
# syntax: `import app.agents.catalog` raised ModuleNotFoundError, so no
# application code could reach the discipline hats at all.


class TestFlagOffNoop:
    """Adapter must return None/empty when FORK_HATS_ENABLED is unset."""

    def test_env_flag_default_is_off(self):
        """Default (unset) should be disabled."""
        flag = os.environ.get("FORK_HATS_ENABLED", "")
        if flag:
            pytest.skip(f"FORK_HATS_ENABLED={flag} -- skipping no-op test")

        from app.agents.activation import HATS_ENABLED
        assert HATS_ENABLED is False, f"HATS_ENABLED should be False by default, got {HATS_ENABLED}"

    def test_adapter_returns_none_when_disabled(self):
        """select_hat_for_message must return None when disabled."""
        from app.agents.activation import HatActivationAdapter, HATS_ENABLED
        if HATS_ENABLED:
            pytest.skip("HATS_ENABLED is true -- skipping no-op test")

        adapter = HatActivationAdapter()
        result = adapter.select_hat_for_message("What is the critical path?")
        assert result is None, f"Expected None when disabled, got {result}"

    def test_score_all_hats_empty_when_disabled(self):
        """score_all_hats must return empty dict when disabled."""
        from app.agents.activation import HatActivationAdapter, HATS_ENABLED
        if HATS_ENABLED:
            pytest.skip("HATS_ENABLED is true -- skipping no-op test")

        adapter = HatActivationAdapter()
        scores = adapter.score_all_hats("Any message")
        assert scores == {}, f"Expected empty dict, got {scores}"

    def test_get_active_hat_names_empty_when_disabled(self):
        """get_active_hat_names must return empty list when disabled."""
        from app.agents.activation import HatActivationAdapter, HATS_ENABLED
        if HATS_ENABLED:
            pytest.skip("HATS_ENABLED is true -- skipping no-op test")

        adapter = HatActivationAdapter()
        names = adapter.get_active_hat_names("Any message")
        assert names == [], f"Expected empty list, got {names}"

    def test_module_level_select_hat_returns_none(self):
        """Module-level select_hat() returns None when disabled."""
        from app.agents.activation import select_hat, HATS_ENABLED
        if HATS_ENABLED:
            pytest.skip("HATS_ENABLED is true -- skipping no-op test")

        result = select_hat("What is the cost buildup?")
        assert result is None

    def test_module_level_quick_score_returns_empty(self):
        """Module-level quick_score() returns empty dict when disabled."""
        from app.agents.activation import quick_score, HATS_ENABLED
        if HATS_ENABLED:
            pytest.skip("HATS_ENABLED is true -- skipping no-op test")

        scores = quick_score("What is the cost buildup?")
        assert scores == {}

    def test_is_enabled_returns_false(self):
        """is_enabled() must return False when flag is off."""
        from app.agents.activation import HatActivationAdapter, HATS_ENABLED
        if HATS_ENABLED:
            pytest.skip("HATS_ENABLED is true -- skipping no-op test")

        adapter = HatActivationAdapter()
        assert adapter.is_enabled() is False

    def test_catalog_loads_regardless_of_flag(self):
        """Catalog should load even when hats are disabled -- it's import-time."""
        from app.agents.catalog import get_agent_catalog, get_base, list_hats

        catalog = get_agent_catalog()
        assert len(catalog) > 0, "Catalog should load"

        base = get_base()
        assert base.id == "fork.base"

        hats = list_hats()
        assert len(hats) == 8, f"Expected 8 hats, found {len(hats)}"

    def test_manifests_validate_regardless_of_flag(self):
        """Manifests should validate even when hats are disabled."""
        from app.agents.catalog import get_agent_catalog
        catalog = get_agent_catalog()
        for mid, manifest in catalog.items():
            assert manifest.id == mid
            assert manifest.name
            assert manifest.discipline


class TestFlagOnBehavior:
    """When FORK_HATS_ENABLED=1, adapter should be active."""

    def test_env_flag_on_enables_adapter(self, monkeypatch):
        """When env var is set, HATS_ENABLED should be True."""
        monkeypatch.setenv("FORK_HATS_ENABLED", "1")
        flag = os.environ.get("FORK_HATS_ENABLED", "").strip()
        enabled = flag in ("1", "true", "True", "TRUE", "yes")
        assert enabled is True

    def test_various_true_values(self, monkeypatch):
        """Various true-like values should enable hats."""
        true_values = ["1", "true", "True", "TRUE", "yes"]
        for val in true_values:
            monkeypatch.setenv("FORK_HATS_ENABLED", val)
            flag = os.environ.get("FORK_HATS_ENABLED", "").strip()
            enabled = flag in ("1", "true", "True", "TRUE", "yes")
            assert enabled is True, f"Value '{val}' should enable hats"

    def test_various_false_values(self, monkeypatch):
        """Various false-like values should not enable hats."""
        false_values = ["", "0", "false", "False", "FALSE", "no", "2"]
        for val in false_values:
            monkeypatch.setenv("FORK_HATS_ENABLED", val)
            flag = os.environ.get("FORK_HATS_ENABLED", "").strip()
            enabled = flag in ("1", "true", "True", "TRUE", "yes")
            assert enabled is False, f"Value '{val}' should not enable hats"
