"""
K-T0a: the hat router is always on. FORK_HATS_ENABLED is removed -- no
value of it (unset, 0, 1) turns the router off.
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


FLAG_VALUES = [None, "0", "1", "false"]


def _set_flag(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FORK_HATS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("FORK_HATS_ENABLED", value)


class TestHatsAlwaysOn:
    """The router answers whatever the (removed) flag says."""

    @pytest.mark.parametrize("value", FLAG_VALUES)
    def test_enabled_whatever_the_removed_flag_says(self, monkeypatch, value):
        _set_flag(monkeypatch, value)
        from app.agents.activation import HatActivationAdapter, hats_enabled

        assert hats_enabled() is True
        assert HatActivationAdapter().is_enabled() is True

    @pytest.mark.parametrize("value", FLAG_VALUES)
    @pytest.mark.parametrize("message", [
        "What is the critical path of the programme?",
        "Check the variation claim against the contract clause",
    ])
    def test_scores_every_hat_whatever_the_flag(self, monkeypatch, value, message):
        _set_flag(monkeypatch, value)
        from app.agents.activation import HatActivationAdapter
        from app.agents.catalog import list_hats

        scores = HatActivationAdapter().score_all_hats(message)
        assert scores, "router returned no scores: it is off"
        assert set(scores) <= {h.discipline.value for h in list_hats()}

    @pytest.mark.parametrize("rel", ["app/agents/activation.py", "app/agents/runtime.py"])
    def test_no_code_reads_the_removed_flag(self, rel):
        src = (Path(__file__).resolve().parents[1] / rel).read_text()
        assert 'environ.get("FORK_HATS_ENABLED"' not in src

    @pytest.mark.parametrize("flag", ["0", "1"])
    def test_catalog_loads_regardless_of_flag(self, monkeypatch, flag):
        """Catalog loads the same whatever the removed flag says."""
        monkeypatch.setenv("FORK_HATS_ENABLED", flag)
        from app.agents.catalog import get_agent_catalog, get_base, list_hats

        catalog = get_agent_catalog()
        assert len(catalog) > 0, "Catalog should load"

        base = get_base()
        assert base.id == "fork.base"

        hats = list_hats()
        assert len(hats) == 8, f"Expected 8 hats, found {len(hats)}"

    @pytest.mark.parametrize("flag", ["0", "1"])
    def test_manifests_validate_regardless_of_flag(self, monkeypatch, flag):
        """Manifests validate the same whatever the removed flag says."""
        monkeypatch.setenv("FORK_HATS_ENABLED", flag)
        from app.agents.catalog import get_agent_catalog
        catalog = get_agent_catalog()
        for mid, manifest in catalog.items():
            assert manifest.id == mid
            assert manifest.name
            assert manifest.discipline

