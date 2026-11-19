"""
K-T0a: the hat router is always on. The old on/off env flag is removed;
no product code reads it (guarded below), so the tests no longer set it.
"""

from __future__ import annotations

import re
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


REMOVED_FLAG = "FORK_HATS_ENABLED"
APP_DIR = Path(__file__).resolve().parents[1] / "app"


class TestHatsAlwaysOn:
    """The router is on with no flag of any kind."""

    def test_router_is_enabled(self):
        from app.agents.activation import HatActivationAdapter, hats_enabled

        assert hats_enabled() is True
        assert HatActivationAdapter().is_enabled() is True

    @pytest.mark.parametrize("message", [
        "What is the critical path of the programme?",
        "Check the variation claim against the contract clause",
    ])
    def test_scores_every_hat(self, message):
        from app.agents.activation import HatActivationAdapter
        from app.agents.catalog import list_hats

        scores = HatActivationAdapter().score_all_hats(message)
        assert scores, "router returned no scores: it is off"
        assert set(scores) <= {h.discipline.value for h in list_hats()}

    def test_no_code_reads_the_removed_flag(self):
        """No product module reads the removed flag from the environment."""
        read = re.compile(
            r"(?:getenv|environ\.get|environ\[)\s*\(?\s*[\"']"
            + re.escape(REMOVED_FLAG)
        )
        readers = sorted(
            str(p.relative_to(APP_DIR))
            for p in APP_DIR.rglob("*.py")
            if read.search(p.read_text(encoding="utf-8", errors="replace"))
        )
        assert not readers, f"the removed flag is read again in: {readers}"

    @pytest.mark.parametrize("discipline", ["planning", "safety"])
    def test_catalog_loads(self, discipline):
        """The catalog loads the base and the eight hats, this one among them."""
        from app.agents.catalog import get_agent_catalog, get_base, list_hats

        catalog = get_agent_catalog()
        assert len(catalog) > 0, "Catalog should load"

        base = get_base()
        assert base.id == "fork.base"

        hats = list_hats()
        assert len(hats) == 8, f"Expected 8 hats, found {len(hats)}"
        assert discipline in {h.discipline.value for h in hats}

    @pytest.mark.parametrize("manifest_id", ["fork.base", "fork.hat.contracts"])
    def test_manifests_validate(self, manifest_id):
        """Every manifest validates under its own id, this one among them."""
        from app.agents.catalog import get_agent_catalog
        catalog = get_agent_catalog()
        assert manifest_id in catalog
        for mid, manifest in catalog.items():
            assert manifest.id == mid
            assert manifest.name
            assert manifest.discipline

