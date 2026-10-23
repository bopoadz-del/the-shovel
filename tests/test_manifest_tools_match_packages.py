"""Each hat's manifest lists exactly the tools its package registers, and
every owner with registered tools or formulas has a manifest."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from app.agents.core import tool_registry
from app.lib import formula_registry

MANIFESTS = Path(__file__).resolve().parents[1] / "app" / "agents" / "manifests"


def _manifests():
    return {m["discipline"]: m for m in
            (json.loads(p.read_text(encoding="utf-8")) for p in sorted(MANIFESTS.glob("*.json")))}


def test_available_tools_equal_the_package_registry():
    for discipline, m in _manifests().items():
        assert sorted(m.get("available_tools", [])) == tool_registry.tools_of(discipline), discipline


def test_every_owner_has_a_manifest():
    manifests = _manifests()
    tool_owners = {o for o in tool_registry.OWNERS if tool_registry.tools_of(o)}
    formula_owners = {s.owner for s in formula_registry.all_specs()}
    assert (tool_owners | formula_owners) - set(manifests) == set()


def test_every_manifest_discipline_has_a_package():
    for discipline in _manifests():
        pkg = "app.agents.base" if discipline == "base" else f"app.agents.hats.{discipline}"
        assert importlib.util.find_spec(pkg) is not None, discipline


def test_the_manifests_load():
    from app.agents import catalog

    hats = {h.discipline.value for h in catalog.list_hats()}
    assert "design" in hats
