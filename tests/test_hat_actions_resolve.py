"""Every action a hat manifest allows resolves to machinery that runs it.

A manifest action is written in one of three forms, and each form resolves
against the live registry that answers it -- never against a hand-kept list:

  * ``construction_calc:<name>`` -- <name> is a declared calculation
    (``formula_registry.calculators()``, what construction_calc runs);
  * ``construction:<route>`` -- <route> is a key of the construction
    container's route table and names a method the container has;
  * a plain name -- a tool registered in ``app.agents.core.tool_registry``.

Anything else is a dead action: a name the model can be told about and that
nothing answers. Plain names that only an alias table in the dispatcher
knew were exactly that (no tool is offered under them); they are written as
the reference they always meant, so the count of dead actions is zero by
construction and a new one fails here.

A hat still only orders and focuses; nothing here restricts a tool.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

MANIFEST_DIR = Path("app/agents/manifests")
CONTAINER_SRC = Path("app/containers/construction/__init__.py")


def _container_routes() -> set[str]:
    """Keys of ConstructionContainer.route's ``handlers`` table whose value is
    ``self.<method>`` and whose method the class really has."""
    from app.containers.construction import ConstructionContainer

    tree = ast.parse(CONTAINER_SRC.read_text(encoding="utf-8"))
    routes: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "handlers" for t in node.targets):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name) and value.value.id == "self"
                    and callable(getattr(ConstructionContainer, value.attr, None))):
                routes.add(key.value)
    return routes


def _registries():
    from app.agents.core import tool_registry
    from app.lib import formula_registry

    return set(formula_registry.calculators()), _container_routes(), tool_registry.get


def unresolved(actions, calcs, routes, tool_get) -> list[str]:
    """The actions in ``actions`` that resolve to nothing."""
    dead = []
    for action in actions:
        kind, sep, target = action.partition(":")
        if sep and kind == "construction_calc":
            ok = target in calcs
        elif sep and kind == "construction":
            ok = target in routes
        elif not sep:
            ok = tool_get(action) is not None
        else:
            ok = False
        if not ok:
            dead.append(action)
    return dead


def _manifests() -> dict[str, list[str]]:
    out = {}
    for path in sorted(MANIFEST_DIR.glob("fork.*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[path.name] = list(data.get("allowed_actions") or [])
    return out


def test_the_walk_reads_every_manifest_and_every_form():
    manifests = _manifests()
    assert len(manifests) >= 6, sorted(manifests)
    every = [a for acts in manifests.values() for a in acts]
    assert len(every) >= 25, every
    kinds = {a.partition(":")[0] if ":" in a else "plain" for a in every}
    assert {"construction_calc", "construction", "plain"} <= kinds, kinds


def test_the_registries_the_walk_reads_are_populated():
    calcs, routes, tool_get = _registries()
    assert len(calcs) >= 50 and len(routes) >= 20
    assert tool_get("construction_calc") is not None


def test_every_allowed_action_in_every_manifest_resolves():
    calcs, routes, tool_get = _registries()
    dead = {name: unresolved(acts, calcs, routes, tool_get)
            for name, acts in _manifests().items()}
    dead = {k: v for k, v in dead.items() if v}
    assert not dead, f"allowed actions that resolve to nothing: {dead}"


def test_no_manifest_declares_an_action_twice():
    for name, acts in _manifests().items():
        assert len(acts) == len(set(acts)), (name, acts)


# -- the resolver itself, on invented inputs --------------------------------

_CALCS = {"alpha_volume", "beta_float"}
_ROUTES = {"gamma_route", "delta_route"}
_TOOLS = {"epsilon_tool", "zeta_tool"}


def _tool_get(name):
    return object() if name in _TOOLS else None


@pytest.mark.parametrize("action", [
    "construction_calc:alpha_volume", "construction_calc:beta_float",
    "construction:gamma_route", "construction:delta_route",
    "epsilon_tool", "zeta_tool",
])
def test_a_reference_to_registered_machinery_resolves(action):
    assert unresolved([action], _CALCS, _ROUTES, _tool_get) == []


@pytest.mark.parametrize("action", [
    "construction_calc:omega_missing", "construction_calc:gamma_route",
    "construction:omega_route", "construction:alpha_volume",
    "omega_planner", "alpha_volume",
    "block:epsilon_tool", "other:zeta_tool",
])
def test_a_name_nothing_answers_is_dead(action):
    assert unresolved([action], _CALCS, _ROUTES, _tool_get) == [action]


def test_the_dead_are_reported_in_order_and_the_live_dropped():
    acts = ["epsilon_tool", "omega_one", "construction_calc:beta_float",
            "construction:omega_two", "construction:delta_route"]
    assert unresolved(acts, _CALCS, _ROUTES, _tool_get) == ["omega_one", "construction:omega_two"]
