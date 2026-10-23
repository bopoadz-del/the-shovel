"""Every tool an agent advertises must be one the runtime can dispatch.

Audit 2026-08-12, agents subsystem. The direct analogue of
tests/test_construction_actions_reachable.py, which found three container
actions the router could not reach.

An agent's `tool_definitions()` is what the LLM is shown and what it will
therefore try to call. `Agent._run_tool_call()` is what actually executes.
Nothing tied the two together, so a tool could be advertised and not
dispatchable -- the agent would emit a call the runtime rejects, and the model
would either loop or apologise to the user. That is the same failure the
container audit found, where `_suggest_next_action` recommended
`generate_construction_report` and `route()` answered "Unknown action".

Dispatchable names are read from the DISPATCHER, not from a hand-kept list.
An earlier pass of this audit checked advertised tools against an assumed set
of three synthetic names taken from a docstring and reported all 14 agents as
broken; `list_project_documents`, `fetch_document` and `construction_calc` are
all handled in `_run_tool_call` (runtime.py:4512, 4555, 4712) and the docstring
was simply incomplete. Deriving from the code is what makes the check true.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RUNTIME = Path("app/agents/runtime.py")


def _synthetic_tool_names() -> set[str]:
    """Names `_run_tool_call` handles itself, parsed from its comparisons.

    Picks up `if name == "x"`, `name in ("x", "y")` and equivalents, so a new
    synthetic tool counts the moment it is dispatched rather than when someone
    remembers to list it here.
    """
    from tests import agent_source

    target = None
    for path in agent_source.files():  # runtime.py and app/agents/core/
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "_run_tool_call":
                target = node
                break
        if target is not None:
            break
    assert target is not None, "_run_tool_call not found -- has the runtime been restructured?"

    names: set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "name":
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                    if isinstance(comparator.value, str):
                        names.add(comparator.value)
                elif isinstance(op, ast.In) and isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                    names |= {e.value for e in comparator.elts
                              if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return names


def _container_route_keys() -> set[str]:
    src = Path("app/containers/construction/__init__.py").read_text(encoding="utf-8")
    start = src.index("handlers = {")
    block = src[start:src.index("\n        }", start)]
    return set(re.findall(r'"([a-z0-9_]+)":\s*self\.', block))


def _registered_tool_names() -> set[str]:
    """Tools owned by a package (app.agents.base / app.agents.hats.<hat>) and
    dispatched through app.agents.core.tool_registry (F-DRIVER Phase A)."""
    from app.agents.core import tool_registry

    tool_registry.load()
    return set(tool_registry._TOOLS)


def _dispatchable() -> set[str]:
    from app.blocks import BLOCK_REGISTRY

    return (set(BLOCK_REGISTRY) | _container_route_keys() | _synthetic_tool_names()
            | _registered_tool_names())


def test_the_synthetic_tool_set_was_actually_parsed():
    """Guards the parser. If it matched nothing, the real test below would
    pass by declaring everything dispatchable -- the failure mode where a
    check gets weaker exactly when it stops working."""
    synthetic = _synthetic_tool_names()
    assert len(synthetic) >= 3, (
        f"only {sorted(synthetic)} parsed out of _run_tool_call -- the "
        "dispatch shape has changed and this file is no longer checking anything"
    )


def test_every_advertised_tool_is_dispatchable():
    """The whole point: what the model is shown, the runtime must accept."""
    from app.agents.runtime import get_agent, load_agents

    agents = load_agents()
    assert agents, "no agents loaded"

    dispatchable = _dispatchable()
    undispatchable: dict[str, list[str]] = {}
    for agent_id in sorted(agents):
        agent = get_agent(agent_id)
        advertised = [
            (tool.get("function") or {}).get("name") or tool.get("name")
            for tool in agent.tool_definitions(project_id="fence-check")
        ]
        assert advertised, f"{agent_id} advertises no tools at all"
        missing = sorted({n for n in advertised if n and n not in dispatchable})
        if missing:
            undispatchable[agent_id] = missing

    assert not undispatchable, (
        "these agents advertise tools the runtime cannot dispatch, so the model "
        "will emit calls that are rejected:\n"
        + "\n".join(f"  {a}: {', '.join(t)}" for a, t in sorted(undispatchable.items()))
    )


def test_no_agent_advertises_a_duplicate_tool():
    """Two tools with one name is ambiguous to the model and to the dispatcher.

    Cheap to check, and it cannot be seen by reading any single manifest --
    duplicates arise when a block name collides with a synthetic tool.
    """
    from app.agents.runtime import get_agent, load_agents

    for agent_id in sorted(load_agents()):
        names = [
            (tool.get("function") or {}).get("name") or tool.get("name")
            for tool in get_agent(agent_id).tool_definitions(project_id="fence-check")
        ]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        assert not duplicates, f"{agent_id} advertises {duplicates} more than once"


@pytest.mark.parametrize("project_id", [None, "fence-check"])
def test_project_scoped_tools_appear_only_with_a_project(project_id):
    """`search_project_documents` is documented as project-only.

    Advertising a document search with no project to search would invite the
    model to call it and get nothing back, which reads to a user as "the
    platform cannot find my documents".
    """
    from app.agents.runtime import get_agent, load_agents

    for agent_id in sorted(load_agents()):
        names = {
            (tool.get("function") or {}).get("name") or tool.get("name")
            for tool in get_agent(agent_id).tool_definitions(project_id=project_id)
        }
        if "search_project_documents" in names:
            assert project_id is not None, (
                f"{agent_id} advertises search_project_documents with no project_id"
            )
