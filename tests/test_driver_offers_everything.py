"""U-3: in driver mode a hat changes the ORDER of the tools offered, never
which tools are offered (owner rule 1: a hat is focus, never access).

The package tools are offered directly here (the workflow catalogue is
emptied by the test), so every package tool is visible in the offer.
"""
from __future__ import annotations

import pytest

from app.agents.core import tool_registry
from app.agents.driver import routes, tools
from tests.test_driver_mode import _agent


def _package_tools() -> set:
    return {n for owner in tool_registry.OWNERS for n in tool_registry.tools_of(owner)}


def _hat_tools() -> set:
    """Every tool a hat package owns (base tools are offered as before; one,
    delegation, depends on the agent, not on a hat)."""
    return {n for owner in tool_registry.OWNERS if owner != "base"
            for n in tool_registry.tools_of(owner)}


def _hats_with_tools() -> list:
    return [o for o in tool_registry.OWNERS if o != "base" and tool_registry.tools_of(o)]


def _offer(monkeypatch, hat):
    monkeypatch.setattr(routes, "catalogue", lambda: [])
    disciplines = [o for o in tool_registry.OWNERS if o != "base"]
    names = [t["function"]["name"] for t in tools.offered(_agent(), "synthetic-proj", hat, disciplines)]
    assert names[0] == tools.SELECT_HAT
    return names[1:]


def test_at_least_two_hats_own_tools():
    assert len(_hats_with_tools()) >= 2


@pytest.mark.parametrize("pick", [0, -1])
def test_every_package_tool_is_offered_whichever_hat(monkeypatch, pick):
    hat = _hats_with_tools()[pick]
    offered = _offer(monkeypatch, hat)
    assert _hat_tools() <= set(offered)
    assert len(offered) == len(set(offered)), "a tool is offered twice"


def test_before_any_hat_every_package_tool_is_offered(monkeypatch):
    assert _hat_tools() <= set(_offer(monkeypatch, None))


@pytest.mark.parametrize("pick", [0, -1])
def test_the_worn_hat_changes_the_order_not_the_set(monkeypatch, pick):
    hat = _hats_with_tools()[pick]
    worn = _offer(monkeypatch, hat)
    bare = _offer(monkeypatch, None)
    assert set(worn) == set(bare)
    own = set(tool_registry.tools_of(hat))
    assert worn[:len(own)] and set(worn[:len(own)]) == own


@pytest.mark.parametrize("pick", [0, -1])
def test_base_comes_before_the_other_hats(monkeypatch, pick):
    hat = _hats_with_tools()[pick]
    worn = _offer(monkeypatch, hat)
    owner = {n: tool_registry.get(n).owner for n in worn if tool_registry.get(n)}
    ranks = [0 if owner.get(n) == hat else 1 if owner.get(n) == "base" else 2
             for n in worn if n in owner]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize("hat", [None, "a-hat-with-no-tools"])
def test_lead_first_adds_and_removes_nothing(hat):
    given = [{"function": {"name": n}} for n in sorted(_package_tools())]
    out = tools.lead_first(given, hat)
    assert sorted(t["function"]["name"] for t in out) == sorted(_package_tools())


@pytest.mark.parametrize("pick", [0, -1])
def test_lead_rank_puts_the_worn_hat_first_then_base_then_the_rest(pick):
    hat = _hats_with_tools()[pick]
    other = next(h for h in _hats_with_tools() if h != hat)
    own_tool = tool_registry.tools_of(hat)[0]
    base_tool = tool_registry.tools_of("base")[0]
    other_tool = tool_registry.tools_of(other)[0]
    assert tools.lead_rank(own_tool, hat) == 0
    assert tools.lead_rank(base_tool, hat) == 1
    assert tools.lead_rank(other_tool, hat) == 2
    assert tools.lead_rank(own_tool, None) == 2  # no hat worn: no lead


@pytest.mark.parametrize("pick", [0, -1])
def test_a_hat_tool_that_is_a_workflow_has_one_door(monkeypatch, pick):
    """Every hat tool is now offered directly; one that is also a catalogue
    workflow is offered through run_workflow only, never both."""
    hat = _hats_with_tools()[pick]
    shared = tool_registry.tools_of(hat)[0]
    monkeypatch.setattr(routes, "catalogue", lambda: [shared, "widget_register"])
    monkeypatch.setattr(routes, "cards", lambda: {})
    disciplines = [o for o in tool_registry.OWNERS if o != "base"]
    names = [t["function"]["name"]
             for t in tools.offered(_agent(), "synthetic-proj", None, disciplines)]
    assert routes.RUN_WORKFLOW in names
    assert shared not in names
    others = _hat_tools() - {shared}
    assert others <= set(names)
