"""The workflow catalogue the driver's model reads: every entry has a card of
one shape (what it produces, when to use it, what it needs, its inputs with
display labels), no two entries read alike, an alias of a workflow is not a
second entry, and a workflow that refuses for a missing input says so instead
of reporting success. Synthetic cards and workflows where the real ones are
not needed."""
from __future__ import annotations

import asyncio

import pytest

from app.agents.driver import routes
from tests.conftest import construction_kit_markers

_GOOD = {"produces": "A widget register for the stated works",
         "use_when": "The user asks to list the widgets fitted so far",
         "needs": "The widget schedule or a list of widgets",
         "inputs": {"widgets": "Widgets"}}


def test_every_card_on_file_has_the_required_shape():
    declared = routes.cards()
    assert routes.card_gaps(sorted(declared), declared) == []


def test_no_two_cards_on_file_read_alike():
    assert routes.card_overlaps(routes.cards()) == []


@pytest.mark.parametrize("card,gap", [
    ({**_GOOD, "produces": ""}, "produces missing"),
    ({**_GOOD, "use_when": "Widgets"}, "use_when missing"),
    ({k: v for k, v in _GOOD.items() if k != "needs"}, "needs missing"),
    ({**_GOOD, "inputs": {}}, "no inputs declared"),
    ({**_GOOD, "inputs": {"widgets": ""}}, "has no display label"),
    ({**_GOOD, "inputs": {"widgets": "widgets"}}, "has no display label"),
])
def test_the_shape_gate_flags_a_thin_card(card, gap):
    gaps = routes.card_gaps(["widget_register"], {"widget_register": card})
    assert any(gap in g for g in gaps), gaps


def test_the_shape_gate_flags_a_missing_and_a_stale_card():
    gaps = routes.card_gaps(["widget_register"], {"gadget_register": _GOOD})
    assert any(g.startswith("widget_register:") for g in gaps)
    assert any(g.startswith("gadget_register:") and "not in the catalogue" in g for g in gaps)


def test_the_overlap_gate_flags_two_cards_that_read_alike():
    twin = {**_GOOD, "produces": "A widget register of the stated works"}
    other = {**_GOOD, "produces": "A cost report for the gadget order",
             "use_when": "The user asks what the gadget order will cost"}
    flagged = routes.card_overlaps({"widget_register": _GOOD, "widget_log": twin, "gadget_cost": other})
    assert len(flagged) == 1 and flagged[0].startswith("widget_log / widget_register")


def test_a_catalogue_entry_carries_its_card(monkeypatch):
    monkeypatch.setattr(routes, "cards", lambda: {"widget_register": _GOOD})
    line = routes._entry("widget_register")
    assert line == ("- widget_register: produces a widget register for the stated works. "
                    "Use when the user asks to list the widgets fitted so far. "
                    "Needs the widget schedule or a list of widgets. Inputs: Widgets (widgets).")


@pytest.mark.parametrize("status,ok", [("error", False), ("success", True), (None, True)])
def test_a_workflow_that_refuses_is_not_reported_as_done(monkeypatch, status, ok):
    from app.core import predefined_reasoning

    monkeypatch.setattr(routes, "catalogue", lambda: ["widget_register"])
    monkeypatch.setattr(routes, "cards", lambda: {"widget_register": _GOOD})

    async def fake_run_workflow(action, ctx, session):
        return {"handled": True, "status": status,
                "answer": "No widget list supplied: cannot build a widget register."}

    monkeypatch.setattr(predefined_reasoning, "run_workflow", fake_run_workflow)
    out = asyncio.run(routes.run({"workflow": "widget_register"}, "list the widgets", None, None, None))
    assert out["ok"] is ok
    if not ok:
        assert out["missing"] == "No widget list supplied: cannot build a widget register."
        assert "documents" in out["next"] and "params" in out["next"]
        # The refusal names the workflow's own inputs, so the model can gather
        # them and call again instead of writing the deliverable itself.
        assert out["inputs"] == {"widgets": "Widgets"} and out["needs"]
        assert "yourself" in out["next"]


@construction_kit_markers[0]
@construction_kit_markers[1]
def test_every_workflow_in_the_catalogue_has_a_card():
    workflows = routes.catalogue()
    assert workflows, "the construction container did not load: no workflows"
    assert routes.card_gaps(workflows, routes.cards()) == []


@construction_kit_markers[0]
@construction_kit_markers[1]
def test_an_action_name_bound_to_another_workflows_method_is_not_a_second_entry():
    from app.dependencies import get_block_instance

    impl = get_block_instance("construction").get_actions()
    methods = [getattr(impl[w], "__func__", impl[w]) for w in routes.catalogue() if w in impl]
    assert len(methods) == len(set(methods))


def test_a_workflow_has_one_door(monkeypatch):
    """A package tool named like a catalogue workflow is the same workflow by a
    second door: the model is offered run_workflow for it, never both."""
    from app.agents.core import tool_registry
    from app.agents.driver import tools

    shared = sorted(set(tool_registry._TOOLS))[:1]
    assert shared, "no package tools registered"
    monkeypatch.setattr(routes, "catalogue", lambda: shared + ["widget_register"])
    monkeypatch.setattr(routes, "cards", lambda: {})
    from tests.test_driver_mode import _agent

    hat = tool_registry._TOOLS[shared[0]].owner
    offered = [t["function"]["name"] for t in tools.offered(_agent(), None, hat, [hat])]
    assert routes.RUN_WORKFLOW in offered
    assert shared[0] not in offered
