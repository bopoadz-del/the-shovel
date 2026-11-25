"""The model's run_workflow tool does not hand back an unbuilt workflow as a deliverable.

``run_workflow`` reports ``built`` False when a registry plan ran and staged
nothing its renderer reads (the schedule template found no scope it could
structure). Both chat doors already hand such a turn to the agent. The model's
own ``run_workflow`` tool (driver mode) still returned ``ok: True`` with the
renderer's "could not build" line as the answer, so the model read a dead end
as a finished deliverable.

Now the tool says ``ok: False`` with the failing steps in the engine's own
words and no answer; a workflow that built something is returned as before.
Every input is invented for this file; two inputs per decision, each with its
other side.
"""
from __future__ import annotations

import asyncio

import pytest

import app.agents.driver.routes as driver_routes
from tests import _wbs_fakes

# The user's own small network or own quantities + rates: nothing the
# template's structurer is given to build from (it is stubbed to find nothing).
OWN_FIGURE_ASKS = (
    "Footings 6 days, then blockwork 11 days and the roof 7 days in parallel, "
    "then finishes 5 days. Critical path and the float on the roof?",
    "Piling 900 m at 45 m per day, then pile caps 60 m3 at 12 m3 per day, one "
    "crew each, in that order. Durations and the programme, please.",
)
# Briefs the template's structurer turns into activities (input-derived fake).
BRIEF_ASKS = (
    "Produce the schedule for a riverside pumping station with intake, wet well and control room.",
    "Build a baseline programme for a two-storey clinic with a pharmacy wing.",
)


def _template_finds_nothing(monkeypatch):
    """The template's brief structurer returns no activities."""
    from app.core import llm_client

    async def nothing(system, user, **kwargs):
        return {"activities": []}
    monkeypatch.setattr(llm_client, "complete_json", nothing)


def _tool(action: str, text: str):
    return asyncio.run(driver_routes.run({"workflow": action}, text, None, None, None))


@pytest.mark.parametrize("action", ["generate_wbs", "resource_histogram"])
@pytest.mark.parametrize("text", OWN_FIGURE_ASKS)
def test_an_unbuilt_workflow_is_not_ok(text, action, monkeypatch):
    _template_finds_nothing(monkeypatch)
    monkeypatch.setattr(driver_routes, "catalogue", lambda: [action])
    out = _tool(action, text)
    assert out["ok"] is False, out
    assert "answer" not in out                      # the dead-end line is not a deliverable
    assert "no schedulable scope" in out["not_built"]  # the engine's own words
    assert "calculators" in out["next"]
    # Not a missing input: the driver must not re-ask or re-run for one.
    assert "missing" not in out and "needs_input" not in out


@pytest.mark.parametrize("text", BRIEF_ASKS)
def test_a_built_schedule_is_still_ok(text, monkeypatch):
    _wbs_fakes.install(monkeypatch)
    monkeypatch.setattr(driver_routes, "catalogue", lambda: ["generate_wbs"])
    out = _tool("generate_wbs", text)
    assert out["ok"] is True and out["answer"].startswith("Schedule built:"), out
    assert "not_built" not in out


@pytest.mark.parametrize("text", BRIEF_ASKS)
def test_a_built_histogram_is_still_ok(text, monkeypatch):
    _wbs_fakes.install(monkeypatch, min_activities=6)
    monkeypatch.setattr(driver_routes, "catalogue", lambda: ["resource_histogram"])
    out = _tool("resource_histogram", text)
    assert out["ok"] is True and out["answer"].startswith("Manpower histogram"), out
    assert "not_built" not in out


@pytest.mark.parametrize("named, offered", [
    ("generate_wbs", ["resource_histogram"]),
    ("resource_histogram", ["generate_wbs"]),
])
def test_a_workflow_outside_the_catalogue_is_not_run(named, offered, monkeypatch):
    """The tool runs only a workflow the catalogue offers; any other name is
    refused before the engine is reached, and the offered one still runs."""
    from app.core import predefined_reasoning

    ran = []

    async def engine(action, context, session):
        ran.append(action)
        return {"handled": True, "built": True, "answer": "done"}
    monkeypatch.setattr(predefined_reasoning, "run_workflow", engine)
    monkeypatch.setattr(driver_routes, "catalogue", lambda: offered)
    out = _tool(named, OWN_FIGURE_ASKS[0])
    assert out == {"ok": False, "error": f"No workflow named {named!r}."}
    assert ran == []
    # The offered workflow is reached through the same door.
    assert _tool(offered[0], OWN_FIGURE_ASKS[0])["ok"] is True
    assert ran == offered


def _engine_returns(monkeypatch, action, result):
    from app.core import predefined_reasoning

    async def engine(_action, context, session):
        return dict(result)
    monkeypatch.setattr(predefined_reasoning, "run_workflow", engine)
    monkeypatch.setattr(driver_routes, "catalogue", lambda: [action])


@pytest.mark.parametrize("action, answer", [
    ("payment_certificate", "Certificate 4: net due 18,250.00 after retention."),
    ("cost_estimate", "Estimate: 3 items, total 96,400.00."),
])
def test_a_result_that_carries_no_built_flag_is_still_ok(action, answer, monkeypatch):
    """A container action's result has no ``built`` key at all. Only an
    explicit ``built`` False is a dead end; an absent flag is a deliverable."""
    _engine_returns(monkeypatch, action, {"handled": True, "status": "success",
                                          "answer": answer, "exports": []})
    out = _tool(action, OWN_FIGURE_ASKS[1])
    assert out["ok"] is True and out["answer"] == answer, out
    assert "not_built" not in out


@pytest.mark.parametrize("action, refusal", [
    ("payment_certificate", "The certificate needs the contract value."),
    ("cost_estimate", "The estimate needs at least one rate."),
])
def test_a_refusal_that_carries_no_built_flag_asks_for_the_input(action, refusal, monkeypatch):
    """A refusal without a ``built`` flag stays a missing-input turn, not a dead end."""
    _engine_returns(monkeypatch, action, {"handled": True, "status": "error",
                                          "answer": refusal, "exports": []})
    out = _tool(action, OWN_FIGURE_ASKS[0])
    assert out["ok"] is False and out["missing"] == refusal, out
    assert "not_built" not in out


@pytest.mark.parametrize("reasons, expected", [
    ([], "the workflow staged nothing to deliver"),
    (["structure: no activities", "render: empty tree"],
     "structure: no activities; render: empty tree"),
])
def test_an_unbuilt_workflow_always_says_why(reasons, expected, monkeypatch):
    """Every failing step is named; with none recorded the tool still says
    that nothing was staged, never an empty reason."""
    _engine_returns(monkeypatch, "generate_wbs", {"handled": True, "built": False,
                                                  "unbuilt_reasons": reasons,
                                                  "status": "success", "answer": "x"})
    out = _tool("generate_wbs", OWN_FIGURE_ASKS[0])
    assert out["ok"] is False and out["not_built"] == expected, out
