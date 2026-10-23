"""Addressing a specialist agent by name must keep the turn with that agent.

Live-fire find 2026-08-15 (F24). A resource-costing request sent to the
quantity-surveyor agent -- whose text happened to contain the word
"manpower" -- was answered in three seconds by the canned manpower-histogram
deliverable flow, three separate sessions in a row, with the costing never
running. Mechanism: the agents router runs keyword routing on every message
and, when the routed action is a predefined workflow, the predefined path
replaces the WHOLE turn regardless of which agent the caller addressed.

The agent-picker contract is that choosing a specialist is deliberate.
Predefined interception is now allowed only for the generalist entry points
(heavy-reasoning, project-assistant, smart-orchestrator); a specialist turn
always reaches the specialist's own tool loop.
"""
from __future__ import annotations

from app.routers.agents import _PREDEFINED_GENERALISTS, _predefined_may_intercept


def test_specialists_are_never_intercepted():
    for name in (
        "quantity-surveyor", "safety-officer", "contracts-manager",
        "construction-pm", "bim-analyst", "document-analyst",
        "validation", "self-coding", "supervision-proposal", "learning",
    ):
        assert _predefined_may_intercept(name) is False, name


def test_generalist_entry_points_may_still_use_predefined_flows():
    for name in ("heavy-reasoning", "project-assistant", "smart-orchestrator"):
        assert _predefined_may_intercept(name) is True, name


def test_the_generalist_set_is_the_single_source():
    assert _PREDEFINED_GENERALISTS == {
        "heavy-reasoning", "project-assistant", "smart-orchestrator",
    }


def test_the_gate_is_wired_into_the_dispatch_condition():
    """The router's _use_predefined must consult the helper -- a refactor
    that drops the call silently reopens the hijack."""
    import inspect

    import app.routers.agents as m
    src = inspect.getsource(m)
    idx = src.find("_use_predefined = bool(")
    assert idx != -1
    assert "_predefined_may_intercept(requested_agent_name)" in src[idx:idx + 900]
