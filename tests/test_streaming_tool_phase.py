"""The pre-first-token tool phase is a core module of its own.

app/agents/core/streaming.py hit the 1500-line core-module limit that
tests/test_core_files_stay_small.py enforces, so the deterministic
predispatch phase moved, unchanged, to app/agents/core/streaming_tool_phase.
These pin that module on its own terms -- the budget's deterministic rank
cap and the phase's SSE stream -- so the move keeps behaviour and the module
is exercised directly rather than only through a whole streamed turn.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents import runtime
from app.agents.core import streaming_tool_phase as phase


class _Agent:
    name = "construction"


def _collect(coro) -> list[dict]:
    async def _drain():
        return [event async for event in coro]

    return asyncio.run(_drain())


def _drive(monkeypatch, file_probe):
    """Run the phase with every predispatch probe stubbed; only the file
    probe returns whatever the caller chose."""

    async def _file_probe(*a, **k):
        return file_probe

    async def _nothing(*a, **k):
        return None

    returns = {"_predispatch_file_tool": _file_probe}
    for name in (
        "_predispatch_wbs_duration_override",
        "_predispatch_resource_histogram",
        "_predispatch_look_ahead",
        "_predispatch_wir_form",
        "_predispatch_remaining_deliverables",
        "_predispatch_formula_calc",
    ):
        returns[name] = _nothing
    for name, fn in returns.items():
        monkeypatch.setattr(runtime, name, fn)
    results: list[dict] = []
    noted: list[str] = []
    state: dict = {}
    events = _collect(
        phase._run_tool_phase(
            _Agent(), [], "proj-1", "what does the spec say?", None,
            stream_tool_results=results,
            note_tool=lambda name: noted.append(name),
            state=state,
        )
    )
    return events, results, noted, state


def test_the_budget_is_inert_by_default(monkeypatch):
    monkeypatch.delenv("TOOL_PHASE_MAX_PROBES", raising=False)
    monkeypatch.delenv("TOOL_PHASE_TIME_BUDGET_MS", raising=False)
    budget = phase._make_tool_phase_budget(lambda: 0.0)
    assert isinstance(budget, phase._ToolPhaseBudget)
    assert all(budget.allow(key) for key in phase._TOOL_PHASE_PRIORITY)
    assert not budget.skipped


@pytest.mark.parametrize(
    "cap,expected",
    [(2, list(phase._TOOL_PHASE_PRIORITY[:2])), (0, [])],
)
def test_the_count_cap_keeps_the_highest_priority_probes(monkeypatch, cap, expected):
    monkeypatch.setenv("TOOL_PHASE_MAX_PROBES", str(cap))
    budget = phase._make_tool_phase_budget(lambda: 0.0)
    allowed = [key for key in phase._TOOL_PHASE_PRIORITY if budget.allow(key)]
    assert allowed == expected


@pytest.mark.parametrize("cap,expected", [(7, True), (0, False)])
def test_the_constructor_alone_establishes_the_cap(cap, expected):
    budget = phase._ToolPhaseBudget.__new__(phase._ToolPhaseBudget)
    budget.__init__(cap, 0.0, lambda: 0.0)
    assert budget.allow(phase._TOOL_PHASE_PRIORITY[0]) is expected


@pytest.mark.parametrize(
    "probe_result,expected_tools",
    [
        ({"name": "fetch_document", "result": {"filename": "spec.pdf"}},
         ["fetch_document"]),
        (None, []),
    ],
)
def test_the_phase_streams_each_probe_and_records_it(
    monkeypatch, probe_result, expected_tools
):
    events, results, noted, state = _drive(monkeypatch, probe_result)

    assert events[0]["type"] == "status"
    assert events[0]["label"] == phase._TOOL_PHASE_STATUS
    assert [event["tool"] for event in events if event["type"] == "tool_result"] == expected_tools
    assert noted == expected_tools
    assert state["probes"][0] == probe_result
    assert state["probes"][1:] == (None,) * 6
    if probe_result is None:
        assert results == []
    else:
        assert results[0] is probe_result
