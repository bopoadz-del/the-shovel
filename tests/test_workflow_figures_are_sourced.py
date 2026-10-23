"""A figure a platform workflow produced is sourced by that workflow's run:
the exit check keeps it (lane 1: EVM CV / EAC and cash-flow month rows were
drafted and then removed). Synthetic workflow, scripted model."""
from __future__ import annotations

from tests.lane5_core_support import answer_text, call, drive, install_driver_stream


def test_a_workflows_own_figures_survive_the_exit_check(monkeypatch):
    from app.agents.driver import routes
    from app.core import predefined_reasoning

    install_driver_stream(monkeypatch)
    monkeypatch.setattr(routes, "catalogue", lambda: ["widget_cost_review"])
    monkeypatch.setattr(routes, "cards", lambda: {})

    async def fake_run_workflow(action, ctx, session):
        return {"handled": True, "status": "success",
                "answer": "Cost variance AED -5,000; estimate at completion AED 211,111.11 (CPI 0.947); month 3 AED 48,200."}

    monkeypatch.setattr(predefined_reasoning, "run_workflow", fake_run_workflow)
    events, _ = drive(monkeypatch, "Review the widget costs", [
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_cost_review"}, "w1")]},
        {"content": "The cost variance is AED -5,000, the estimate at completion is AED 211,111.11 and month 3 is AED 48,200."},
    ])
    text = answer_text(events)
    assert "211,111.11" in text and "48,200" in text
    end = events[-1]
    assert not end["exit_check"]["figures_removed"]


def test_a_payment_certificate_takes_a_stated_zero_retention():
    """Lane 1: retention 0 given to the workflow was replaced by 10%."""
    import asyncio

    import pytest

    from tests.conftest import is_construction_kit_enabled
    if not is_construction_kit_enabled():
        pytest.skip("requires the construction kit")
    from app.dependencies import get_block_instance

    con = get_block_instance("construction")
    envelope = {"contract_value": 250000, "work_done_percent": 100, "retention_percent": 0}
    out = asyncio.run(con.payment_certificate(envelope, {}))
    assert out.get("status") == "success", out
    assert out["deductions"]["retention_held"] == 0  # no 10% default taken
    assert out["payment"]["net_due_this_period"] == 250000
