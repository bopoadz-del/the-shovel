"""Item 7 (ORDERS_31OCT_0405Z ruling): one status for an input the asked output
needs but did not get. A required numeric that is provided-but-unparseable OR
blank OR omitted-when-needed returns status "needs_input" naming the field --
never "error" with a flag riding alongside. Omitted when the output does not
need it (daily-only) returns success. General coercion rule, no per-value branch.
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas as cf
from app.agents.runtime import _tool_result_errored


# ── reason split: an unreadable value ASKS, a wrong (in-range-failing) value
#    ERRORS. Pins _input_rejection_envelope directly (the ruling's two mutants:
#    "value-domain routed to needs_input" RED, "blank routed to error" RED). ──
@pytest.mark.parametrize("reason,status", [
    ("empty", "needs_input"),        # blank -> ask (mutant "blank->error" RED)
    ("not_a_number", "needs_input"),  # garbled -> ask
    ("not_whole", "needs_input"),     # fraction where a whole is wanted -> ask
    ("out_of_range", "error"),        # a wrong value -> error (mutant "->needs_input" RED)
])
def test_rejection_reason_decides_status(reason, status):
    env = cf._input_rejection_envelope(
        "delay_damages_daily",
        [{"parameter": "days", "label": "days", "reason": reason,
          "lo": 1, "hi": 365, "value": "x"}],
        [],
    )
    assert env["status"] == status, env
    assert env.get("needs_input") is (True if status == "needs_input" else None)


def test_missing_required_always_asks():
    env = cf._input_rejection_envelope("delay_damages_daily", [], ["days"])
    assert env["status"] == "needs_input"
    assert env.get("needs_input") is True


# ── the ask cases: unparseable / blank both -> needs_input naming the field ──
@pytest.mark.parametrize("days", ["days3", ""])
def test_unusable_required_numeric_asks(days):
    r = cf.run_calculation("delay_damages_daily", {"amount_per_day": "5000", "days": days})
    assert r.get("status") == "needs_input", r          # one status, not "error"
    assert r.get("status") != "error"
    assert r.get("needs_input") is True
    assert "days" in (r.get("question") or ""), r        # the field is named
    # blank routes through the same bind path, which must still name the missing
    # required field in its error text (kills drop_rule on _bind_error_envelope).
    if days == "":
        assert "missing required" in (r.get("error") or "")


# ── the run cases: a usable value, or an omission the output doesn't need ────
@pytest.mark.parametrize("params", [
    {"amount_per_day": "5000", "days": "60"},   # valid total
    {"amount_per_day": "5000"},                 # daily-only: days not needed
])
def test_usable_or_unneeded_runs(params):
    assert cf.run_calculation("delay_damages_daily", params).get("status") == "success"


# ── a nested needs_input is a non-result the loop must still catch (nudge) ───
@pytest.mark.parametrize("node,errored", [
    ({"ok": True, "result": {"status": "needs_input"}}, True),   # mutant ==\"error\" RED
    ({"ok": True, "result": {"status": "success"}}, False),
    ("not-a-dict", False),                                       # kills drop_rule (isinstance guard)
])
def test_tool_result_errored_flags_nested_needs_input(node, errored):
    assert _tool_result_errored(node) is errored
