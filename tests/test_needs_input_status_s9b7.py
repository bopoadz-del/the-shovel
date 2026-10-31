"""Item 7 (ORDERS_31OCT_0405Z ruling): one status for an input the asked output
needs but did not get. A required numeric that is provided-but-unparseable OR
blank OR omitted-when-needed returns status "needs_input" naming the field --
never "error" with a flag riding alongside. Omitted when the output does not
need it (daily-only) returns success. General coercion rule, no per-value branch.
"""
from __future__ import annotations

import json

import pytest

import app.agents.base.tools as _base_tools  # noqa: F401 -- selects tools.py mutants for CI-M
import app.lib.construction_formulas as cf
from app.agents.answer_exit import _error_texts
from app.agents.runtime import _tool_result_errored
from tests.test_construction_calc_tool import _agent, _run


# ── status split (ORDERS_31OCT_0825Z FINAL): a blank/unreadable value unifies
#    to status "needs_input"; an out-of-range value keeps its pre-existing
#    contract (status "error" + the needs_input flag + scale), and a wrong unit
#    stays "error" too. Every rejection still flags the ask. Pins
#    _input_rejection_envelope directly -- mutant "coercion/blank routed to
#    error" RED; mutant "out-of-range status flipped to needs_input" RED. ──
@pytest.mark.parametrize("reason,status", [
    ("empty", "needs_input"),         # blank -> unified ask (mutant "ask->error" RED)
    ("not_a_number", "needs_input"),  # garbled -> unified ask
    ("not_whole", "needs_input"),     # fraction where a whole is wanted -> unified ask
    ("out_of_range", "error"),        # out-of-range keeps status error (mutant "->needs_input" RED)
    ("unit", "error"),                # a wrong unit stays error
])
def test_rejection_reason_decides_status(reason, status):
    env = cf._input_rejection_envelope(
        "delay_damages_daily",
        [{"parameter": "days", "label": "days", "reason": reason,
          "lo": 1, "hi": 365, "value": "x"}],
        [],
    )
    assert env["status"] == status, env
    # Every rejection still invites re-supply -- the flag is unchanged from
    # before item 7; only the status token moved for the coercion/blank class.
    assert env.get("needs_input") is True, env


@pytest.mark.parametrize("missing", [["days"], ["amount_per_day", "days"]])
def test_missing_required_always_asks(missing):
    env = cf._input_rejection_envelope("delay_damages_daily", [], missing)
    assert env["status"] == "needs_input"
    assert env.get("needs_input") is True


# ── a value the caller GAVE but the calculator rejected as out of domain keeps
#    status "error" (with the ask flag); an ABSENT input asks (needs_input).
#    ORDERS_31OCT_0945Z. Mutant "given-invalid routed to needs_input" RED. ──
@pytest.mark.parametrize("name,params,status", [
    # given-but-invalid rate (calculator raises "must be greater than zero")
    # -> status error WITH the ask flag (the 3139 given-invalid branch).
    ("productivity_manpower_duration",
     {"quantity": 4800, "productivity_rate": "-120 m2 per gang-day"}, "error"),
    # absent input -> ask (same partial-result branch, no value rejected).
    ("supervision_ratio", {"concrete_m3": 5000}, "needs_input"),
])
def test_given_invalid_value_errors_absent_asks(name, params, status):
    env = cf.run_calculation(name, params)
    assert env.get("status") == status, env
    if status == "error":
        assert env.get("needs_input") is True, env        # still flags the ask
        assert "greater than zero" in (env.get("error") or ""), env


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


# ── the construction_calc TOOL envelope: an ask (needs_input) is not ok, a
#    delivered answer is. Exercises handle_construction_calc directly (this
#    module imports app.agents.base.tools, so CI-M selects it for that function's
#    mutants). Two inputs: an ask vs a run. ──
def _calc_tool(user_message, args):
    agent = _agent(["construction"])
    tc = {"id": "c1", "function": {
        "name": "construction_calc", "arguments": json.dumps(args)}}
    return _run(agent._run_tool_call(tc, user_message=user_message))


@pytest.mark.parametrize("args,user_message,ok,status", [
    ({"action": "construction_calc", "calculation": "delay_damages_daily"},
     "delay_damages_daily", False, "needs_input"),                  # ask -> not ok
    ({"action": "construction_calc", "calculation": "delay_damages_daily",
      "amount_per_day": "5000", "days": "60"},
     "delay_damages_daily 5000 per day for 60 days", True, "success"),  # run -> ok
])
def test_construction_calc_tool_ok_tracks_needs_input(args, user_message, ok, status):
    env = _calc_tool(user_message, args)
    assert env["name"] == "construction_calc", env
    assert env.get("ok") is ok, env               # mutant "ok True for needs_input" RED
    assert env["result"]["status"] == status, env


# ── answer_exit treats a needs_input tool result as a FAILED tool round, so the
#    model's mechanics reply is replaced by the tool's question (not narrated as
#    an answer). Imports _error_texts directly so CI-M selects answer_exit's
#    mutants. Two inputs: an ask carries its text; a success carries none. ──
@pytest.mark.parametrize("payload,has_text", [
    ({"status": "needs_input", "error": "The days of delay needs a number."}, True),
    ({"status": "error", "error": "boom"}, True),
    ({"status": "success", "error": "ignored"}, False),
])
def test_error_texts_treats_needs_input_as_failed(payload, has_text):
    assert bool(_error_texts(payload)) is has_text   # mutant dropping needs_input RED


@pytest.mark.parametrize("user_message", [
    "interim payment certificate", "prepare the IPC",
])
def test_construction_calc_tool_ipc_ask_dispatches_payment_certificate(user_message):
    # No calculation + an IPC ask hits the first `if` in handle_construction_calc,
    # which dispatches the payment certificate (name "payment_certificate").
    # Drop that `if` (CI-M drop_rule) and the ask falls through to the plain
    # calculator path -> name "construction_calc": this asserts it RED.
    env = _calc_tool(user_message, {"action": "construction_calc"})
    assert env["name"] == "payment_certificate", env
