"""S-12a: the envelope_provenance badge rides on the calc figure entry.

figure_provenance credited a calculator figure with only {source, formula,
inputs}. The "Why this answer" panel needs WHERE the number came from -- the
envelope_provenance badge (provenance = kernel / reference_table / improvised,
the standard it states, whether it is certified) and the default lines the
formula had to supply. These pin that the badge now rides both on the credit
and, threaded through, on the figure entry; dropping the thread-through (the
`if credit.envelope:` block) reddens the entry test, and dropping the credit
`envelope=` reddens the credit test.
"""
from __future__ import annotations

import json

import pytest

from app.agents import provenance_trail as pt
from app.agents.citation_provenance import (
    _calculator_credits,
    build_evidence,
    figure_provenance,
)
from app.lib.construction_formulas import envelope_provenance, run_calculation


def _calc_messages(calculation: str, params: dict, envelope: dict, ask: str) -> list[dict]:
    args = json.dumps({"calculation": calculation, "params": params})
    return [
        {"role": "user", "content": ask},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_s12a",
                         "function": {"name": "construction_calc", "arguments": args}}]},
        {"role": "tool", "tool_call_id": "call_s12a", "name": "construction_calc",
         "content": json.dumps(envelope)},
    ]


def _calc_entry(entries: list[dict]) -> dict:
    calc = [e for e in entries if e.get("source") == pt.SOURCE_CALCULATOR]
    assert calc, entries
    return calc[0]


# Two distinct kernels (both registered -> provenance "kernel", certified True),
# each paired with a UNIT-bearing figure the run actually returned, so
# _figure_mentions detects it (bare sub-1 decimals are not figures).
_CASES = [
    ("dewatering_uplift_check", {"water_depth": 23, "raft_thickness": 2.0, "floor_count": 5},
     "The counter weight is {counter_weight_t_m2} T/m2."),
    ("formwork_striking_time", {"concrete_strength_7h": 2.0},
     "Formwork may be struck after {recommended_hours} hours."),
]


@pytest.mark.parametrize("calculation, params, tmpl", _CASES)
def test_the_credit_carries_the_envelope_badge(calculation, params, tmpl):
    # Credit level (deterministic, no figure-matching): envelope_provenance is
    # computed at credit time and stored verbatim on the credit.
    env = run_calculation(calculation, dict(params))
    messages = _calc_messages(calculation, params, env, f"run the {calculation} check")
    ev = build_evidence(None, messages, tool_passages=False)
    credits = _calculator_credits(ev)
    assert credits, "no calculator credit built"
    badge = envelope_provenance(calculation, env.get("result"))
    assert credits[0].envelope == badge
    assert badge["provenance"] in {"kernel", "reference_table", "improvised"}
    assert badge["certified"] is (badge["provenance"] == "kernel")


@pytest.mark.parametrize("calculation, params, tmpl", _CASES)
def test_the_entry_carries_the_badge_threaded_through(calculation, params, tmpl):
    env = run_calculation(calculation, dict(params))
    answer = tmpl.format(**env["result"])
    messages = _calc_messages(calculation, params, env, f"run the {calculation} check")

    _text, entries = figure_provenance(answer, None, messages, enforce=False)
    entry = _calc_entry(entries)
    badge = envelope_provenance(calculation, env.get("result"))
    assert entry.get("provenance") == badge["provenance"]
    assert entry.get("standard") == badge["standard"]
    assert isinstance(entry.get("standard"), str) and entry["standard"].strip()
    assert entry.get("certified") == badge["certified"]
    assert entry["certified"] is (entry["provenance"] == "kernel")


@pytest.mark.parametrize("calculation, params, _tmpl", _CASES)
def test_only_a_tool_run_record_becomes_a_credit(calculation, params, _tmpl):
    # A calc envelope that arrives as a USER message (not a tool result) is not
    # a calculator credit -- only tool_run records are. Dropping the
    # `if rec.kind != "tool_run"` guard (drop_rule) would credit it, so this
    # reddens the mutant.
    env = run_calculation(calculation, dict(params))
    user_carries_envelope = [{"role": "user", "content": json.dumps(env)}]
    ev = build_evidence(None, user_carries_envelope, tool_passages=False)
    assert _calculator_credits(ev) == [], "a non-tool_run record was credited"
