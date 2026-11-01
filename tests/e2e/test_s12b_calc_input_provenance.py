"""S-12b (e2e) — a construction CALCULATION answer carries, for every figure,
the inputs the calculator used, each stamped with WHERE its value came from.

This is the data the "Why this answer" panel renders. A calculation ask routes
to LLM + calculator: the calculator runs, and figure_provenance credits each
answer figure with its calc inputs as a LIST of {name, value, unit, source}.
Source is one of: user_text (the operator stated it), document:<chunk_id> (it
came from a retrieved chunk), project_data (stamped at compose time), or
"unresolved" (neither stated, retrieved, nor stamped). Per ruling A an
unstamped operand is NEVER guessed as "derived" — that would invent a fact.

Deterministic and LLM-free, like test_f3: it drives the calculator + provenance
layer directly, so what it proves is the plumbing under the answer, not what a
model writes. Full dotted imports so the mutation gate selects this module.
"""
from __future__ import annotations

import json

import pytest

import app.agents.citation_provenance as cp  # noqa: F401
from app.agents import provenance_trail as pt
from app.agents.citation_provenance import figure_provenance
from app.lib.construction_formulas import run_calculation

_ALLOWED_PREFIXES = ("user_text", "project_data", "unresolved", "document:")


def _calc_turn(question, name, params):
    result = run_calculation(name, params)
    msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "t1", "type": "function",
            "function": {"name": "construction_calc",
                         "arguments": json.dumps({"calculation": name, "params": params})}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "construction_calc",
         "content": json.dumps(result)},
    ]
    return msgs, result


def _excerpt(doc_id, name, text):
    body = (f"[doc_id={doc_id} chunk=0 score=0.900 class=project_corpus "
            f"layer=project_corpus page=12 src={name}] {text}")
    return {"role": "system", "content": "REFERENCE CONTEXT\n" + body}


def _stamp(msgs, **sources):
    env = json.loads(msgs[-1]["content"])
    env["_input_sources"] = dict(sources)
    msgs[-1]["content"] = json.dumps(env)


def _calc_inputs(prov):
    cals = [e for e in prov if e.get("source") == pt.SOURCE_CALCULATOR and e.get("inputs")]
    assert cals, prov
    return cals[0]["inputs"]


@pytest.mark.parametrize("depth,grounded", [(1.15, True), (9.99, False)])
def test_calc_inputs_are_a_list_of_stamped_sources_never_derived(depth, grounded):
    # length stated by the operator (user_text); width lifted from a retrieved
    # spec chunk (document:<chunk_id>); depth stamped project_data at compose
    # time when grounded, or left ungrounded (-> "unresolved") when not.
    question = "What is the dig volume for the pit in the spec, 3.4 m long?"
    params = {"length_m": 3.4, "width_m": 2.6, "depth_m": depth}
    msgs, result = _calc_turn(question, "excavation_volume", params)
    ctx = _excerpt("spec_a", "Earthworks Spec.pdf", "The pit is 2.6 m wide.")
    if grounded:
        _stamp(msgs, depth_m="project_data")
    bank = result["result"]["bank_volume_m3"]
    _out, prov = figure_provenance(f"The bank volume is {bank} m3.", ctx, msgs, enforce=False)

    inputs = _calc_inputs(prov)
    # shape: a LIST of {name, value, unit, source}
    assert isinstance(inputs, list) and inputs, inputs
    for row in inputs:
        assert set(row) >= {"name", "value", "unit", "source"}, row
        assert row["source"].startswith(_ALLOWED_PREFIXES), row
        assert row["source"] != "derived", row  # ruling A

    by_name = {r["name"]: r for r in inputs}
    assert by_name["length_m"]["source"] == "user_text", by_name["length_m"]
    assert by_name["width_m"]["source"].startswith("document:spec_a"), by_name["width_m"]
    if grounded:
        assert by_name["depth_m"]["source"] == "project_data", by_name["depth_m"]
    else:
        assert by_name["depth_m"]["source"] == "unresolved", by_name["depth_m"]
