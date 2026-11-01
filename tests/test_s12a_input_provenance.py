"""S-12a: each calculator input carries {name, value, unit, source}. The unit
is the declared registry Param; the source is user_text (the operator stated
it), document:<chunk_id> (it appears in a retrieved chunk), or derived (a
model-supplied / intermediate value). Synthetic documents and figures only.
Full-dotted import so the mutation gate selects this.
"""
from __future__ import annotations

import json

import pytest

import app.agents.citation_provenance as cp  # noqa: F401
from app.agents import provenance_trail as pt
from app.agents.citation_provenance import figure_provenance
from app.lib.construction_formulas import run_calculation


def _calc_turn(question, name, params):
    result = run_calculation(name, params)
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "t1", "type": "function",
            "function": {"name": "construction_calc",
                         "arguments": json.dumps({"calculation": name, "params": params})}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "construction_calc",
         "content": json.dumps(result)},
    ], result


def _excerpt(doc_id, page, name, text):
    body = (f"[doc_id={doc_id} chunk=0 score=0.900 class=project_corpus "
            f"layer=project_corpus page={page} src={name}] {text}")
    return {"role": "system", "content": "REFERENCE CONTEXT\n" + body}


def _calc_inputs(prov):
    cals = [e for e in prov if e.get("source") == pt.SOURCE_CALCULATOR and e.get("inputs")]
    assert cals, prov
    return {r["name"]: r for r in cals[0]["inputs"]}


@pytest.mark.parametrize("name,value", [("length_m", 3.4), ("depth_m", 1.15)])
def test_user_stated_inputs_are_user_text_with_a_declared_unit(name, value):
    msgs, result = _calc_turn(
        "Dig a pit 3.4 m by 2.6 m and 1.15 m deep?", "excavation_volume",
        {"length_m": 3.4, "width_m": 2.6, "depth_m": 1.15})
    bank = result["result"]["bank_volume_m3"]
    _out, prov = figure_provenance(f"The bank volume is {bank} m3.", None, msgs, enforce=False)
    rows = _calc_inputs(prov)
    assert rows[name]["value"] == value
    assert rows[name]["source"] == "user_text"
    assert rows[name]["unit"] == "m"


@pytest.mark.parametrize("doc_id", ["spec_a", "spec_b"])
def test_input_from_a_retrieved_chunk_cites_that_document(doc_id):
    ctx = _excerpt(doc_id, 12, "Earthworks Spec.pdf",
                   "The pit is 3.4 m by 2.6 m by 1.15 m.")
    msgs, result = _calc_turn(
        "What is the dig volume for the pit in the spec?", "excavation_volume",
        {"length_m": 3.4, "width_m": 2.6, "depth_m": 1.15})
    bank = result["result"]["bank_volume_m3"]
    _out, prov = figure_provenance(f"The bank volume is {bank} m3.", ctx, msgs, enforce=False)
    rows = _calc_inputs(prov)
    assert rows["length_m"]["source"].startswith(f"document:{doc_id}"), rows["length_m"]


@pytest.mark.parametrize("depth", [7.77, 9.99])
def test_model_supplied_value_not_in_evidence_is_derived(depth):
    # length/width are the operator's words; depth is a value neither stated nor
    # in any chunk -> derived. One calc exercises user_text AND derived.
    msgs, result = _calc_turn(
        "Dig a pit 3.4 m by 2.6 m?", "excavation_volume",
        {"length_m": 3.4, "width_m": 2.6, "depth_m": depth})
    bank = result["result"]["bank_volume_m3"]
    _out, prov = figure_provenance(f"The bank volume is {bank} m3.", None, msgs, enforce=False)
    rows = _calc_inputs(prov)
    assert rows["depth_m"]["source"] == "derived", rows["depth_m"]
    assert rows["length_m"]["source"] == "user_text"
