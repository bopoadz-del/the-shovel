"""Every figure in an answer is credited to its source: the provenance record.

Sources: the user's own words; a calculator run this turn (formula and
inputs); a retrieved excerpt (layer, document, page). Built from the turn's
evidence objects (one mechanism, ``citation_provenance``). Enforced, a figure
with no source is removed; recorded only, the answer is unchanged.
Synthetic documents and figures only.
"""
from __future__ import annotations

import json

from app.agents import provenance_trail as pt
from app.agents.citation_provenance import figure_provenance
from app.lib.construction_formulas import run_calculation


def _user(text):
    return [{"role": "user", "content": text}]


def _calc_turn(question, name, params):
    """A real calculator run, as the conversation records it."""
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


def _excerpts(*chunks):
    body = "\n\n".join(
        f"[doc_id={d} chunk=0 score=0.900 class=project_corpus layer={layer} page={page} src={name}] {text}"
        for d, layer, page, name, text in chunks)
    return {"role": "system", "content": "REFERENCE CONTEXT\n" + body}


SPEC = ("d_spec", "own", 41, "Roofing Specification Rev B.pdf",
        "Clause 5.3 Membrane laps shall be 150 mm minimum.")
HANDBOOK = ("d_hand", "general_knowledge", 17, "Waterproofing Handbook.pdf",
            "Typical membrane lap width is 100 mm for torch-applied sheets.")


def test_user_value_is_credited_to_the_user():
    _out, prov = figure_provenance("With a 120 mm lap you need more rolls.", None,
                                   _user("Using a 120 mm lap, how many rolls?"))
    assert prov == [{"figure": "120 mm", "source": pt.SOURCE_USER}]


def test_calculator_result_is_credited_with_formula_and_inputs():
    q = "How much do I dig for a pit 3.4 m by 2.6 m and 1.15 m deep?"
    messages, result = _calc_turn(q, "excavation_volume",
                                  {"length_m": 3.4, "width_m": 2.6, "depth_m": 1.15})
    bank = result["result"]["bank_volume_m3"]
    out, prov = figure_provenance(f"The bank volume is {bank} m3.", None, messages)
    (entry,) = prov
    assert entry["source"] == pt.SOURCE_CALCULATOR and entry["formula"] == "excavation_volume"
    assert entry["inputs"]


def test_document_figures_are_credited_with_layer_document_and_page():
    ctx = _excerpts(SPEC, HANDBOOK)
    _out, prov = figure_provenance(
        "Laps must be at least 150 mm; the handbook's typical value is 100 mm.", ctx,
        _user("What lap width applies on this roof?"))
    by_fig = {e["figure"]: e for e in prov}
    assert (by_fig["150 mm"]["source"], by_fig["150 mm"]["doc_id"], by_fig["150 mm"]["page"]) == (
        pt.SOURCE_PROJECT, "d_spec", 41)
    assert (by_fig["100 mm"]["source"], by_fig["100 mm"]["doc_id"], by_fig["100 mm"]["page"]) == (
        pt.SOURCE_GENERAL, "d_hand", 17)


def test_an_unsourced_figure_is_removed_when_enforced():
    answer = "You will dig 10.17 m3. Allow for working space."
    out, prov = figure_provenance(answer, None, _user("How much do I dig for the pit?"))
    assert "10.17" not in out and "working space" in out
    assert prov == []


def test_recording_only_leaves_the_answer_unchanged_and_marks_the_gap():
    answer = "You will dig 10.17 m3."
    out, prov = figure_provenance(answer, None, _user("How much do I dig?"), enforce=False)
    assert out == answer
    assert prov == [{"figure": "10.17 m3", "source": None}]


def test_the_old_path_records_without_rewriting():
    from app.agents.runtime import _record_figure_provenance

    audit: dict = {}
    answer = "You will dig 10.17 m3."
    assert _record_figure_provenance(answer, None, _user("How much?"), audit) == answer
    assert audit["provenance"] == [{"figure": "10.17 m3", "source": None}]


def test_the_assistant_message_stores_its_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import agent_memory

    agent_memory.init_db()
    conv = agent_memory.get_or_create_conversation("conv-prov-test", "heavy-reasoning")
    entries = [{"figure": "150 mm", "source": pt.SOURCE_PROJECT, "doc_id": "d_spec", "page": 41}]
    pt.LAST_PROVENANCE.set(entries)
    msg = agent_memory.append_message(conv["id"], "assistant", "Laps are 150 mm.")
    pt.LAST_PROVENANCE.set(None)
    assert msg["provenance"] == entries
    assert agent_memory.append_message(conv["id"], "user", "thanks")["provenance"] is None
