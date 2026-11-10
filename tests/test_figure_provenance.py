"""Every figure in an answer is credited to its source: the provenance record.

Sources: the user's own words; a calculator run this turn (formula and
inputs); a retrieved excerpt (layer, document, page). Built from the turn's
evidence objects (one mechanism, ``citation_provenance``). Enforced, a figure
with no source is removed; recorded only, the answer is unchanged.
Synthetic documents and figures only.
"""
from __future__ import annotations

import json

import pytest

from app.agents import provenance_trail as pt
from app.agents.citation_provenance import (
    EvidenceRecord,
    _answer_cites_record,
    figure_provenance,
)
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


@pytest.mark.parametrize(
    "figure,doc_text",
    [
        ("10%", "Clause 4.2 Performance Security: 10% of the Accepted Contract Amount."),
        ("5%", "Clause 14.3 Retention: 5% of each Interim Payment Certificate."),
    ],
)
def test_a_contract_figure_folded_into_the_turn_is_the_project_not_user_input(figure, doc_text):
    """S1a: a contract figure gets folded into the operator turn (RAG fold), so
    user_vals is polluted with the document's numbers. The figure still came
    from the project document and must be credited to it -- never user_input,
    the one value that tells the owner "you gave me this number"."""
    ctx = _excerpts(("d_con", "own", 1, "harbourline_contract.pdf", doc_text))
    # The question echoes the clause text, so the figure is also in user_vals.
    msgs = _user(f"What does the contract require? {doc_text}")
    _out, prov = figure_provenance(f"The contract states {figure}.", ctx, msgs, enforce=False)
    by = {e["figure"]: e for e in prov}
    assert figure in by, by
    assert by[figure]["source"] == pt.SOURCE_PROJECT
    assert by[figure]["source"] != pt.SOURCE_USER


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


@pytest.mark.parametrize(
    "answer,question,figure,params",
    [
        (
            "That design strength of 35 MPa is not in the project documents.",
            "What is the design strength for fck = 35 MPa and γc = 1.5?",
            "35 MPa",
            {"length_m": 3.0, "width_m": 2.0, "depth_m": 35},
        ),
        (
            "The trench is 6 m deep; I cannot run that excavation without a width.",
            "How much do I dig for a 6 m trench?",
            "6 m",
            {"length_m": 6.0, "width_m": 2.0, "depth_m": 3.0},
        ),
    ],
)
def test_an_unrelated_calculator_does_not_steal_a_users_figure(
    answer, question, figure, params,
):
    """A tool whose result the answer never uses is not a source (M13).

    The calculator is given a value equal to the answer's figure, but it
    returns a different number the answer does not state. Under the old pool
    (result ∪ inputs) the figure was credited to that unrelated tool and
    surfaced in "Why this answer"; the figure came from the operator's own
    words and must stay user_input.
    """
    result = run_calculation("excavation_volume", params)
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "t1", "type": "function",
            "function": {"name": "construction_calc",
                         "arguments": json.dumps({"calculation": "excavation_volume",
                                                  "params": params})}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "construction_calc",
         "content": json.dumps(result)},
    ]
    _out, prov = figure_provenance(answer, None, messages, enforce=False)
    assert prov, prov
    assert all(e.get("source") != pt.SOURCE_CALCULATOR for e in prov), prov
    by_fig = {e["figure"]: e for e in prov}
    assert by_fig[figure]["source"] == pt.SOURCE_USER, by_fig


@pytest.mark.parametrize(
    "answer,cited_id,cited_name,cited_text,other_name,other_text,figure",
    [
        (
            "Per ASCE 7-16, the minimum lap is 150 mm.",
            "d_asce", "ASCE 7-16.pdf", "Clause 5.3 laps shall be 150 mm minimum.",
            "FIDIC Red Book 2017.pdf", "Typical lap is 150 mm for membranes.",
            "150 mm",
        ),
        (
            "Per ACI 318-19, the minimum cover is 40 mm.",
            "d_aci", "ACI 318-19.pdf", "Minimum cover for slabs is 40 mm.",
            "Eurocode 2.pdf", "Nominal cover is 40 mm for XC4.",
            "40 mm",
        ),
    ],
)
def test_a_figure_is_credited_to_the_cited_chunk_not_an_uncited_one(
    answer, cited_id, cited_name, cited_text, other_name, other_text, figure,
):
    """An uncited chunk holding the figure must not steal the credit (M13/R8).

    Both chunks contain the figure and the OTHER one is retrieved first, so
    the old first-match loop credited it. The answer cites the second chunk
    by name; provenance must follow the citation, not retrieval order.
    """
    ctx = _excerpts(
        ("d_other", "own", 3, other_name, other_text),
        (cited_id, "own", 5, cited_name, cited_text),
    )
    _out, prov = figure_provenance(answer, ctx, _user("What is the minimum?"))
    by_fig = {e["figure"]: e for e in prov}
    assert figure in by_fig, by_fig
    assert by_fig[figure]["doc_id"] == cited_id, by_fig
    assert by_fig[figure]["source"] == pt.SOURCE_PROJECT


@pytest.mark.parametrize(
    "answer, chunk_index, source_name",
    [
        (
            "See chunk 40 for the membrane lap width.",
            40,
            "Roofing Specification Rev B.pdf",
        ),
        (
            "The design factor is stated in chunks 12 of the handbook.",
            12,
            "Waterproofing Handbook.pdf",
        ),
    ],
)
def test_a_chunk_cited_by_index_alone_credits_the_record(
    answer, chunk_index, source_name
):
    """A record cited by chunk index is credited even when its document
    name never appears in the answer.

    ``_answer_cites_record`` has two citation paths: chunk index and
    document name. A ``drop_rule`` mutant of the index path would make a
    chunk cited only as "chunk 40" stop being credited whenever the answer
    does not also name the document -- this pins the index branch.
    """
    rec = EvidenceRecord(
        kind="retrieval",
        text="synthetic chunk text",
        source_name=source_name,
        chunk_index=chunk_index,
    )
    assert _answer_cites_record(answer, rec) is True


@pytest.mark.parametrize(
    "chunk,question,answer,figure,subject_words",
    [
        (
            "The initial setting period is 7 days. The final setting period is 28 days. "
            "The hardening period is 42 days.",
            "What is the final setting period?",
            "The final setting period is 28 days.",
            "28 days",
            ("final", "setting"),
        ),
        (
            "The curing time is 2 hours. The mixing time is 3 hours. The drying time is 4 hours.",
            "What is the mixing time?",
            "The mixing time is 3 hours.",
            "3 hours",
            ("mixing",),
        ),
    ],
)
def test_a_sentence_naming_the_subject_keeps_the_project_tier(
    chunk, question, answer, figure, subject_words,
):
    """A chunk holding competing figures keeps the project tier when the
    cited sentence names the ask's subject (B11 sentence-level grounding).

    ``figure_provenance`` records the chunk's own sentence for the figure;
    because that sentence names the subject, the credit is not capped.
    """
    ctx = _excerpts(("d_spec", "own", 5, "Specification.pdf", chunk))
    _out, prov = figure_provenance(answer, ctx, _user(question))
    (entry,) = prov
    assert entry["source"] == pt.SOURCE_PROJECT
    assert entry["figure"] == figure
    assert entry["sentence"]
    low = entry["sentence"].lower()
    for w in subject_words:
        assert w in low


@pytest.mark.parametrize(
    "chunk,question,answer,figure,competing",
    [
        (
            "The initial setting period is 7 days. The final setting period is 28 days. "
            "The hardening period is 42 days.",
            "What is the hardening period?",
            "The final setting period is 28 days.",
            "28 days",
            {"7 days", "42 days"},
        ),
        (
            "The curing time is 2 hours. The curing depth is 150 mm. The drying time is 4 hours.",
            "What is the curing time?",
            "The drying time is 4 hours.",
            "4 hours",
            {"2 hours"},
        ),
    ],
)
def test_a_sentence_not_naming_the_subject_caps_to_general_and_lists_competing(
    chunk, question, answer, figure, competing,
):
    """A chunk holding competing figures is capped to the general tier and
    lists the competing figures when the cited sentence names another (or no)
    subject (B11 sentence-level grounding).

    The sentence carrying the stated figure names a different subject, so the
    ambiguity is surfaced instead of one figure being silently asserted.
    """
    ctx = _excerpts(("d_spec", "own", 5, "Specification.pdf", chunk))
    _out, prov = figure_provenance(answer, ctx, _user(question))
    (entry,) = prov
    assert entry["source"] == pt.SOURCE_GENERAL
    assert entry["figure"] == figure
    assert set(entry["competing"]) == competing
