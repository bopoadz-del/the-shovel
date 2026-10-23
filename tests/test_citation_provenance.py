"""Fence tests for the citation-provenance gate.

THE FIXTURE, and where it comes from. The F2 and G1 raw turns were captured
during the A-H gate battery on live ``13b2bf7`` (2026-08-31). The scored
evidence pack ``FLEET_OPS/artifacts/gate_battery_13b2bf7_2026-08-31.md``
records the load-bearing strings verbatim:

    F2 | PASS on the override, carries a Sev-1 | ... the answer cites
    ``BOQ context: Bill 03 - Demolition and Site Clearance (AB-2022-202)``
    -- the WRONG contract.

    G1 | FAIL -- asserted Schedule 10 "sets out any applicable Works
    Guarantees" and quoted CONTRACT TEMPLATE wording.

The surrounding prose of those two turns was lost with the workspace and the
platform's token has expired, so the answers around the citations here are
reconstructed; THE CITATION STRINGS THEMSELVES ARE THE RECORDED ONES. That
distinction is the point of the test, and it is stated rather than papered
over: the gate is asserted against the exact defect, not against a
paraphrase of it. The live re-proof is the full re-battery on the deployed
SHA, which is the next-but-one item in the owner's order.

Everything the gate's own logic rests on was verified literally on 13b2bf7,
not inferred:

* ``generate_wbs`` (schedule.py:2235) -- zero boq/bill/quantit/contract
  tokens in its body; it has no BOQ input to read.
* The string ``BOQ context`` does not occur anywhere in the repository, so
  nothing composed that line.
* ``construction_calc`` (construction/__init__.py:2164),
  ``resource_histogram`` (schedule.py:724), ``commissioning_checklist``
  (schedule.py:652), ``cash_flow_forecast`` (boq.py:1709) and ``wir_form``
  (documents.py:2112) likewise carry no corpus/retrieval reference.
"""

import pytest

from app.agents.citation_provenance import (
    UNVERIFIED_NOTE,
    Evidence,
    EvidenceRecord,
    build_evidence,
    gate,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# The recorded F2 citation, byte for byte from the evidence pack.
F2_CITATION = "BOQ context: Bill 03 - Demolition and Site Clearance (AB-2022-202)"

F2_ANSWER = f"""Tree-removal override applied and the schedule recomputed.

| metric | value |
|---|---|
| activities | 204 |
| working days | 721 |
| critical activities | 44 |

The override replaced the template duration for tree removal before CPM, so
the recomputed ES/EF/total float match the number you gave rather than the
rule-of-thumb default.

{F2_CITATION}
"""

# generate_wbs's tool call and result as the runtime records them: name on the
# tool message, arguments on the preceding assistant message. Note what is NOT
# in the arguments -- there is no BOQ field to pass.
F2_MESSAGES = [
    {"role": "user", "content": "redo it with 3 days per tree removal"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "function": {
                "name": "generate_wbs",
                "arguments": '{"brief": "infrastructure package", '
                             '"duration_overrides": [{"match": "tree removal", "days": 3}]}',
            },
        }],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "generate_wbs",
        "content": '{"status": "success", "wbs_id": "wbs-9f21ab30", '
                   '"project_type": "building", "actual_count": 204}',
    },
]

# G1's recorded assertion. Retrieval DID happen on that turn -- the failure was
# that template wording was quoted as the contract's own, which is a source
# CLASS defect, not an unbacked attribution.
G1_ANSWER = """Schedule 10 sets out any applicable Works Guarantees, including the
form of guarantee to be provided and the period for which each guarantee
remains in force.
"""

G1_RAG = {
    "role": "system",
    "content": (
        "AUTHORITATIVE REFERENCE CONTEXT - the material below was retrieved.\n"
        "[doc_id=abc123 chunk=0 score=0.712 src=AB-2023-101_Conditions.pdf] "
        "Schedule 10 sets out any applicable Works Guarantees.\n\n"
        "[doc_id=abc123 chunk=1 score=0.664 src=AB-2023-101_Conditions.pdf] "
        "The Guarantor shall provide the guarantee in the form annexed."
    ),
}


def _rag(*chunks: str, src: str = "AB-2023-101_Vol1.pdf", cls: str = "") -> dict:
    cls_s = f" class={cls}" if cls else ""
    body = "\n\n".join(
        f"[doc_id=d{i} chunk={i} score=0.700{cls_s} src={src}] {c}"
        for i, c in enumerate(chunks)
    )
    return {"role": "system", "content": "AUTHORITATIVE REFERENCE CONTEXT\n" + body}


# --------------------------------------------------------------------------
# F2 -- the Sev-1 fence
# --------------------------------------------------------------------------

def test_f2_fabricated_boq_citation_is_removed():
    out = gate(F2_ANSWER, None, F2_MESSAGES)
    assert "BOQ context" not in out
    assert "AB-2022-202" not in out
    assert UNVERIFIED_NOTE.strip() in out


def test_f2_answer_content_survives_intact():
    """The gate removes the attribution, never the answer."""
    out = gate(F2_ANSWER, None, F2_MESSAGES)
    for kept in ("204", "721", "44", "Tree-removal override applied",
                 "recomputed ES/EF/total float"):
        assert kept in out, kept


def test_f2_markdown_table_is_not_mangled():
    """A blanket separator sweep would eat every table pipe in the answer."""
    out = gate(F2_ANSWER, None, F2_MESSAGES)
    assert "| metric | value |" in out
    assert "|---|---|" in out
    assert "| activities | 204 |" in out


def test_f2_leaves_no_orphan_debris_where_the_citation_stood():
    out = gate(F2_ANSWER, None, F2_MESSAGES)
    body = out.replace(UNVERIFIED_NOTE, "")
    assert "Bill 03" not in body
    assert not body.rstrip().endswith("|")
    assert "\n\n\n" not in body


def test_generate_wbs_run_is_not_a_corpus_read():
    ev = build_evidence(None, F2_MESSAGES)
    assert ev.any_corpus_read() is False
    assert ev.boq_backed() is False
    assert "generate_wbs" in ev.tool_names()


def test_f2_citation_alone_fires_the_guard():
    """The recorded string on its own, with nothing else in the turn."""
    out = gate(F2_CITATION, None, F2_MESSAGES)
    assert "AB-2022-202" not in out
    assert UNVERIFIED_NOTE.strip() in out


# --------------------------------------------------------------------------
# G1 -- what this gate can honestly do, and what it cannot
# --------------------------------------------------------------------------

def test_g1_is_not_mangled_by_this_gate():
    """G1's defect is source CLASS (template quoted as the contract), which
    needs the class tag from numbered item 2. Retrieval backed the wording,
    so this gate must leave the turn exactly as it found it -- a gate that
    edits an answer it cannot judge is worse than one that abstains."""
    assert gate(G1_ANSWER, G1_RAG, [{"role": "user", "content": "What does Schedule 10 contain?"}]) == G1_ANSWER


def test_class_tagged_template_chunk_cannot_back_a_contract_attribution():
    """Forward fence for item 2: the moment the marker carries
    ``class=template``, an attribution resting only on it is unbacked. The
    enforcement is here already so item 2 only has to emit the tag."""
    rag = _rag("Schedule 10 sets out any applicable Works Guarantees.",
               src="fidic_template.md", cls="template")
    answer = "Schedule 10 covers Works Guarantees.\nSource: AB-2023-101\n"
    out = gate(answer, rag, [])
    assert "Source: AB-2023-101" not in out
    assert UNVERIFIED_NOTE.strip() in out


def test_project_corpus_chunk_does_back_the_same_attribution():
    rag = _rag("Schedule 10 sets out any applicable Works Guarantees.",
               src="AB-2023-101_Conditions.pdf")
    answer = "Schedule 10 covers Works Guarantees.\nSource: AB-2023-101\n"
    assert gate(answer, rag, []) == answer


# --------------------------------------------------------------------------
# Mutation probe -- plant a fake Source line, the guard fires
# --------------------------------------------------------------------------

@pytest.mark.parametrize("planted", [
    "Source: AB-2019-004 Particular Conditions",
    "**Source:** AB-2021-990",
    "- Source: AB-2020-117, clause 14.3",
    "Sources: AB-2018-001 and AB-2018-002",
    "BOQ context: Bill 07 - Roads (AB-2024-311)",
    "Bill of Quantities context: Bill 12 (AB-2015-002)",
])
def test_planted_fake_attributions_all_fire_the_guard(planted):
    answer = f"The retention is 10% of the Accepted Contract Amount.\n\n{planted}\n"
    out = gate(answer, None, F2_MESSAGES)
    assert "The retention is 10%" in out
    assert UNVERIFIED_NOTE.strip() in out
    for tok in ("AB-2019-004", "AB-2021-990", "AB-2020-117", "AB-2018-001",
                "AB-2024-311", "AB-2015-002", "BOQ context"):
        assert tok not in out, tok


@pytest.mark.parametrize("cue", [
    "per", "from", "see", "under", "according to", "as set out in", "taken from",
])
def test_planted_prose_attribution_fires_when_nothing_read_the_corpus(cue):
    answer = f"The retention is 10%, {cue} AB-2022-202.\n"
    out = gate(answer, None, F2_MESSAGES)
    assert "AB-2022-202" not in out
    assert "The retention is 10%" in out


# --------------------------------------------------------------------------
# The gate must not fire on honest answers
# --------------------------------------------------------------------------

def test_backed_attribution_survives():
    rag = _rag("The Accepted Contract Amount is SAR 1,754,504,456.25.",
               src="AB-2023-101_ContractData.pdf")
    answer = "The Accepted Contract Amount is SAR 1,754,504,456.25.\nSource: AB-2023-101 Contract Data 1.1.1\n"
    assert gate(answer, rag, []) == answer


def test_id_the_user_named_is_never_stripped():
    """The operator's own words are evidence. An id they typed is theirs."""
    msgs = [{"role": "user", "content": "What is the retention under AB-2022-202?"}]
    answer = "Retention under AB-2022-202 is 10%.\nSource: AB-2022-202\n"
    assert gate(answer, None, msgs) == answer


def test_standards_reference_is_not_a_contract_id():
    """ISO-9001-2015 matches the contract-id SHAPE exactly. A blanket strip
    would eat it; the attribution cue is what keeps it safe."""
    answer = "The QA system shall comply with ISO-9001-2015 and BS-5950-2000.\n"
    assert gate(answer, None, F2_MESSAGES) == answer


def test_answer_with_no_attribution_is_returned_unchanged():
    answer = "204 activities, 688 working days, 44 on the critical path.\n"
    assert gate(answer, None, F2_MESSAGES) is answer


def test_tool_self_declaration_survives():
    """R3 requires the template scheduler to declare itself at the glass. A
    citation naming the tool that actually ran is rendered, not invented."""
    answer = ("204 activities.\n"
              "Source: generate_wbs template scaffold, project_type inferred: "
              "building, not derived from this project's BOQ\n")
    assert gate(answer, None, F2_MESSAGES) == answer


def test_cash_flow_forecast_handed_a_boq_may_cite_one():
    """The tool reads a BOQ from its OWN payload -- that citation is honest.
    This is why the registry marks 'does not retrieve', not 'never sees a
    bill'."""
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1",
            "function": {"name": "cash_flow_forecast",
                         "arguments": '{"contract_value": 60000000, '
                                      '"boq": [{"code": "D599.5", "amount": 10568024.0}]}'},
        }]},
        {"role": "tool", "tool_call_id": "c1", "name": "cash_flow_forecast",
         "content": '{"status": "success", "peak_monthly_billing": 4200000}'},
    ]
    answer = "Peak monthly billing is SAR 4.2M.\n\nBOQ context: the priced bill you supplied\n"
    assert gate(answer, None, msgs) == answer


def test_unknown_tool_is_assumed_to_read_the_corpus():
    """Fail-safe by default: a tool nobody has verified must never cause a
    legitimate attribution to be stripped by omission."""
    msgs = [{"role": "tool", "name": "some_future_retriever", "content": "..."}]
    ev = build_evidence(None, msgs)
    assert ev.any_corpus_read() is True


# --------------------------------------------------------------------------
# Evidence objects
# --------------------------------------------------------------------------

def test_retrieval_records_are_built_per_chunk():
    """Per-chunk, never one blob: the cost gate learned on 2026-07-14 that a
    mixed bundle reads as rate-semantic as a whole and grounds numbers it
    should not. One bill chunk must not make its neighbours look like bills."""
    rag = _rag("Bill of Quantities - priced bill, Part 2.",
               "General arrangement drawing, levels in mm.")
    ev = build_evidence(rag, [])
    retr = [r for r in ev.records if r.kind == "retrieval"]
    assert len(retr) == 2
    assert retr[0].carries_boq() is True
    assert retr[1].carries_boq() is False


def test_marker_src_and_class_are_parsed():
    rag = _rag("text", src="AB-2023-101_Vol1.pdf", cls="knowledge_base")
    rec = build_evidence(rag, []).records[0]
    assert rec.source_name == "AB-2023-101_Vol1.pdf"
    assert rec.source_class == "knowledge_base"
    assert rec.citable is False


def test_flat_rag_context_without_markers_still_counts_as_a_corpus_read():
    ev = build_evidence({"role": "system", "content": "some retrieved prose"}, [])
    assert ev.any_corpus_read() is True


def test_corpus_reading_tool_ids_are_citable():
    msgs = [{"role": "tool", "name": "search_project_documents",
             "content": "found AB-2023-101_Vol1.pdf"}]
    ev = build_evidence(None, msgs)
    assert "ab-2023-101" in ev.citable_ids()


def test_template_scheduler_ids_are_not_citable():
    msgs = [{"role": "tool", "name": "generate_wbs",
             "content": '{"brief": "works under AB-2022-202"}'}]
    ev = build_evidence(None, msgs)
    assert ev.citable_ids() == set()


# --------------------------------------------------------------------------
# A gate must never break an answer
# --------------------------------------------------------------------------

def test_gate_never_raises_on_junk_input():
    for bad in (None, "", "   ", 0):
        assert gate(bad, None, None) == bad
    assert gate("text", {"content": None}, [None, 3, "x"]) == "text"


def test_gate_survives_an_exploding_evidence_builder(monkeypatch):
    import app.agents.citation_provenance as cp
    monkeypatch.setattr(cp, "build_evidence", lambda *a, **k: 1 / 0)
    assert cp.gate(F2_ANSWER, None, F2_MESSAGES) == F2_ANSWER


def test_flag_off_passes_everything_through(monkeypatch):
    monkeypatch.setenv("CITATION_PROVENANCE_GATE", "0")
    assert gate(F2_ANSWER, None, F2_MESSAGES) == F2_ANSWER


def test_gate_is_wired_into_postprocess_answer():
    """The module is inert unless the answer path calls it."""
    import inspect
    from app.agents import runtime
    src = inspect.getsource(runtime._postprocess_answer)
    assert "citation_provenance" in src


# --------------------------------------------------------------------------
# Parenthesised ids -- the attribution form that carries no keyword
# --------------------------------------------------------------------------

def test_standalone_parenthesised_id_is_stripped_when_unbacked():
    """"Bill 03 - Demolition and Site Clearance (AB-2022-202)" is an
    attribution even with no "Source:" and no "BOQ context:" in front of it."""
    rag = _rag("Bill 03 covers demolition and site clearance.",
               src="AB-2023-101_BOQ.pdf")
    answer = "The demolition scope sits in Bill 03 (AB-2022-202).\n"
    out = gate(answer, rag, [])
    assert "AB-2022-202" not in out
    assert "The demolition scope sits in Bill 03" in out
    assert UNVERIFIED_NOTE.strip() in out


def test_standalone_parenthesised_id_survives_when_evidence_names_it():
    rag = _rag("Bill 03 covers demolition and site clearance.",
               src="AB-2023-101_BOQ.pdf")
    answer = "The demolition scope sits in Bill 03 (AB-2023-101).\n"
    assert gate(answer, rag, []) == answer


def test_parenthesised_id_is_stripped_when_nothing_read_the_corpus():
    answer = "Tree removal is item 3.2 (AB-2022-202).\n"
    out = gate(answer, None, F2_MESSAGES)
    assert "AB-2022-202" not in out
    assert "Tree removal is item 3.2" in out


def test_parenthesised_id_the_user_named_survives_with_no_corpus_read():
    msgs = F2_MESSAGES + [{"role": "user", "content": "check AB-2022-202 please"}]
    answer = "Tree removal is item 3.2 (AB-2022-202).\n"
    assert gate(answer, None, msgs) == answer


def test_attribution_inside_a_table_row_does_not_destroy_the_row():
    """The strip lands mid-row. Only the removed cell's own separator may go;
    every other pipe in that row is the table's structure."""
    answer = (
        "| phase | days | note |\n"
        "|---|---|---|\n"
        "| Demolition | 42 | BOQ context: Bill 03 (AB-2022-202) |\n"
        "| Earthworks | 60 | template default |\n"
    )
    out = gate(answer, None, F2_MESSAGES)
    body = out.replace(UNVERIFIED_NOTE, "")
    assert "AB-2022-202" not in body
    assert "BOQ context" not in body
    assert "| phase | days | note |" in body
    assert "|---|---|---|" in body
    assert "| Earthworks | 60 | template default |" in body
    demolition = [l for l in body.splitlines() if "Demolition" in l][0]
    assert demolition.count("|") >= 3, demolition
    assert "42" in demolition


def test_inline_separators_on_a_non_table_line_survive_the_strip():
    """The trailing pipe left dangling by a strip is debris; the separators
    between the metrics before it are the line's own formatting. Only the
    orphan goes."""
    answer = ("Activities: 204 | Working days: 721 | "
              "BOQ context: Bill 03 (AB-2022-202)\n")
    out = gate(answer, None, F2_MESSAGES)
    body = out.replace(UNVERIFIED_NOTE, "").rstrip()
    assert "BOQ context" not in body
    assert "AB-2022-202" not in body
    assert body == "Activities: 204 | Working days: 721"
