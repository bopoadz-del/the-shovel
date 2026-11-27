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


@pytest.mark.parametrize("number,pool,passages,expected", [
    (4321.0, {55.0, 2.0}, [[10.0, 20.0]], False),   # in neither: the model's own number
    (55.0, {55.0, 2.0}, [], True),                  # the user wrote it
    (20.0, set(), [[10.0, 20.0]], True),            # a retrieved passage carries it
    (987.5, set(), [], False),
    (-43.0, {-43.0}, [], True),                     # given with its sign
    (-43.0, {43.0}, [], False),                     # the sign was turned: nobody gave -43
    (43.0, {-43.0}, [], False),                     # nor +43 when -43 was given
    (-20.0, set(), [[10.0, -20.0]], True),
    (-20.0, set(), [[10.0, 20.0]], False),
])
def test_given_to_us_is_true_only_for_a_number_the_user_or_a_source_gave(number, pool, passages, expected):
    from app.agents.citation_provenance import _given_to_us

    assert _given_to_us(number, pool, passages) is expected


@pytest.mark.parametrize("message,number,expected", [
    ("the trench is 40 m long and 1.5 m wide", 40.0, "user"),
    ("the trench is 40 m long and 1.5 m wide", 1.5, "user"),
    ("the trench is 40 m long and 1.5 m wide", 4321.0, ""),
    ("work it out again", 40.0, ""),
])
def test_given_check_holds_a_number_to_the_messages_so_far(message, number, expected):
    from app.agents.citation_provenance import given_check

    assert given_check([{"role": "user", "content": message}])(number) == expected


@pytest.mark.parametrize("text,figures", [
    ("The formation level is -43 m. It rises by 10 m.", [-43.0, 10.0]),
    ("between 40-43 m, grid B-12", [40.0, 43.0, 12.0]),          # a range and a compound: no minus
    ("depth .5 m and -.25", [0.5, -0.25]),                        # a decimal written without its zero
    ("level \u22127 m then +2.5", [-7.0, 2.5]),                    # the typographic minus
    ("15 x 8 x -0.2, rate 1,250 per m", [15.0, 8.0, -0.2, 1250.0]),
    ("no figure here", []), ("", []), (None, []),
])
def test_given_figures_reads_each_figure_with_the_sign_it_was_written_with(text, figures):
    from app.agents.citation_provenance import _given_figures

    assert _given_figures(text) == figures


@pytest.mark.parametrize("message,number,expected", [
    ("The formation level is -43 m. Raise it by 10 m.", -43.0, "user"),
    ("The formation level is -43 m. Raise it by 10 m.", 43.0, ""),       # the sign turned
    ("The formation level is 43 m. Lower it by 10 m.", -43.0, ""),
    ("The pipe is 40-43 m long, .5 m deep.", 43.0, "user"),
    ("The pipe is 40-43 m long, .5 m deep.", 0.5, "user"),
    ("The pipe is 40-43 m long, .5 m deep.", -43.0, ""),
])
def test_given_check_holds_a_number_to_its_sign_as_the_user_wrote_it(message, number, expected):
    from app.agents.citation_provenance import given_check

    assert given_check([{"role": "user", "content": message}])(number) == expected


@pytest.mark.parametrize("produced,number,expected", [([{120.0}], 120.0, True), ([{120.0}], 121.0, False),
                                                      ([], 120.0, False)])
def test_given_pools_take_user_numbers_produced_numbers_and_passages(produced, number, expected):
    from app.agents.citation_provenance import _given_pools, _given_to_us, build_evidence

    ev = build_evidence(None, [{"role": "user", "content": "rate 67 per m3"}], tool_passages=True)
    pool, passages = _given_pools(ev, produced)
    assert _given_to_us(67.0, pool, passages) is True
    assert _given_to_us(number, pool, passages) is expected


# --- a given is held to the QUANTITY that was written, not to its number alone ---

@pytest.mark.parametrize("text,quantities", [
    ("A trench is 12 m long, 3 m wide and 2 m deep.", [(12.0, "m"), (3.0, "m"), (2.0, "m")]),
    ("The formation level is -43 m. Raise it by 10 m: what now?", [(-43.0, "m"), (10.0, "m")]),
    ("The depth is 2.5 m, not 2 m.", [(2.5, "m"), (2.0, "m")]),
    ("12mm bars at 200 thick, 5% waste, 20 kN/m2", [(12.0, "mm"), (200.0, ""), (5.0, "%"), (20.0, "kN/m2"), (2.0, None)]),
    ("cost at 67 per m3, then 1,250 m3", [(67.0, ""), (3.0, None), (1250.0, "m3"), (3.0, None)]),
    ("grade M25 on grid B-12, 7 working days", [(25.0, ""), (12.0, ""), (7.0, "working days")]),
    ("no figure here", []), ("", []), (None, []),
])
def test_given_quantities_reads_each_figure_with_the_unit_written_after_it(text, quantities):
    from app.agents.citation_provenance import _given_figures, _given_quantities

    assert _given_quantities(text) == quantities
    # the same figures, in the same order, as the number reader gives
    assert [n for n, _unit in _given_quantities(text)] == _given_figures(text)


@pytest.mark.parametrize("unit,kind", [
    ("m", ("length", 1.0, 0.0)), ("mm", ("length", 0.001, 0.0)), ("km", ("length", 1000.0, 0.0)),
    ("m3", ("volume", 1.0, 0.0)),
    ("°C", ("temperature", 1.0, 0.0)), ("°F", ("temperature", 5 / 9, -160 / 9)),   # an offset is kept
    ("m*m", ("area", 1.0, 0.0)), ("mm*m", ("area", 0.001, 0.0)),                   # multiplied out
    ("m2/(m2/day)", ("time", 24.0, 0.0)), ("m3/(m3/h)", ("time", 1.0, 0.0)),
    ("kN.m", ("[m^2 kg^1 s^-2]", 1000.0, 0.0)), ("N*mm", ("[m^2 kg^1 s^-2]", 0.001, 0.0)),
    ("", None), (None, None), ("long", None), ("per m3", None), ("m*long", None), ("m*(m", None),
    ("working days", None),     # needs a calendar to convert: the number check alone
    ("m*working days", None),   # nor does a product with it
])
def test_unit_kind_is_the_kind_and_scale_of_a_unit_a_scale_alone_converts(unit, kind):
    from app.agents.citation_provenance import _unit_kind

    assert _unit_kind(unit) == kind


_TRENCH = "A trench is 12 m long, 3 m wide and 2 m deep. The rate is 67 per m3 and the slab is 200 thick."


@pytest.mark.parametrize("number,unit,expected", [
    (12.0, "m", "user"),          # as written
    (12000.0, "mm", "user"),      # the same quantity in another unit
    (0.012, "km", "user"),
    (12.0, "", "user"),           # a bare number: the number check, as before
    (12.0, "mm", ""),             # the unit changed: 12 mm is not what the user wrote
    (12.0, "km", ""),
    (12000.0, "m", ""),           # the number rescaled under the same unit
    (2.0, "mm", ""), (3.0, "cm", ""),
    (12.0, "m2", ""),             # written as a length, used as an area
    (3.0, "m3", ""),              # the 3 of "per m3" is no figure anyone gave
    (67.0, "currency", "user"),   # written bare: the unit is a reading of it
    (200.0, "mm", "user"), (200.0, "m", "user"),
    (4321.0, "m", ""), (4321.0, "", ""),
])
def test_given_check_holds_a_given_with_a_unit_to_the_quantity_the_user_wrote(number, unit, expected):
    from app.agents.citation_provenance import given_check

    assert given_check([{"role": "user", "content": _TRENCH}])(number, unit) == expected


@pytest.mark.parametrize("message,number,unit,expected", [
    ("The formation level is -43 m. Raise it by 10 m.", -43.0, "m", "user"),
    ("The formation level is -43 m. Raise it by 10 m.", 43.0, "m", ""),
    ("The formation level is -43 m. Raise it by 10 m.", -43000.0, "mm", "user"),
    ("The formation level is -43 m. Raise it by 10 m.", -43.0, "mm", ""),
    ("Two small bars, 1 mm and 2 mm.", 1.0, "mm", "user"),
    ("Two small bars, 1 mm and 2 mm.", 1.5, "mm", ""),      # close in metres, not the same quantity
    ("Two small bars, 1 mm and 2 mm.", 0.001, "m", "user"),
])
def test_a_quantity_is_compared_at_the_givens_own_scale_with_its_sign(message, number, unit, expected):
    from app.agents.citation_provenance import given_check

    assert given_check([{"role": "user", "content": message}])(number, unit) == expected


@pytest.mark.parametrize("number,unit,users,tools,passages,expected", [
    (5.0, "m", [], [(5.0, "")], [], True),                  # a calculator's bare result, read as metres
    (5.0, "m", [], [(5.0, "mm")], [], False),               # a workflow wrote 5 mm
    (5000.0, "mm", [], [(5.0, "m")], [], True),
    (7.25, "m", [], [], [[(7.25, "m")]], True),             # a passage wrote it
    (7.25, "mm", [], [], [[(7.25, "m")]], False),
    (7.25, "m", [], [], [[(7.25, "")]], True),
    (7.25, "m", [], [], [[(7.25, None)]], False),           # a digit inside a unit gives nothing
    (7.25, "", [(7.25, "m")], [], [], True),                # no unit on the given: the number check
    (-7.25, "", [(7.25, "m")], [], [], False),
    (67.0, "currency", [(67.0, "SAR/m3")], [], [], True),   # a price per something is that number of money
    (67.0, "SAR", [], [], [[(67.0, "currency/t")]], True),
    (67.0, "currency", [(67.0, "SAR")], [], [], True),
    (67.0, "currency", [(67.0, "m")], [], [], False),       # a length is not money
    (67.0, "currency", [(67.0, "m3/day")], [], [], False),  # nor is a rate of anything else
    (68.0, "currency", [(67.0, "SAR/m3")], [], [], False),
    (67.0, "m3", [(67.0, "SAR/m3")], [], [], False),        # a price is not the thing it is per
    (67.0, "currency/m3", [(67.0, "SAR")], [], [], False),  # an amount is not a price per m3
])
def test_quantity_given_over_users_tools_and_passages(number, unit, users, tools, passages, expected):
    from app.agents.citation_provenance import _quantity_given

    assert _quantity_given(number, unit, users, tools, passages) is expected


@pytest.mark.parametrize("answer,number,unit,expected", [
    ("The formation level is -43 m.", -43.0, "m", "calculator"),
    ("The formation level is -43 m.", 43.0, "m", ""),          # the sign turned
    ("The formation level is -43 m.", 43.0, "", ""),
    ("The formation level is -43 m.", -43.0, "", "calculator"),
    ("Cut depth -7 m at chainage 120.", -7.0, "m", "calculator"),
    ("Cut depth -7 m at chainage 120.", 7.0, "m", ""),
])
def test_a_workflows_answer_is_read_with_its_signs_for_the_given_check(answer, number, unit, expected):
    import json

    from app.agents.citation_provenance import given_check

    messages = [
        {"role": "user", "content": "What is the level?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "w1", "type": "function", "function": {"name": "run_workflow", "arguments": "{}"}}]},
        {"role": "tool", "name": "run_workflow", "tool_call_id": "w1",
         "content": json.dumps({"ok": True, "workflow": "survey_levels", "answer": answer})},
    ]
    assert given_check(messages)(number, unit) == expected


@pytest.mark.parametrize("tool,payload", [
    ("read_memory", {"ok": True, "result": {"name": "datum", "data": {"level_m": 99.5}}}),
    ("list_memory", {"ok": True, "result": {"notes": [{"name": "datum", "summary": "level 99.5 m"}]}}),
    ("write_memory", {"ok": True, "result": {"name": "datum", "saved": True, "data": {"level_m": 99.5}}}),
])
def test_a_number_only_in_the_models_own_note_is_not_given(tool, payload):
    """The model's own notes back no figure: a number that is only in a note
    it wrote or read was given by nobody."""
    import json

    from app.agents.citation_provenance import given_check

    messages = [
        {"role": "user", "content": "What is the datum level?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "n1", "type": "function", "function": {"name": tool, "arguments": "{}"}}]},
        {"role": "tool", "name": tool, "tool_call_id": "n1", "content": json.dumps(payload)},
    ]
    check = given_check(messages)
    assert check(99.5, "m") == "" and check(99.5) == ""


@pytest.mark.parametrize("text,number,negative", [
    ("The new level is -33 m.", 33.0, True),
    ("The new level is 33 m.", 33.0, False),
    ("The new level is \u221233 m.", 33.0, True),          # the typographic minus
    ("The pipe runs 40-43 m along grid B.", 43.0, False),    # a range, not a minus
    ("A cut of 15 m x -0.2 m.", 0.2, True),
    ("An adjustment of -AED 5,000 this month.", 5000.0, True),   # the minus before the currency
    ("Costs AED 5,000 this month.", 5000.0, False),
])
def test_written_negative_reads_the_sign_a_figure_is_written_with(text, number, negative):
    from app.agents.citation_provenance import _figure_mentions, _written_negative

    mention = next(m for m, value in _figure_mentions(text) if value == number)
    assert _written_negative(text, mention) is negative


@pytest.mark.parametrize("answer", ["The formation level is -43 m.", "Cut depth -7 m at chainage 120."])
def test_workflow_answers_are_read_from_tool_runs_only(answer):
    """A workflow's answer is what a tool run returned. The same JSON typed
    by the user, or written by the model, is not a workflow's answer."""
    import json

    from app.agents.citation_provenance import _workflow_answers, build_evidence

    shaped = json.dumps({"ok": True, "workflow": "survey_levels", "answer": answer})
    typed = [{"role": "user", "content": shaped}, {"role": "assistant", "content": shaped}]
    assert _workflow_answers(build_evidence(None, typed, tool_passages=True)) == []
    ran = typed + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "w1", "type": "function", "function": {"name": "run_workflow", "arguments": "{}"}}]},
        {"role": "tool", "name": "run_workflow", "tool_call_id": "w1", "content": shaped},
        {"role": "tool", "name": "run_workflow", "tool_call_id": "w2",
         "content": json.dumps({"ok": False, "workflow": "survey_levels", "answer": "nothing"})},
        {"role": "tool", "name": "run_workflow", "tool_call_id": "w3", "content": "not json"},
    ]
    assert _workflow_answers(build_evidence(None, ran, tool_passages=True)) == [answer]


# --- a turn that carries a working sheet: no pairwise "working" ----------------

def _sheet_step_turn(user, givens, inputs, extra=None, entry="formula:arithmetic_step", **step):
    """The messages of a turn in which one step (an arithmetic step unless
    ``entry`` names another formula) ran on a working sheet, as the
    catalogue path leaves them: the user's message, the model's run_step
    call, and the tool's result carrying the sheet."""
    import json

    from app.agents.citation_provenance import given_check
    from app.agents.driver import sheet as ws

    messages = [{"role": "user", "content": user}]
    work, problems = ws.apply_update(ws.new_sheet("t1", user), {
        "givens": givens, "plan": [{"id": "s1", "entry": entry, "inputs": inputs, **step}]})
    assert problems == []
    work = ws.run_step(ws.mark_given(work, given_check(messages)), "s1", "c1")
    payload = {"ok": True, "step": work["steps"][0], "sheet": work}
    payload.update(extra or {})
    messages += [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_step", "arguments": "{\"step\": \"s1\"}"}}]},
        {"role": "tool", "name": "run_step", "tool_call_id": "c1", "content": json.dumps(payload)},
    ]
    return messages, work


def _level_turn(level, rise):
    messages, work = _sheet_step_turn(
        f"The formation level is {level} m. Raise it by {rise} m: what is the new level?",
        [{"id": "g1", "value": level, "unit": "m"}, {"id": "g2", "value": rise, "unit": "m"}],
        {"operator": "add", "a": "g1", "b": "g2"})
    assert work["steps"][0]["result"]["value"] == pytest.approx(level + rise)
    return messages


@pytest.mark.parametrize("level,rise,said", [
    (-43, 10, "33 m"), (-7, 2.5, "4.5 m"),        # the minus dropped in the reply
    (43, 10, "-53 m"), (7, 2.5, "-9.5 m"),        # a minus code never wrote
    (-43, 10, "53 m"), (43, 10, "33 m"),          # the sizes of the givens worked another way
])
def test_on_a_sheet_turn_a_step_result_with_its_sign_turned_is_withheld(level, rise, said):
    from app.agents.citation_provenance import figure_provenance

    messages = _level_turn(level, rise)
    out, entries = figure_provenance(f"The new level is {said}.", None, messages, tool_passages=True)
    assert said not in out and said.lstrip("-") not in out
    assert entries == []
    _same, record = figure_provenance(f"The new level is {said}.", None, messages, enforce=False,
                                      tool_passages=True)
    assert [e["source"] for e in record] == [None]


@pytest.mark.parametrize("level,rise,said", [
    (-43, 10, "-33 m"), (-7, 2.5, "-4.5 m"), (43, 10, "53 m"), (7, 2.5, "9.5 m"),
    (-43, 10, "\u221233 m"),                       # the typographic minus
])
def test_on_a_sheet_turn_a_step_result_with_the_sign_code_wrote_is_shown(level, rise, said):
    from app.agents.citation_provenance import figure_provenance

    answer = f"The new level is {said}."
    out, entries = figure_provenance(answer, None, _level_turn(level, rise), tool_passages=True)
    assert out == answer
    assert [(e["source"], e.get("step"), e.get("working")) for e in entries] == [("calculator", "s1", None)]


@pytest.mark.parametrize("level,rise,answer", [
    (-43, 10, "The new level is -33 m (\u221243 m + 10 m = \u221233 m)."),
    (-7, 2.5, "The new level is -4.5 m (\u22127 m + 2.5 m = \u22124.5 m)."),
    (43, 10, "The new level is 53 m (43 m + 10 m = 53 m)."),
])
def test_on_a_sheet_turn_a_result_restated_with_its_givens_is_shown_whole(level, rise, answer):
    from app.agents.citation_provenance import figure_provenance

    out, entries = figure_provenance(answer, None, _level_turn(level, rise), tool_passages=True)
    assert out == answer
    assert [(e["source"], e.get("step")) for e in entries] == [
        ("calculator", "s1"), ("user_input", None), ("user_input", None), ("calculator", "s1")]
    assert all(not e.get("working") for e in entries)


@pytest.mark.parametrize("a,b", [(-5, 5), (12.5, -12.5), (0, 0)])
def test_on_a_sheet_turn_a_step_result_of_zero_is_credited_to_its_step(a, b):
    """Zero is not a negative result: "0 m" carries no minus and is the
    step's own figure."""
    from app.agents.citation_provenance import figure_provenance

    messages = _level_turn(a, b)
    out, entries = figure_provenance("The new level is 0 m.", None, messages, tool_passages=True)
    assert out == "The new level is 0 m."
    assert [(e["figure"], e["source"], e.get("step")) for e in entries] == [("0 m", "calculator", "s1")]


@pytest.mark.parametrize("level,rise,worked", [(43, 10, "430 m"), (7, 2.5, "17.5 m")])
def test_pairwise_working_is_allowed_only_on_a_turn_without_a_working_sheet(level, rise, worked):
    """The same workflow result and the same reply, twice: a turn with no
    sheet credits a product of the user's own numbers as the calculator's
    working, as it always did; the turn that carries a sheet withholds it,
    because there the product would have been a step."""
    import json

    from app.agents.citation_provenance import figure_provenance

    user = f"The formation level is {level} m. Raise it by {rise} m: what is the new level?"
    ran = {"ok": True, "workflow": "survey_levels", "answer": f"The new level is {level + rise} m."}
    plain = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "w1", "type": "function", "function": {"name": "run_workflow", "arguments": "{}"}}]},
        {"role": "tool", "name": "run_workflow", "tool_call_id": "w1", "content": json.dumps(ran)},
    ]
    on_sheet, _work = _sheet_step_turn(
        user, [{"id": "g1", "value": level, "unit": "m"}, {"id": "g2", "value": rise, "unit": "m"}],
        {"operator": "add", "a": "g1", "b": "g2"}, extra=ran)
    answer = f"The new level is {level + rise} m. The two multiplied give {worked}."
    out, entries = figure_provenance(answer, None, plain, tool_passages=True)
    assert out == answer
    assert [(e["figure"], e.get("working")) for e in entries] == [(f"{level + rise} m", None), (worked, True)]
    out, entries = figure_provenance(answer, None, on_sheet, tool_passages=True)
    assert worked not in out and f"{level + rise} m" in out
    assert [e["figure"] for e in entries] == [f"{level + rise} m"]


# --- a step's result is credited only to a figure whose unit agrees with it ---

@pytest.mark.parametrize("tail,unit", [
    (" m long, 3 m wide", "m"), ("mm bars", "mm"), (" kN/m2.", "kN/m2"), (" working days of float", "working days"),
    (" kg per m3", "kg per m3"),                   # a rate: "per" is no unit of its own
    (" m in depth", "m"), (" m in", "m"),          # two units side by side are not one: not minutes
    (" m. In all", "m"), (" mm in total", "mm"),
    (" long", ""), (" per m3", ""), (", then 4", ""), ("", ""), (None, ""),
])
def test_unit_written_after_reads_the_unit_that_follows_a_figure(tail, unit):
    from app.agents.citation_provenance import _unit_written_after

    assert _unit_written_after(tail) == unit


@pytest.mark.parametrize("message,number,unit,expected", [
    ("The trench is 12 m in length and 2 m in depth.", 12.0, "m", "user"),
    ("The trench is 12 m in length and 2 m in depth.", 2000.0, "mm", "user"),
    ("The trench is 12 m in length and 2 m in depth.", 12.0, "min", ""),     # nobody gave minutes
    ("The pour took 12 min in all.", 12.0, "min", "user"),
    ("The pour took 12 min in all.", 12.0, "m", ""),
])
def test_given_check_reads_a_unit_followed_by_a_word_that_is_also_a_unit(message, number, unit, expected):
    from app.agents.citation_provenance import given_check

    assert given_check([{"role": "user", "content": message}])(number, unit) == expected


@pytest.mark.parametrize("text,number,unit", [
    ("The new level is 207 m.", 207.0, "m"),
    ("The area is 48 m*m.", 48.0, "m*m"),                       # the whole token, not the "m" the pattern knows
    ("A density of 5 kN/m3 of fill.", 5.0, "kN/m3"),
    ("That is **120 m³** of spoil.", 120.0, "m³"),    # markdown marks are not part of the unit
    ("It is 12 m, then more.", 12.0, "m"),                 # a no-break space before the unit
    ("It is 33 m—that is all.", 33.0, "m"),
    ("A waste of 25%.", 25.0, "%"),
    ("An output of 3 m3/day.", 3.0, "m3/day"),
    ("A 40 m-long trench.", 40.0, "m-long"),
    ("It costs $8,040 in total.", 8040.0, "currency"),          # money: the currency stands before the figure
    ("It costs AED 450/m2 laid.", 450.0, "currency/m2"),
    ("It costs AED 450 per m2 laid.", 450.0, "currency/m2"),
    ("It costs AED 450 per square metre.", 450.0, "currency"),  # per what is in words the converter cannot read
])
def test_unit_written_with_reads_the_unit_a_figure_is_written_with(text, number, unit):
    from app.agents.citation_provenance import _figure_mentions, _unit_written_with

    mention = next(m for m, value in _figure_mentions(text) if value == number)
    assert _unit_written_with(text, mention) == unit


@pytest.mark.parametrize("value,written,size,unit,pure,expected", [
    (207.0, "m", 207.0, "m", False, True),
    (0.207, "km", 207.0, "m", False, True),                 # the same quantity at another scale
    (20700.0, "cm", 207.0, "m", False, True),
    (72000.0, "L", 72.0, "m3", False, True),
    (207.0, "mm", 207.0, "m", False, False),                # the same number in another unit
    (0.207, "m", 207.0, "m", False, False),                 # the same unit, another number
    (207.0, "kg", 207.0, "m", False, False),                # another kind
    (207.0, "m2", 207.0, "m", False, False),
    (207.0, "", 207.0, "m", False, False),                  # no unit written for a result in m
    (12.0, "tons", 12.0, "t", False, False),                # a unit the converter does not know
    (207.0, "m", 207.0, "-", True, False),                  # a pure number given a unit
    (207.0, "currency", 207.0, "-", True, False),
    (207.0, "", 207.0, "-", True, True),                    # ... and written with none
    (208.0, "", 207.0, "-", True, False),
    (8.0, "days", 8.0, "-", False, True),                   # no unit declared for the output: the number alone
    (8000.0, "mm", 8.0, "-", False, True),
    (9.0, "days", 8.0, "-", False, False),
    (48.0, "m2", 48.0, "m*m", False, True),                 # a product of units, multiplied out: an area
    (48.0, "m", 48.0, "m*m", False, False),
    (49.0, "m2", 48.0, "m*m", False, False),
    (150.0, "m", 150.0, "m*m", False, False),
    (150.0, "sqm", 150.0, "m*m", False, True),
    (3.0, "days", 3.0, "m2/(m2/day)", False, True),         # a quantity over its rate: a time
    (72.0, "hours", 3.0, "m2/(m2/day)", False, True),
    (3.0, "m", 3.0, "m2/(m2/day)", False, False),
    (3.0, "hours", 3.0, "m2/(m2/day)", False, False),
    (120.0, "kN", 120.0, "kN/m2*m2", False, True),          # a pressure over an area: a force
    (120.0, "kPa", 120.0, "kN/m2*m2", False, False),
    (90.0, "kN.m", 90.0, "kN.m", False, True),              # no family: compared with itself
    (90000.0, "N.m", 90.0, "kN.m", False, True),
    (90.0, "kN", 90.0, "kN.m", False, False),
    (90.0, "kN.m.s", 90.0, "kN.m.s.foo", False, False),     # neither side compares: the unit as written
    (90.0, "kN.m.foo", 90.0, "kN.m.foo", False, True),
    (90.0, "", 90.0, "kN.m.foo", False, False),             # ... never the number alone
    (68.0, "°F", 20.0, "°C", False, True),                  # an offset unit converts with its offset
    (20.0, "°F", 20.0, "°C", False, False),
    (20.0, "°C", 68.0, "°F", False, True),
    (20.0, "°C", 20.0, "°F", False, False),
    (8040.0, "currency", 8040.0, "currency", False, True),
    (8040.0, "m", 8040.0, "currency", False, False),
    (450.0, "currency/m2", 450.0, "currency/m2", False, True),
    (450.0, "currency", 450.0, "currency/m2", False, True),     # money by its currency alone states a money rate
    (451.0, "currency", 450.0, "currency/m2", False, False),
    (450.0, "currency/t", 450.0, "currency/m2", False, False),  # per something else
    (450.0, "currency/m2", 450.0, "currency", False, False),
])
def test_states_result_holds_a_figure_to_the_unit_of_the_step_result(value, written, size, unit, pure, expected):
    from app.agents.citation_provenance import _states_result

    assert _states_result(value, written, size, unit, pure) is expected


_RISE_IN_MM = "The level is {level} m. Raise it by {rise} mm: what is the new level?"


@pytest.mark.parametrize("level,rise,said", [
    (-43, 250, "207 m"), (-43, 250, "207 mm"), (12, 500, "512 m"), (12, 500, "512 kg"), (12, 500, "AED 512"),
])
def test_on_a_sheet_turn_a_pure_number_result_written_with_a_unit_is_withheld(level, rise, said):
    """Both givens written as pure numbers: each number is the user's, the
    step adds them, and the result is a pure number. No unit is its unit."""
    from app.agents.citation_provenance import figure_provenance

    messages, work = _sheet_step_turn(
        _RISE_IN_MM.format(level=level, rise=rise),
        [{"id": "g1", "value": level, "unit": "-"}, {"id": "g2", "value": rise, "unit": "-"}],
        {"operator": "add", "a": "g1", "b": "g2"})
    assert [g.get("checked") for g in work["givens"]] == ["user", "user"]
    assert work["steps"][0]["result"] == {"name": "result", "value": float(level + rise), "unit": "-"}
    out, entries = figure_provenance(f"The new level is {said}.", None, messages, tool_passages=True)
    assert said not in out
    assert entries == []
    _same, record = figure_provenance(f"The new level is {said}.", None, messages, enforce=False,
                                      tool_passages=True)
    assert [e["source"] for e in record] == [None]


@pytest.mark.parametrize("level,rise,said,credited", [
    (200, 7, "207 m", True), (200, 7, "0.207 km", True), (200, 7, "20700 cm", True),
    (200, 7, "207 mm", False), (200, 7, "0.207 m", False), (200, 7, "207 kg", False), (200, 7, "207 m2", False),
    (-43, 10, "-33 m", True), (-43, 10, "-0.033 km", True), (-43, 10, "-33 mm", False), (-43, 10, "-33 km", False),
])
def test_on_a_sheet_turn_a_step_result_is_credited_only_in_a_unit_that_agrees(level, rise, said, credited):
    from app.agents.citation_provenance import figure_provenance

    answer = f"The new level is {said}."
    out, entries = figure_provenance(answer, None, _level_turn(level, rise), tool_passages=True)
    assert (out == answer) is credited
    assert (said in out) is credited
    assert [(e["source"], e.get("step"), e.get("unit")) for e in entries] == (
        [("calculator", "s1", "m")] if credited else [])


@pytest.mark.parametrize("late,early,said", [(20, 12, "8 days"), (31, 30.5, "0.5 days"), (20, 12, "8 m")])
def test_on_a_sheet_turn_an_output_with_no_declared_unit_is_held_to_its_number_alone(late, early, said):
    """The registry declares no unit for this output, and the formula does
    not return one: nothing says what unit its figure carries, so the unit
    the reply writes is not held (the last case is what that leaves open)."""
    from app.agents.citation_provenance import figure_provenance

    messages, work = _sheet_step_turn(
        f"Early start is {early}, early finish {early + 3}, late start {late} and late finish {late + 3}. "
        "What is the total float?",
        [{"id": "g1", "value": early, "unit": "-"}, {"id": "g2", "value": early + 3, "unit": "-"},
         {"id": "g3", "value": late, "unit": "-"}, {"id": "g4", "value": late + 3, "unit": "-"}],
        {"early_start": "g1", "early_finish": "g2", "late_start": "g3", "late_finish": "g4"},
        entry="formula:critical_path_float", output="total_float")
    assert work["steps"][0]["result"] == {"name": "total_float", "value": float(late - early), "unit": "-"}
    answer = f"The total float is {said}."
    out, entries = figure_provenance(answer, None, messages, tool_passages=True)
    assert out == answer
    assert [(e["source"], e.get("step")) for e in entries] == [("calculator", "s1")]


@pytest.mark.parametrize("udl,span,said,credited", [
    (20, 6, "90 kN.m", True), (20, 6, "90 kN", False), (20, 6, "91 kN.m", False), (8, 5, "25 kN.m", True),
    (8, 5, "25 kN", False), (8, 5, "25 m", False),
])
def test_on_a_sheet_turn_a_moment_is_credited_only_in_a_unit_of_a_moment(udl, span, said, credited):
    """A moment in kN.m: no family of the converter measures it, so the
    converter multiplies it out to its powers of the base quantities, and a
    figure in a force or a length is not that moment."""
    from app.agents.citation_provenance import figure_provenance

    messages, work = _sheet_step_turn(
        f"A simply supported beam carries {udl} kN/m over a span of {span} m. What is the maximum moment?",
        [{"id": "g1", "value": udl, "unit": "kN/m"}, {"id": "g2", "value": span, "unit": "m"}],
        {"udl_w_kn_m": "g1", "span_m": "g2"}, entry="formula:beam_moment_simple")
    assert work["steps"][0]["result"]["unit"] == "kN.m"
    answer = f"The maximum moment is {said}."
    out, entries = figure_provenance(answer, None, messages, tool_passages=True)
    assert (said in out) is credited
    assert [e.get("step") for e in entries] == (["s1"] if credited else [])


# --- a figure's magnitude word and a compound unit -----------------------------

@pytest.mark.parametrize("unit,dimension,scale", [
    ("m*m", "area", 1.0), ("m·mm", "area", 0.001), ("m×m×m", "volume", 1.0),
    ("m2/(m2/day)", "time", 24.0), ("m3/(m3/h)", "time", 1.0), ("kN/m2*m2", "force", 1000.0),
    ("SAR/m3*m3", "currency", 1.0), ("m2/m2", "fraction", 1.0),
    ("kPa*m2", "force", 1000.0), ("MPa*mm2", "force", 1.0),             # a pressure on an area
    ("kN.m", "[m^2 kg^1 s^-2]", 1000.0), ("N*mm", "[m^2 kg^1 s^-2]", 0.001), ("m*h", "[m^1 s^1]", 3600.0),
])
def test_parse_compound_unit_multiplies_out_a_product_or_quotient_of_units(unit, dimension, scale):
    from app.agents.base.formulas.construction_formulas_planning import parse_compound_unit

    parsed = parse_compound_unit(unit)
    assert (parsed.dimension, parsed.offset) == (dimension, 0.0)
    assert parsed.scale == pytest.approx(scale)


@pytest.mark.parametrize("unit", [
    "", None, "m*", "*m", "m*(m", "m*m)", "(m*m", "m*long", "m*working days", "°F*m", "m//m", "m**m",
])
def test_parse_compound_unit_is_none_for_what_does_not_multiply_out(unit):
    """A part that is no unit, a family that does not multiply (an offset
    unit, working time that needs a calendar), or brackets that do not
    close: no unit, so nothing is compared by it."""
    from app.agents.base.formulas.construction_formulas_planning import parse_compound_unit

    assert parse_compound_unit(unit) is None


@pytest.mark.parametrize("text,factor,taken", [
    ("It costs AED 1,225 thousand.", 1000.0, len(" thousand")),
    ("It costs SAR 8.04k in all.", 1000.0, len("k")),
    ("It costs AED 1.2 million.", 1e6, len(" million")),
    ("It costs AED 1.2 mn.", 1e6, len(" mn")),          # money has no meganewton
    ("It costs USD 3 bn.", 1e9, len(" bn")),
    ("It costs AED 1,225.", 1.0, 0),
    ("It is 12 km long.", 1.0, 0),                       # a unit, not a magnitude
    ("It carries 12 kN.", 1.0, 0),
    ("It costs AED 12 per m3.", 1.0, 0),
])
def test_magnitude_written_reads_the_word_after_a_figure(text, factor, taken):
    from app.agents.citation_provenance import _figure_mentions, _magnitude_written

    (mention, _value), *_rest = _figure_mentions(text)
    assert _magnitude_written(text, mention) == (factor, taken)


@pytest.mark.parametrize("word,money,factor", [
    ("mn", False, 1.0), ("mn", True, 1e6),      # a meganewton, unless money is written
    ("k", False, 1e3), ("k", True, 1e3),        # no unit of the converter: a magnitude either way
    ("kg", True, 1.0), ("kg", False, 1.0),
])
def test_magnitude_written_leaves_a_unit_word_to_the_unit_unless_money_is_written(word, money, factor):
    import re

    from app.agents.citation_provenance import _magnitude_written

    text = f"{'AED ' if money else ''}12 {word} more"
    mention = re.compile(r"(?P<pre>AED )?(?P<num>12)").match(text)
    assert _magnitude_written(text, mention)[0] == factor


@pytest.mark.parametrize("text,unit", [
    ("It costs AED 1,225 thousand.", "currency"),
    ("It costs AED 8.04k per m2.", "currency/m2"),
    ("It costs AED 1.5 million/t.", "currency/t"),
    ("It costs AED 1,225 thousand per m3.", "currency/m3"),
])
def test_unit_written_with_reads_past_a_magnitude_word(text, unit):
    from app.agents.citation_provenance import _figure_mentions, _unit_written_with

    (mention, _value), *_rest = _figure_mentions(text)
    assert _unit_written_with(text, mention) == unit


def _cost_turn(quantity, rate):
    messages, work = _sheet_step_turn(
        f"Excavate {quantity:,} m3 at {rate} per m3. What does it cost?",
        [{"id": "g1", "value": quantity, "unit": "m3"}, {"id": "g2", "value": rate, "unit": "-"}],
        {"quantity": "g1", "unit_rate": "g2"}, entry="formula:unit_cost_total")
    assert work["steps"][0]["result"]["value"] == pytest.approx(quantity * rate)
    return messages


@pytest.mark.parametrize("quantity,rate,said,credited", [
    (1250, 980, "AED 1,225,000", True), (1250, 980, "AED 1,225 thousand", True),
    (1250, 980, "AED 1,225k", True), (1250, 980, "AED 1.225 million", True), (1250, 980, "AED 1.225 mn", True),
    (1250, 980, "AED 1,225", False),                  # a thousand times too small
    (1250, 980, "AED 1,225 million", False),          # a thousand times too large
    (1250, 980, "AED 1,300 thousand", False),
    (670, 12, "SAR 8,040", True), (670, 12, "SAR 8.04k", True), (670, 12, "SAR 8.04 thousand", True),
    (670, 12, "SAR 8.04", False), (670, 12, "SAR 8.04 million", False), (670, 12, "SAR 8,040k", False),
])
def test_on_a_sheet_turn_money_written_with_its_magnitude_word_states_the_step_result(quantity, rate, said,
                                                                                      credited):
    """The magnitude word is part of how the figure is written: "AED 1,225
    thousand" is 1,225,000 and states that result; "AED 1,225" does not."""
    from app.agents.citation_provenance import figure_provenance

    answer = f"The cost is {said}."
    out, entries = figure_provenance(answer, None, _cost_turn(quantity, rate), tool_passages=True)
    assert (said in out) is credited
    assert [e.get("step") for e in entries] == (["s1"] if credited else [])


@pytest.mark.parametrize("a,b,said,credited", [
    (15, 10, "150 m2", True), (15, 10, "150 m", False),
    (12, 4, "48 sqm", True), (12, 4, "48 m", False), (12, 4, "48 m3", False),
])
def test_on_a_sheet_turn_a_product_of_two_lengths_is_credited_only_as_an_area(a, b, said, credited):
    from app.agents.citation_provenance import figure_provenance

    messages, work = _sheet_step_turn(
        f"A slab is {a} m by {b} m. What is its area?",
        [{"id": "g1", "value": a, "unit": "m"}, {"id": "g2", "value": b, "unit": "m"}],
        {"operator": "multiply", "a": "g1", "b": "g2"})
    assert work["steps"][0]["result"]["value"] == pytest.approx(a * b)
    answer = f"The area is {said}."
    out, entries = figure_provenance(answer, None, messages, tool_passages=True)
    assert (said in out) is credited
    assert [e.get("step") for e in entries] == (["s1"] if credited else [])


@pytest.mark.parametrize("area,rate,said,credited", [
    (150, 50, "3 days", True), (150, 50, "72 hours", True), (150, 50, "3 m", False), (150, 50, "3 hours", False),
    (240, 60, "4 days", True), (240, 60, "4 m2", False),
])
def test_on_a_sheet_turn_a_quantity_over_its_rate_is_credited_only_as_a_time(area, rate, said, credited):
    from app.agents.citation_provenance import figure_provenance

    messages, work = _sheet_step_turn(
        f"Lay {area} m2 at {rate} m2/day. How long does it take?",
        [{"id": "g1", "value": area, "unit": "m2"}, {"id": "g2", "value": rate, "unit": "m2/day"}],
        {"operator": "divide", "a": "g1", "b": "g2"})
    assert work["steps"][0]["result"]["value"] == pytest.approx(area / rate)
    answer = f"It takes {said}."
    out, entries = figure_provenance(answer, None, messages, tool_passages=True)
    assert (said in out) is credited
    assert [e.get("step") for e in entries] == (["s1"] if credited else [])


@pytest.mark.parametrize("message,number,unit,expected", [
    ("The slab is 20 m long.", 20.0, "°F", ""),           # a temperature is not a length
    ("The slab is 20 m long.", 20.0, "kelvin", ""),
    ("It was cured at 20 °C.", 20.0, "°F", ""),          # 20 °C is 68 °F
    ("It was cured at 20 °C.", 68.0, "°F", "user"),
    ("It was cured at 30 °C.", 86.0, "°F", "user"),
    ("It was cured at 30 °C.", 30.0, "°F", ""),
    ("It was cured at 30 °C.", 30.0, "°C", "user"),
])
def test_a_temperature_is_given_only_at_its_own_offset(message, number, unit, expected):
    from app.agents.citation_provenance import given_check

    assert given_check([{"role": "user", "content": message}])(number, unit) == expected
