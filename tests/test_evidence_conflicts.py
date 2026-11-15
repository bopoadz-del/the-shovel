"""C2-C4: conflicts between documents in an answer's evidence.

Every input here is invented for this file and differs from the C1 exam's
cases (tests/conflict_exam/cases.json); expected values are written by hand.
Each decision is shown on two inputs or more, one each way where it has two.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.evidence_conflicts import detect, extract, same


def _row(doc: str, text: str, sec: str = "p. 1") -> dict:
    return {"doc_name": doc, "page_or_section": sec, "snippet": text}


# ── normalisation: values that are the same, and values that are not ───────

@pytest.mark.parametrize("a, b", [
    ("Wall height 3.6 m.", "Wall height 3600 mm."),
    ("Topsoil strip 450 m³.", "Topsoil strip 450 m3."),
    ("Curing period 2 weeks.", "Curing period 14 days."),
    ("Bearing pressure 150 kPa.", "Bearing pressure 0.15 N/mm2."),
    ("Budget USD 1.2 million.", "Budget USD 1,200,000."),
    ("Handover on 3 November 2029.", "Handover on 2029-11-03."),
    ("Warranty period 1 year.", "Warranty period 12 months."),
])
def test_same_after_normalisation(a, b):
    va, vb = extract(a), extract(b)
    assert len(va) == 1 and len(vb) == 1
    assert same(va[0], vb[0])


@pytest.mark.parametrize("a, b", [
    ("Wall height 3.6 m.", "Wall height 3500 mm."),
    ("Curing period 2 weeks.", "Curing period 10 days."),
    ("Budget USD 1.2 million.", "Budget USD 1,250,000."),
    ("Handover on 3 November 2029.", "Handover on 2029-11-30."),
    ("Warranty period 1 year.", "Warranty period 18 months."),
    ("Budget USD 1,000.", "Budget EUR 1,000."),
])
def test_different_after_normalisation(a, b):
    va, vb = extract(a), extract(b)
    assert len(va) == 1 and len(vb) == 1
    assert not same(va[0], vb[0])


@pytest.mark.parametrize("text", ["Inspection on 04/05/2030.", "Inspection on 07/08/2031."])
def test_ambiguous_numeric_date_is_no_value(text):
    assert [v for v in extract(text) if v.kind == "date"] == []


# ── detection ───────────────────────────────────────────────────────────────

_DIFFER = [
    ("The canopy steel is grade S355.",
     [_row("Quayside Steel Specification.pdf", "Canopy steelwork shall be grade S355."),
      _row("Quayside Canopy Drawing S-501.pdf", "Canopy steelwork: grade S275.")],
     {("Quayside Steel Specification.pdf", "S355"), ("Quayside Canopy Drawing S-501.pdf", "S275")}),
    ("The car park ramp gradient is 8%.",
     [_row("Northgate Design Statement.pdf", "The car park ramp gradient is 8%."),
      _row("Northgate Ramp Section A-330.pdf", "Car park ramp gradient 10% maximum.")],
     {("Northgate Design Statement.pdf", "8%"), ("Northgate Ramp Section A-330.pdf", "10%")}),
]


@pytest.mark.parametrize("answer, rows, pairs", _DIFFER)
def test_differing_documents_are_a_conflict(answer, rows, pairs):
    found = detect(answer, rows)
    assert len(found) == 1
    assert {(v["document"], v["value"]) for v in found[0]["values"]} == pairs
    assert found[0]["governs"] is None


_NO_CONFLICT = [
    # agree after a unit change
    ("The parapet is 1.1 m high.",
     [_row("Eastfield Specification.pdf", "Parapet height 1100 mm."),
      _row("Eastfield Roof Drawing A-410.pdf", "Parapet height 1.1 m.")]),
    # a different element in the second document
    ("The lift pit depth is 1.5 m.",
     [_row("Eastfield Specification.pdf", "Lift pit depth 1.5 m."),
      _row("Eastfield Stair Drawing A-520.pdf", "Stair landing width 1.8 m.")]),
    # the second value was never stated in the answer and is about another thing
    ("Retention is 10%.",
     [_row("Eastfield Contract Data.pdf", "Retention: 10%."),
      _row("Eastfield Tender Appendix.pdf", "Advance payment: 15%.")]),
    # one document only
    ("The pile length is 18 m.",
     [_row("Eastfield Piling Report.pdf", "Pile length 18 m. Pile length 24 m at the north wall.")]),
]


@pytest.mark.parametrize("answer, rows", _NO_CONFLICT)
def test_no_conflict(answer, rows):
    assert detect(answer, rows) == []


@pytest.mark.parametrize("answer", ["", "No figure here."])
def test_answer_without_a_value_is_no_conflict(answer):
    rows = [_row("A Specification.pdf", "Slab depth 200 mm."), _row("A Drawing.pdf", "Slab depth 250 mm.")]
    assert detect(answer, rows) == []


@pytest.mark.parametrize("rows", [None, [{"snippet": "x"}], "not rows"])
def test_bad_evidence_never_raises(rows):
    assert detect("Slab depth 200 mm.", rows) == []


# ── precedence (C3): read from the evidence, never built in ─────────────────

_PREC_ROWS = [
    _row("Quayside Steel Specification.pdf", "Canopy steelwork shall be grade S355."),
    _row("Quayside Canopy Drawing S-501.pdf", "Canopy steelwork: grade S275."),
]


@pytest.mark.parametrize("statement, governs", [
    ("Should the Drawings and the Specification disagree, the Drawings shall prevail.",
     "Quayside Canopy Drawing S-501.pdf"),
    ("Where the Specification and the Drawings conflict, the Specification shall govern.",
     "Quayside Steel Specification.pdf"),
])
def test_stated_precedence_names_the_winner(statement, governs):
    rows = _PREC_ROWS + [_row("Quayside General Conditions.pdf", statement)]
    found = detect("The canopy steel is grade S355.", rows)
    assert found and found[0]["governs"] == governs
    assert found[0]["basis_document"] == "Quayside General Conditions.pdf"


@pytest.mark.parametrize("statement", [
    "Where the Bill of Quantities and the Programme differ, the Programme shall prevail.",
    "The Contractor shall keep the site tidy.",
    # an ordered list that is not a precedence statement
    "The documents are listed in order of issue: Specification, Drawings.",
    "Submit the copies in this order: Drawings, Specification.",
])
def test_statement_not_covering_the_documents_names_no_winner(statement):
    rows = _PREC_ROWS + [_row("Quayside General Conditions.pdf", statement)]
    found = detect("The canopy steel is grade S355.", rows)
    assert found and found[0]["governs"] is None


@pytest.mark.parametrize("first, second", [
    ("Should the Drawings and the Specification disagree, the Drawings shall prevail.",
     "Where the Specification and the Drawings conflict, the Specification shall govern."),
    ("Where the Specification and the Drawings conflict, the Specification shall govern.",
     "Should the Drawings and the Specification disagree, the Drawings shall prevail."),
])
def test_contradictory_precedence_statements_name_no_winner(first, second):
    rows = _PREC_ROWS + [_row("Quayside General Conditions.pdf", first),
                         _row("Quayside Particular Conditions.pdf", second)]
    found = detect("The canopy steel is grade S355.", rows)
    assert found and found[0]["governs"] is None
    assert found[0]["basis_document"] is None


# ── what links an answer to the evidence ────────────────────────────────────

@pytest.mark.parametrize("answer, rows, pairs", [
    # the only shared word is a plural in the evidence and a singular in the answer
    ("Gully spacing is 30 m.",
     [_row("Westbank Drainage Specification.pdf", "Gullies at 30 m."),
      _row("Westbank Drainage Layout D-210.pdf", "Gullies at 25 m.")],
     {("Westbank Drainage Specification.pdf", "30 m"), ("Westbank Drainage Layout D-210.pdf", "25 m")}),
    ("The gallery clear height is 4.2 m.",
     [_row("Westbank Museum Brief.pdf", "Galleries: 4.2 m."),
      _row("Westbank Section A-301.pdf", "Galleries: 3.9 m.")],
     {("Westbank Museum Brief.pdf", "4.2 m"), ("Westbank Section A-301.pdf", "3.9 m")}),
    # a labelled clause with no number in it
    ("The roof membrane is torch-on bitumen.",
     [_row("Westbank Roofing Specification.pdf", "Roof membrane: torch-on bitumen."),
      _row("Westbank Roof Plan A-120.pdf", "Roof membrane: liquid polyurethane.")],
     {("Westbank Roofing Specification.pdf", "torch-on bitumen"),
      ("Westbank Roof Plan A-120.pdf", "liquid polyurethane")}),
    ("Door ironmongery finish is satin stainless steel.",
     [_row("Westbank Door Schedule.pdf", "Door ironmongery finish: satin stainless steel."),
      _row("Westbank Hardware Specification.pdf", "Door ironmongery finish: polished brass.")],
     {("Westbank Door Schedule.pdf", "satin stainless steel"),
      ("Westbank Hardware Specification.pdf", "polished brass")}),
])
def test_conflict_found_through_word_forms_and_labelled_clauses(answer, rows, pairs):
    found = detect(answer, rows)
    assert len(found) == 1
    assert {(v["document"], v["value"]) for v in found[0]["values"]} == pairs


@pytest.mark.parametrize("answer, rows", [
    # evidence about another thing entirely: no word of the answer in either document
    ("The slab depth is 200 mm.",
     [_row("Westbank Kerb Detail.pdf", "Kerb upstand 200 mm."),
      _row("Westbank External Works.pdf", "Kerb upstand 150 mm.")]),
    ("The handrail height is 1.1 m.",
     [_row("Westbank Fence Detail.pdf", "Boundary fence 1.1 m."),
      _row("Westbank Landscape Plan.pdf", "Boundary fence 1.8 m.")]),
    # a passage with no document name is not a document that can disagree
    ("The slab depth is 200 mm.",
     [_row("Westbank Structural Specification.pdf", "Slab depth 200 mm."),
      _row("Westbank Slab Drawing S-110.pdf", "Slab depth 200 mm."),
      {"page_or_section": "p. 4", "snippet": "Slab depth 250 mm."}]),
    ("The kerb upstand is 150 mm.",
     [_row("Westbank External Works.pdf", "Kerb upstand 150 mm."),
      {"doc_name": "", "snippet": "Kerb upstand 125 mm."},
      _row("Westbank Kerb Detail.pdf", "Kerb upstand 150 mm.")]),
])
def test_unlinked_or_unnamed_evidence_is_no_conflict(answer, rows):
    assert detect(answer, rows) == []


@pytest.mark.parametrize("answer", [123, b"Slab depth 200 mm."])
def test_non_text_answer_is_no_conflict_and_no_error(answer, caplog):
    rows = [_row("A Specification.pdf", "Slab depth 200 mm."), _row("A Drawing.pdf", "Slab depth 250 mm.")]
    with caplog.at_level("WARNING", logger="app.agents.evidence_conflicts"):
        assert detect(answer, rows) == []
    assert caplog.records == []


# ── C4: the answer contract carries it ──────────────────────────────────────

@pytest.mark.parametrize("value_rows, statement_at, winner_value, winner_doc", [
    # the precedence statement sits inside one of the two documents that disagree
    ([("Quayside Steel Specification.pdf", "Canopy steelwork shall be grade S355."),
      ("Quayside Canopy Drawing S-501.pdf", "Canopy steelwork: grade S275.")],
     0, "S355", "Quayside Steel Specification.pdf"),
    ([("Quayside Steel Specification.pdf", "Canopy steelwork shall be grade S355."),
      ("Quayside Canopy Drawing S-501.pdf", "Canopy steelwork: grade S275.")],
     1, "S275", "Quayside Canopy Drawing S-501.pdf"),
])
def test_contract_says_which_value_governs_and_cites_each_document_once(
        value_rows, statement_at, winner_value, winner_doc):
    statements = ["Where the Specification and the Drawings conflict, the Specification shall govern.",
                  "Should the Drawings and the Specification disagree, the Drawings shall prevail."]
    rows = [_row(doc, text + (" " + statements[i] if i == statement_at else ""))
            for i, (doc, text) in enumerate(value_rows)]
    res = answer_contract.apply("The canopy steel is grade S355.", sources=rows)
    assert len(res.conflicts) == 1
    assert res.conflicts[0]["governs"] == winner_doc
    assert f"{winner_value} governs" in res.text
    assert "No document states" not in res.text
    # the basis is one of the two documents: it keeps that document's marker
    docs = [e["document"] for e in res.citations]
    assert sorted(docs) == sorted(doc for doc, _t in value_rows)

@pytest.mark.parametrize("answer, rows, pairs", _DIFFER)
def test_contract_states_every_value_and_carries_the_conflict(answer, rows, pairs):
    res = answer_contract.apply(answer, sources=rows)
    assert len(res.conflicts) == 1
    assert {(v["document"], v["value"]) for v in res.conflicts[0]["values"]} == pairs
    for _doc, value in pairs:
        assert value in res.text
    # sources travel as markers, never as file names in the text
    for doc, _value in pairs:
        assert doc not in res.text
    markers = {e["marker"] for e in res.citations}
    assert {v["marker"] for v in res.conflicts[0]["values"]} <= markers


@pytest.mark.parametrize("answer, rows", _NO_CONFLICT)
def test_contract_without_conflict_is_unchanged(answer, rows):
    res = answer_contract.apply(answer, sources=rows)
    assert res.conflicts == []
    assert res.text == answer
