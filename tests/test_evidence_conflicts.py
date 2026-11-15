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
])
def test_statement_not_covering_the_documents_names_no_winner(statement):
    rows = _PREC_ROWS + [_row("Quayside General Conditions.pdf", statement)]
    found = detect("The canopy steel is grade S355.", rows)
    assert found and found[0]["governs"] is None


# ── C4: the answer contract carries it ──────────────────────────────────────

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
