"""A document answer's later paragraph sourced only from other documents is context.

Rule 2 bounded padding by shape (offers, restatements, notes, closing
requests) but nothing bounded a document answer that, after answering in
its first paragraph, went on to a paragraph drawn from another document
("The grade itself is defined in the specification ... [2]"). The contract
now reads every later paragraph of a turn that did not compute and is not
asking:

* it goes when every citation in it names a document the first paragraph
  does not cite and it states no value of a kind the first paragraph
  states; its citation entry goes with it and the markers left are
  renumbered with no gap;
* it stays when it cites a document the first paragraph cites, when it
  cites nothing, when it states a value of a kind the first paragraph
  states (a comparison or a conflict), when it is a list item or table row,
  when the first paragraph cites nothing or introduces a list, on a turn
  that computed, and on an asking turn.

Every input here is written by the test; none is a project file or an exam
question.
"""
from __future__ import annotations

import pytest

from app.agents import answer_contract


def _apply(text, **kw):
    return answer_contract.apply(text, sources=[], **kw)


# ── context from another document, no value of the lead's kind: it goes ────

@pytest.mark.parametrize("text, gone, document", [
    (("The billed area of the podium screed is 4,180 m² [source: Podium Ledger.xlsx].\n\n"
      "The screed mix is specified at 25 N/mm² with a 50 mm minimum depth [source: Screed Specification.pdf]."),
     "screed mix", "Podium Ledger.xlsx"),
    (("No unit rate for the louvre panels is in the priced ledger [source: Annex Ledger.xlsx].\n\n"
      "The louvre finish is set out in the facade schedule as anodised, 1.5 mm thick [source: Facade Schedule.pdf]."),
     "anodised", "Annex Ledger.xlsx"),
])
def test_other_source_context_goes(text, gone, document):
    out = _apply(text)
    # The answer is the first paragraph exactly as the contract renders it on
    # its own: the context paragraph goes and nothing else changes. Comparing
    # with that rendering, not with a literal, keeps the other rules' wording
    # of the lead (absence form, sentence marks) out of this rule's test.
    assert out.text == _apply(text.split("\n\n")[0]).text
    assert gone not in out.text and "[1]" in out.text
    assert [c["document"] for c in out.citations] == [document]
    assert [c["marker"] for c in out.citations] == [1]


@pytest.mark.parametrize("text, expect, documents", [
    (("Gate G2 opens on 3 May 2027 [source: Gate Register.pdf].\n\n"
      "The gate leaf is 6 m wide [source: Gate Drawing.pdf].\n\n"
      "The register is reissued monthly [source: Gate Register.pdf]."),
     "Gate G2 opens on 3 May 2027 [1].\n\nThe register is reissued monthly [1].",
     ["Gate Register.pdf"]),
    (("The jetty deck is rated at 15 kPa [source: Jetty Basis.pdf].\n\n"
      "Its piles are 900 mm diameter [source: Pile Layout.pdf].\n\n"
      "The rating includes a crane outrigger case [source: Jetty Basis.pdf] [source: Crane Note.pdf]."),
     "The jetty deck is rated at 15 kPa [1].\n\nThe rating includes a crane outrigger case [1] [2].",
     ["Jetty Basis.pdf", "Crane Note.pdf"]),
])
def test_markers_left_are_renumbered_with_no_gap(text, expect, documents):
    out = _apply(text)
    assert out.text == expect
    assert [c["document"] for c in out.citations] == documents
    assert [c["marker"] for c in out.citations] == list(range(1, len(documents) + 1))


# ── it stays ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    ("The billed area of the east car park is 2,900 m² [source: Car Park Ledger.xlsx].\n\n"
     "The billed area of the west car park is 3,400 m² [source: West Ledger.xlsx]."),
    ("The retaining wall concrete is grade C40 [source: Wall Specification.pdf].\n\n"
     "The wall drawing shows grade C35 for the same pour [source: Wall Drawing.pdf]."),
])
def test_a_value_of_the_leads_kind_from_another_document_stays(text):
    out = _apply(text)
    assert out.text.count("\n\n") == 1
    assert len(out.citations) == 2


@pytest.mark.parametrize("text", [
    ("The lift shaft wall is 250 mm thick [source: Core Drawing.pdf].\n\n"
     "Its reinforcement is shown at 200 mm centres [source: Core Drawing.pdf]."),
    ("The plant room floor is epoxy coated [source: Finish Schedule.pdf].\n\n"
     "The coating is a two-pack system rated for 5 kN point loads."),
])
def test_same_document_or_uncited_paragraph_stays(text):
    out = _apply(text)
    assert out.text.count("\n\n") == 1


@pytest.mark.parametrize("text", [
    ("The ledger carries these drainage lines [source: Drain Ledger.xlsx]:\n\n"
     "Pipe 150 mm, 420 m [source: Drain Sheet.pdf]."),
    ("The pump schedule lists two pumps [source: Pump Schedule.pdf].\n\n"
     "- Duty pump P1, 30 kW [source: Pump Data.pdf]"),
])
def test_a_list_after_the_lead_stays(text):
    out = _apply(text)
    assert out.text.count("\n\n") == 1


@pytest.mark.parametrize("text", [
    ("The plinth finish is not stated.\n\n"
     "The plinth is 600 mm high [source: Plinth Drawing.pdf]."),
    ("No hoarding height is given in the brief.\n\n"
     "The hoarding panels are 2.4 m long [source: Hoarding Detail.pdf]."),
])
def test_a_lead_that_cites_nothing_keeps_everything(text):
    out = _apply(text)
    assert out.text.count("\n\n") == 1


@pytest.mark.parametrize("text", [
    ("The ramp slab is 300 mm thick [source: Ramp Drawing.pdf].\n\n"
     "The ramp gradient is 1 in 12 [source: Access Statement.pdf]."),
    ("The tank holds 120 m³ [source: Tank Sheet.pdf].\n\n"
     "The tank lining is 2 mm HDPE [source: Lining Data.pdf]."),
])
def test_a_computed_or_asking_turn_keeps_everything(text):
    assert _apply(text, computed=True).text.count("\n\n") == 1
    assert _apply(text, asking=True).text.count("\n\n") == 1


@pytest.mark.parametrize("text", [
    ("The culvert is 2.4 m wide [source: Culvert Drawing.pdf].\n\n"
     "The drawing also names its headwall parts [source: Culvert Drawing.pdf]:\n\n"
     "Wingwalls cast in grade C32 [source: Headwall Sheet.pdf]."),
    ("The basement is 11 m deep [source: Basement Section.pdf].\n\n"
     "The section lists the waterproofing layers [source: Basement Section.pdf]:\n\n"
     "Bentonite sheet bonded to the slab [source: Membrane Data.pdf]."),
])
def test_a_paragraph_after_one_that_introduces_a_list_stays(text):
    out = _apply(text)
    assert out.text.count("\n\n") == 2
