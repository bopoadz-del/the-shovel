"""The answer contract on the exit step: two invented answers per rule.

Every answer here is constructed in the test; none is a recorded probe.
"""

from __future__ import annotations

import logging

import pytest

from app.agents import answer_contract, answer_exit


def _lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.strip()]


PADDED_CALC_A = (
    "Maximum bending moment M = 45 kN·m.\n"
    "M = wL²/8 = 10 × 6² / 8 = 45 kN·m\n"
    "This is a deterministic statics calculation and the result is a certified calculation.\n"
    "Two notes on the inputs:\n"
    "- Load factors were not applied; a factored design load would raise the moment.\n"
    "- Self-weight of the member is not included.\n"
    "Source: Simply supported beam moment — platform calculator (load 10 kN/m, span 6 m)\n"
    "Let me know if you would like the shear force as well.\n"
    "3135 of 3203 project documents indexed\n"
)

PADDED_CALC_B = (
    "Here is the result you asked for, computed by the platform calculator.\n"
    "Concrete volume = 12.6 m³\n"
    "V = L × W × D = 7 × 4 × 0.45 = 12.6 m³\n"
    "A few caveats:\n"
    "- Wastage allowances vary between contractors.\n"
    "In summary, the slab needs 12.6 m³ of concrete.\n"
    "Source: Concrete volume — platform calculator (length 7 m, width 4 m, depth 0.45 m)\n"
    "120 of 480 project documents indexed\n"
)

ROWS = [
    {"doc_name": "Earthworks Specification.pdf", "page_or_section": "p. 12", "page": 12,
     "layer": "own", "layer_label": "Project document", "snippet": "compacted to 95% MDD"},
    {"doc_name": "Bill of Quantities.xlsx", "page_or_section": "", "layer": "general_knowledge",
     "layer_label": "Knowledge base", "snippet": "Item D.4 excavation"},
]


# ── Rule 1 + 2: a padded calculation is reduced to figure plus working ──────

@pytest.mark.parametrize("text, first, second, gone", [
    (PADDED_CALC_A, "Maximum bending moment M = 45 kN·m", "M = wL²/8 = 10 × 6² / 8 = 45 kN·m",
     ("notes", "load factor", "self-weight", "let me know", "deterministic")),
    (PADDED_CALC_B, "Concrete volume = 12.6 m³", "V = L × W × D = 7 × 4 × 0.45 = 12.6 m³",
     ("summary", "wastage", "caveat", "here is the result")),
])
def test_padded_calculation_is_figure_plus_working(text, first, second, gone):
    out = answer_contract.apply(text)
    lines = _lines(out.text)
    assert lines[0].startswith(first)
    assert lines[1].startswith(second)
    assert len(lines) == 2
    for word in gone:
        assert word not in out.text.lower()
    # The calculation's citation keeps its marker even though its line went.
    assert out.text.count("[1]") == 1


# ── Rule 2: an assumption the working depends on survives ───────────────────

@pytest.mark.parametrize("text, kept, gone", [
    (("Excavated volume = 54 m³\nV = 6 × 3 × 3 = 54 m³\nNotes:\n"
     "- Assumed a trench depth of 3 m, since no depth was given.\n- Bulking factors vary by soil type.\n"),
     "Assumed a trench depth of 3 m", "Bulking"),
    (("Reaction at each support = 30 kN\nR = wL/2 = 10 × 6 / 2 = 30 kN\nTwo notes:\n"
     "- Assumption: the beam is simply supported at both ends.\n"
     "- You may wish to consider a dynamic amplification of 1.2 for plant.\n"),
     "Assumption: the beam is simply supported", "dynamic amplification"),
])
def test_assumption_the_result_depends_on_survives(text, kept, gone):
    out = answer_contract.apply(text, computed=True)
    assert kept in out.text
    assert gone not in out.text


@pytest.mark.parametrize("text, kept", [
    ("Floor load = 60 kN\nW = 10 × 6 = 60 kN\nNotes:\n- Load of 10 kN/m taken as the office live load.\n",
     "Load of 10 kN/m taken as the office live load."),
    ("Area = 24 m²\nA = 4 × 6 = 24 m²\nOne note:\n- Width of 4 m read from the gridline spacing.\n",
     "Width of 4 m read from the gridline spacing."),
])
def test_note_stating_a_working_value_survives_without_the_word_assume(text, kept):
    assert kept in answer_contract.apply(text, computed=True).text


@pytest.mark.parametrize("text, kept", [
    ("Delay damages are 25,000 per day.\nNotes:\n- The cap is 10% of the contract sum.\n",
     "The cap is 10% of the contract sum."),
    ("The defects period is 365 days.\nTwo notes:\n- Retention release is 50% at taking over.\n",
     "Retention release is 50% at taking over."),
])
def test_note_with_a_figure_on_a_lookup_turn_survives(text, kept):
    assert kept in answer_contract.apply(text).text


@pytest.mark.parametrize("text, kept", [
    ("Slab volume = 9 m³\nV = 3 × 3 × 1 = 9 m³\nNotes:\n- Wastage varies.\n\nPour within 90 minutes of batching.\n",
     "Pour within 90 minutes of batching."),
    ("Trench volume = 8 m³\nV = 4 × 2 × 1 = 8 m³\nCaveats:\n- Soil bulks.\n\nShore any trench deeper than 1.5 m.\n",
     "Shore any trench deeper than 1.5 m."),
])
def test_a_notes_block_ends_at_the_blank_line(text, kept):
    assert kept in answer_contract.apply(text, computed=True).text


@pytest.mark.parametrize("text, kept", [
    ("Mortar volume = 2.4 m³\nV = 120 × 0.02 = 2.4 m³\nIn summary, order 3 m³ to allow for loss.\n",
     "order 3 m³"),
    ("Rebar mass = 785 kg\nM = 0.1 × 7850 = 785 kg\nOverall, specify 800 kg on the order.\n",
     "800 kg"),
])
def test_a_summary_with_a_new_figure_survives(text, kept):
    assert kept in answer_contract.apply(text, computed=True).text


# ── Rule 3: machinery words and index counts are removed ────────────────────

@pytest.mark.parametrize("text, first", [
    (("The search returned three passages from the retrieved documents on blockwork.\n"
     "Blockwork output is 10 m² per mason-day for 200 mm blocks.\n"), "Blockwork output is 10 m²"),
    (("I searched the project documents for plastering rates.\n"
     "Plastering output is 12 m² per day for two coats.\n"), "Plastering output is 12 m²"),
])
def test_search_narration_lead_goes_figures_stay(text, first):
    out = answer_contract.apply(text)
    assert _lines(out.text)[0].startswith(first)
    assert "search" not in out.text.lower()


@pytest.mark.parametrize("text, figures, gone, coverage", [
    (("The delay damages are 25,000 per day, calculated using the pre-dispatch calculator.\n"
     "Based on the retrieved documents, the cap is 10% of the contract sum.\n6 of 9 project documents indexed\n"),
     ("25,000 per day", "10% of the contract sum"), ("pre-dispatch", "calculator", "retrieved", "indexed", "6 of 9"),
     {"indexed": 6, "total": 9}),
    ("The deterministic calculation gives 12.5 m³ of concrete.\n",
     ("12.5 m³",), ("deterministic",), None),
    ("The moment is 45 kN·m using the platform calculator with a span of 6 m.\n",
     ("45 kN·m", "6 m"), (), None),
    ("The load is 10 kN per m via the platform calculator.\n",
     ("10 kN per m",), ("calculator",), None),
])
def test_a_figure_and_its_unit_survive_machinery_in_the_same_sentence(text, figures, gone, coverage):
    out = answer_contract.apply(text)
    for fig in figures:
        assert fig in out.text
    for word in gone:
        assert word not in out.text.lower()
    assert out.coverage == coverage


@pytest.mark.parametrize("text, kept", [
    ("Curing is by wet hessian. The search returned three passages on formwork.", "Curing is by wet hessian."),
    ("Formwork is struck by hand. The retrieved documents cover formwork.", "Formwork is struck by hand."),
])
def test_a_plain_sentence_beside_machinery_stays(text, kept):
    out = answer_contract.apply(text)
    assert out.text == kept


@pytest.mark.parametrize("text", [
    "The certified amount is 1,200,000 and the gas pipeline runs 40 m.\n",
    "The interim certificate was certified on 3 March; the tool hire is 400 per week.\n",
])
def test_domain_words_are_not_machinery(text):
    assert answer_contract.apply(text).text == text


@pytest.mark.parametrize("text", [
    "**Cover to reinforcement**\n\n- Footings: 75 mm\n- Slabs: 25 mm\n\n\nColumns take 40 mm.  \n",
    "The programme has 14 activities.\nThe critical path runs through the roof.\n",
])
def test_an_answer_with_none_of_the_patterns_passes_byte_for_byte(text):
    out = answer_contract.apply(text)
    assert out.text == text
    assert out.citations == [] and out.coverage is None


# ── Rule 4: source lines become markers plus structured entries ─────────────

@pytest.mark.parametrize("text, expect, document, section, passage, tier", [
    ("Fill is compacted to 95% of maximum dry density.\nSource: Earthworks Specification.pdf, p. 12\n",
     "Fill is compacted to 95% of maximum dry density. [1]",
     "Earthworks Specification.pdf", "p. 12", "compacted to 95% MDD", "Project document"),
    ("Excavation is measured as item D.4.\nSource: Bill of Quantities.xlsx\n",
     "Excavation is measured as item D.4. [1]",
     "Bill of Quantities.xlsx", "", "Item D.4 excavation", "Knowledge base"),
])
def test_source_line_becomes_marker_and_entry_with_row_tier(text, expect, document, section, passage, tier):
    out = answer_contract.apply(text, sources=ROWS)
    assert out.text == expect
    [entry] = out.citations
    assert (entry["marker"], entry["document"], entry["section"], entry["passage"], entry["tier"]) == (
        1, document, section, passage, tier)


@pytest.mark.parametrize("text, tiers", [
    (("Excavation is measured in item D.4 (Bill of Quantities.xlsx) and the bedding depth is 150 mm "
     "[Pipe Bedding Detail.pdf, p. 3].\nSources: Bill of Quantities.xlsx\n"),
     {"Bill of Quantities.xlsx": "Knowledge base", "Pipe Bedding Detail.pdf": None}),
    (("Fill goes in 200 mm layers (Earthworks Specification.pdf, p. 12) under the slab "
     "[Slab Notes.docx].\n"),
     {"Earthworks Specification.pdf": "Project document", "Slab Notes.docx": None}),
])
def test_bracketed_file_citations_become_numbered_markers(text, tiers):
    out = answer_contract.apply(text, sources=ROWS)
    for ext in (".xlsx", ".pdf", ".docx", "Source"):
        assert ext not in out.text
    assert "[1]" in out.text and "[2]" in out.text
    # A citation no Sources row names is never given a tier.
    assert {e["document"]: e["tier"] for e in out.citations} == tiers


@pytest.mark.parametrize("text, document, passage, coverage", [
    (PADDED_CALC_A, "Simply supported beam moment", "load 10 kN/m, span 6 m", {"indexed": 3135, "total": 3203}),
    (PADDED_CALC_B, "Concrete volume", "length 7 m, width 4 m, depth 0.45 m", {"indexed": 120, "total": 480}),
])
def test_calculation_credit_is_a_calculation_entry_never_a_document_tier(text, document, passage, coverage):
    out = answer_contract.apply(text, sources=ROWS)
    [entry] = out.citations
    assert (entry["kind"], entry["document"], entry["passage"], entry["tier"]) == (
        "calculation", document, passage, None)
    assert out.coverage == coverage


@pytest.mark.parametrize("text, documents", [
    ("Cover is 50 mm.\nSources: Concrete Spec.docx, p. 7; Bar Schedule.xlsx\n",
     ["Concrete Spec.docx", "Bar Schedule.xlsx"]),
    ("Lap is 40 bar diameters.\nSource: Detailing Manual.pdf, clause 8.4; Site Memo 12.pdf\n",
     ["Detailing Manual.pdf", "Site Memo 12.pdf"]),
])
def test_a_source_line_listing_two_documents_gives_two_markers(text, documents):
    out = answer_contract.apply(text)
    assert [e["document"] for e in out.citations] == documents
    assert out.text.endswith("[1][2]")


@pytest.mark.parametrize("text", [
    "Bearing pressure is 150 kPa.\nSource: — platform calculator (load 600 kN, area 4 m²)\n",
    "Wall area is 36 m².\nSource: - platform calculator (length 12 m, height 3 m)\n",
])
def test_a_source_line_naming_no_document_gives_no_entry(text):
    out = answer_contract.apply(text)
    assert out.citations == []
    assert "[1]" not in out.text and "calculator" not in out.text


@pytest.mark.parametrize("text", [
    "The period is 42 days (Programme Notes.pdf, p. 2).\nSource: Programme Notes.pdf, p. 2\n",
    "Cover is 50 mm [Concrete Spec.docx, p. 7].\nSources: Concrete Spec.docx, p. 7\n",
])
def test_a_document_cited_inline_and_on_a_source_line_gets_one_marker(text):
    out = answer_contract.apply(text)
    assert out.text.count("[1]") == 1 and "[2]" not in out.text
    assert len(out.citations) == 1


# ── Rule 5: honesty unchanged ───────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    ("The documents do not state a curing period for the slab.\n"
    "Provide the concrete specification or the method statement and I will read the period from it.\n"),
    ("I could not find a defects notification period in this project's documents. "
    "Upload the Particular Conditions to answer this."),
    "The search returned no clause on delay damages.",
    "I searched the project documents and found nothing on curing periods. Provide the specification.",
])
def test_an_answer_that_is_only_a_refusal_stays(text):
    assert answer_contract.apply(text).text == text


@pytest.mark.parametrize("text", [
    "The search returned three passages from the retrieved documents.",
    "This is a deterministic calculation by the platform calculator.",
])
def test_answer_that_is_only_machinery_is_not_emptied(text):
    assert answer_contract.apply(text).text == text


@pytest.mark.parametrize("text", [
    "The contract does not fix a curing period.\nClause 4 sets cover at 50 mm.\n",
    "Those drawings are not in this project.\nThe nearest is sheet 12, issued for tender.\n",
])
def test_a_turn_that_did_not_compute_keeps_its_order(text):
    assert answer_contract.apply(text).text == text


@pytest.mark.parametrize("value", [None, {"answer": "Cover is 50 mm."}])
def test_a_non_text_answer_passes_without_an_error(value, caplog):
    with caplog.at_level(logging.ERROR, logger="app.agents.answer_contract"):
        out = answer_contract.apply(value)
    assert out.text is value
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


# ── Length per question class ───────────────────────────────────────────────

@pytest.mark.parametrize("text, first, most", [
    (("You asked about the defects notification period.\n"
     "The defects notification period is 365 days from the Taking-Over Certificate.\n"
     "Source: Particular Conditions.pdf, clause 11.1\n"
     "I hope this helps. If you would like, I can summarise the other periods too.\n"
     "In summary, the period is one year.\n"),
     "The defects notification period is 365 days", 3),
    (PADDED_CALC_A, "Maximum bending moment M = 45 kN·m", 2),
    (PADDED_CALC_B, "Concrete volume = 12.6 m³", 2),
])
def test_length_per_question_class(text, first, most):
    out = answer_contract.apply(text)
    assert len(_lines(out.text)) <= most
    assert _lines(out.text)[0].startswith(first)


# ── The exit step carries the structured payload ────────────────────────────

@pytest.mark.parametrize("content, first, coverage", [
    (PADDED_CALC_B, "Concrete volume = 12.6 m³", {"indexed": 120, "total": 480}),
    (PADDED_CALC_A, "Maximum bending moment M = 45 kN·m", {"indexed": 3135, "total": 3203}),
])
def test_end_event_carries_citations_and_coverage(content, first, coverage):
    out = answer_exit.check_end_event({"type": "end", "content": content, "sources": []}, "", answer_exit.Turn())
    assert _lines(out["content"])[0].startswith(first)
    assert "indexed" not in out["content"] and "Source" not in out["content"]
    assert out["coverage"] == coverage
    assert out["citations"][0]["kind"] == "calculation"


@pytest.mark.parametrize("answer, row, coverage", [
    (("Fill is compacted to 95% of maximum dry density.\nSource: Earthworks Specification.pdf, p. 12\n"
     "9 of 10 project documents indexed"), ROWS[0], {"indexed": 9, "total": 10}),
    (("Excavation is measured as item D.4.\nSource: Bill of Quantities.xlsx\n"
     "4 of 7 project documents indexed"), ROWS[1], {"indexed": 4, "total": 7}),
])
def test_result_envelope_carries_citations_and_coverage(answer, row, coverage):
    out = answer_exit.check_result({"answer": answer, "sources": [dict(row)]}, answer_exit.Turn())
    assert out["answer"].endswith("[1]")
    assert out["coverage"] == coverage
    assert out["citations"][0]["document"] == row["doc_name"]


@pytest.mark.parametrize("env", [{}, {"sources": []}])
def test_the_exit_contract_leaves_an_envelope_without_text_alone(env):
    out = answer_exit._apply_contract(dict(env), "answer", [], answer_exit.Turn())
    assert out == env


# ── A turn asking for inputs keeps its question as asked ────────────────────

@pytest.mark.parametrize("missing, question", [
    (["span"], "The beam needs the span. What span should I use?"),
    (["rate", "quantity"], "The line cost needs the rate and quantity. What values should I use for the rate and quantity?"),
])
def test_input_question_is_left_as_asked(missing, question):
    turn = answer_exit.Turn(questions=[question], awaiting_inputs=missing)
    out = answer_exit.check_end_event({"type": "end", "content": question + "\n4 of 5 project documents indexed"},
                                      "", turn)
    assert out["content"] == question
    assert out["coverage"] == {"indexed": 4, "total": 5}
