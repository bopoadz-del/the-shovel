"""The answer contract on the exit step: two invented answers per rule.

Every answer here is constructed in the test; none is a recorded probe.
"""

from __future__ import annotations

from app.agents import answer_contract, answer_exit


def _lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.strip()]


# ── Rule 1 + 2: a padded calculation is reduced to figure plus working ──────

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


def test_padded_calculation_a_is_figure_plus_working():
    out = answer_contract.apply(PADDED_CALC_A)
    lines = _lines(out.text)
    assert lines[0].startswith("Maximum bending moment M = 45 kN·m")
    assert lines[1].startswith("M = wL²/8 = 10 × 6² / 8 = 45 kN·m")
    assert len(lines) == 2
    assert "notes" not in out.text.lower() and "load factor" not in out.text.lower()


def test_padded_calculation_b_opens_on_the_figure():
    out = answer_contract.apply(PADDED_CALC_B)
    lines = _lines(out.text)
    assert lines[0].startswith("Concrete volume = 12.6 m³")
    assert lines[1].startswith("V = L × W × D = 7 × 4 × 0.45 = 12.6 m³")
    assert len(lines) == 2
    assert "summary" not in out.text.lower() and "wastage" not in out.text.lower()


# ── Rule 2: an assumption that changes the result survives ──────────────────

def test_assumption_stating_a_working_value_survives():
    text = (
        "Excavated volume = 54 m³\n"
        "V = 6 × 3 × 3 = 54 m³\n"
        "Notes:\n"
        "- Assumed a trench depth of 3 m, since no depth was given.\n"
        "- Bulking factors vary by soil type.\n"
    )
    out = answer_contract.apply(text, computed=True)
    assert "Assumed a trench depth of 3 m" in out.text
    assert "Bulking" not in out.text


def test_labelled_assumption_without_value_survives_a_caveat_does_not():
    text = (
        "Reaction at each support = 30 kN\n"
        "R = wL/2 = 10 × 6 / 2 = 30 kN\n"
        "Two notes:\n"
        "- Assumption: the beam is simply supported at both ends.\n"
        "- You may wish to consider a dynamic amplification of 1.2 for plant.\n"
    )
    out = answer_contract.apply(text, computed=True)
    assert "Assumption: the beam is simply supported" in out.text
    assert "dynamic amplification" not in out.text


# ── Rule 3: machinery words and index counts are removed ────────────────────

def test_search_narration_lead_paragraph_goes_figures_stay():
    text = (
        "The search returned three passages from the retrieved documents on blockwork.\n"
        "Blockwork output is 10 m² per mason-day for 200 mm blocks and 12 m² per mason-day for 150 mm blocks.\n"
    )
    out = answer_contract.apply(text)
    lines = _lines(out.text)
    assert lines[0].startswith("Blockwork output is 10 m²")
    assert "search" not in out.text.lower() and "retrieved" not in out.text.lower()


def test_machinery_clause_inside_a_figure_sentence_keeps_the_figure():
    text = (
        "The delay damages are 25,000 per day, calculated using the pre-dispatch calculator.\n"
        "Based on the retrieved documents, the cap is 10% of the contract sum.\n"
        "6 of 9 project documents indexed\n"
    )
    out = answer_contract.apply(text)
    assert "25,000 per day" in out.text and "10% of the contract sum" in out.text
    low = out.text.lower()
    for word in ("pre-dispatch", "calculator", "retrieved", "indexed", "6 of 9"):
        assert word not in low
    assert out.coverage == {"indexed": 6, "total": 9}


def test_domain_words_are_not_machinery():
    text = "The certified amount is 1,200,000 and the gas pipeline runs 40 m.\n"
    assert answer_contract.apply(text).text == text.strip()


# ── Rule 4: source lines become markers plus structured entries ─────────────

ROWS = [
    {"doc_name": "Earthworks Specification.pdf", "page_or_section": "p. 12", "page": 12,
     "layer": "own", "layer_label": "Project document", "snippet": "compacted to 95% MDD"},
    {"doc_name": "Bill of Quantities.xlsx", "page_or_section": "", "layer": "general_knowledge",
     "layer_label": "Knowledge base", "snippet": "Item D.4 excavation"},
]


def test_source_line_becomes_marker_and_entry_with_row_tier():
    text = (
        "Fill is compacted to 95% of maximum dry density.\n"
        "Source: Earthworks Specification.pdf, p. 12\n"
    )
    out = answer_contract.apply(text, sources=ROWS)
    assert out.text == "Fill is compacted to 95% of maximum dry density. [1]"
    [entry] = out.citations
    assert entry["marker"] == 1
    assert entry["document"] == "Earthworks Specification.pdf"
    assert entry["section"] == "p. 12"
    assert entry["passage"] == "compacted to 95% MDD"
    assert entry["tier"] == "Project document"


def test_bracketed_file_citations_become_numbered_markers():
    text = (
        "Excavation is measured in item D.4 (Bill of Quantities.xlsx) and the bedding depth is 150 mm "
        "[Pipe Bedding Detail.pdf, p. 3].\n"
        "Sources: Bill of Quantities.xlsx\n"
    )
    out = answer_contract.apply(text, sources=ROWS)
    assert ".xlsx" not in out.text and ".pdf" not in out.text and "Source" not in out.text
    assert "[1]" in out.text and "[2]" in out.text
    docs = {e["document"]: e for e in out.citations}
    assert docs["Bill of Quantities.xlsx"]["tier"] == "Knowledge base"
    # A citation no Sources row names is never given a tier.
    assert docs["Pipe Bedding Detail.pdf"]["tier"] is None
    assert docs["Pipe Bedding Detail.pdf"]["section"] == "p. 3"
    assert len(out.citations) == 2


def test_calculation_credit_is_a_calculation_entry_never_a_document_tier():
    out = answer_contract.apply(PADDED_CALC_A)
    [entry] = out.citations
    assert entry["kind"] == "calculation"
    assert entry["document"] == "Simply supported beam moment"
    assert entry["passage"] == "load 10 kN/m, span 6 m"
    assert entry["tier"] is None
    assert out.coverage == {"indexed": 3135, "total": 3203}


# ── Rule 5: honesty unchanged ───────────────────────────────────────────────

def test_refusal_stays_a_refusal():
    text = (
        "The documents do not state a curing period for the slab.\n"
        "Provide the concrete specification or the method statement and I will read the period from it.\n"
    )
    out = answer_contract.apply(text)
    assert out.text == text.strip()


def test_answer_that_is_only_machinery_is_not_emptied():
    text = "The search returned nothing from the retrieved documents."
    assert answer_contract.apply(text).text == text


# ── Length per question class ───────────────────────────────────────────────

def test_length_per_question_class():
    lookup = (
        "You asked about the defects notification period.\n"
        "The defects notification period is 365 days from the Taking-Over Certificate.\n"
        "Source: Particular Conditions.pdf, clause 11.1\n"
        "I hope this helps. If you would like, I can summarise the other periods too.\n"
        "In summary, the period is one year.\n"
    )
    out = answer_contract.apply(lookup)
    assert len(_lines(out.text)) <= 3
    assert _lines(out.text)[0].startswith("The defects notification period is 365 days")
    for padded in (PADDED_CALC_A, PADDED_CALC_B):
        assert len(_lines(answer_contract.apply(padded).text)) <= 3


# ── The exit step carries the structured payload ────────────────────────────

def test_end_event_carries_citations_and_coverage(monkeypatch):
    turn = answer_exit.Turn()
    event = {"type": "end", "content": PADDED_CALC_B, "sources": []}
    out = answer_exit.check_end_event(event, "", turn)
    assert _lines(out["content"])[0].startswith("Concrete volume = 12.6 m³")
    assert "indexed" not in out["content"] and "Source" not in out["content"]
    assert out["coverage"] == {"indexed": 120, "total": 480}
    assert out["citations"][0]["kind"] == "calculation"


def test_result_envelope_carries_citations_and_coverage():
    turn = answer_exit.Turn()
    result = {"answer": "Fill is compacted to 95% of maximum dry density.\n"
                        "Source: Earthworks Specification.pdf, p. 12\n9 of 10 project documents indexed",
              "sources": [dict(ROWS[0])]}
    out = answer_exit.check_result(result, turn)
    assert out["answer"].endswith("[1]")
    assert out["coverage"] == {"indexed": 9, "total": 10}
    assert out["citations"][0]["document"] == "Earthworks Specification.pdf"


# ── A turn asking for inputs keeps its question as asked ────────────────────

def test_input_question_is_left_as_asked_two_turns():
    for missing, question in (
        (["span"], "The beam needs the span. What span should I use?"),
        (["rate", "quantity"], "The line cost needs the rate and quantity. What values should I use for the rate and quantity?"),
    ):
        turn = answer_exit.Turn(questions=[question], awaiting_inputs=missing)
        out = answer_exit.check_end_event({"type": "end", "content": question + "\n4 of 5 project documents indexed"},
                                          "", turn)
        assert out["content"] == question
        assert out["coverage"] == {"indexed": 4, "total": 5}


def test_a_document_cited_inline_and_on_a_source_line_gets_one_marker():
    for text in (
        "The period is 42 days (Programme Notes.pdf, p. 2).\nSource: Programme Notes.pdf, p. 2\n",
        "Cover is 50 mm [Concrete Spec.docx, p. 7].\nSources: Concrete Spec.docx, p. 7\n",
    ):
        out = answer_contract.apply(text)
        assert out.text.count("[1]") == 1 and "[2]" not in out.text
        assert len(out.citations) == 1
