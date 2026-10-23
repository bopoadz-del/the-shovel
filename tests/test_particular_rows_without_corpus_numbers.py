"""Particular rows are recognised by label + value shape, not one contract's numbers.

The per-particular recognisers used to carry one contract's clause numbers
(the guarantee at one sub-clause, Time for Completion at another), one
contract's schedule number for the guarantee form, one project's place names
for a group of milestones and one corpus's file-naming convention. A
different contract numbers its particulars differently, so each of those
was a photograph. These tests use a contract whose guarantee sits in clause
9.2 with its form in Schedule 3, whose Time for Completion is clause 2.3, and
whose milestones are grouped by districts nobody has heard of. Every name and
figure is invented.
"""
from __future__ import annotations

import pytest

from app.core.rag import retriever as ret

PREFIX = "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"

# ── parent company guarantee ──────────────────────────────────────────────

PCG_ROW_92 = PREFIX + "9.2 | Parent Company Guarantee | Not required |"
PCG_VALUE_92 = PREFIX + "9.2 | Parent Company Guarantee | 7.5% of the Accepted Contract Amount |"
PCG_FORM_SCHEDULE_3 = (
    "Schedule 3 Form of Parent Company Guarantee. In consideration of the "
    "award of the "
    "Contract, the guarantor irrevocably undertakes to pay 25% of its "
    "shareholders' funds on first demand."
)
SCHEDULE_8_LIST = (
    "Schedule 8 | Programme of Works | as attached. The Parent Company "
    "Guarantee is listed in Schedule 3."
)


def test_the_honest_line_cites_the_clause_the_chunk_carries():
    line = ret.format_pcg_honest_line(PCG_ROW_92)
    assert "9.2" in line and "not required" in line.lower(), line
    line = ret.format_pcg_honest_line(PCG_VALUE_92)
    assert "7.5%" in line and "9.2" in line, line


def test_the_honest_line_cites_no_clause_when_the_chunk_carries_none():
    line = ret.format_pcg_honest_line("Parent Company Guarantee: Not required.")
    assert not any(ch.isdigit() for ch in line), line


def test_a_guarantee_form_in_any_schedule_is_a_template():
    assert ret.chunk_states_pcg_form_template(PCG_FORM_SCHEDULE_3)
    assert not ret.chunk_states_pcg_contract_data(PCG_FORM_SCHEDULE_3)


def test_a_schedule_number_alone_does_not_make_a_form():
    assert not ret.chunk_states_pcg_form_template(SCHEDULE_8_LIST)


def test_a_bare_clause_number_is_not_a_guarantee_mention():
    text = PREFIX + "4.3.7 Professional Indemnity Insurance\nNo."
    assert not ret.chunk_states_pcg_contract_data(text)


# ── time for completion ───────────────────────────────────────────────────

TFC_ROW_23 = "2.3 Time for Completion: 730 days from the Commencement Date"
TFC_PROSE = (
    "The Contractor shall submit the as-built records within 90 days of the "
    "Time for Completion."
)


def test_a_numbered_time_for_completion_row_is_the_particular():
    assert ret.chunk_states_time_for_completion(TFC_ROW_23)
    assert ret.extract_time_for_completion_days(TFC_ROW_23) == "730 days"


def test_prose_about_the_time_for_completion_is_not_the_row():
    assert not ret.chunk_states_time_for_completion(TFC_PROSE)


# ── a named group of milestones ───────────────────────────────────────────

@pytest.mark.parametrize("ask, name", [
    ("Which Harbourside Quarter milestone has the longest Time for Completion?",
     "harbourside quarter"),
    ("In the Old Mill District, which milestone has the shortest Time for "
     "Completion?", "old mill district"),
    ("Among the Riverside Precinct milestones, which is longest?", "riverside precinct"),
])
def test_a_milestone_group_is_read_from_the_question(ask, name):
    assert ret.extract_asked_community_name(ask).lower() == name


def test_a_question_with_no_group_names_none():
    assert ret.extract_asked_community_name(
        "Which milestone has the longest Time for Completion?"
    ) == ""


# ── document kind by name, not one corpus's naming convention ─────────────

@pytest.mark.parametrize("name, kind", [
    ("Lakeview Conditions of Contract.pdf", True),
    ("Lakeview Particular Conditions.pdf", True),
    ("Lakeview_Contract Data.pdf", True),
    ("Lakeview Cond of Contract (complete).pdf", True),
    ("Vol 3 Concrete Works Specification.pdf", False),
    ("Vol 2 Construction Drawings.pdf", False),
])
def test_a_conditions_volume_is_named_as_one(name, kind):
    assert ret.filename_looks_like_conditions_volume(name) is kind


# ── query expansion carries a vocabulary, not one document's spelling ─────

def test_cover_expansion_is_a_vocabulary_not_a_misspelling():
    ask = "Per the project specification, what is the minimum concrete cover for slabs?"
    for text in (ret.retrieval_lexical_query(ask), ret.numeric_requirement_expansion(ask)):
        assert "casted" not in text.lower(), text


# ── defects notification period / delay damages ───────────────────────────

def test_the_numbered_defects_row_wins_whatever_its_clause_number():
    assert ret.extract_defects_notification_period(
        "Defects notification under the subcontract: 30 days after taking-over.\n\n"
        "6.4 Defects Notification Period: 545 days from the Taking-Over Certificate"
    ) == "545 days"


def test_a_numbered_delay_damages_clause_qualifies_its_volume():
    clause = (
        "11.3 Delay Damages. The Contractor shall pay delay damages at the "
        "rate stated in the particulars."
    )
    assert ret._doc_qualifies_for_late_aca_scan(clause, "Lakeview Volume 1.pdf")
