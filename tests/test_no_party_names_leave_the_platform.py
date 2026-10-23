"""No party names leave this RAG: no Employer, Contractor, Engineer, Consultant.

Owner ruling, 2026-09-19: "No names at all from this RAG. No employer, project
name, no consultant, no contractor, no engineer."

The assistant's prompt has always said so (rule 9: "use the fact, not the
name"). One path was built to do the opposite: asked who the Engineer is, the
platform injected "State ONLY that firm's name" and, if the model still did
not, a graft wrote "The Engineer is <firm>" in front of the answer. Employer
and Contractor asks had no such machinery and leaked or refused by luck.

Three layers, because a rule the model is merely ASKED to follow is not a
rule:

1. the instruction given to the model no longer asks for a name;
2. the graft no longer writes one;
3. whatever the model says, names of parties that appear in the retrieved
   excerpts are removed from the final answer -- derived from the documents
   themselves, not from a hand-kept list, so the next contract is covered
   the day it is uploaded.
"""
from __future__ import annotations

import pytest

from app.core.party_names import (
    extract_party_names,
    party_names_withheld,
    withhold_party_names,
)

EXCERPT = (
    "[doc_id=cd chunk=4 score=7.100] CONTRACT DATA particulars — filled-in amount "
    "[XX-2099-001_Contract Data.pdf].\nCONTRACT DATA\n"
    "1.3.1(b): | Party and Engineer details and addresses for Communications: | |\n"
    "|: | Employer | Exampleton Gate Company Limited |\n"
    "|: | Engineer EXAMPLECO(EX2M Arabia Limited): | |\n"
    "|: | Contractor Al-Sample Investment & Contracting Company: | |\n"
    "4.3.3(a): | Value of Performance Bond: 10 % of the Accepted Contract Amount | |\n"
)
FLAT_EXCERPT = (
    "[doc_id=gc chunk=0 score=1.500] Contract No. XX-2099-001 Construction Contract "
    "between Exampleton Gate Company Limited as the Employer and Al-Sample "
    "Investment & Contracting Company as the Contractor Volume 4 of 6"
)


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    monkeypatch.delenv("RAG_WITHHOLD_PARTY_NAMES", raising=False)


def test_the_rule_is_on_unless_someone_turns_it_off(monkeypatch):
    assert party_names_withheld()
    monkeypatch.setenv("RAG_WITHHOLD_PARTY_NAMES", "0")
    assert not party_names_withheld()


def test_the_parties_are_read_from_the_documents_themselves():
    found = dict(extract_party_names(EXCERPT + "\n\n" + FLAT_EXCERPT))
    assert found["Exampleton Gate Company Limited"] == "the Employer"
    assert found["Al-Sample Investment & Contracting Company"] == "the Contractor"
    assert any(name.startswith("EXAMPLECO") and role == "the Engineer"
               for name, role in found.items())


@pytest.mark.parametrize(
    "answer, gone, kept",
    [
        ("The Engineer under this contract is **EXAMPLECO (EX2M Arabia Limited)**.",
         ["EXAMPLECO", "EX2M"], "the Engineer"),
        ("The Employer is Exampleton Gate Company Limited, per clause 1.3.1(b).",
         ["Exampleton"], "clause 1.3.1(b)"),
        ("Notices go to Al-Sample Investment & Contracting Company (the Contractor).",
         ["Al-Sample"], "the Contractor"),
        # The short form a model actually writes, not the registered name.
        ("EXAMPLECO signed it, and Exampleton Gate approved.", ["EXAMPLECO", "Exampleton"], "approved"),
    ],
)
def test_a_name_in_the_answer_is_replaced_by_the_role(answer, gone, kept):
    out = withhold_party_names(answer, EXCERPT)
    for word in gone:
        assert word not in out, out
    assert kept in out


@pytest.mark.parametrize(
    "answer",
    [
        "The Performance Bond is 10 % of the Accepted Contract Amount (clause 4.3.3(a)).",
        "The Contractor shall give notice to the Engineer within 28 days.",
        "Milestone 5 is 731 days from the date the Contractor is given right of access.",
        "A limited company must provide a Parent Company Guarantee if required.",
    ],
)
def test_an_answer_with_no_party_name_is_returned_exactly(answer):
    """Facts are the product. Roles, clauses, figures and ordinary words that
    happen to occur in a company's name ("Company", "Limited") are untouched."""
    assert withhold_party_names(answer, EXCERPT) == answer


def test_switched_off_nothing_is_removed(monkeypatch):
    monkeypatch.setenv("RAG_WITHHOLD_PARTY_NAMES", "0")
    answer = "The Engineer is EXAMPLECO (EX2M Arabia Limited)."
    assert withhold_party_names(answer, EXCERPT) == answer


# ── the model is no longer ASKED for the name ─────────────────────────────

def _context(question):
    from app.core.rag.inject import format_chunks_as_system_message
    from app.core.rag.vector_store import Chunk

    chunk = Chunk(chunk_id="c", project_id="p", doc_id="cd", chunk_index=4,
                  text=EXCERPT.split("] ", 1)[1], score=7.1)
    return format_chunks_as_system_message([chunk], 1, query=question)["content"]


@pytest.mark.parametrize(
    "question",
    [
        "Who is the Engineer under this contract?",
        "Who is the Employer under this contract?",
        "Who is the Contractor under this contract?",
        "Which firm is the Consultant on this project?",
    ],
)
def test_a_who_is_question_gets_the_withheld_instruction_not_a_naming_one(question):
    ctx = _context(question)
    assert "PARTY NAMES WITHHELD" in ctx
    assert "State ONLY that firm's name" not in ctx
    assert "ENGINEER IDENTITY" not in ctx


def test_with_the_rule_off_the_old_engineer_instruction_is_back(monkeypatch):
    monkeypatch.setenv("RAG_WITHHOLD_PARTY_NAMES", "0")
    ctx = _context("Who is the Engineer under this contract?")
    assert "ENGINEER IDENTITY" in ctx and "PARTY NAMES WITHHELD" not in ctx


# ── and the graft no longer WRITES it ─────────────────────────────────────

def _graft(answer, question):
    from app.agents.runtime import _graft_asked_contract_particular

    rag = {"role": "system", "content": "Project excerpts:\n" + EXCERPT}
    return _graft_asked_contract_particular(answer, rag, [{"role": "user", "content": question}])


def test_the_graft_states_where_the_engineer_is_named_and_not_who():
    out = _graft("I don't have that in the retrieved excerpts.",
                 "Who is the Engineer under this contract?")
    assert "EXAMPLECO" not in out and "EX2M" not in out
    assert "Contract Data" in out and "withheld" in out.lower()


def test_with_the_rule_off_the_graft_names_the_engineer_as_before(monkeypatch):
    monkeypatch.setenv("RAG_WITHHOLD_PARTY_NAMES", "0")
    out = _graft("I don't have that in the retrieved excerpts.",
                 "Who is the Engineer under this contract?")
    assert "EXAMPLECO" in out


def test_the_final_answer_path_removes_a_name_the_model_wrote_anyway():
    """End of the line: _postprocess_answer is what every non-streamed answer
    passes through, and it holds the excerpts."""
    from app.agents.runtime import _postprocess_answer

    rag = {"role": "system", "content": "Project excerpts:\n" + EXCERPT}
    msgs = [{"role": "user", "content": "Who issues instructions on site?"}]
    out = _postprocess_answer(
        "Instructions are issued by EXAMPLECO (EX2M Arabia Limited) as the Engineer.", rag, msgs)
    assert "EXAMPLECO" not in out and "EX2M" not in out
    assert "Engineer" in out


# ── a streamed line is scrubbed BEFORE it leaves ──────────────────────────

def test_a_streamed_segment_has_its_names_removed_before_it_is_sent():
    from app.agents.runtime import _withhold_names_in_streamed_segment

    rag = {"role": "system", "content": "Project excerpts:\n" + EXCERPT}
    out = _withhold_names_in_streamed_segment(
        "- Notices are served on EXAMPLECO (EX2M Arabia Limited).\n", rag)
    assert "EXAMPLECO" not in out and "EX2M" not in out and "Notices are served" in out


def test_both_streaming_flush_points_go_through_it():
    """Structural: a third flush point added later without the scrub is the
    leak this whole file exists to prevent."""
    import inspect
    from app.agents import runtime

    src = inspect.getsource(runtime.Agent._chat_stream_impl)
    flushes = src.count("_sanitize_inline_paths(_sanitize_citation_labels(")
    scrubbed = src.count("_withhold_names_in_streamed_segment(seg, _rag_sys_msg)")
    assert flushes >= 2 and scrubbed >= 2, (flushes, scrubbed)


# ── the acronym correspondence uses ───────────────────────────────────────

ACRONYM_EXCERPT = (
    "[doc_id=cd chunk=4 score=7.100] CONTRACT DATA\n"
    "|: | Contractor Bexample Trading & Haulage Company: | |\n"
)


def test_a_partys_initials_are_a_name_too():
    out = withhold_party_names(
        "The land was delivered to BTHC in March, and BTHC shall hand it back.", ACRONYM_EXCERPT)
    assert "BTHC" not in out and out.count("the Contractor") == 2


def test_initials_never_touch_an_ordinary_word_or_a_lower_case_match():
    text = "The bthc register and the word Both are untouched; so is BTHCX-001."
    assert withhold_party_names(text, ACRONYM_EXCERPT) == text


# ── a contract TERM is not a party ────────────────────────────────────────
#
# Live 562ee32 -- a regression this file's first version shipped, caught by
# re-running Set 1 the moment it deployed. C1 (order of precedence) came back
# as "3. Schedule of Project the Employer ... the Employer Environment the
# Employer for Contract". The extractor had read "Employer Requirements XYZC
# General Specification" as role + name (there was an ALL-CAPS token in it,
# and that was the whole test), and then offered every word of that "name",
# "Requirements" included, as something to remove.
#
# A contract is full of role-led TERMS: Employer Requirements, Contractor
# Documents, Engineer Instructions, Employer's Representative. None is a party.

TERMS_EXCERPT = (
    "[doc_id=s chunk=1 score=1.000] 1.5.1(d) the Specification: Post Tender "
    "Clarifications; Tender Addenda; Schedule of Project Requirements; Particular "
    "Specification; XYZC Environment Requirements for Contract & XYZC Sustainability "
    "Requirements; XYZC HSE Specification; Employer Requirements XYZC General "
    "Specification. The Employer's Representative shall attend. Contractor Documents "
    "XYZC shall be submitted. Engineer Instructions ABCD shall be in writing. "
    "Contractor Personnel and Contractor Equipment remain on Site."
)
PRECEDENCE_ANSWER = (
    "1. Post Tender Clarifications 2. Tender Addenda 3. Schedule of Project "
    "Requirements, followed by the Particular Specification, the Environment "
    "Requirements for Contract, the HSE Specification and the General Specification. "
    "Engineer Instructions must be in writing; Contractor Documents are submitted."
)


def test_role_led_contract_terms_are_not_read_as_parties():
    assert extract_party_names(TERMS_EXCERPT) == []


def test_the_live_precedence_answer_comes_back_untouched():
    assert withhold_party_names(PRECEDENCE_ANSWER, TERMS_EXCERPT) == PRECEDENCE_ANSWER


def test_a_real_party_beside_those_terms_is_still_found_and_nothing_else_is():
    both = TERMS_EXCERPT + "\n\n" + EXCERPT
    names = dict(extract_party_names(both))
    assert set(names.values()) == {"the Employer", "the Engineer", "the Contractor"}
    out = withhold_party_names(PRECEDENCE_ANSWER + " Signed by EXAMPLECO.", both)
    assert "EXAMPLECO" not in out
    assert "Schedule of Project Requirements" in out
    assert "Environment Requirements for Contract" in out


@pytest.mark.parametrize(
    "word", ["Requirements", "Specification", "Company", "Limited", "Investment",
             "Contracting", "Gate", "Arabia", "Instructions", "General"],
)
def test_an_ordinary_word_of_a_name_is_never_removed_on_its_own(word):
    """Only what IDENTIFIES: the whole name, its distinctive opening word, an
    all-capitals short form, a bracketed alias, the initials."""
    sentence = f"The {word} clause applies to every {word.lower()} in the schedule."
    assert withhold_party_names(sentence, EXCERPT + "\n\n" + FLAT_EXCERPT) == sentence


# ── an item code is not a firm ────────────────────────────────────────────
#
# Found by REPLAYING the scrub over all 77 real answers with their real
# excerpts, before shipping the fix above -- the check that should have been
# run before the first version. Bills say "as directed by the Engineer D110
# General site clearance ...". "All capitals" was accepted as a firm's name,
# so item codes D110 and D599.9 came back as "the Engineer" in BOQ answers.

BOQ_EXCERPT = (
    "[doc_id=b chunk=3 score=4.000] complete as directed by the Engineer D110 "
    "General site clearance ha 158 186,328.00 29,439,824.00 Removal of Asbestos "
    "including handling, complete as directed by the Engineer D599.9 m 262.00 Rate "
    "Only as instructed by the Engineer CESMM4 Class D applies. Employer BOQ Rev B."
)
BOQ_ANSWER = ("D110: quantity 158 ha @ 186,328.00 = 29,439,824. Item D599.9 is Rate "
              "Only under CESMM4 Class D (BOQ Rev B).")


def test_item_codes_and_standards_are_not_read_as_the_engineer():
    assert extract_party_names(BOQ_EXCERPT) == []
    assert withhold_party_names(BOQ_ANSWER, BOQ_EXCERPT) == BOQ_ANSWER


def test_a_firm_written_in_capitals_is_still_found_through_its_registered_name():
    names = dict(extract_party_names(EXCERPT))
    assert any(n.startswith("EXAMPLECO") for n in names)
