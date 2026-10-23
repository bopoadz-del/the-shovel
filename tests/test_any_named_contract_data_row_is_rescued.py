"""Any Contract Data row the question names is fetched -- not only five of them.

Live on b64bbd2, project master_corpus, three runs each, all 0/3:

    "What is the value of the Performance Bond?"
    "How many Milestones are there and what is the Time for Completion for
     Milestone 5?"
    "What is the approved method of electronic communication under the
     contract?"

Every answer was an honest "not in the retrieved excerpts". Every answer IS in
the corpus, in a correctly labelled ``CONTRACT DATA particulars`` chunk:

    4.3.3(a): | Value of Performance Bond: 10 % of the Accepted Contract Amount
    |: | | Milestone 5 | 731days from the date the Contractor is given ...
    electronic communication: Aconex

Asked WITH the words "Contract Data" in the question those chunks score 5.0
and take all five slots. Asked the way a person asks, they are not in the
result at all -- a volume's table-of-contents page and a drawing title block
win instead.

The cause is the shape of the rescue, not any one row. A scanned Contract
Data table embeds badly, so cosine never puts it in the candidate pool, and
every bonus downstream only re-scores what is already in the pool. The
out-of-pool fetch that fixes this existed -- for the Accepted Contract Amount,
Time for Completion, delay damages, the Defects Notification Period, the
Engineer, the Parent Company Guarantee and the Commencement Date. Seven rows,
each added after that row failed live. A Contract Data sheet has about sixty.

These tests pin the CLASS: a question that names a filled Contract Data row
gets that row, whichever row it is.
"""
from __future__ import annotations

import pytest

from app.core.rag.vector_store import Chunk

ACTIVE = "p_master"
CD_DOC = "cdreal"
TOC_DOC = "vol4"
GC_DOC = "gc"
DRAWING_DOC = "dwg"

CD_NAME = "XX-2099-001_Vol 1.0_Cond of Contract (complete)_Contract Data.pdf"
LABEL = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage "
    f"[{CD_NAME}].\nCONTRACT DATA\n"
)

# Row shapes copied from the live index; every name and figure is synthetic.
CD_BOND = LABEL + (
    "|: | Example Signatory (signatory@example.com) | |\n"
    "3.1: | The Engineer's duties: as set out in the Conditions | |\n"
    "| 4.3.3(a): | Value of Performance Bond: 10 % of the Accepted Contract Amount | |\n"
    "4.3.7: | Parent Company Guarantee Required: No | |\n"
)
CD_MILESTONE_LIST = LABEL + (
    "1.1.50: | | Milestones (if applicable) | Milestone 1 | North Quarter 1a\n"
    "|: | | Milestone 2 | North Quarter 1b\n"
    "|: | | Milestone 3 | South Quarter 2a\n"
    "|: | | Milestone 4 | South Quarter 2b\n"
    "|: | | Milestone 5 | East Quarter 3a\n"
    "|: | | Milestone 6 | East Quarter 3b\n"
    "1.1.67: | | Sections (if applicable) | Not applicable |\n"
)
CD_MILESTONE_TIMES = LABEL + (
    "1.1.75: | | Time for Completion (for\n"
    "the whole of the Works): 900 days from the Commencement Date |\n"
    "1.1.75: | | Time for Completion (by\n"
    "Milestone, if applicable): Milestone 1 | 300 days from the date the "
    "Contractor is given right of access to North Quarter 1a.\n"
    "|: | | Milestone 4 | 480days from the date the Contractor is given right "
    "of access to South Quarter 2b.\n"
    "|: | | Milestone 5 | 640days from the date the Contractor is given right "
    "of access to East Quarter 3a.\n"
)
CD_COMMS = LABEL + (
    "1.1.78: | Nominated Subcontractor | Notapplicable | |\n"
    "1.3.1(a): | Approved methods of\n"
    "electronic communication: ExampleDocs | |\n"
)
CD_INSURANCE = LABEL + (
    "18.2: | Deductible per occurrence: SAR 250,000 | |\n"
    "18.3: | Minimum third party insurance: SAR 20,000,000 | |\n"
)
CD_ACA = LABEL + (
    "| Clause (as: | Description | Data |\n"
    "1.1.1: | | Accepted Contract Amount: SAR 1,000,000,000.00 excluding VAT |\n"
    "1.1.27: | | Defects Notification Period: 365 days |\n"
)
CD_MILESTONE_DAMAGES = LABEL + (
    "8.8.1: | | Delay Damages (if\n"
    "applicable per Milestone): Milestone | Delay Damages\n"
    "|: | | Milestone 1 | 0.015% of the Contract Price per calendar day\n"
    "|: | | Milestone 2 | 0.015% of the Contract Price per calendar day\n"
)
CD_CHUNKS = (CD_BOND, CD_MILESTONE_LIST, CD_MILESTONE_TIMES, CD_COMMS,
             CD_INSURANCE, CD_ACA, CD_MILESTONE_DAMAGES)

# What actually won the live top-5.
TOC_PAGE = (
    "Contract No. XX-2099-001 Construction Contract between Example Employer "
    "and Example Contractor Volume 4 of 6 Schedules (2 of 2) TABLE OF CONTENTS "
    "VOLUME 1 CONTRACT AGREEMENT CONTRACT DATA CONDITIONS OF CONTRACT VOLUME 4 "
    "SCHEDULE 8 FORM OF PERFORMANCE BOND SCHEDULE 5 MILESTONES"
)
GC_BOND_CLAUSE = (
    "deliver to the Employer a duly executed Advance Payment Bond and a duly "
    "executed Performance Bond. The Advance Payment Bond and Performance Bond "
    "shall both be payable on demand, unconditional and irrevocable, in the "
    "amounts stated in the Contract Data."
)
GC_NOTICES_CLAUSE = (
    "1.3 Communications. Wherever these Conditions provide for the giving of "
    "notices, these communications shall be in writing and delivered by hand, "
    "or transmitted using any of the approved systems of electronic "
    "transmission as stated in the Contract Data."
)
GC_MILESTONE_DEFINITION = (
    "1.1.50 “Milestone” shall have the meaning given to such term in Schedule 5 "
    "[Project Schedule]. The Contractor shall complete each Milestone within "
    "the Time for Completion stated in the Contract Data."
)
DRAWING_TITLE = (
    "INFRASTRUCTURE DESIGN P.O. Box 000 Example Consultant DATUM ORIENTATION "
    "GEODETIC DATUM: WGS-84 PROJECT SYSTEM: UTM GENERAL ARRANGEMENT"
)
SPEC_CONCRETE = (
    "03 30 00 Cast-in-place concrete. The minimum compressive strength of "
    "concrete for culverts shall be 40 MPa at 28 days. Testing shall be "
    "carried out by an approved laboratory."
)

A7 = "What is the value of the Performance Bond?"
A4 = ("How many Milestones are there and what is the Time for Completion for "
      "Milestone 5?")
A8 = "What is the approved method of electronic communication under the contract?"
E2 = "If Milestone 1 is 30 days late, what are the milestone delay damages?"


def _chunk(cid, doc_id, score, text, index=0):
    return Chunk(chunk_id=cid, project_id=ACTIVE, doc_id=doc_id,
                 chunk_index=index, text=text, score=score)


def _cd_chunks():
    return [_chunk(f"cd{i}", CD_DOC, 0.0, text, index=i)
            for i, text in enumerate(CD_CHUNKS)]


@pytest.fixture
def ret(monkeypatch):
    """The live shape: cosine returns everything EXCEPT the Contract Data."""
    from app.core.rag import retriever

    semantic = [
        _chunk("toc", TOC_DOC, 0.74, TOC_PAGE),
        _chunk("gcbond", GC_DOC, 0.73, GC_BOND_CLAUSE, index=1),
        _chunk("gcnotice", GC_DOC, 0.73, GC_NOTICES_CLAUSE, index=2),
        _chunk("gcms", GC_DOC, 0.72, GC_MILESTONE_DEFINITION, index=3),
        _chunk("dwg", DRAWING_DOC, 0.71, DRAWING_TITLE),
        _chunk("spec", "spec", 0.70, SPEC_CONCRETE),
    ]
    names = {CD_DOC: CD_NAME, TOC_DOC: "Vol 4 - Schedule (2 of 3).pdf",
             GC_DOC: "general_conditions.pdf", DRAWING_DOC: "GA-001.pdf",
             "spec": "033000 Concrete.pdf"}

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in semantic if c.project_id == project_id][:k]

    def fake_chunks_for_docs(self, project_id, doc_ids, k_per_doc=12, **_kw):
        return _cd_chunks() if CD_DOC in (doc_ids or []) else []

    def fake_title_phrase(pid, phrase, limit=8):
        if "contract data" in (phrase or "").lower():
            return [{"id": CD_DOC, "original_name": CD_NAME, "file_path": ""}]
        return []

    vs = "app.core.rag.vector_store.VectorStore."
    monkeypatch.setattr(vs + "search", fake_search)
    monkeypatch.setattr(vs + "identifier_search",
                        lambda self, pid, identifiers, k=20: [])
    monkeypatch.setattr(vs + "chunks_for_docs", fake_chunks_for_docs)
    monkeypatch.setattr(vs + "chunks_containing_all",
                        lambda self, pid, needles, k=20, **_kw: [])
    monkeypatch.setattr(vs + "count", lambda self, pid=None: 11)
    monkeypatch.setattr(vs + "_verify_embedding_identity", lambda self: None)
    monkeypatch.setattr(retriever, "_doc_name_for_id",
                        lambda did: names.get(did, ""), raising=False)
    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase",
                        fake_title_phrase)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return retriever


def _texts(ret, question):
    chunks, _ = ret.retrieve_with_filter(question, ACTIVE, k=5)
    return [c.text for c in chunks]


def test_a7_the_performance_bond_row_leads(ret):
    texts = _texts(ret, A7)
    assert texts, "nothing retrieved"
    assert "Value of Performance Bond: 10 %" in texts[0]


def test_a4_the_milestone_5_duration_is_retrieved(ret):
    blob = "\n".join(_texts(ret, A4))
    assert "Milestone 5 | 640days" in blob


def test_a4_the_milestone_list_is_retrieved_too(ret):
    """The question has two halves; "how many" is answered by the list."""
    blob = "\n".join(_texts(ret, A4))
    assert "Milestones (if applicable)" in blob


def test_a8_the_electronic_communication_row_leads(ret):
    """Not a figure, and not a field any regex names. Still a filled row."""
    texts = _texts(ret, A8)
    assert texts
    assert "electronic communication: ExampleDocs" in texts[0]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What is the deductible per occurrence?", "Deductible per occurrence"),
        ("What is the minimum third party insurance?", "third party insurance"),
        ("Who is the Nominated Subcontractor?", "Nominated Subcontractor"),
    ],
)
def test_a_row_nobody_has_asked_about_yet_is_rescued(ret, question, expected):
    """Shape-invariance: rows that have never failed live, so no special case
    exists for them and none should be needed."""
    assert expected in "\n".join(_texts(ret, question))


def test_e2_a_percentage_of_the_contract_price_brings_the_price_with_it(ret):
    """Live 24d1c0c, 0/3. The rate row was retrieved at rank 1 and the answer
    stopped, correctly, at "0.45% of the Contract Price -- and the Contract
    Price is not in the retrieved context". A rate is half an answer to a
    question about money; the other half is one row up the same sheet."""
    blob = "\n".join(_texts(ret, E2))
    assert "Milestone 1 | 0.015% of the Contract Price" in blob
    assert "Accepted Contract Amount: SAR 1,000,000,000.00" in blob


def test_a7_a_percentage_of_the_aca_brings_the_aca_too(ret):
    """Not only delay damages: any row valued as a share of the contract sum."""
    blob = "\n".join(_texts(ret, A7))
    assert "Value of Performance Bond: 10 %" in blob
    assert "Accepted Contract Amount: SAR 1,000,000,000.00" in blob


def test_the_base_amount_never_outranks_the_row_that_was_asked_for(ret):
    assert "Value of Performance Bond" in _texts(ret, A7)[0]
    assert "0.015% of the Contract Price" in _texts(ret, E2)[0]


def _crowd_the_pool_with_copies_of_the_rate_row(monkeypatch):
    """Live 39d6b8d: the corpus holds the executed contract, the unsigned
    one and an OCR of each. Every copy carries the same 8.8.1 row, every copy
    earns the same lift, and together they fill all five slots."""
    copies = [
        _chunk(f"copy{i}", f"copy{i}", 0.90 - i / 100,
               "8.8.1 Delay Damages (if applicable per Milestone) Milestone Delay "
               "Damages Milestone 1 0.015% of the Contract Price per calendar day "
               f"Milestone 2 0.015% of the Contract Price per calendar day copy {i}")
        for i in range(6)
    ]
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.search",
        lambda self, pid, qvec, k, query_text=None: copies[:k],
    )


def test_e2_the_sum_keeps_a_slot_when_copies_of_the_rate_fill_the_rest(
    ret, monkeypatch,
):
    """The first E2 fix fetched the sum and gave it a bonus. Live it still
    came sixth of five. A reservation, not a bigger number: any constant
    large enough to clear that crowd is fitted to one corpus."""
    _crowd_the_pool_with_copies_of_the_rate_row(monkeypatch)
    texts = _texts(ret, E2)
    assert len(texts) == 5
    assert any("Accepted Contract Amount: SAR 1,000,000,000.00" in t for t in texts)
    assert "0.015% of the Contract Price" in texts[0]


def test_a_plain_rate_question_in_the_same_crowd_reserves_nothing(
    ret, monkeypatch,
):
    _crowd_the_pool_with_copies_of_the_rate_row(monkeypatch)
    texts = _texts(ret, "What are the milestone Delay Damages for Milestone 1?")
    assert not any("Accepted Contract Amount: SAR" in t for t in texts)


# ── a phrase that IS a row's label names that row ─────────────────────────
#
# Unseen Set 3, live d8d9573. "What is the Contract Date?" and "Who is the VT
# Subcontractor?" both missed the sheet: the first was answered from the
# Conditions' definition ("means the date stated in the Contract Data"), the
# second from ANOTHER contract's glossary. Counting content words cannot see
# either label: "contract" is on every line so it is not counted, and "VT"
# is two letters. But "Contract Date" and "VT Subcontractor", whole, are
# exactly what is printed in the label cell.

CD_DATES = LABEL + (
    "1.1.10: | | Contract Date | [Insert date of Letter of Award or date of NOA] |\n"
    "1.1.78: | VT Subcontractor | Not applicable | |\n"
    "14.6.3: | | Minimum Amount of Interim Payment Certificate | Not applicable |\n"
)


@pytest.mark.parametrize(
    "question, row",
    [
        ("What is the Contract Date?", "Contract Date | [Insert date"),
        ("Who is the VT Subcontractor?", "VT Subcontractor | Not applicable"),
    ],
)
def test_a_label_phrase_names_its_row(ret, monkeypatch, question, row):
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, pid, doc_ids, k_per_doc=12, **_kw: [
            _chunk("dates", CD_DOC, 0.0, CD_DATES, 2)],
    )
    assert ret.named_particulars_row_match(question, CD_DATES) > 0
    assert row in "\n".join(_texts(ret, question))


@pytest.mark.parametrize(
    "question",
    [
        # The phrase is in the chunk, but as part of a VALUE, not a label.
        "What is the Contract Price?",
        # Stopword-led: "the engineer" is not a label phrase.
        "Who is the Engineer under this contract?",
        "What is the date of the site visit?",
    ],
)
def test_a_phrase_that_is_not_a_label_names_nothing(ret, question):
    text = LABEL + (
        "8.8.1: | | Delay Damages: 0.1% of the Contract Price per calendar day |\n"
        "3.1: | The Engineer's duties: as set out in the Conditions | |\n"
        "1.1.10: | | Contract Date | [Insert date of Letter of Award] |\n"
    )
    assert ret.named_particulars_row_match(question, text) == 0, question


def test_the_sheets_own_heading_is_not_a_row_label(ret):
    """Found on the real index: "...listed in the Contract Data?" matched all
    ten chunks, because every one opens with the CONTRACT DATA heading."""
    assert "contract data" not in ret._label_phrases(
        "What are the major items of Plant listed in the Contract Data?")
    assert ret.named_particulars_row_match(
        "What is stated in the Contract Data?", CD_INSURANCE) == 0


def test_a_label_phrase_with_nothing_beside_it_is_still_not_an_answer(ret):
    bare = LABEL + "1.1.10: | | Contract Date | |\n1.1.1: | | Accepted Contract Amount: SAR 5 |\n"
    assert ret.named_particulars_row_match("What is the Contract Date?", bare) == 0


def test_the_reservation_itself_takes_the_contract_sum_and_only_when_asked(ret):
    """The reservation on its own, with no fence in front of it to mask a
    mistake. Ranked: the rate, then an insurance row that is ALSO an amount of
    money, then the sum."""
    rate = _chunk("rate", CD_DOC, 4.0, CD_MILESTONE_DAMAGES, 8)
    insurance = _chunk("ins", CD_DOC, 3.0, CD_INSURANCE, 9)
    aca = _chunk("aca", CD_DOC, 2.0, CD_ACA, 0)
    ranked = [rate, insurance, aca]

    kept = [rate]
    assert ret.reserve_monetary_base_row(E2, kept, ranked)
    assert kept == [aca], "an insurance deductible is money, but not the sum"

    kept = [rate]
    plain = "What are the milestone Delay Damages for Milestone 1?"
    assert not ret.reserve_monetary_base_row(plain, kept, ranked)
    assert kept == [rate]


def test_a_plain_rate_question_still_gets_the_rate_and_nothing_else(ret):
    """The control for the E2 exception. "What ARE the delay damages" is
    answered by the rate; only a question that applies a duration to it
    ("30 days late") needs the sum as well."""
    texts = _texts(ret, "What are the milestone Delay Damages for Milestone 1?")
    assert texts and "0.015% of the Contract Price" in texts[0]
    assert not any("Accepted Contract Amount: SAR" in t for t in texts)


@pytest.mark.parametrize(
    "question",
    [
        "If Milestone 1 is 30 days late, what are the milestone delay damages?",
        "Milestone 2 finishes 6 weeks behind; what delay damages apply?",
        "What are the delay damages for 14 calendar days of delay to Milestone 1?",
    ],
)
def test_any_phrasing_of_a_delay_duration_keeps_the_sum(ret, question):
    assert any("Accepted Contract Amount: SAR" in t for t in _texts(ret, question))


def test_a_row_that_is_not_a_share_of_anything_brings_nothing_extra(ret):
    assert not any("Accepted Contract Amount: SAR" in t for t in _texts(ret, A8))


def test_the_rescue_does_not_flood_the_top_k(ret):
    """It fetches the rows the question names, not the Contract Data file."""
    texts = _texts(ret, A7)
    rescued = [t for t in texts if t.startswith("CONTRACT DATA particulars")]
    assert 1 <= len(rescued) <= 2
    assert not any("Deductible per occurrence" in t for t in texts)


def test_a_label_repeated_on_every_page_still_takes_two_slots_at_most(
    ret, monkeypatch,
):
    """A bound volume repeats its running header; five windows that all match
    must not push every other source out of a five-slot answer."""
    pages = [
        _chunk(f"page{i}", CD_DOC, 0.0,
               CD_BOND + f"Page {i} of 46 | Volume 1\n", index=i)
        for i in range(5)
    ]
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, pid, doc_ids, k_per_doc=12, **_kw: pages,
    )
    texts = _texts(ret, A7)
    assert sum(t.startswith("CONTRACT DATA particulars") for t in texts) == 2


@pytest.mark.parametrize(
    "question",
    [
        "What is the minimum compressive strength of concrete for culverts?",
        "Which laboratory carries out concrete testing?",
        # One shared word ("Milestone") is a coincidence, not a named row.
        "Who attended the milestone review workshop?",
        # Half the question's words are in the chunk, but never two on one
        # row: "insurance" is on a row, "register" is on none.
        "What is the insurance register?",
        # Two words on one row ("third party"), in a question about
        # something else entirely.
        "Who attended the third party barbecue on Friday evening?",
    ],
)
def test_a_question_that_names_no_row_gets_no_contract_data(ret, question):
    """The control. A rescue that fired on any shared word would put a
    Contract Data chunk into every answer on the platform."""
    assert not any(t.startswith("CONTRACT DATA particulars")
                   for t in _texts(ret, question))


def test_a_definition_question_stays_on_the_glossary(ret):
    assert not any(
        t.startswith("CONTRACT DATA particulars")
        for t in _texts(ret, "What does Performance Bond mean?")
    )


def test_an_unfilled_row_is_not_an_answer(ret, monkeypatch):
    """A key with nothing beside it names the row and states nothing."""
    blank = LABEL + "4.3.3(a): | Value of Performance Bond | |\n"
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, pid, doc_ids, k_per_doc=12, **_kw: [
            _chunk("blank", CD_DOC, 0.0, blank)],
    )
    assert not any(t.startswith("CONTRACT DATA particulars")
                   for t in _texts(ret, A7))


def test_the_rest_of_a_bare_label_is_not_its_value(ret):
    """``Time for Completion for the whole of the Works`` with no cell beside
    it: the words after "whole" are still the label. Caught by the year-lock
    suite, which elects a contract on exactly this predicate — a bare key in
    an older contract's window must not take the pool from the newer one."""
    bare = LABEL + (
        "1.1.1 Accepted Contract Amount including VAT: SAR 1,000,000.00\n"
        "1.1.75 Time for Completion for the whole of the Works\n"
        "8.8 Delay Damages for the whole of the Works: 0.1% per day\n"
    )
    ask = "What is the Time for Completion for the whole of the Works?"
    assert ret.named_particulars_row_match(ask, bare) == 0
    # ...while the row below it, which IS filled, still counts for its own ask.
    assert ret.named_particulars_row_match(
        "What are the Delay Damages for the whole of the Works?", bare) > 0


def test_a_key_wrapped_over_two_lines_is_one_row(ret):
    assert ret.named_particulars_row_match(A4, CD_MILESTONE_TIMES) > 0


# ── a table that runs on into the next chunk is still one row ─────────────
#
# Unseen Set 3, live 5011b62 -- three failures, one cause:
#   A4  "Which Milestones have a Time for Completion of 397 days?"  (M1 and M7)
#   A5  "Which area does Milestone 9 cover and what is its Time for Completion?"
#   F1  "Among the Northern Community milestones, which have the longest ...?"
#
# Row 1.1.75 lists ten milestones. The page breaks after Milestone 5, and so
# does the chunk. The second half opens with the table's repeated header and
# then "Milestone 7 | 397 days ..." -- it carries NO "Time for Completion"
# label, so nothing that looks for the row by its label can find it. Every
# answer stopped, honestly, at Milestone 5.

CD_TIMES_CONTINUED = LABEL + (
    "amended): | Description | Data | |\n"
    "|: | Milestone 6 | | 640days from the date the Contractor is given right "
    "of access to East Quarter 3b.\n"
    "|: | Milestone 7 | | 300 days from the date the Contractor is given right "
    "of access to Northern Quarter 4a.\n"
    "|: | Milestone 9 | | 520days from the date the Contractor is given right "
    "of access to Northern Quarter 4c.\n"
    "1.1.78: | Nominated Subcontractor | Notapplicable | |\n"
)
CD_UNRELATED_NEXT = LABEL + (
    "14.2.1: | Advance Payment: 10% of the Accepted Contract Amount | |\n"
    "14.3(c): | Percentage of Retention: 10% | |\n"
)


def _sheet_with_continuation(monkeypatch):
    chunks = [
        _chunk("list", CD_DOC, 0.0, CD_MILESTONE_LIST, 0),
        _chunk("times", CD_DOC, 0.0, CD_MILESTONE_TIMES, 1),
        _chunk("cont", CD_DOC, 0.0, CD_TIMES_CONTINUED, 2),
        _chunk("other", CD_DOC, 0.0, CD_UNRELATED_NEXT, 3),
    ]
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore.chunks_for_docs",
        lambda self, pid, doc_ids, k_per_doc=12, **_kw: chunks,
    )


@pytest.mark.parametrize(
    "question, needle",
    [
        ("Which area does Milestone 9 cover and what is its Time for Completion?",
         "Milestone 9 | | 520days"),
        ("Which Milestones have a Time for Completion of 300 days?",
         "Milestone 7 | | 300 days"),
        ("Among the Northern Quarter milestones, which has the longest Time for Completion?",
         "Northern Quarter 4c"),
    ],
)
def test_the_second_half_of_the_table_comes_with_the_first(ret, monkeypatch, question, needle):
    _sheet_with_continuation(monkeypatch)
    assert needle in "\n".join(_texts(ret, question)), question


def test_the_first_half_is_still_there(ret, monkeypatch):
    """300 days is Milestone 1 AND Milestone 7: both halves are the answer."""
    _sheet_with_continuation(monkeypatch)
    blob = "\n".join(_texts(ret, "Which Milestones have a Time for Completion of 300 days?"))
    assert "Milestone 1 | 300 days" in blob and "Milestone 7 | | 300 days" in blob


def test_a_chunk_that_merely_follows_is_not_a_continuation(ret, monkeypatch):
    """The next chunk along is only part of the row if the NUMBERING runs on.
    The retention rows that come after the milestones do not."""
    _sheet_with_continuation(monkeypatch)
    blob = "\n".join(_texts(ret, "Which area does Milestone 9 cover and what is its Time for Completion?"))
    assert "Percentage of Retention" not in blob


def test_a_question_about_another_row_does_not_drag_the_milestones_in(ret, monkeypatch):
    _sheet_with_continuation(monkeypatch)
    assert "Milestone 9" not in "\n".join(_texts(ret, A7))


def test_another_documents_numbering_is_not_this_tables_second_half(ret):
    """A different contract's sheet can open on "Milestone 6" too. The second
    half of a table is in the same document as the first."""
    parent = _chunk("times", CD_DOC, 0.0, CD_MILESTONE_TIMES, 1)
    same_doc = _chunk("cont", CD_DOC, 0.0, CD_TIMES_CONTINUED, 2)
    other_doc = _chunk("foreign", "another-contract", 0.0, CD_TIMES_CONTINUED, 2)

    found = ret._enumeration_continuations(parent, [parent, other_doc, same_doc])

    assert [c.chunk_id for c in found] == ["cont"]
