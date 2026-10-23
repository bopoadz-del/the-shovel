"""A heading that SAYS "Contract Data" is not a Contract Data table.

Live on 2d9d9c0 the ``CONTRACT DATA particulars`` label -- and the retrieval
bonus that rides on it -- sat on three documents, none of them the contract:

  * two sets of progress-meeting minutes, whose agenda item "Contract Data"
    lists document transmittals
    ("15: Performance Bond form | Issued via <portal> on <date>")
  * a DIFFERENT contract, a consultancy agreement whose Part A is headed
    "Contract Particulars"

They outranked the real Contract Data PDF on every contract question. Five
chunks are returned; the real clause never arrived. The assistant therefore
answered "not in the retrieved excerpts" -- 3 runs out of 3 -- to the value of
the Performance Bond, the number of Milestones, the approved method of
electronic communication, and whether Sections apply. All four are stated in
that PDF. It never invented a figure; it was handed meeting minutes and said
so. The second document is the dangerous one: it can answer a question about
THIS contract with ANOTHER contract's terms.

The fix was verified before it was written down here, against the owner's
real files and against the exact text that was mislabelled on live (old rule
True, new rule False, for all three; the real PDF accepted in both of the
forms its text arrives in). Everything below is synthetic -- same shapes,
invented names and figures -- because real contract values do not belong in a
repository.
"""
from __future__ import annotations

import re

import pytest

from app.core.contract_data_chunks import (
    contract_data_heading_spans,
    contract_data_particulars_chunks,
    contract_data_spans,
    section_states_contract_particulars,
)

# -- the three shapes that were wrongly labelled ---------------------------

MEETING_MINUTES = """PROGRESS MEETING No. 3
Attendees and apologies
Contract Data
Document Description | Remarks
1: Conditions of Contract | Issued via the portal on 9/1/24
2: Contract Data | Issued via the portal on 9/1/24
13: Forms and Security | Issued via the portal on 9/1/24
14: Advance Payment Bond form | Issued via the portal on 9/1/24
15: Performance Bond form | Issued via the portal on 9/1/24
16: Parent Company Guarantee | Issued via the portal on 9/1/24
20: Pre-Tender Clarification | To Follow
Community District Commencement Completion
A North Quarter NA NA
B South Quarter Jan-24 May-25
9. AOB
"""

OTHER_CONTRACT = """PROFESSIONAL SERVICES AGREEMENT
PART A: CONTRACT PARTICULARS
RECITALS:
(A) The Client intends to develop the Project and wishes to appoint a
specialist to provide the Services for the Project.
(B) The Consultant acknowledges that the Client has entered into this
Agreement relying on the skill, care and expertise of the Consultant.
1: Client | Northfield Development Company
2: Consultant | Harbour Design Partners
3: Services | As described in Schedule 1
4: Key Personnel | As listed in Schedule 4
5: Governing language | English
"""

PROGRESS_REPORT = """MONTHLY PROGRESS REPORT
Contract Data
The Contractor notes that Retention of 10% has been applied to the
certificate this month. No other contractual matters to report.
"""

# -- the real shape, in the two forms its text arrives in ------------------

REAL_CONTRACT_DATA = """CONTRACT DATA
Clause (as amended) Description Data
1.1.1 Accepted Contract Amount SAR 8,640,000.00 (Eight Million Six Hundred
Forty Thousand Saudi Riyals)
1.1.27 Defects Notification Period 365 days from the issuance of Taking Over
Certificate
1.1.67 Sections (if applicable) Not applicable
1.1.75 Time for Completion (for the whole of the Works) 310 days from the
Commencement Date
1.3.1(a) Approved methods of electronic communication ProjectMail
4.3.3(a) Value of Performance Bond 10 % of the Accepted Contract Amount
4.3.7 Parent Company Guarantee Required No
8.8.1 Delay Damages (for the whole of the Works) 0.1% of the Contract Price
per calendar day
14.3.2(c) Percentage of Retention 10% of each Interim Payment Certificate
"""

# Some PDF table extraction yields one cell fragment per line. This is the
# form the real document is actually stored in on live.
REAL_ONE_WORD_PER_LINE = re.sub(r"[ \t]+", "\n", REAL_CONTRACT_DATA)


@pytest.mark.parametrize(
    "name,text",
    [
        ("meeting minutes", MEETING_MINUTES),
        ("a different contract", OTHER_CONTRACT),
        ("a progress report", PROGRESS_REPORT),
    ],
)
def test_a_heading_alone_no_longer_earns_the_label(name, text):
    # The heading IS there -- that is the whole trap -- and the old rule
    # labelled on the strength of it.
    assert contract_data_heading_spans(text), f"{name}: fixture lost its heading"

    assert contract_data_spans(text, "Contract Data.pdf") == [], name
    assert contract_data_particulars_chunks(text, "Contract Data.pdf") == [], name


@pytest.mark.parametrize(
    "form,text",
    [("clean", REAL_CONTRACT_DATA), ("one word per line", REAL_ONE_WORD_PER_LINE)],
)
def test_a_real_contract_data_table_still_earns_it(form, text):
    """The control. A gate that also rejected the real table would turn a
    ranking defect into a total loss of the Contract Data path."""
    assert contract_data_spans(text, "Contract Data.pdf"), form
    chunks = contract_data_particulars_chunks(text, "Contract Data.pdf")
    assert chunks, form
    assert any("Performance Bond" in c.replace("\n", " ") for c in chunks), form


def test_a_transmittal_line_names_a_key_but_states_no_particular():
    """"15: Performance Bond form | Issued via the portal" -- a high-signal
    key with no value behind it. This is the exact line that let meeting
    minutes outrank the contract on "What is the value of the Performance
    Bond?"."""
    line = "15: Performance Bond form | Issued via the portal on 9/1/24"
    assert not section_states_contract_particulars(line)


def test_one_stated_particular_is_not_a_table():
    """"Retention of 10%" turns up in a progress report. Two DIFFERENT
    particulars, each with a real value, is what a Contract Data table has."""
    assert not section_states_contract_particulars("Retention of 10% was applied.")
    assert section_states_contract_particulars(
        "Retention 10% of each certificate. Time for Completion 310 days."
    )


def test_the_same_particular_twice_does_not_count_twice():
    text = "Performance Bond 10% of the amount. Performance Security 10% again."
    assert not section_states_contract_particulars(text)


def test_not_applicable_does_not_count_as_a_stated_value():
    """Meeting minutes are full of "NA". It is a fine value for one row of a
    table already known to be Contract Data, and useless for deciding whether
    a section is one."""
    text = "Time for Completion NA. Delay Damages not applicable. Retention n/a."
    assert not section_states_contract_particulars(text)


def test_the_filename_fallback_is_gated_too():
    """A file NAMED "Contract Data" whose body is a dense list of
    key | value rows used to qualify with no heading at all."""
    rows = "\n".join(
        f"{n}: Document {n} | Issued via the portal on 9/1/24" for n in range(1, 12)
    )
    assert contract_data_spans(rows, "Contract Data transmittal.pdf") == []


def test_empty_and_whitespace_sections_are_not_particulars():
    assert not section_states_contract_particulars("")
    assert not section_states_contract_particulars("   \n\t  ")
