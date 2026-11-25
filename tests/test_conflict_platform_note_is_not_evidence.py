"""A note the platform writes into a passage is not that document's text.

A tool's bracketed note (``[bill item rows: ... matches the item "..."]``,
``[context_truncated: ...]``) can quote the search item, and the search item
can carry a value the user typed. Read as document text, that value became a
second document's value and the answer led with "The documents differ on
this", naming a document that never states it. The rule is
``evidence_plan.is_platform_note``, the one the evidence loop already uses
to keep a note from covering its own plan item.

Every document, value and question here is invented for this file; the
expected values are written by hand. Each decision is shown on two inputs
or more, one each way.
"""

from __future__ import annotations

import pytest

from app.agents.evidence_conflicts import detect


def _row(doc: str, text: str) -> dict:
    return {"doc_name": doc, "page_or_section": "p. 1", "snippet": text}


_ITEM_NOTE = ("[bill item rows: every priced row of the project's bill(s) whose description "
              "matches the item \"{item}\", one per line, each led by the bill section it "
              "sits in (3 rows, not sampled).]")
_CUT_NOTE = "[context_truncated: the search for \"{item}\" returned more passages than shown]"

# (answer, governing passage, other document's lead line, search item quoting the typed value)
_CASES = [
    ("Late completion charges for 12 days are 36,000.00: 0.06% of the contract sum you gave "
     "(5,000,000.00) is 3,000.00 per day.",
     "CONTRACT PARTICULARS\n6.4.2 Late Completion Charges: 0.2% of the Contract Sum per calendar day",
     "The Supervisor shall give notice of any instruction affecting the Works.",
     "late completion charges rate (0.06% per day) of the contract sum"),
    ("The advance payment on the 9,000,000 you gave is 900,000.00: 10% of the contract sum.",
     "CONTRACT PARTICULARS\n14.2 Advance Payment: 15% of the Contract Sum, repaid by deduction",
     "Interim payments are certified monthly by the Supervisor.",
     "advance payment percentage (10%) of the contract sum"),
    ("The retention held on the 3,200,000 you gave is 128,000.00: 4% of that sum.",
     "CONTRACT PARTICULARS\n14.3 Retention Percentage (of each interim payment): 7% of the amount certified",
     "Interim payments are certified monthly by the Supervisor.",
     "Contract Particulars retention percentage (4%) of each interim payment"),
]


def _values(found: list[dict]) -> list[tuple[str, str]]:
    return [(v["value"], v["document"]) for c in found for v in c["values"]]


@pytest.mark.parametrize("note", [_ITEM_NOTE, _CUT_NOTE])
@pytest.mark.parametrize("answer, governing, lead, item", _CASES)
def test_a_value_only_a_platform_note_quotes_is_no_document_value(answer, governing, lead, item, note):
    rows = [_row("Harbour Wall Conditions.pdf", governing),
            _row("Harbour Wall General Terms.pdf", lead + "\n" + note.format(item=item))]
    assert detect(answer, rows) == []


@pytest.mark.parametrize("answer, governing, lead, item", _CASES)
def test_the_same_value_in_the_documents_own_text_still_differs(answer, governing, lead, item):
    """Not a blanket drop: when the second document's own text states the
    typed value, the two documents do differ and the conflict stands."""
    own = item[0].upper() + item[1:] + "."
    rows = [_row("Harbour Wall Conditions.pdf", governing),
            _row("Harbour Wall General Terms.pdf", lead + "\n" + own)]
    found = detect(answer, rows)
    docs = {d for _v, d in _values(found)}
    assert docs == {"Harbour Wall Conditions.pdf", "Harbour Wall General Terms.pdf"}, _values(found)


@pytest.mark.parametrize("label", ["[Section 4 - Retention] ", "[Bill No. 7 > Preliminaries] "])
def test_a_bracketed_section_label_leading_document_text_is_still_text(label):
    """Only a whole line shaped as a platform note is dropped; a line led by
    a section label (the document's own heading path) keeps its values."""
    answer, governing, lead, item = _CASES[2]
    rows = [_row("Harbour Wall Conditions.pdf", governing),
            _row("Harbour Wall General Terms.pdf",
                 lead + "\n" + label + "Retention percentage (4%) of each interim payment.")]
    docs = {d for _v, d in _values(detect(answer, rows))}
    assert docs == {"Harbour Wall Conditions.pdf", "Harbour Wall General Terms.pdf"}


_STEEL_ROWS = [
    _row("Pier Deck Specification.pdf", "Deck steelwork shall be grade S355."),
    _row("Pier Deck Drawing D-210.pdf", "Deck steelwork: grade S275."),
]
_RANKINGS = [
    ("Should the Drawings and the Specification disagree, the Drawings shall prevail",
     "Pier Deck Drawing D-210.pdf"),
    ("Where the Specification and the Drawings conflict, the Specification shall govern",
     "Pier Deck Specification.pdf"),
]


@pytest.mark.parametrize("statement, _governs", _RANKINGS)
def test_a_precedence_statement_only_a_platform_note_quotes_names_no_winner(statement, _governs):
    """``_precedence`` reads the same row text: a ranking that sits only in a
    note quoting the search item is no document's precedence statement."""
    note = _CUT_NOTE.format(item=statement)
    rows = _STEEL_ROWS + [_row("Pier General Conditions.pdf", "Notices shall be in writing.\n" + note)]
    found = detect("The deck steel is grade S355.", rows)
    assert found and found[0]["governs"] is None
    assert found[0]["basis_document"] is None


@pytest.mark.parametrize("statement, governs", _RANKINGS)
def test_the_same_precedence_statement_in_the_documents_own_text_names_the_winner(statement, governs):
    rows = _STEEL_ROWS + [_row("Pier General Conditions.pdf", "Notices shall be in writing.\n" + statement + ".")]
    found = detect("The deck steel is grade S355.", rows)
    assert found and found[0]["governs"] == governs
    assert found[0]["basis_document"] == "Pier General Conditions.pdf"


@pytest.mark.parametrize("case, cell", [(0, "Late Completion Charges: 0.06%"),
                                        (2, "Retention Percentage: 4%")])
def test_document_lines_around_a_dropped_note_stay_separate_lines(case, cell):
    """Dropping a note keeps the document's own lines apart: a table cell
    that ends in its value is not glued to the next line's first word (which
    would hide the value and lose a real conflict)."""
    answer, governing, lead, item = _CASES[case]
    other = "\n".join([lead, cell, _CUT_NOTE.format(item=item),
                       "Interim payments are certified monthly by the Supervisor."])
    rows = [_row("Harbour Wall Conditions.pdf", governing), _row("Harbour Wall General Terms.pdf", other)]
    docs = {d for _v, d in _values(detect(answer, rows))}
    assert docs == {"Harbour Wall Conditions.pdf", "Harbour Wall General Terms.pdf"}
