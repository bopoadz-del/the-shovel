"""The procedure catalogue ships procedure KINDS; a live document's code is
resolved at run time from the corpus. Synthetic names throughout."""
from __future__ import annotations

import json
import re

from app.core import procedure_catalogue as pc

NAMES = [
    "XYZ-101_Change Control.pdf",
    "XYZ-240 Design Review and Acceptance.docx",
    "AB-2001-101_Vol 1.pdf",               # a contract id, not a procedure code
    "Site Diary 2031-04-02.pdf",
    "QRS_330_HSE Audit.pdf",
]


def test_shipped_catalogue_carries_no_document_codes():
    text = pc.CATALOGUE_PATH.read_text(encoding="utf-8")
    assert not re.search(r"\b[A-Z]{2,6}-\d{3}\b", text.replace("RFI-", "").replace("NCR-", "")), (
        "a code-shaped token is shipped in the catalogue"
    )
    data = json.loads(text)
    for kind, rec in data["procedures"].items():
        assert rec["kind"] == kind
        assert rec["label"] and rec["match_phrases"]


def test_kind_for_name_reads_the_title():
    assert pc.kind_for_name("XYZ-101_Change Control.pdf") == "change_management"
    assert pc.kind_for_name("XYZ-240 Design Review and Acceptance.docx") == "design_review"
    assert pc.kind_for_name("QRS_330_HSE Audit.pdf") == "hse_audit"
    assert pc.kind_for_name("Site Diary 2031-04-02.pdf") is None


def test_document_code_is_the_leading_controlled_code_only():
    assert pc.document_code("XYZ-101_Change Control.pdf") == "XYZ-101"
    assert pc.document_code("QRS_330_HSE Audit.pdf") == "QRS-330"
    assert pc.document_code("AB-2001-101_Vol 1.pdf") is None
    assert pc.document_code("Change Control.pdf") is None


def test_resolve_uses_the_live_code_else_the_kind_id():
    got = pc.resolve_procedure("change_management", NAMES)
    assert got == {"procedure_id": "XYZ-101", "procedure_document": "XYZ-101_Change Control.pdf",
                   "procedure_kind": "change_management"}
    assert pc.resolve_procedure("interim_payment", NAMES)["procedure_id"] == "interim_payment"
    assert pc.resolve_procedure("change_management", None)["procedure_id"] == "change_management"


def test_a_typed_code_is_expanded_with_its_kind_label():
    out = pc.expand_codes("what does XYZ-101 say about type B changes?", NAMES)
    assert "change management" in out.lower()
    assert pc.expand_codes("what does XYZ-999 say?", NAMES) == "what does XYZ-999 say?"
    assert pc.expand_codes("what does XYZ-101 say?", []) == "what does XYZ-101 say?"


def test_a_year_is_not_a_document_code():
    assert pc.document_code("FIDIC 2017 Change Management Guide.pdf") is None
    assert pc.document_code("XYZ-2101_Change Control.pdf") == "XYZ-2101"
