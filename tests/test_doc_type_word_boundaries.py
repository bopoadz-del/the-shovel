"""A filename keyword must match a WORD, not a substring.

Live-fire phase 3, T5: a real Work Inspection Request --
"MRH-QMS-C-01(02) Work Inspection Request (WIR).xlsx" -- classified as
`specification`, because the keyword "spec" matched INSIDE the word
"In-spec-tion". Every inspection document in the operator's QMS (WIRs,
inspection procedures, inspection reports) would misfile as a spec.

Mechanism: `classify` used bare substring membership (`kw in filename`).
Now keywords match at word boundaries, with hyphen/underscore/parenthesis
treated as separators since real controlled-document names are full of them.

Also registers the QA/QC types the operator's QMS actually uses -- NCR, WIR,
ITP, MAR/MIR submittals -- so those documents classify to their own type
instead of leaning on generic 'report'.
"""
from __future__ import annotations

from app.core.doc_types import classify


def test_the_live_failure_inspection_is_not_a_spec():
    out = classify("MRH-QMS-C-01(02) Work Inspection Request (WIR).xlsx")
    assert out["name"] != "specification", out
    assert out["name"] == "inspection_request", out


def test_a_real_spec_still_classifies_as_spec():
    out = classify("AB-2023-101_Vol 2 - Specification (3 of 11).pdf")
    assert out["name"] == "specification", out


def test_spec_at_word_boundary_still_matches():
    assert classify("structural spec rev2.pdf")["name"] == "specification"


def test_ncr_and_itp_have_their_own_types():
    ncr = classify("MRH-QMS-C-03(03) Non-Conformance Report (NCR).xlsx")
    assert ncr["name"] == "ncr", ncr
    itp = classify("1. LUM103W_ITP_inside tank pipe and fittings.xlsx")
    assert itp["name"] == "itp", itp


def test_report_keyword_no_longer_swallows_ncr():
    """'Non-Conformance Report' contains 'report'; the specific type must win
    over the generic one."""
    out = classify("Non-Conformance Report NCR-0042.xlsx")
    assert out["name"] == "ncr", out


def test_unmatched_stays_unrecognised_not_guessed():
    out = classify("holiday_photos_batch7.zip")
    assert out["name"] == "unrecognised"
    assert out["needs_user_confirmation"] is True
