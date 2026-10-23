"""A complete drawing number is recognised by its SHAPE, not by one
originator's code: any originator token in either token order is accepted,
and a half-match is not."""
from __future__ import annotations

import pytest

from app.blocks.drawing_qto import _is_full_drawing_number


@pytest.mark.parametrize("dn", [
    "AB-CDE-001-0000-KLM-DWG-TM-200-0000001-A",   # originator after the zone
    "AB-CDE-001-KLM-0000-DWG-WS-600-0000001-C",   # tokens 4-5 swapped
    "XY-QRS-123-0000-OPQR-DWG-EL-210-1234567",    # 4-letter originator, no revision
])
def test_full_numbers_from_any_originator_are_full(dn):
    assert _is_full_drawing_number(dn)
    assert _is_full_drawing_number(dn.lower())


@pytest.mark.parametrize("dn", [
    "", None,
    "AB-CDE-001-KLM",                 # half-match from a title-block fragment
    "AB-CDE-001-0000-KLM-DWG",        # truncated
    "SECTION A-A",                    # a title, not a number
    "AB-CDE-001-0000-KLM-DWG-TM-200-00 01-A",  # broken by a space
])
def test_partial_or_malformed_numbers_are_not_full(dn):
    assert not _is_full_drawing_number(dn)


@pytest.mark.parametrize("dn,rev", [
    ("AB-CDE-001-0000-KLM-DWG-TM-200-0000001-A", "A"),
    ("AB-CDE-001-0000-KLM-DWG-TM-200-0000001-P01", "P01"),
    ("AB-CDE-001-0000-KLM-DWG-TM-200-0000001", None),   # sequence number
    ("AB-CDE-001-0000-KLM", None),                      # originator token, not a revision
    ("", None),
])
def test_revision_is_read_from_the_tail_by_shape(dn, rev):
    from app.blocks.drawing_qto import _revision_from_number_tail

    assert _revision_from_number_tail(dn) == rev
