"""The CESMM OCR collapse must not read the times sign of a dimension chain
as an item class letter.

normalize_cesmm_item_codes collapses ``D 549.2`` to ``D549.2`` at index
time and at query time. A dimension written "a x b x 0.25" has the same
shape after the last number ("x 0.25"), so it was collapsed to "x0.25";
the query identifier extractor then kept "x0.25" as a document
reference, and a self-contained quantity ask was refused as a missing
reference. A letter x written straight after a number (with or without
its unit) is a times sign. Class X written as a code keeps collapsing.
"""

import pytest

from app.core.rag.retriever import extract_query_identifiers
from app.core.rag.vector_store import normalize_cesmm_item_codes as n


@pytest.mark.parametrize("text", [
    "Footing 3 x 2 x 0.45 m",
    "pad 1.2 X 1.2 X 0.60 deep",
])
def test_times_sign_after_a_number_is_kept(text):
    assert n(text) == text


@pytest.mark.parametrize("text", [
    "Wall 10 m x 3 m x 0.20 m",
    "beam 6 mm x 0.35 thick",
])
def test_times_sign_after_a_number_with_its_unit_is_kept(text):
    assert n(text) == text


@pytest.mark.parametrize("text, collapsed", [
    ("rate for X 312.4", "rate for X312.4"),
    ("X 101.1 and D 549.2", "X101.1 and D549.2"),
])
def test_class_x_code_not_after_a_number_still_collapses(text, collapsed):
    assert n(text) == collapsed


@pytest.mark.parametrize("text, collapsed", [
    ("Bill 2 D 549.2", "Bill 2 D549.2"),
    ("page 7 E 425.1 item", "page 7 E425.1 item"),
])
def test_other_class_letters_after_a_number_still_collapse(text, collapsed):
    assert n(text) == collapsed


@pytest.mark.parametrize("ask", [
    "Footing 3 x 2 x 0.45 m. Concrete?",
    "Wall 10 m x 3 m x 0.20 m, blocks?",
])
def test_dimension_ask_yields_no_reference_identifier(ask):
    assert extract_query_identifiers(ask) == []


@pytest.mark.parametrize("ask, code", [
    ("What is the rate for X 312.4?", "x312.4"),
    ("Quantity of X 101.1 please", "x101.1"),
])
def test_class_x_code_is_still_a_reference_identifier(ask, code):
    assert code in extract_query_identifiers(ask)
