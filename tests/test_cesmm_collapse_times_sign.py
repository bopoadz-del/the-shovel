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

from app.core.rag.retriever import extract_query_identifiers
from app.core.rag.vector_store import normalize_cesmm_item_codes as n


def test_times_sign_after_a_number_is_kept():
    assert n("Footing 3 x 2 x 0.45 m") == "Footing 3 x 2 x 0.45 m"
    assert n("pad 1.2 X 1.2 X 0.60 deep") == "pad 1.2 X 1.2 X 0.60 deep"


def test_times_sign_after_a_number_with_its_unit_is_kept():
    assert n("Wall 10 m x 3 m x 0.20 m") == "Wall 10 m x 3 m x 0.20 m"
    assert n("beam 6 mm x 0.35 thick") == "beam 6 mm x 0.35 thick"


def test_class_x_code_not_after_a_number_still_collapses():
    assert n("rate for X 312.4") == "rate for X312.4"
    assert n("X 101.1 and D 549.2") == "X101.1 and D549.2"


def test_other_class_letters_after_a_number_still_collapse():
    assert n("Bill 2 D 549.2") == "Bill 2 D549.2"
    assert n("page 7 E 425.1 item") == "page 7 E425.1 item"


def test_dimension_ask_yields_no_reference_identifier():
    assert extract_query_identifiers("Footing 3 x 2 x 0.45 m. Concrete?") == []
    assert extract_query_identifiers("Wall 10 m x 3 m x 0.20 m, blocks?") == []


def test_class_x_code_is_still_a_reference_identifier():
    assert "x312.4" in extract_query_identifiers("What is the rate for X 312.4?")
    assert "x101.1" in extract_query_identifiers("Quantity of X 101.1 please")
