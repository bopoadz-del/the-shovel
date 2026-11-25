"""A value's subject is read from its own field, not the whole row.

A spreadsheet row ("A | B | 1.5 | 81000") or a field record
("Sheet: X; Item: Y; Quantity: Z") has no sentence break. Before this
change the subject of a value in such a row was every cell before it, and
the clarifying question printed that row as the option's subject (hundreds
of words per option). The subject now comes from the field next to the
value. Number-only cells are stepped over. Prose with no field separator
reads exactly as before.
"""

from __future__ import annotations

import pytest

from app.agents.clarify_cases import cases_from_passages, clarifying_question


def _subject_of(text: str, value: str) -> str:
    cases = [c for c in cases_from_passages([("a.xlsx", text)]) if c.value == value]
    assert cases, f"no case read for {value!r}"
    return cases[0].subject.lower().strip()


# Fails on main: the subject was the whole row up to the value.
@pytest.mark.parametrize(
    "text,value,expected",
    [
        (
            "Item | Rate | Qty Site clearance works | 1.5 | 1.4 | Excavation depth | 2.5 m | 900",
            "2.5 m",
            "excavation depth",
        ),
        (
            "Zone | Area Plant room north | 14 | Ceiling void height | 600 mm | 3",
            "600 mm",
            "ceiling void height",
        ),
    ],
)
def test_table_row_subject_is_the_nearest_worded_cell(text, value, expected):
    subject = _subject_of(text, value)
    assert subject == expected
    assert "|" not in subject


# Fails on main: the record's earlier fields were part of the subject.
@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("Sheet: Costs; Item: Pump base; Thickness: 150 mm; Unit: nr", "150 mm", "thickness:"),
        ("Block: West; Level: Roof; Parapet height: 1100 mm; Notes: none", "1100 mm", "parapet height:"),
    ],
)
def test_record_field_subject_is_its_own_label(text, value, expected):
    subject = _subject_of(text, value)
    assert subject.rstrip(":") == expected.rstrip(":")
    assert ";" not in subject


# Number-only cells between the label and the value are stepped over.
@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("Slab thickness | 12 | 7 | 250 mm", "250 mm", "slab thickness"),
        ("Wall height | 3 | 3.2 m", "3.2 m", "wall height"),
    ],
)
def test_number_cells_are_stepped_over(text, value, expected):
    assert _subject_of(text, value) == expected


# The other side: prose with no field separator reads as it always did.
@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("The concrete cover to reinforcement shall be 40 mm.", "40 mm",
         "concrete cover to reinforcement"),
        ("The trench depth below ground level is 1.2 m.", "1.2 m",
         "trench depth below ground level"),
    ],
)
def test_prose_without_separators_is_unchanged(text, value, expected):
    assert _subject_of(text, value) == expected


# A semicolon between clauses separates two subjects in prose too.
@pytest.mark.parametrize(
    "text,first,second",
    [
        ("The cover is 50 mm; the slab is 200 mm.", ("50 mm", "cover"), ("200 mm", "slab")),
        ("The kerb is 150 mm; the verge is 2 m.", ("150 mm", "kerb"), ("2 m", "verge")),
    ],
)
def test_semicolon_clauses_keep_their_own_subject(text, first, second):
    assert _subject_of(text, first[0]) == first[1]
    assert _subject_of(text, second[0]) == second[1]


# End to end: an option built from a table row does not carry the row.
@pytest.mark.parametrize(
    "ask,draft,rows",
    [
        ("what is the height?", "The height is 3 m.", [
            ("a.xlsx", "Ref | Desc Blockwork wall partitions to plant room | 4 | Wall height | 3 m"),
            ("b.xlsx", "Ref | Desc Precast cladding panels to stair core | 9 | Panel height | 4.5 m"),
        ]),
        ("what is the depth?", "The depth is 600 mm.", [
            ("c.xlsx", "Code | Item Ground beams to grid lines | 2 | Beam depth | 600 mm"),
            ("d.xlsx", "Code | Item Retaining wall along east boundary | 5 | Stem depth | 450 mm"),
        ]),
    ],
)
def test_option_subject_is_short_for_a_table_row(ask, draft, rows):
    question = clarifying_question(ask, draft, cases_from_passages(rows))
    assert question is not None
    assert "|" not in question
    assert "Desc" not in question and "Item " not in question


# A cell whose words are all stopwords (a pronoun: "they are 25 mm") refers
# to a thing but does not name one. Read as the subject, "they" matched no
# asked thing. It is stepped over like a number cell, so the subject is the
# worded cell before it.
@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("Item: movement joints | they are 25 mm wide", "25 mm", "item: movement joints"),
        ("Screed to plant room | it is 75 mm thick", "75 mm", "screed to plant room"),
    ],
)
def test_pronoun_cell_is_stepped_over(text, value, expected):
    assert _subject_of(text, value) == expected


# Must not fire: a cell with one content word beside the stopwords still
# names its thing, and stays the subject.
@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("Item: movement joints | their width is 25 mm", "25 mm", "their width"),
        ("Screed to plant room | the depth is 75 mm", "75 mm", "depth"),
    ],
)
def test_worded_cell_beside_stopwords_still_names(text, value, expected):
    assert _subject_of(text, value) == expected


# Degenerate: every cell before the value names nothing. The reader falls
# back to the words after the value, as it does for a row of number cells.
@pytest.mark.parametrize(
    "text,value,expected",
    [
        ("they | it is 25 mm wide movement joints", "25 mm", "wide movement joints"),
        ("we | them | 40 mm concrete cover", "40 mm", "concrete cover"),
    ],
)
def test_all_cells_name_nothing_falls_back_after(text, value, expected):
    assert _subject_of(text, value) == expected
