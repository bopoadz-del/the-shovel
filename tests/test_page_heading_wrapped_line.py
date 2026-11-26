"""A wrapped line of prose is not the heading a page is filed under.

A PDF page wraps its sentences at a fixed width. A wrapped line that opens on a
number ("300 mm thick ...") or a single capital ("A minimum cover ...") and
stops mid-sentence passes the heading-like shape test, so it named the page
ahead of the page's own section heading. The line after it opens on a
lowercase word: that is the sign of a sentence carried on, and such a line is
skipped. Every input below is invented; two inputs per decision.
"""
from __future__ import annotations

import math

import pytest

from app.core.chunk_bounds import WRAP_FILL, _wrapped_at_margin, heading_line_for_page, page_heading_lines


def _page(opener: str, carry: str, heading: str) -> str:
    return f"{opener}\n{carry}\n{heading}\nThe works are measured net as fixed.\n"


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("450 mm deep ground beams are cast against a blinding of", "lean mix concrete and sand.", "3.4 GROUND BEAMS"),
        ("75 kg of fixings are allowed for each tonne of steel", "erected on the upper floors.", "11 Steel Fixings"),
    ],
)
def test_wrapped_line_opening_on_a_number_does_not_name_the_page(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A sealant joint is formed between each frame and the", "surrounding masonry opening.", "2.6 Sealants"),
        ("C class bricks are laid below the damp proof course in", "every external wall.", "14 BRICKWORK"),
    ],
)
def test_wrapped_line_opening_on_one_capital_does_not_name_the_page(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize("marker", ["a) all joints are struck flush", "(iv) the samples are labelled"])
def test_heading_followed_by_a_lowercase_list_item_still_names_the_page(marker):
    page = f"7.1 Pointing Works\n{marker}\nb) the face is brushed clean\n"
    assert heading_line_for_page(page) == "7.1 Pointing Works"


@pytest.mark.parametrize(
    "heading,body",
    [
        ("9 ROOF COVERINGS", "Tiles are fixed with two nails each."),
        ("B Drainage Schedule", "Pipes are laid to the falls shown."),
    ],
)
def test_heading_followed_by_a_capitalised_line_still_names_the_page(heading, body):
    assert heading_line_for_page(f"{heading}\n{body}\n") == heading


@pytest.mark.parametrize("heading", ["## Levels of the site", "# 5 Notes on the works"])
def test_markdown_heading_followed_by_a_lowercase_line_still_names_the_page(heading):
    page = f"{heading}\nrecorded by the surveyor before stripping.\n"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize(
    "first_head,second_head",
    [("4.1 Excavation", "4.2 FILLING"), ("6 PAINTING", "7 Glazing")],
)
def test_every_page_of_a_document_is_named_by_its_own_section(first_head, second_head):
    page_one = _page("200 mm of topsoil is stripped and stockpiled for", "reuse in the landscaped areas.", first_head)
    page_two = _page("A mist coat is applied to new plaster before the", "two finishing coats.", second_head)
    text = page_one + page_two
    headings = page_heading_lines(text, [(1, 0), (2, len(page_one))])
    assert headings == {1: first_head, 2: second_head}


# --- grader faults on 38e67f0 (REPORT_OPUSGRADE_S616): each decision, two inputs ---


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A minimum cover of 25 mm is kept to the outer", "ties.", "5.3 COVER TO TIES"),
        ("Joints are raked out and left open for later", "use) as directed.", "8 Pointing"),
    ],
)
def test_short_closing_word_is_not_a_list_marker(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "heading,body",
    [
        ("5.2 MIXING WATER", "pH of the water is tested weekly."),
        ("12 Brick Sizes", "mm figures below are work sizes."),
    ],
)
def test_capitalised_heading_over_a_lowercase_unit_line_still_names_the_page(heading, body):
    assert heading_line_for_page(f"{heading}\n{body}\n") == heading


@pytest.mark.parametrize(
    "heading,body",
    [
        ("8 Fire stopping", "mm gaps are sealed with mastic."),
        ("2 Floor screeds", "kg of cement per bag is printed."),
    ],
)
def test_heading_with_as_many_capitals_as_lowercase_words_is_not_prose(heading, body):
    assert heading_line_for_page(f"{heading}\n{body}\n") == heading


@pytest.mark.parametrize("marker", ["iv. the samples are labelled", "xii) the face is brushed clean", "(b) the edges are sealed"])
def test_sentence_case_heading_over_a_list_item_still_names_the_page(marker):
    page = f"7.1 Pointing of joints\n{marker}\n"
    assert heading_line_for_page(page) == "7.1 Pointing of joints"


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("600 mm wide strip footings are poured on the", "same day as excavated.", "2.2 FOOTINGS"),
        ("A damp proof membrane is lapped 150 mm at all", "joints and taped.", "6 Membranes"),
    ],
)
def test_blank_line_inside_a_wrapped_sentence_is_skipped(opener, carry, heading):
    page = f"{opener}\n\n{carry}\n{heading}\n"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize("heading", ["3 Cover to the bars", "9 Making good of finishes"])
def test_prose_shaped_heading_on_the_last_line_still_names_the_page(heading):
    assert heading_line_for_page(f"Notes are given below.\n{heading}\n") == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A lintel bearing is as set out in appendix", "iv.", "10 LINTELS"),
        ("150 mm kerbs are laid to the profile of detail", "c.", "7.4 Kerbs"),
    ],
)
def test_bare_letter_or_numeral_that_ends_a_sentence_is_not_a_list_marker(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("C Grade is used", "throughout the frame.", "4 STEEL"),
        ("A Mesh is laid", "over the membrane.", "6 SLABS"),
    ],
)
def test_the_opener_is_not_counted_as_a_capitalised_word(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize("marker", ["viii) the joints are raked out", "xiii. the bays are cast in turn"])
def test_four_letter_roman_numeral_is_a_list_marker(marker):
    page = f"7.1 Pointing of joints\n{marker}\n"
    assert heading_line_for_page(page) == "7.1 Pointing of joints"


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A sealant bead is run where the frames are made", "up. Joints are then tooled.", "8 SEALANTS"),
        ("300 mm wide bays are laid and the forms are struck", "in. Each bay is then cured.", "5.2 Slab Bays"),
    ],
)
def test_two_letter_word_that_ends_a_sentence_is_not_a_list_marker(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("2 Bars at 150 200 250 centres to", "each face of the wall.", "2.1 REINFORCEMENT"),
        ("7 Ties of 6 8 10 mm at", "spacings shown on the drawings.", "7.3 Links"),
    ],
)
def test_figures_in_a_wrapped_line_are_not_counted_as_capitalised_words(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


# --- grader fault on 2335d3c (REPORT_OPUSGRADE_S616): a short sentence-case
# heading over a lowercase line had room for that line's first word, so the
# page did not wrap there; it is a heading. Two inputs per decision. ---


@pytest.mark.parametrize(
    "heading,body",
    [
        ("6.1 Quality of water", "pH of mixing water shall be 6 to 8."),
        ("3.2 Cover to reinforcement", "mm values below are nominal."),
        ("7.1 Pointing of joints", "e.g. flush, struck or weathered as shown."),
        ("5.4 Loads on the frame", "kN values are unfactored."),
    ],
)
def test_sentence_case_heading_over_a_lowercase_line_still_names_the_page(heading, body):
    assert heading_line_for_page(f"{heading}\n{body}\n") == heading


@pytest.mark.parametrize(
    "first_head,body",
    [("6.1 Quality of water", "pH of mixing water shall be 6 to 8."),
     ("4.3 Laps in the bars", "mm laps are measured from the bend.")],
)
def test_sentence_case_heading_names_its_page_through_the_document_path(first_head, body):
    page_one = f"{first_head}\n{body}\n"
    page_two = _page("A mist coat is applied to new plaster before the", "two finishing coats.", "7 Glazing")
    headings = page_heading_lines(page_one + page_two, [(1, 0), (2, len(page_one))])
    assert headings == {1: first_head, 2: "7 Glazing"}


@pytest.mark.parametrize(
    "line,word",
    [("3 The joints are left open", "until"), ("A blinding is laid under the slab", "to")],
)
def test_wrap_is_read_at_the_measured_margin(line, word):
    carry = f"{word} the works are done."
    reach = len(line) + 1 + len(word)
    # The page's lines run wider than the line, a space and the next word fill
    # at WRAP_FILL: the word had room, so the line was not wrapped and names the page.
    roomy = "x" * (math.floor(reach / WRAP_FILL) + 1)
    page = f"{line}\n{carry}\n{roomy}\n9 SECTION HEAD\n"
    assert heading_line_for_page(page) == line
    # The widest the lines can run with the word still reaching WRAP_FILL of them:
    # the line wrapped.
    tight = "x" * math.floor(reach / WRAP_FILL)
    page = f"{line}\n{carry}\n{tight}\n9 SECTION HEAD\n"
    assert heading_line_for_page(page) == "9 SECTION HEAD"


@pytest.mark.parametrize(
    "heading,body",
    [("6.1 Quality of water", "pH of mixing water shall be 6 to 8 and is to"),
     ("5.4 Loads on the frame", "kN values given in the schedule are not yet")],
)
def test_a_wider_paragraph_below_a_heading_does_not_make_it_a_wrap(heading, body):
    page = f"{heading}\n{body}\nchecked by the engineer.\n"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize(
    "line,carry",
    [("A blinding of lean mix is laid under the whole", "slab before the steel is fixed."),
     ("300 mm of hardcore is rolled in layers below the", "ground floor slab as shown.")],
)
def test_a_wrapped_line_is_measured_against_the_rest_of_the_page(line, carry):
    # The last line is the shortest; the margin is read from the lines above it.
    page = f"{line}\n{carry}\n2 HARDCORE\nRates include for all the labours and the plant needed.\nend.\n"
    assert heading_line_for_page(page) == "2 HARDCORE"


@pytest.mark.parametrize(
    "line,word",
    [("3 The joints are left open", "against"), ("A blinding is laid under the", "whole")],
)
def test_a_line_that_fills_exactly_the_wrap_share_is_wrapped(line, word):
    # line + space + word is 34 characters, exactly 0.85 of a 40-character
    # page: it reaches the share, so the line wrapped there.
    assert len(line) + 1 + len(word) == 34
    page = f"{line}\n{word} the works are done.\n{'x' * 40}\n9 SECTION HEAD\n"
    assert heading_line_for_page(page) == "9 SECTION HEAD"


@pytest.mark.parametrize(
    "heading,body",
    [("4.2 Fixing of the panels", "mm joints are left between them as"),
     ("8.1 Testing of the drains", "pH of the outfall is checked by the")],
)
def test_a_heading_three_quarters_as_wide_as_the_page_is_not_a_wrap(heading, body):
    # The heading and the next word fill between 70 % and 85 % of the page's
    # measured width: room was left on the line, so it is a heading.
    reach = len(heading) + 1 + len(body.split()[0])
    width = len(body)
    assert 0.70 * width <= reach < WRAP_FILL * width
    page = f"{heading}\n{body}\nframe as shown.\n"
    assert heading_line_for_page(page) == heading


# A page whose other lines are short (a schedule under its heading, a short
# page, a narrow column) has a narrow measured width. A numbered heading in
# sentence case fills that width, but it is still the page's heading.
@pytest.mark.parametrize(
    "page,heading",
    [
        ("5.4 Loads on the frame\nkN values are unfactored\nDead 2.5\nLive 1.5\nWind 0.8\nNotes\n",
         "5.4 Loads on the frame"),
        ("9.2 Rates for the pumps\nm3 per hour at the inlet\nPump A 40\nPump B 55\nEnd\n",
         "9.2 Rates for the pumps"),
        ("7.1 Pointing of joints\ne.g. flush, struck or weathered\nas shown.\n", "7.1 Pointing of joints"),
        ("6.3 Curing of the slab\nmm of water lost is to be\nmade good.\n", "6.3 Curing of the slab"),
        ("3.2 Cover to reinforcement bars\nmm values below are nominal and\nare kept at all faces of the\n"
         "members by spacers of a type\nthe engineer has approved.\n", "3.2 Cover to reinforcement bars"),
        ("8.3 Joints in the blockwork wall\nmm gaps are filled with mortar as\nthe courses are laid, and are\n"
         "struck flush on the face of the\nwall when it is finished.\n", "8.3 Joints in the blockwork wall"),
    ],
)
def test_numbered_sentence_case_heading_on_a_page_of_short_lines_names_the_page(page, heading):
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize(
    "first,head",
    [("5.4 Loads on the frame\nkN values are unfactored\nDead 2.5\nLive 1.5\nNotes\n", "5.4 Loads on the frame"),
     ("9.2 Rates for the pumps\nm3 per hour at the inlet\nPump A 40\nPump B 55\nEnd\n", "9.2 Rates for the pumps")],
)
def test_short_line_page_keeps_its_heading_through_the_document_path(first, head):
    second = _page("A mist coat is applied to new plaster before the", "two finishing coats.", "7 Glazing")
    headings = page_heading_lines(first + second, [(1, 0), (2, len(first))])
    assert headings == {1: head, 2: "7 Glazing"}


@pytest.mark.parametrize(
    "line,carry",
    [
        # A quantity opener: the word after the number is a lowercase unit.
        ("3 mm joints are left in the", "frame as shown."),
        # A unit symbol that opens on a capital is not a capitalised word.
        ("30 N/mm2 concrete is placed in", "the pile caps."),
        ("300 Pa of suction is held on", "the duct run."),
        # A decimal is a quantity too; its unit may open on a capital.
        ("2.5 MPa grout is pumped into", "the ducts."),
        ("0.5 litres of primer is used per", "square metre."),
        # A count followed by a capitalised noun is not a section number.
        ("4 Bolts are fixed at each of", "the base plates."),
    ],
)
def test_a_number_and_a_unit_on_a_page_of_short_lines_is_still_a_wrap(line, carry):
    # The sentence-case rule must not fire: these lines open on a quantity.
    page = f"{line}\n{carry}\n4 PILE CAPS\nAs shown.\n"
    assert heading_line_for_page(page) == "4 PILE CAPS"


@pytest.mark.parametrize(
    "line,wrapped",
    [
        # A capital and two lowercase letters is a word; a capital and one is a symbol.
        ("4.1 The joints are left in the", False),
        ("4.1 Th joints are left in the", True),
    ],
)
def test_the_capitalised_word_needs_two_lowercase_letters(line, wrapped):
    page = f"{line}\nframe as shown.\n4 PILE CAPS\nAs shown.\n"
    assert heading_line_for_page(page) == ("4 PILE CAPS" if wrapped else line)


@pytest.mark.parametrize(
    "line,word",
    [("A blinding is laid under the slab", "to"), ("C bricks are laid below the course", "in")],
)
@pytest.mark.parametrize("pad", ["      ", "\t\t\t"])
def test_trailing_padding_is_not_counted_toward_the_margin(line, word, pad):
    # The page's lines run just wider than the line and the next word reach at
    # WRAP_FILL: the line was not wrapped, and its padding does not change that.
    reach = len(line) + 1 + len(word)
    roomy = "x" * (math.floor(reach / WRAP_FILL) + 1)
    assert len(line + pad) + 1 + len(word) >= WRAP_FILL * len(roomy)
    page = f"{line}{pad}\n{word} the works are done.\n{roomy}\n9 SECTION HEAD\n"
    assert heading_line_for_page(page) == line


@pytest.mark.parametrize("heading", ["A CONCRETE cover", "C BRICK wall"])
def test_the_word_after_a_capital_opener_is_counted(heading):
    # One capitalised word against one lowercase word is not prose, even over a
    # lowercase line at the margin.
    page = f"{heading}\nmm gaps\n{'x' * len(heading)}\nend.\n"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize(
    "line,carry",
    [("3 The joints are left open", "until the works are done."),
     ("A blinding is laid under the slab", "to the levels shown.")],
)
def test_a_page_with_no_line_to_measure_shows_no_wrap(line, carry):
    # A prose-shaped line that is not a numbered sentence-case heading, over a
    # lowercase last line: no other line shows the margin, so nothing shows a wrap.
    assert heading_line_for_page(f"{line}\n{carry}\n") == line
    # The same two lines with a measured line at the margin below: a wrap.
    margin = "x" * len(line)
    assert heading_line_for_page(f"{line}\n{carry}\n{margin}\n9 SECTION HEAD\n") == "9 SECTION HEAD"


# --- grader faults on ab655f5 (REPORT_OPUSGRADE_S616): a heading the sentence-case
# rule does not take (a one-level number, a letter opener, a short first word)
# over a lowercase line on a page of short lines is shaped like a wrap. It gives
# way to a later heading, but a page with no other heading keeps it. ---


@pytest.mark.parametrize(
    "page,heading",
    [
        ("4 Loads on the frame\nkN values are unfactored\nDead 2.5\nLive 1.5\nWind 0.8\nNotes\n",
         "4 Loads on the frame"),
        ("3 Cover to reinforcement bars\nmm values below are nominal and\nshall be increased by 10 mm where\n"
         "exposure is severe.\n", "3 Cover to reinforcement bars"),
        ("A Loads on the frame\nkN values are unfactored\nDead 2.5\nLive 1.5\nWind 0.8\nNotes\n",
         "A Loads on the frame"),
        ("8.2 On site testing of concrete\nmm values are nominal here\nSlump 75\nCube 40\nNotes\n",
         "8.2 On site testing of concrete"),
        ("2.1 In situ concrete to the walls\nmm joints are left where shown\nC32 40\nEnd\n",
         "2.1 In situ concrete to the walls"),
    ],
)
def test_wrap_shaped_heading_with_no_later_heading_still_names_the_page(page, heading):
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize(
    "line,carry,later",
    [("4 Loads on the frame", "kN values are unfactored", "4.1 DEAD LOADS"),
     ("8.2 On site testing of concrete", "mm values are nominal here", "8.3 CUBES")],
)
def test_wrap_shaped_line_gives_way_to_a_later_heading(line, carry, later):
    # The same wrap-shaped line: alone it names the page, with a heading below it gives way.
    assert heading_line_for_page(f"{line}\n{carry}\nDead 2.5\nNotes\n") == line
    assert heading_line_for_page(f"{line}\n{carry}\n{later}\nDead 2.5\nNotes\n") == later


@pytest.mark.parametrize(
    "first,head",
    [("4 Loads on the frame\nkN values are unfactored\nDead 2.5\nLive 1.5\nNotes\n", "4 Loads on the frame"),
     ("A Pumps on the site\nm3 per hour at the inlet\nPump 40\nEnd\n", "A Pumps on the site")],
)
def test_wrap_shaped_heading_names_its_page_through_the_document_path(first, head):
    second = _page("450 mm deep ground beams are cast against a blinding of", "lean mix concrete and sand.",
                   "3.4 GROUND BEAMS")
    headings = page_heading_lines(first + second, [(1, 0), (2, len(first))])
    assert headings == {1: head, 2: "3.4 GROUND BEAMS"}


@pytest.mark.parametrize(
    "line,carry,furniture",
    [("4 Loads on the frame", "kN values are unfactored", "SPECIFICATION VOLUME"),
     ("A Pumps on the site", "m3 per hour at the inlet", "ISSUED FOR TENDER")],
)
def test_wrap_shaped_line_is_kept_ahead_of_page_furniture(line, carry, furniture):
    def running(text: str) -> bool:
        return text.strip() == furniture

    page = f"{furniture}\n{line}\n{carry}\nDead 2.5\nNotes\n"
    assert heading_line_for_page(page, running=running) == line
    # Negative: furniture alone still names the page.
    assert heading_line_for_page(f"{furniture}\nDead 2.5\n", running=running) == furniture


@pytest.mark.parametrize(
    "one,two",
    [("4 Loads on the frame", "5 Laps in the bars"), ("A Pumps on the site", "B Pipes to the tank")],
)
def test_the_first_of_two_wrap_shaped_lines_names_the_page(one, two):
    page = f"{one}\nkN values are unfactored\n{two}\nmm values are nominal\nNotes\n"
    assert heading_line_for_page(page) == one


@pytest.mark.parametrize(
    "heading,item,item2",
    [("3 Fixings to be used on site", "xiii) stainless steel bolts", "xiv) galvanised washers"),
     ("6 Anchors to be set in the wall", "viii) resin anchors are drilled", "xii) the holes are cleaned")],
)
def test_four_letter_numeral_under_a_heading_with_a_later_heading(heading, item, item2):
    # A four-letter numeral opens a list item: the heading is not carried on,
    # so the later heading does not take the page.
    page = f"{heading}\n{item}\n{item2}\n9 SECTION HEAD\nNotes are given below\n"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize(
    "one,two",
    [("SPECIFICATION VOLUME", "ISSUED FOR TENDER"), ("PART B WORKS", "REVISION C")],
)
def test_a_page_of_furniture_only_keeps_the_first_furniture_line(one, two):
    def running(text: str) -> bool:
        return text.strip() in (one, two)

    assert heading_line_for_page(f"{one}\n{two}\nDead 2.5\n", running=running) == one
    assert heading_line_for_page(f"{two}\n{one}\nDead 2.5\n", running=running) == two


@pytest.mark.parametrize(
    "line,following",
    [("3 The joints are left open", ["until the works are done."]),
     ("A blinding is laid under the slab", ["", "to the levels shown."])],
)
def test_wrapped_at_margin_shows_no_wrap_with_no_measured_width(line, following):
    # No other line measures the margin (width 0): nothing shows a wrap, however
    # long the line. With a measured width the same line and word do reach it.
    assert _wrapped_at_margin(line, following, 0) is False
    assert _wrapped_at_margin(line, following, len(line)) is True
