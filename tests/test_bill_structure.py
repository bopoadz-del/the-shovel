"""A scanned bill's section columns from its rows' heading paths (app.core.bill_structure)
and the reader that uses them (app.core.bill_aggregate).

Every bill here is invented in the test: a page header carried onto every
row, a page label whose numbers change per page, a section title printed on
the first page of its section only, wrapped preamble lines, a sub-heading.
Each decision is driven by at least two inputs with different outcomes.
"""

from __future__ import annotations

import pytest

from app.core import bill_aggregate as agg
from app.core import bill_structure as bs

PREAMBLE = ["Supply and lay", "Provide and fix", "Excavate for", "Form and cast"]
WRAPPED = ["including joints", "including supports", "including testing"]
SUBS = ["Pipes in trench", "Valves and chambers", "Fittings", "Bedding"]


def _bill(sections, *, header=("Acme Holdings", "Ref 4471"), label="Tender 9 Sheet Q/{s}/{p}",
          carried=True, unread=(), titled_pages=(1,), extra_above=None):
    """Rows of a scanned bill: ``sections`` is ``[(number, title, [rows per page])]``.

    Returns ``(paths, truth)`` with ``truth`` the (section number, page) of each row.
    """
    paths, truth = [], []
    first = label.format(s=sections[0][0], p=1)
    i = 0
    for number, title, pages in sections:
        for p, n_rows in enumerate(pages, 1):
            for r in range(n_rows):
                path = list(header)
                if carried:
                    path.append(first)
                if i not in unread:
                    path.append(label.format(s=number, p=p))
                if p in titled_pages and title:
                    path.append(title if p == 1 else f"{title} (cont'd)")
                if extra_above and (number, p) in extra_above:
                    path.append(extra_above[(number, p)])
                path += [PREAMBLE[r % 4], WRAPPED[i % 3], SUBS[i % 4]]
                paths.append(path)
                truth.append((str(number), str(p)))
                i += 1
    return paths, truth


TWO = [(2, "EARTHWORKS", [3, 3, 2]), (3, "DRAINAGE WORKS", [2, 4])]
THREE = [(4, "CONCRETE WORK", [2, 2]), (5, "STEEL FRAME", [3]), (7, "ROOF COVERINGS", [2, 2, 2])]


@pytest.mark.parametrize("sections", [TWO, THREE], ids=["two", "three"])
def test_section_key_and_page_come_from_the_varying_label_fields(sections):
    paths, truth = _bill(sections)
    found = bs.structure(paths)
    assert [(bs.key_text(r.section_key), r.page) for r in found.rows] == truth
    assert found.labelled == len(paths) and found.inherited == 0 and found.unkeyed == 0


@pytest.mark.parametrize("label", ["Lot 7 Bill {s} p {p}", "Contract 12/{s}-{p}"])
def test_constant_label_fields_are_not_part_of_the_key(label):
    paths, truth = _bill(TWO, label=label)
    found = bs.structure(paths)
    assert [bs.key_text(r.section_key) for r in found.rows] == [s for s, _ in truth]


@pytest.mark.parametrize("sections", [TWO, THREE], ids=["two", "three"])
def test_running_header_lines_never_reach_a_column(sections):
    paths, _ = _bill(sections)
    found = bs.structure(paths)
    for row in found.rows:
        for text in (row.section_title, row.sub_heading, row.preamble):
            assert "Acme" not in text and "4471" not in text and "Sheet" not in text
    assert "Acme Holdings" in found.running_words
    assert "Sheet" not in found.running_words  # the label is not a running header's word


@pytest.mark.parametrize("unread, inherited", [((3, 4), 2), ((8,), 1)])
def test_an_unreadable_label_inherits_the_key_before_it_and_is_counted(unread, inherited):
    paths, truth = _bill(TWO, unread=unread)
    found = bs.structure(paths)
    assert found.inherited == inherited
    assert found.labelled == len(paths) - inherited
    for i in unread:
        assert found.rows[i].key_inherited
        assert bs.key_text(found.rows[i].section_key) == truth[i - 1][0]
    assert not found.rows[0].key_inherited


@pytest.mark.parametrize("unread, unkeyed", [((0, 1), 2), ((0,), 1)])
def test_rows_before_any_readable_label_have_no_key(unread, unkeyed):
    paths, _ = _bill(TWO, unread=unread)
    found = bs.structure(paths)
    assert found.unkeyed == unkeyed
    assert [r.section_key for r in found.rows[:unkeyed]] == [None] * unkeyed
    assert found.rows[unkeyed].section_key == ("2",)


def test_the_first_pages_label_carried_onto_every_row_is_not_a_rows_own():
    carried, _ = _bill(TWO, unread=(5,))
    found = bs.structure(carried)
    # Row 5 prints only the carried first-page label: it inherits, it is not "page 1 of 2".
    assert found.rows[5].key_inherited
    assert found.rows[5].page == ""
    assert found.rows[4].page == "2"


@pytest.mark.parametrize("misread, read", [("Tender 9 Sheet O/3/2", True), ("Item 3 of 2", False)])
def test_an_ocr_reading_of_the_label_is_read_and_another_line_is_not(misread, read):
    paths, _ = _bill(TWO, unread=(9,))
    paths[9].append(misread)
    paths[9][-1], paths[9][-2] = paths[9][-2], paths[9][-1]  # keep the sub-heading last
    found = bs.structure(paths)
    assert found.rows[9].key_inherited is (not read)
    assert found.rows[9].page == ("2" if read else "")


@pytest.mark.parametrize("sections", [TWO, THREE], ids=["two", "three"])
def test_a_title_printed_on_one_page_belongs_to_every_row_of_its_section(sections):
    paths, truth = _bill(sections, titled_pages=(1,))
    found = bs.structure(paths)
    titles = {number: title for number, title, _ in sections}
    for row, (number, _page) in zip(found.rows, truth):
        assert row.section_title == titles[int(number)]


@pytest.mark.parametrize("pages", [(1, 2), (2,)])
def test_a_continued_title_is_its_original(pages):
    paths, truth = _bill(TWO, titled_pages=pages)
    found = bs.structure(paths)
    assert {r.section_title for r, t in zip(found.rows, truth) if t[0] == "2"} == {"EARTHWORKS"}


def test_the_next_sections_title_on_a_boundary_page_stays_with_its_own_section():
    # The boundary page prints the next title on 2 rows; its own section on 3.
    sections = [(2, "EARTHWORKS", [3, 3, 2]), (3, "DRAINAGE WORKS", [3, 2]), (4, "ROADWORKS", [3, 3])]
    paths, truth = _bill(sections, extra_above={(2, 3): "DRAINAGE WORKS"})
    found = bs.structure(paths)
    assert {r.section_title for r, t in zip(found.rows, truth) if t[0] == "2"} == {"EARTHWORKS"}
    assert {r.section_title for r, t in zip(found.rows, truth) if t[0] == "3"} == {"DRAINAGE WORKS"}


def test_a_heading_printed_in_most_sections_names_none_of_them():
    # Printed on the first page of every section: fewer than half the rows,
    # so not a running header, but in most sections.
    above = {(s, 1): "GENERAL REQUIREMENTS" for s, _t, _pages in THREE}
    paths, _ = _bill(THREE, extra_above=above)
    found = bs.structure(paths)
    assert all("GENERAL REQUIREMENTS" not in r.section_title for r in found.rows)
    one = {(4, 1): "GENERAL REQUIREMENTS"}
    found_one = bs.structure(_bill(THREE, extra_above=one)[0])
    assert "GENERAL REQUIREMENTS" in found_one.rows[0].section_title


@pytest.mark.parametrize("line, base", [
    ("EARTHWORKS (cont'd)", "EARTHWORKS"),
    ("EARTHWORKS [cont’d]", "EARTHWORKS"),
    ("EARTHWORKS - contd.", "EARTHWORKS"),
    ("EARTHWORKS (continued)", "EARTHWORKS"),
    ("EARTHWORKS (cont d|", "EARTHWORKS"),
    ("DISCONTINUED ITEMS", "DISCONTINUED ITEMS"),
    ("Contractor", "Contractor"),
    ("(cont'd)", ""),
])
def test_base_heading(line, base):
    assert bs.base_heading(line) == base


@pytest.mark.parametrize("sections", [TWO, THREE], ids=["two", "three"])
def test_wrapped_preamble_lines_are_one_element_and_the_last_level_is_the_sub_heading(sections):
    found = bs.structure(_bill(sections)[0])
    for i in (0, 1):
        row = found.rows[i]
        assert row.sub_heading == SUBS[i]
        assert row.preamble == f"{PREAMBLE[i]} {WRAPPED[i]}"


def test_a_bill_without_a_page_label_has_no_keys():
    paths = [["Acme Holdings", "Groundworks", f"Item {i}"] for i in range(4)]
    found = bs.structure(paths)
    assert found.label is None and found.unkeyed == 4
    assert bs.log_line(found).startswith("bill_structure rows=4 sections=1 ")


# ---- the reader -------------------------------------------------------------

def _rows(paths, descriptions, qty="2", unit="No"):
    return [agg.SheetRow(path=tuple(p), ref="", description=d, unit=unit, quantity=qty, rate="", amount="")
            for p, d in zip(paths, descriptions)]


def _culvert_bill():
    paths, truth = _bill(TWO)
    descriptions = [f"Culvert unit type {i % 3}" for i in range(len(paths))]
    return _rows(paths, descriptions), truth


@pytest.mark.parametrize("section, count", [("earthworks", 8), ("drainage works", 6)])
def test_a_section_word_counts_every_row_of_the_section_not_only_titled_pages(section, count):
    rows, _ = _culvert_bill()
    res = agg.aggregate(rows, terms="culvert", section=section)
    assert res["count"] == count
    assert [s["count"] for s in res["by_section"].values()] == [count]


def test_the_raw_pile_would_have_kept_the_titled_page_only():
    rows, _truth = _culvert_bill()
    pile = [r for r in rows if "earthworks" in agg._words(" ".join(r.path))]
    assert len(pile) == 3  # the first page of the section; the reader counts all 8
    assert agg.aggregate(rows, terms="culvert", section="earthworks")["count"] == 8


@pytest.mark.parametrize("question, keys, count", [
    ("How many culvert units in the drainage works?", ["3"], 6),
    ("How many culvert units in the earthworks?", ["2"], 8),
    ("How many culvert units in the whole bill?", None, 14),
])
def test_the_users_question_names_the_sections_whatever_the_model_passed(question, keys, count):
    rows, _ = _culvert_bill()
    res = agg.aggregate(rows, terms="culvert", question=question)
    assert res["sections_named_by_question"] == keys
    assert res["count"] == count


def test_a_question_word_in_every_title_names_no_section():
    paths, _ = _bill([(2, "WATER MAINS", [2, 2]), (3, "WATER TANKS", [2, 2])])
    rows = _rows(paths, ["Valve"] * len(paths))
    assert agg.question_sections(agg.with_sections(rows), "valves in the water works") is None
    assert agg.question_sections(agg.with_sections(rows), "valves in the water tanks") == {"3"}


def test_a_sheet_with_section_columns_is_read_as_written():
    written = [agg.SheetRow(path=("x",), ref="", description="Culvert", unit="No", quantity="1",
                            rate="", amount="", section_key="9", section_title="OUTFALLS",
                            sub_heading="Pipes")]
    assert agg.with_sections(written) == written
    derived = agg.with_sections(_culvert_bill()[0])
    assert {r.section_key for r in derived} == {"2", "3"}


@pytest.mark.parametrize("section, count", [("groundworks", 4), ("roofing", 0)])
def test_a_bill_without_a_label_matches_its_section_words_on_the_path(section, count):
    paths = [["Acme Holdings", "Groundworks", f"Item {i}"] for i in range(4)]
    rows = _rows(paths, ["Culvert"] * 4)
    assert agg.aggregate(rows, terms="culvert", section=section)["count"] == count


@pytest.mark.parametrize("description, matched", [
    ("Size 1.5 x 1.5m, depth 2.0-2.5m", True),
    ("Size 1.8 x 1.8m, depth 5.0-5.5m", False),
])
def test_a_decimal_number_is_one_word(description, matched):
    rows = _rows([["Bill", "Manholes"]], [description])
    assert agg.aggregate(rows, terms="1.5 x 1.5m")["count"] == (1 if matched else 0)


@pytest.mark.parametrize("text, words", [
    ("1.5 x 1.5m", {"1.5", "x", "1.5m"}),
    ("28,496.00 SAR", {"28,496.00", "sar"}),
    ("K137.1 pipe", {"k137", "1", "pipe"}),
])
def test_word_reading(text, words):
    assert agg._words(text) == words
