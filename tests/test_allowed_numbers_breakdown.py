"""The allowed-numbers hold MEASUREMENT (``would_withhold``) never flags what
a correct answer normally says (grader, 19 Nov 03:40Z, on #469 at 280f2bc).
The hold only measures (advisor, 19 Nov 04:34Z): every text here survives
whole, and "kept"/"withheld" below is the measurement's reading.

* A BREAKDOWN beside the total ('N No: A No in one section and B No in the
  other'; 'N m: A m + B m + C m') is not a restated total: only the claim
  BOUND to the total or correction wording is judged, and a clause ends at a
  colon and an opening bracket.
* The wording tables are narrow: 'should be', 'rather', 'actually',
  'overall', 'combined' state no total.
* A count of sections and a section's own subtotal are ALLOWED BY RULE: they
  break the tool's own selected rows down by the rows' own section field.
* A page number, an item reference and a year are ALLOWED BY RULE: they are
  locators, not figures. A number bound to one of the tool's units is always
  a figure.
* A range end is allowed by rule (a) when the question states it and by rule
  (d) when a selected row prints it.

Every bill is invented, with no digit in the descriptions, so a number the
test expects to pass passes by RULE and never because the payload happens to
carry the digit (asserted per case). Through ``exit_frames.with_answer_exit``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import answer_exit
from app.core import bill_aggregate as ba
from app.routers import exit_frames


def _row(ref, description, qty, unit, path, page):
    return ba.SheetRow(path=tuple(path), ref=ref, description=description, unit=unit,
                       quantity=qty, rate="", amount="", pages=[page])


MANHOLES = [
    _row("A", "Manhole, shallow band", "30", "No", ("MANHOLES", "TYPE ONE"), 7),
    _row("B", "Manhole, deep band", "12", "No", ("MANHOLES", "TYPE ONE"), 7),
    _row("D", "Manhole, shallow band", "8", "No", ("MANHOLES", "TYPE TWO"), 9),
    _row("E", "Manhole, shallow band", "6", "No", ("MANHOLES", "TYPE THREE"), 11),
    _row("G", "Gully, shallow band", "13", "No", ("GULLIES",), 15),   # not selected
]
KERBS = [
    _row("K", "Precast kerb, straight", "320", "m", ("KERBS", "STRAIGHT"), 21),
    _row("L", "Precast kerb, curved", "85", "m", ("KERBS", "CURVED"), 22),
    _row("M", "Precast kerb, curved dropped", "40", "m", ("KERBS", "CURVED"), 22),
    _row("N", "Footway edging", "112", "m", ("EDGINGS",), 24),          # not selected
]
CASES = {
    "manholes": (MANHOLES, "manhole", "how many manholes are there?"),
    "kerbs": (KERBS, "kerb", "what length of kerb is billed?"),
}


def _bill(rows, terms):
    result = ba.aggregate(rows, terms=terms)
    return {"name": "bill_rows_aggregate", "ok": True,
            "result": {"status": "success", "bills": [
                {"document_id": "doc-a", "filename": "bill.xlsx", "status": "success", **result}]}}


def _turn(case, question=None):
    rows, terms, asked = CASES[case]
    turn = answer_exit.Turn(question=question or asked)
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(rows, terms), turn)
    return turn


def _in_payload(case, number):
    rows, terms, _ = CASES[case]
    return any(answer_exit._same(number, n) for n in answer_exit._payload_numbers(_bill(rows, terms)))


def _frame(event):
    return f"data: {json.dumps(event)}\n\n"


def _stream_end(case, text, question=None):
    """The streamed answer's end content and the hold MEASUREMENT's reading
    of it (``would_withhold``), through the wrapper the chat routes use."""
    rows, terms, asked = CASES[case]
    readings = []
    measure = answer_exit.would_withhold

    def spy(t, turn=None):
        reading = measure(t, turn)
        readings.append(reading)
        return reading

    async def scripted():
        answer_exit.note_tool_result("bill_rows_aggregate", _bill(rows, terms))
        half = len(text) // 2
        for tok in (text[:half], text[half:]):
            yield _frame({"type": "token", "content": tok})
        yield _frame({"type": "end"})

    async def run():
        frames = [f async for f in exit_frames.with_answer_exit(scripted(), question or asked)]
        return json.loads(frames[-1][len("data: "):])["content"]
    answer_exit.would_withhold = spy
    try:
        out = asyncio.run(run())
    finally:
        answer_exit.would_withhold = measure
    assert text in out            # nothing is ever withheld: the text survives whole
    return out, readings[-1]


def _kept(case, text, question=None):
    _, reading = _stream_end(case, text, question)
    return reading["would_withhold"] is False


def _withheld(case, text, question=None):
    _, reading = _stream_end(case, text, question)
    return reading["would_withhold"] is True


# --- the two breakdown replies of the grader ---------------------------------

@pytest.mark.parametrize("case,text,derived", [
    ("manholes", ("In total there are 56 No: 42 No in the first section, 8 No in the second "
                  "and 6 No in the third."), 42),
    ("kerbs", "The total length is 445 m: 320 m straight and 125 m curved.", 125),
])
def test_a_total_with_its_section_breakdown_is_kept(case, text, derived):
    assert not _in_payload(case, derived)      # the section subtotal: by rule
    assert _kept(case, text)


@pytest.mark.parametrize("case,text", [
    ("manholes", "The total is 56 No: 30 No + 12 No + 8 No + 6 No."),
    ("kerbs", "The total length is 445 m: 320 m + 85 m + 40 m."),
])
def test_a_total_with_its_row_sum_is_kept(case, text):
    assert _kept(case, text)


@pytest.mark.parametrize("case,text", [
    ("manholes", "The total is 30 No + 12 No + 8 No + 6 No = 56 No."),
    ("kerbs", "The sum is 320 m + 85 m + 40 m = 445 m."),
])
def test_the_terms_of_a_sum_before_its_total_are_not_the_total(case, text):
    assert _kept(case, text)


# --- the breakdown rule does not open a door ---------------------------------

@pytest.mark.parametrize("case,text", [
    ("manholes", "In total there are 42 No: 30 No and 12 No."),   # a section's sum as THE total
    ("kerbs", "The total length is 320 m (straight only)."),
])
def test_a_wrong_total_before_its_breakdown_is_still_withheld(case, text):
    assert _withheld(case, text)


@pytest.mark.parametrize("case,text", [
    ("manholes", "Total: 42 No."),
    ("kerbs", "Correction: 85 m."),
])
def test_a_label_and_its_claim_after_the_colon_is_still_judged(case, text):
    assert _withheld(case, text)


@pytest.mark.parametrize("case,text", [
    ("manholes", "The gullies add 13 No more."),   # an unselected row's number
    ("kerbs", "The section with the edging comes to 517 m."),
])
def test_a_subtotal_over_an_unselected_row_is_withheld(case, text):
    assert _withheld(case, text)


# --- narrow wording ----------------------------------------------------------

@pytest.mark.parametrize("text,restated", [
    ("The depth should be at least 1.5 m.", False),
    ("Rather, the straight kerb is 320 m.", False),
    ("Actually the curved kerb is 85 m.", False),
    ("Overall the curved kerb is 125 m.", False),
    ("Combined, the curved runs are 125 m.", False),
    ("The total is 320 m.", True),
    ("It should read 320 m.", True),
])
def test_only_total_and_correction_words_bind_a_claim(text, restated):
    assert answer_exit._restates_a_total(text, [(445.0, "m")], [320.0, 85.0, 40.0]) is restated


@pytest.mark.parametrize("text,restated", [
    ("The total is 445 m: 320 m and 125 m.", False),
    ("The total (445 m) splits as 320 m [straight] and 125 m.", False),
    ("445 m in total.", False),
    ("320 m in total.", True),                       # the claim before the wording
    ("In total: 320 m.", True),                      # a label's claim
    ("The total (320 m) is straight only.", True),   # a label's claim in brackets
    ("The total is 320 m + 85 m.", False),           # terms of a sum, rows' own
    ("The total is 300 m + 85 m.", True),            # 300 is no row's number
    ("Total at 10:30 was 320 m.", True),             # a clock's colon is no clause end
])
def test_the_bound_claim(text, restated):
    assert answer_exit._restates_a_total(text, [(445.0, "m")], [320.0, 85.0, 40.0]) is restated


# --- MUST-KEEP: what a correct answer normally says -------------------------

@pytest.mark.parametrize("case,text,number", [
    ("manholes", "They fall in 3 sections.", 3),
    ("kerbs", "They fall in 2 sections.", 2),
])
def test_a_count_of_sections_is_kept_by_rule(case, text, number):
    assert not _in_payload(case, number)
    assert _kept(case, text)


@pytest.mark.parametrize("case,text,number", [
    ("manholes", "The first section holds 2 rows, see page 7.", 2),
    ("kerbs", "The curved section holds 2 rows.", 2),
])
def test_a_rows_per_section_count_is_kept_by_rule(case, text, number):
    assert not _in_payload(case, number)
    assert _kept(case, text)


@pytest.mark.parametrize("case,text,number", [
    ("manholes", "The schedule continues on page 27.", 27),
    ("kerbs", "See pp. 31 for the edging notes.", 31),
])
def test_a_page_number_is_kept_by_rule(case, text, number):
    assert not _in_payload(case, number)
    assert _kept(case, text)


@pytest.mark.parametrize("case,text,number", [
    ("manholes", "These are priced under item 4.2.1 of the bill.", 4.2),
    ("kerbs", "Clause 14 of the specification covers kerbs.", 14),
])
def test_an_item_reference_is_kept_by_rule(case, text, number):
    assert not _in_payload(case, number)
    assert _kept(case, text)


@pytest.mark.parametrize("case,text,number", [
    ("manholes", "The bill was issued in 2024.", 2024),
    ("kerbs", "The rates date from 2019.", 2019),
])
def test_a_year_is_kept_by_rule(case, text, number):
    assert not _in_payload(case, number)
    assert _kept(case, text)


@pytest.mark.parametrize("case,question,text", [
    ("manholes", "how many manholes from 4.5 to 6.75 m deep?", "None is billed from 4.5 m to 6.75 m."),
    ("kerbs", "what length of kerb between 2.5 and 3 m wide?", "No kerb row states a width of 2.5 to 3 m."),
])
def test_a_range_end_the_question_states_is_kept_by_rule(case, question, text):
    assert _kept(case, text, question)


# --- the locator and year rules do not open a door --------------------------

@pytest.mark.parametrize("case,text", [
    ("manholes", "There are 2024 No."),          # a year-shaped number bound to the unit
    ("kerbs", "Item K runs 2019 m."),
])
def test_a_year_shaped_number_in_the_tools_unit_is_a_figure(case, text):
    assert _withheld(case, text)


@pytest.mark.parametrize("case,text", [
    ("manholes", "There are 27 more."),           # no locator word
    ("kerbs", "Item K is 4.2 m wide."),           # a plain decimal, not a dotted reference
])
def test_a_number_without_a_locator_is_a_figure(case, text):
    assert _withheld(case, text)


@pytest.mark.parametrize("case,text", [
    ("manholes", "Page 7 lists 27 No."),          # the figure after the locator's number
    ("kerbs", "Section 3 carries 77 m."),
])
def test_a_locator_frees_only_its_own_number(case, text):
    assert _withheld(case, text)


# --- the derived numbers --------------------------------------------------

@pytest.mark.parametrize("case,expected,absent", [
    ("manholes", {3.0, 42.0, 56.0, 2.0, 1.0}, {13.0, 69.0}),
    ("kerbs", {2.0, 125.0, 445.0}, {112.0, 557.0}),
])
def test_derived_numbers_come_from_the_selected_rows_only(case, expected, absent):
    rows, terms, _ = CASES[case]
    derived = set(answer_exit._group_numbers(_bill(rows, terms)))
    assert expected <= derived
    assert not absent & derived


# --- the helpers' own rules, asserted directly ----------------------------
# Each rule below is asserted on inputs built here, with two inputs per
# decision: one the rule must act on, one it must leave alone.

def _cyclic_dict():
    payload = {"rows": [{"section": "doors", "qty": 4}]}
    payload["again"] = payload            # a payload that contains itself
    return payload


def _cyclic_list():
    payload = [{"unit": "m", "section": "kerbs", "qty": 9}]
    payload.append(payload)
    return payload


@pytest.mark.parametrize("make,rows", [(_cyclic_dict, 4), (_cyclic_list, 9)])
def test_a_self_containing_payload_is_walked_to_a_bounded_depth(make, rows):
    found = list(answer_exit._row_lists(make()))
    assert found and len(found) <= 10      # bounded, not a RecursionError
    assert all(rec.get("qty") == rows for recs in found for rec in recs)


@pytest.mark.parametrize("make,unit", [(_cyclic_dict, set()), (_cyclic_list, {"m"})])
def test_the_units_of_a_self_containing_payload_are_read_to_a_bounded_depth(make, unit):
    assert answer_exit._payload_units(make()) == unit


@pytest.mark.parametrize("value,keys", [
    ("  Doors ", ["doors"]),               # a text field is one group, normalised
    ("WINDOWS", ["windows"]),
    (["Ext", "Doors"], [("ext",), ("ext", "doors")]),   # a path is every prefix
    ("   ", []),                           # a blank text is no group
    (7, []),                               # a number is no group
])
def test_the_groups_a_field_puts_a_record_in(value, keys):
    assert answer_exit._group_keys(value) == keys


@pytest.mark.parametrize("records,present,absent", [
    ([{"section": "Doors", "qty": 2}, {"section": "doors", "qty": 3},
      {"section": "Windows", "qty": 4}], {5.0, 2.0, 4.0, 1.0}, {9.0}),
    ([{"trade": "Paint", "area": 11}, {"trade": "Tile", "area": 6},
      {"trade": "Tile", "area": 8}], {11.0, 14.0, 2.0, 1.0}, {25.0}),
])
def test_a_text_field_breaks_the_records_down(records, present, absent):
    derived = answer_exit._group_numbers({"rows": records})
    assert present <= derived
    assert not absent & derived


@pytest.mark.parametrize("records,sums", [
    ([{"section": "Doors", "qty": 2}, {"section": "Doors"},
      {"section": "Doors", "qty": 5}], {7.0}),            # a record without the field
    ([{"section": "Roof", "qty": 6}, {"section": "Roof", "qty": True},
      {"section": "Roof", "qty": 1}], {7.0}),             # a flag is not a quantity
])
def test_a_group_sum_counts_only_numeric_values(records, sums):
    derived = answer_exit._group_numbers({"rows": records})
    assert sums <= derived
    assert 8.0 not in derived


@pytest.mark.parametrize("text,units,figures", [
    ("See page 12 for 30 No.", {"no"}, {30.0}),            # a locator's number
    ("Clause 4.2.1 sets 14 m.", {"m"}, {14.0}),            # a dotted reference
    ("Billed in 2024 as 16 No.", {"no"}, {16.0}),          # a year
    ("There are 2024 No.", {"no"}, {2024.0}),              # bound to a unit: a figure
    ("Page 7 lists 27 No.", set(), {27.0}),
])
def test_the_figures_a_text_states(text, units, figures):
    assert answer_exit._figure_numbers(text, units) == figures
