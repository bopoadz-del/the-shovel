"""The allowed-numbers MEASUREMENT on a computed turn (redo of #411, rule 1).

#411 withheld a reply when it stated any number outside the question's
numbers and the tool's figures; live, that fired on the tool's own selected
rows. This measurement logs, without changing any answer, the numbers outside
that set (``outside_figures``) and outside it plus every number the computed
result prints (``outside_result``). Every bill here is invented in the test.
"""

from __future__ import annotations

import logging

import pytest

from app.agents import answer_exit
from app.core import bill_aggregate as ba


def _row(ref, description, qty, unit="No", path=("CHAMBERS",), page=4):
    return ba.SheetRow(path=tuple(path), ref=ref, description=description, unit=unit,
                       quantity=qty, rate="", amount="", pages=[page])


def _bill(rows, terms):
    result = ba.aggregate(rows, terms=terms)
    return {"name": "bill_rows_aggregate", "ok": True,
            "result": {"status": "success", "bills": [
                {"document_id": "doc-a", "filename": "bill.xlsx", "status": "success", **result}]}}


CHAMBERS = [
    _row("G", "Box chamber 2.4 x 2.4 m, depth 2.0-2.5 m", "14"),
    _row("H", "Box chamber 2.4 x 2.4 m, depth 2.5-3.0 m", "9"),
    _row("J", "Box chamber 2.4 x 2.4 m, depth 3.0-3.5 m", "7"),
    _row("K", "Box chamber 3.1 x 3.1 m, depth 2.0-2.5 m", "5"),
]
KERBS = [
    _row("A1", "Precast kerb type 2, 150 mm", "320", unit="m", path=("KERBS",), page=11),
    _row("A2", "Precast kerb type 4, 300 mm", "85", unit="m", path=("KERBS",), page=12),
]


def _turn(question, payload):
    turn = answer_exit.Turn(question=question)
    answer_exit.note_tool_result("bill_rows_aggregate", payload, turn)
    return turn


@pytest.mark.parametrize("rows,terms,question,text,figures_out,result_out", [
    # A selected row's own quantity: #411 withheld it, the result prints it.
    (CHAMBERS, "2.4", "how many 2.4 x 2.4 m chambers in total?",
     "So 30 No are billed in total. Item G alone carries 14 No.", [14.0], []),
    (KERBS, "kerb", "total length of kerb?",
     "The kerb totals 405 m. Item A2 carries 85 m.", [85.0], []),
    # The selected rows' own depth bands / sizes.
    (CHAMBERS, "2.4", "how many 2.4 x 2.4 m chambers?",
     "They run from 2.0 m down to 3.5 m.", [2.0, 3.5], []),
    (KERBS, "kerb", "how much kerb?",
     "The kerbs are 150 mm and 300 mm wide.", [150.0, 300.0], []),
])
def test_a_selected_rows_own_number_is_outside_the_figures_but_inside_the_result(
        rows, terms, question, text, figures_out, result_out):
    reading = answer_exit.measure_allowed_numbers(text, _turn(question, _bill(rows, terms)))
    assert reading["outside_figures"] == figures_out
    assert reading["outside_result"] == result_out


@pytest.mark.parametrize("rows,terms,question,text,foreign", [
    # An UNSELECTED row's quantity (K, 5 No) is foreign under both rules.
    (CHAMBERS, "2.4", "how many 2.4 x 2.4 m chambers?", "30 No in total, and K adds 5 No.", [5.0]),
    # A total the tool never stated is foreign under both rules.
    (KERBS, "kerb", "total length of kerb?", "The kerb totals 410 m.", [410.0]),
    (CHAMBERS, "2.4", "how many 2.4 x 2.4 m chambers?", "There are 41 No in total.", [41.0]),
])
def test_a_number_no_selected_row_prints_is_outside_both(rows, terms, question, text, foreign):
    reading = answer_exit.measure_allowed_numbers(text, _turn(question, _bill(rows, terms)))
    assert reading["outside_figures"] == foreign
    assert reading["outside_result"] == foreign


@pytest.mark.parametrize("rows,terms,question,text", [
    (CHAMBERS, "2.4", "how many 2.4 x 2.4 m chambers?", "30 No are billed in 3 rows."),
    (KERBS, "kerb", "total length of kerb?", "The kerb totals 405 m over 2 rows."),
    # The question's own numbers are allowed.
    (KERBS, "kerb", "is there kerb over 100 m?", "Yes: 405 m, more than 100 m."),
])
def test_the_tools_figures_and_the_questions_numbers_are_inside_both(rows, terms, question, text):
    reading = answer_exit.measure_allowed_numbers(text, _turn(question, _bill(rows, terms)))
    assert reading["outside_figures"] == []
    assert reading["outside_result"] == []


@pytest.mark.parametrize("text,stated", [
    ("1. 30 No in total\n2. 3 rows", 2),
    ("- 405 m\n- 2 rows", 2),
])
def test_a_list_marker_is_not_a_number(text, stated):
    reading = answer_exit.measure_allowed_numbers(text, _turn("q", _bill(CHAMBERS, "2.4")))
    assert reading["stated"] == stated


@pytest.mark.parametrize("text,value", [
    ("The total is 1,250 No.", 1250.0),
    ("The total is 1 250 No.", 1250.0),
    ("The total is ١٢٥٠ No.", 1250.0),
])
def test_a_grouped_or_arabic_number_is_one_number(text, value):
    reading = answer_exit.measure_allowed_numbers(text, _turn("q", _bill(CHAMBERS, "2.4")))
    assert reading["outside_figures"] == [value]


@pytest.mark.parametrize("tool,payload", [
    ("search_documents", {"ok": True, "result": {"hits": [{"text": "item 14 of 30"}]}}),
    ("remember_fact", {"ok": True, "result": {"key": "x", "value": "41"}}),
])
def test_a_turn_without_a_computed_result_is_not_measured(tool, payload):
    turn = answer_exit.Turn(question="q")
    answer_exit.note_tool_result(tool, payload, turn)
    assert turn.computed_results == []
    assert answer_exit.measure_allowed_numbers("There are 41 No.", turn) is None


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_answer_is_not_measured(text):
    assert answer_exit.measure_allowed_numbers(text, _turn("q", _bill(KERBS, "kerb"))) is None


@pytest.mark.parametrize("rows,terms,text,line", [
    (CHAMBERS, "2.4", "30 No in total. Item G alone carries 14 No.",
     "outside_figures=[14.0] outside_result=[]"),
    (KERBS, "kerb", "405 m in total; 410 m by my count.",
     "outside_figures=[410.0] outside_result=[410.0]"),
])
def test_the_end_event_is_unchanged_and_the_log_carries_numbers_never_the_text(
        rows, terms, text, line, caplog):
    turn = _turn("how many?", _bill(rows, terms))
    event = {"type": "end", "content": text}
    with caplog.at_level(logging.INFO, logger=answer_exit._LOG.name):
        out = answer_exit.check_end_event(event, turn=turn)
    assert out["content"] == text
    logged = [r.getMessage() for r in caplog.records if "allowed_numbers" in r.getMessage()]
    assert len(logged) == 1
    assert line in logged[0]
    assert text not in logged[0]
    assert "bill_rows_aggregate" in logged[0]


@pytest.mark.parametrize("text", [
    "30 No in total. Item G alone carries 14 No.",
    "There are 41 No in total.",
])
def test_the_whole_turn_result_is_unchanged(text):
    turn = _turn("how many?", _bill(CHAMBERS, "2.4"))
    out = answer_exit.check_result({"answer": text, "sources": []}, turn=turn)
    assert out["answer"] == text


@pytest.mark.parametrize("payload", [
    {"ok": True, "result": {"status": "success", "bills": [{"status": "success", "rows": "not-a-list"}]}},
    {"ok": True, "result": {"status": "success", "bills": "nope"}},
])
def test_a_malformed_result_never_breaks_the_turn(payload):
    turn = answer_exit.Turn(question="q")
    answer_exit.note_tool_result("bill_rows_aggregate", payload, turn)
    out = answer_exit.check_result({"answer": "30 No.", "sources": []}, turn=turn)
    assert out["answer"] == "30 No."


@pytest.mark.parametrize("tool,payload", [
    ("search_documents", {"ok": True, "result": {"hits": [{"text": "item 14 of 30"}]}}),
    ("no_such_tool_in_the_registry", {"ok": True, "result": {"value": 41}}),
])
def test_a_tool_that_declares_no_statement_is_skipped_without_a_warning(tool, payload, caplog):
    turn = answer_exit.Turn(question="q")
    with caplog.at_level(logging.WARNING, logger=answer_exit._LOG.name):
        answer_exit.note_tool_result(tool, payload, turn)
    assert turn.computed_results == []
    assert not [r for r in caplog.records if "computed result" in r.getMessage()]


@pytest.mark.parametrize("rows,terms,text", [
    # The payload's "ok": true is a flag, not the number 1.
    (CHAMBERS, "2.4", "Item G is 1 of the rows."),
    (KERBS, "kerb", "1 of them is type 4."),
])
def test_a_true_flag_in_the_result_is_not_the_number_one(rows, terms, text):
    reading = answer_exit.measure_allowed_numbers(text, _turn("q", _bill(rows, terms)))
    assert 1.0 in reading["outside_result"]


def _nested_dict(depth, leaf):
    out = leaf
    for _ in range(depth):
        out = {"k": out}
    return out


def _nested_list(depth, leaf):
    out = leaf
    for _ in range(depth):
        out = [out]
    return out


@pytest.mark.parametrize("payload,inside", [
    (_nested_dict(12, 777), False),
    (_nested_list(12, "item 777"), False),
    (_nested_dict(3, 777), True),
    (_nested_list(3, "item 777"), True),
])
def test_a_number_nested_past_the_walk_bound_is_not_read(payload, inside):
    assert (777.0 in answer_exit._payload_numbers(payload)) is inside
