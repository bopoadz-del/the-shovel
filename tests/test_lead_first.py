"""LEAD FIRST ONLY on a computed turn (advisor, 19 Nov 04:34Z; #469 split).

On a turn whose computed result stated a figure, the tool's own statement
comes first in the answer, once. It is not repeated when the text already
carries it (the computed recovery rendered it). The model's text follows
UNCHANGED, and nothing is withheld. The hold rule of #469 only MEASURES
(``would_withhold``) and logs numbers, never the text. A turn without a
computed figure is untouched.

Every bill is invented here and run through the real ``aggregate`` and the
registry's statement. The stream cases go through
``exit_frames.with_answer_exit``, the wrapper the chat routes answer through.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.agents import answer_exit
from app.core import bill_aggregate as ba
from app.routers import exit_frames


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
# (rows, terms, question, the tool's total as the lead prints it)
CASES = {
    "chambers": (CHAMBERS, "2.4", "how many 2.4 x 2.4 m chambers in total?", "**30 No**"),
    "kerbs": (KERBS, "kerb", "total length of kerb?", "**405 m**"),
}


def _turn(case):
    rows, terms, question, _ = CASES[case]
    turn = answer_exit.Turn(question=question)
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(rows, terms), turn)
    return turn


# --- The lead comes first; the text follows unchanged ------------------------

@pytest.mark.parametrize("case", ["chambers", "kerbs"])
def test_the_tools_lead_comes_before_the_unchanged_text(case):
    turn = _turn(case)
    text = "Here is what I found."
    out = answer_exit.lead_computed_answer(text, turn)
    lead = answer_exit.computed_leads(turn)[0]
    assert CASES[case][3] in lead
    assert out == lead + "\n\n" + text


@pytest.mark.parametrize("case", ["chambers", "kerbs"])
def test_a_text_that_already_carries_the_lead_is_returned_as_it_is(case):
    turn = _turn(case)
    lead = answer_exit.computed_leads(turn)[0]
    text = lead + "\n\nThe model's own note follows."
    out = answer_exit.lead_computed_answer(text, turn)
    assert out == text
    assert out.count(CASES[case][3]) == 1


@pytest.mark.parametrize("case,text", [
    # Texts the #469 hold WITHHELD: a different total; a foreign number.
    ("chambers", "In total there are 14 No."),
    ("kerbs", "Roughly 400 m, give or take."),
])
def test_nothing_is_withheld_even_when_the_rule_would_withhold(case, text):
    turn = _turn(case)
    assert answer_exit.would_withhold(text, turn)["would_withhold"] is True
    out = answer_exit.lead_computed_answer(text, turn)
    assert out.endswith("\n\n" + text)
    assert out.index(CASES[case][3]) < out.index(text)


@pytest.mark.parametrize("tool,payload", [
    ("search_documents", {"ok": True, "result": {"hits": [{"text": "item 14 of 30"}]}}),
    ("no_such_tool_in_the_registry", {"ok": True, "result": {"value": 41}}),
])
def test_a_turn_without_a_computed_figure_is_untouched(tool, payload):
    turn = answer_exit.Turn(question="how many?")
    answer_exit.note_tool_result(tool, payload, turn)
    text = "There are 41 No in total."
    assert answer_exit.lead_computed_answer(text, turn) == text
    assert answer_exit.would_withhold(text, turn) is None


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_text_is_unchanged(text):
    turn = _turn("kerbs")
    assert answer_exit.lead_computed_answer(text, turn) == text
    assert answer_exit.would_withhold(text, turn) is None


@pytest.mark.parametrize("text", ["There are 30 No.", "The kerb is 405 m."])
def test_no_turn_leaves_the_text_alone(text):
    assert answer_exit.lead_computed_answer(text, None) == text
    assert answer_exit.would_withhold(text, None) is None


@pytest.mark.parametrize("turn", [None, answer_exit.Turn(question="how many?")])
def test_no_turn_or_no_result_leads_nothing(turn):
    assert answer_exit.computed_leads(turn) == []


@pytest.mark.parametrize("language,question", [
    ("en", "how many 2.4 x 2.4 m chambers and how much kerb in total?"),
    ("ar", "كم عدد الغرف وما طول البلدور؟"),
])
def test_two_computed_results_lead_in_call_order_and_a_rendered_set_is_kept(language, question):
    from app.agents.core import computed_answer

    turn = answer_exit.Turn(question=question)
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(CHAMBERS, "2.4"), turn)
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(KERBS, "kerb"), turn)
    leads = answer_exit.computed_leads(turn)
    assert len(leads) == 2
    assert answer_exit.lead_computed_answer("ok.", turn) == "\n\n".join([*leads, "ok."])
    header = computed_answer.LABELS[language]["several"].format(n=len(leads))
    text = "\n\n".join([header, *leads, "That makes 999 No overall."])
    assert answer_exit.lead_computed_answer(text, turn) == text
    # The rendered set's header and leads are not judged; the model's line is.
    assert answer_exit.would_withhold(text, turn)["foreign"] == [999.0]


@pytest.mark.parametrize("case", ["chambers", "kerbs"])
def test_only_the_missing_lead_is_added(case):
    turn = _turn(case)
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(*CASES["kerbs" if case == "chambers" else "chambers"][:2]), turn)
    first, second = answer_exit.computed_leads(turn)
    text = first + "\n\nnote."
    assert answer_exit.lead_computed_answer(text, turn) == second + "\n\n" + text


# --- The measurement line: numbers only, never the text ----------------------

@pytest.mark.parametrize("case,text,foreign,restated", [
    ("chambers", "Item G carries 41 No.", [41.0], False),
    ("kerbs", "The total kerb length is 320 m.", [], True),
])
def test_the_measurement_line_logs_numbers_never_the_text(case, text, foreign, restated, caplog):
    with caplog.at_level(logging.INFO, logger=answer_exit.__name__):
        reading = answer_exit.would_withhold(text, _turn(case))
    assert reading == {"foreign": foreign, "restated_total": restated, "would_withhold": True}
    line = next(r.getMessage() for r in caplog.records if "hold_measure" in r.getMessage())
    assert "would_withhold=True" in line
    assert text not in line


@pytest.mark.parametrize("case,text", [
    ("chambers", "Item G carries 14 No, H 9 No and J 7 No."),
    ("kerbs", "Item A1 carries 320 m and A2 carries 85 m."),
])
def test_a_correct_text_measures_as_kept(case, text):
    assert answer_exit.would_withhold(text, _turn(case)) == {
        "foreign": [], "restated_total": False, "would_withhold": False}


# --- A failure never breaks an answer ----------------------------------------

def _boom(*_a, **_k):
    raise RuntimeError("boom")


@pytest.mark.parametrize("case", ["chambers", "kerbs"])
def test_a_failing_measurement_still_leads(case, monkeypatch):
    turn = _turn(case)
    monkeypatch.setattr(answer_exit, "would_withhold", _boom)
    out = answer_exit._lead_quietly("ok.", turn)
    assert out == answer_exit.computed_leads(turn)[0] + "\n\nok."


@pytest.mark.parametrize("text", ["ok.", "There are 30 No."])
def test_a_failing_lead_passes_the_text_unchanged(text, monkeypatch):
    monkeypatch.setattr(answer_exit, "lead_computed_answer", _boom)
    assert answer_exit._lead_quietly(text, _turn("kerbs")) == text


# --- Through the stream route ------------------------------------------------

def _frame(event):
    return f"data: {json.dumps(event)}\n\n"


async def _scripted(case, tokens):
    rows, terms, _, _ = CASES[case]
    # The tool runs inside the turn the wrapper opened, as the agent's does.
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(rows, terms))
    for tok in tokens:
        yield _frame({"type": "token", "content": tok})
    yield _frame({"type": "end"})


def _stream_end(case, tokens):
    async def run():
        frames = [f async for f in exit_frames.with_answer_exit(_scripted(case, tokens), CASES[case][2])]
        return json.loads(frames[-1][len("data: "):])
    return asyncio.run(run())


@pytest.mark.parametrize("case,tokens", [
    ("chambers", ["Item G carries 14 No, ", "H 9 No and J 7 No."]),
    ("kerbs", ["Item A1 carries 320 m ", "and A2 carries 85 m."]),
])
def test_stream_route_leads_and_keeps_the_text(case, tokens):
    content = _stream_end(case, tokens)["content"]
    text = "".join(tokens)
    assert content.endswith("\n\n" + text)
    assert content.index(CASES[case][3]) < content.index(text)


@pytest.mark.parametrize("case,tokens", [
    ("chambers", ["In total ", "there are 14 No."]),
    ("kerbs", ["The kerb comes to ", "about 400 m."]),
])
def test_stream_route_never_withholds(case, tokens):
    content = _stream_end(case, tokens)["content"]
    assert content.endswith("\n\n" + "".join(tokens))
    assert content.count(CASES[case][3]) == 1


@pytest.mark.parametrize("tokens", [
    ["There are 41 ", "No in total."],
    ["Nothing ", "computed here."],
])
def test_stream_route_without_a_computed_result_is_untouched(tokens):
    async def scripted():
        answer_exit.note_tool_result("search_documents", {"ok": True, "result": {"hits": []}})
        for tok in tokens:
            yield _frame({"type": "token", "content": tok})
        yield _frame({"type": "end"})

    async def run():
        frames = [f async for f in exit_frames.with_answer_exit(scripted(), "how many?")]
        return json.loads(frames[-1][len("data: "):])
    end = asyncio.run(run())
    assert end.get("content", "".join(tokens)) == "".join(tokens)
