"""The allowed-numbers HOLD on a computed turn (redo of #411, rule 1, by the
advisor's ruling of 19 Nov 00:26Z).

ALLOWED in the model's text: a number in the question, one of the tool's
stated figures, a number in the tool's lead, or a number printed by the
tool's result (a selected row's quantity, size or depth band). WITHHELD: any
other number, or a DIFFERENT figure stated as a total (total/sum or
self-correction wording) in a unit the tool stated a figure in. A per-row
list in one unit is not a restatement. The tool's lead comes first.

Every bill is invented here and run through the real ``aggregate`` and the
registry's ``bill_rows_statement``; the stream cases go through
``exit_frames.with_answer_exit``, the wrapper the chat routes answer through.
"""

from __future__ import annotations

import asyncio
import json

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


def _held(case, text):
    return answer_exit.hold_computed_answer(text, _turn(case))


# --- ALLOWED: the selected rows' own numbers --------------------------------

@pytest.mark.parametrize("case,text", [
    ("chambers", "Item G carries 14 No, H 9 No and J 7 No."),
    ("kerbs", "Item A1 carries 320 m and A2 carries 85 m."),
])
def test_a_selected_rows_own_quantities_are_kept(case, text):
    out = _held(case, text)
    assert text in out
    assert answer_exit.FIGURE_WITHHELD not in out


@pytest.mark.parametrize("case,text", [
    ("chambers", "They run from 2.0 m down to 3.5 m, all 2.4 x 2.4 m."),
    ("kerbs", "The kerbs are 150 mm and 300 mm, types 2 and 4."),
])
def test_a_selected_rows_own_sizes_and_depth_bands_are_kept(case, text):
    out = _held(case, text)
    assert text in out
    assert answer_exit.FIGURE_WITHHELD not in out


@pytest.mark.parametrize("case,text", [
    ("chambers", "So 30 No are billed in total over 3 rows."),
    ("kerbs", "In total the kerb is 405 m."),
])
def test_the_tools_own_total_stated_as_a_total_is_kept(case, text):
    out = _held(case, text)
    assert text in out
    assert answer_exit.FIGURE_WITHHELD not in out


# --- WITHHELD: a number in none of the allowed sources ----------------------

@pytest.mark.parametrize("case,text", [
    ("chambers", "There are 41 No in all."),            # invented
    ("kerbs", "Roughly 400 m, give or take."),          # the model's rounding
])
def test_a_number_the_tool_never_printed_is_withheld(case, text):
    out = _held(case, text)
    assert text not in out
    assert out.endswith(answer_exit.FIGURE_WITHHELD)


@pytest.mark.parametrize("case,text", [
    # K (5 No) is a row the tool did NOT select: not one of its facts.
    ("chambers", "Item G carries 14 No and K carries 5 No."),
    ("kerbs", "A1 carries 320 m; the footway kerb adds 112 m."),
])
def test_an_unselected_rows_number_is_withheld(case, text):
    assert _held(case, text).endswith(answer_exit.FIGURE_WITHHELD)


# --- WITHHELD: a different figure for the same total ------------------------

@pytest.mark.parametrize("case,text", [
    # 14 No and 320 m are allowed numbers (a row's own) but stated AS the total.
    ("chambers", "In total there are 14 No."),
    ("kerbs", "The total kerb length is 320 m."),
])
def test_a_rows_number_stated_as_the_total_is_withheld(case, text):
    out = _held(case, text)
    assert text not in out
    assert out.endswith(answer_exit.FIGURE_WITHHELD)


@pytest.mark.parametrize("case,text", [
    ("chambers", "There are 30 No. Correction: 14 No."),
    ("kerbs", "That is 405 m. It should read 85 m."),
])
def test_a_self_correction_to_another_allowed_number_is_withheld(case, text):
    assert _held(case, text).endswith(answer_exit.FIGURE_WITHHELD)


@pytest.mark.parametrize("case,text", [
    ("chambers", "Item G carries 14 No.\nItem H carries 9 No.\nItem J carries 7 No."),
    ("kerbs", "- A1: 320 m\n- A2: 85 m"),
])
def test_a_per_row_list_in_one_unit_is_not_a_restatement(case, text):
    assert answer_exit.FIGURE_WITHHELD not in _held(case, text)


@pytest.mark.parametrize("text,totals,restated", [
    ("The total is 14 No.", [(30.0, "No")], True),
    ("The total is 30 No.", [(30.0, "No")], False),
    ("The total is SAR 9,000.", [(12000.0, "SAR")], True),   # the unit before the number
    ("The total is SAR 12,000.", [(12000.0, "SAR")], False),
    ("The total is 14 m.", [(30.0, "No")], False),           # another unit
    ("The total is 14.", [(30.0, "No")], False),             # no unit: not read
    ("The total is 14 No.", [(3.0, "")], False),             # the tool stated no unit
    ("Item G is 14 No.", [(30.0, "No")], False),             # no total wording
    ("المجموع 14 No.", [(30.0, "No")], True),
])
def test_restates_a_total(text, totals, restated):
    assert answer_exit._restates_a_total(text, totals) is restated


# --- The lead comes first ----------------------------------------------------

@pytest.mark.parametrize("case", ["chambers", "kerbs"])
def test_the_tools_lead_comes_before_the_models_text(case):
    total = CASES[case][3]
    text = "Here is what I found."
    out = _held(case, text)
    assert out.index(total) < out.index(text)


@pytest.mark.parametrize("case", ["chambers", "kerbs"])
def test_a_text_that_already_carries_the_lead_does_not_repeat_it(case):
    turn = _turn(case)
    lead = answer_exit.computed_leads(turn)[0]
    out = answer_exit.hold_computed_answer(lead, turn)
    assert out.count(CASES[case][3]) == 1
    assert out == lead


@pytest.mark.parametrize("tool,payload", [
    ("search_documents", {"ok": True, "result": {"hits": [{"text": "item 14 of 30"}]}}),
    ("no_such_tool_in_the_registry", {"ok": True, "result": {"value": 41}}),
])
def test_a_turn_without_a_computed_figure_is_unchanged(tool, payload):
    turn = answer_exit.Turn(question="how many?")
    answer_exit.note_tool_result(tool, payload, turn)
    assert answer_exit.hold_computed_answer("There are 41 No in total.", turn) == "There are 41 No in total."


@pytest.mark.parametrize("text", ["", "   "])
def test_an_empty_text_is_unchanged(text):
    assert _held("kerbs", text) == text


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
def test_stream_route_keeps_the_rows_own_numbers_after_the_lead(case, tokens):
    end = _stream_end(case, tokens)
    text = "".join(tokens)
    assert text in end["content"]
    assert end["content"].index(CASES[case][3]) < end["content"].index(text)
    assert answer_exit.FIGURE_WITHHELD not in end["content"]


@pytest.mark.parametrize("case,tokens", [
    ("chambers", ["In total ", "there are 14 No."]),
    ("kerbs", ["The kerb comes to ", "about 400 m."]),
])
def test_stream_route_withholds_a_different_total_or_a_foreign_number(case, tokens):
    end = _stream_end(case, tokens)
    assert "".join(tokens) not in end["content"]
    assert CASES[case][3] in end["content"]
    assert end["content"].endswith(answer_exit.FIGURE_WITHHELD)


@pytest.mark.parametrize("turn", [None, answer_exit.Turn(question="how many?")])
def test_no_turn_or_no_result_leads_nothing(turn):
    assert answer_exit.computed_leads(turn) == []


@pytest.mark.parametrize("language,question", [
    ("en", "how many 2.4 x 2.4 m chambers and how much kerb in total?"),
    ("ar", "كم عدد الغرف وما طول البلدور؟"),
])
def test_a_withheld_recovery_keeps_the_renderers_several_results_header(language, question):
    from app.agents.core import computed_answer

    turn = answer_exit.Turn(question=question)
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(CHAMBERS, "2.4"), turn)
    answer_exit.note_tool_result("bill_rows_aggregate", _bill(KERBS, "kerb"), turn)
    leads = answer_exit.computed_leads(turn)
    header = computed_answer.LABELS[language]["several"].format(n=len(leads))
    text = "\n\n".join([header, *leads, "That makes 999 No overall."])
    out = answer_exit.hold_computed_answer(text, turn)
    assert out.startswith(header)
    assert "999" not in out
    assert out.endswith(answer_exit.FIGURE_WITHHELD)
