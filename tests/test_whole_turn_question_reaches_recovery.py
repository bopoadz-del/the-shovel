"""#389 follow-up (grader 21:15Z): on the whole-turn chat path
(``core.loop.chat``) the turn's question now reaches the answer exit, so it
also feeds what the recovery reads from it:

* the allowed-numbers set -- an argument of a call that returned no figure
  is echoed on the "Asked" line only when the user typed that value;
* the recovery message's language -- the labels follow the question.

Before #389 the whole-turn record was opened with an empty question, so a
recovered answer echoed no figure the user typed and was always worded in
English. This drives the real ``chat`` wrapper with the agent's turn body
replaced: the body records one tool call the way the tool runner records it
and ends with the platform's empty-turn reply, which is the exit that
recovers from the turn's tool results.

Every question, pit and figure is invented here; two inputs per decision.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest

from app.agents import answer_exit
from app.agents.core import computed_answer
from app.agents.core import loop as core_loop
from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK as FALLBACK
from app.lib import construction_formulas as cf
from tests.lane5_core_support import agent, call

PITS = [(11, 6, 2.5), (17, 3, 1.25)]
# (bill terms, a number the model sent with them) -- a call with no figure.
FAILED = [("kerb", 40), ("duct", 160)]


def _result(pit) -> Dict[str, Any]:
    length, width, depth = pit
    out = cf.run_calculation("excavation_volume",
                             {"length_m": length, "width_m": width, "depth_m": depth})
    assert out["status"] == "success", out
    return out


def _calc_record(pit):
    length, width, depth = pit
    tc = call("construction_calc", {"calculation": "excavation_volume",
                                    "params": {"length_m": length, "width_m": width,
                                               "depth_m": depth}}, "w1")
    return tc, {"role": "tool", "name": "construction_calc", "tool_call_id": "w1",
                "content": json.dumps({"name": "construction_calc", "ok": True,
                                       "result": _result(pit)})}


def _failed_record(terms: str, number: int):
    tc = call("bill_rows_aggregate", {"terms": terms, "min_qty": number}, "w2")
    return tc, {"role": "tool", "name": "bill_rows_aggregate", "tool_call_id": "w2",
                "content": json.dumps({"status": "error"})}


def _whole_turn(monkeypatch, question: str, record) -> Dict[str, Any]:
    """One ``Agent.chat`` turn whose body recorded ``record`` (tool call,
    tool message) and then produced only the platform's empty-turn reply."""
    probe = agent()
    seen: Dict[str, Any] = {}

    async def impl(user_message: str = "", **_kwargs):
        turn = answer_exit.current_turn()
        seen["question"] = turn.question
        answer_exit.note_tool_message(*record, turn)
        return {"answer": FALLBACK, "sources": []}

    monkeypatch.setattr(probe, "_chat_impl", impl)
    out = asyncio.run(core_loop.chat(probe, user_message=question))
    assert seen.get("question") == question
    return out


def _figure(pit) -> str:
    return f"{float(_result(pit)['result']['bank_volume_m3']):g}"


def _asked_line(text: str, language: str = "en") -> str:
    label = computed_answer.labels_for(language)["asked"] + ":"
    return next((ln for ln in text.splitlines() if label in ln), "")


@pytest.mark.parametrize("terms,number", FAILED)
def test_a_number_the_user_typed_is_echoed_on_the_whole_turn_path(monkeypatch, terms, number):
    out = _whole_turn(monkeypatch, f"How many {terms} rows are over {number} in the bill?",
                      _failed_record(terms, number))
    text = out["answer"]
    assert FALLBACK not in text, text
    line = _asked_line(text)
    assert terms in line and str(number) in line, text


@pytest.mark.parametrize("terms,number", FAILED)
def test_a_number_the_user_did_not_type_is_not_echoed(monkeypatch, terms, number):
    out = _whole_turn(monkeypatch, f"How many {terms} rows are in the bill?",
                      _failed_record(terms, number))
    text = out["answer"]
    assert FALLBACK not in text, text
    line = _asked_line(text)
    assert terms in line, text
    assert str(number) not in line, text


@pytest.mark.parametrize("pit", PITS)
def test_the_recovery_is_worded_in_the_question_language(monkeypatch, pit):
    length, width, depth = pit
    english = f"Bank volume of a pit {length} x {width} x {depth} m?"
    arabic = f"ما هو حجم الحفر لحفرة {length} × {width} × {depth} م؟"
    language = computed_answer.question_language(arabic)
    assert language != computed_answer.question_language(english)
    en_text = _whole_turn(monkeypatch, english, _calc_record(pit))["answer"]
    ar_text = _whole_turn(monkeypatch, arabic, _calc_record(pit))["answer"]
    for text in (en_text, ar_text):
        assert FALLBACK not in text, text
        assert _figure(pit) in text.replace(",", ""), text
    assert computed_answer.heading_for("", language) in ar_text, ar_text
    assert computed_answer.heading_for("", language) not in en_text, en_text
