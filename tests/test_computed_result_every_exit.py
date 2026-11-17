"""#392 grader round 3: every exit that could print the generic empty-turn
reply after a successful tool answers from the turn's tool results instead.

Paths covered, each through the code a user's turn runs:
* the wall-clock deadline in ``chat_stream`` (core loop),
* the crash safety net in ``chat_stream``,
* the synthetic end when the inner stream returns with no terminal event,
* ``answer_exit.check_text_or_fallback`` (the exit check),
* ``runtime._nonblank_after_empty_synthesis``.

Also: the bill count/sum tool through the real driver loop (the live case),
an Arabic question through the real loop (no raw English error inside an
Arabic answer), and the scrub of a stated error and echoed arguments (no
stack trace, table name, internal id or file path reaches the user).

Every test builds its own input (invented bills, rows, errors); two inputs
per decision.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from app.agents import answer_exit
from app.agents import runtime
from app.agents.core import computed_answer
from app.agents.core import loop as core_loop
from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK as FALLBACK
from app.lib import construction_formulas as cf
from tests.lane5_core_support import agent, answer_text, call, drive, install_driver_stream

PITS = [(9, 4, 2), (13, 5, 1.5)]


def _bank(length, width, depth) -> float:
    out = cf.run_calculation("excavation_volume",
                             {"length_m": length, "width_m": width, "depth_m": depth})
    assert out["status"] == "success", out
    return float(out["result"]["bank_volume_m3"])


def _calc_call(pit, cid="c1") -> Dict[str, Any]:
    length, width, depth = pit
    return call("construction_calc", {
        "calculation": "excavation_volume",
        "params": {"length_m": length, "width_m": width, "depth_m": depth}}, cid)


# ── the bill count/sum tool through the real driver loop ────────────────────

BILLS = [
    # (terms asked, rows of the invented bill: (ref, description, unit, qty))
    ("kerb", [("A1", "Precast kerb type B", "m", "120"), ("A2", "Precast kerb type C", "m", "35.5"),
              ("A3", "Topsoil strip", "m2", "900")]),
    ("duct", [("D1", "uPVC duct 110 dia", "m", "48"), ("D2", "uPVC duct 160 dia", "m", "12"),
              ("D3", "uPVC duct bend", "nr", "6"), ("D4", "Gully grating", "nr", "3")]),
]


def _install_bill(monkeypatch, filename: str, rows) -> None:
    from app.core import bill_aggregate, projects

    sheet = [bill_aggregate.SheetRow(path=("Bill 1",), ref=ref, description=desc, unit=unit,
                                     quantity=qty, rate="", amount="", pages=[i + 1])
             for i, (ref, desc, unit, qty) in enumerate(rows)]
    monkeypatch.setattr(bill_aggregate, "derived_bill_documents",
                        lambda project_id, document_id=None: [
                            {"id": "doc-1", "original_name": filename, "metadata": {}}])
    monkeypatch.setattr(projects, "materialize_document_path", lambda doc: "invented.xlsx")
    monkeypatch.setattr(bill_aggregate, "read_sheet_rows", lambda path: list(sheet))


def _expected_bill(terms: str, rows):
    words = terms.lower()
    hit = [r for r in rows if words in r[1].lower()]
    by_unit: Dict[str, float] = {}
    for _ref, _desc, unit, qty in hit:
        by_unit[unit] = by_unit.get(unit, 0.0) + float(qty)
    return hit, by_unit


@pytest.mark.parametrize("synthesis", ["", "**\n\n---\n* "])
@pytest.mark.parametrize("terms,rows", BILLS)
def test_the_bill_count_and_sum_tool_answers_an_empty_synthesis(monkeypatch, terms, rows, synthesis):
    filename = f"invented_{terms}_bill.xlsx"
    _install_bill(monkeypatch, filename, rows)
    install_driver_stream(monkeypatch)
    events, seen = drive(monkeypatch, f"How many {terms} items are in the bill and what is the total?", [
        {"content": "", "tool_calls": [call("bill_rows_aggregate", {"terms": terms}, "b1")]},
        {"content": synthesis},
    ], project_id="p-invented")
    text = answer_text(events)
    hit, by_unit = _expected_bill(terms, rows)
    assert FALLBACK not in text and "unable to generate" not in text.lower(), text
    assert f"**{len(hit)}**" in text, text
    for unit, total in by_unit.items():
        assert f"**{computed_answer._figure_text(total)} {unit}**" in text, text
    assert filename in text
    for ref, *_ in hit:
        assert ref in text
    # The tool ran inside the loop: the model saw its result.
    assert any(m.get("role") == "tool" and m.get("name") == "bill_rows_aggregate"
               for m in seen[-1]["messages"])


# ── an Arabic question through the real loop ────────────────────────────────

ARABIC_FAILURES = [
    # (tool call, English the tool wrote that must not appear, Arabic that must)
    (call("construction_calc", {"calculation": "zz_no_such_calculation", "params": {}}, "a1"),
     "Unknown calculation", computed_answer.LABELS["ar"]["tool_error"]),
    (call("bill_rows_aggregate", {"terms": "kerb"}, "a2"),
     "no_derived_bill_sheet", computed_answer.REASONS["ar"]["no_derived_bill_sheet"]),
]


@pytest.mark.parametrize("tool_call,english,arabic", ARABIC_FAILURES)
def test_an_arabic_question_gets_no_raw_english_error(monkeypatch, tool_call, english, arabic):
    from app.core import bill_aggregate

    monkeypatch.setattr(bill_aggregate, "derived_bill_documents", lambda *a, **k: [])
    install_driver_stream(monkeypatch)
    events, _ = drive(monkeypatch, "كم عدد بنود الحفر وما مجموع الكميات؟", [
        {"content": "", "tool_calls": [tool_call]},
        {"content": ""},
    ], project_id="p-invented")
    text = answer_text(events)
    assert FALLBACK not in text and "unable to generate" not in text.lower(), text
    assert english not in text, text
    assert arabic in text, text
    assert computed_answer.LABELS["ar"]["not_returned"] in text


_EN, _AR = computed_answer.LABELS["en"], computed_answer.LABELS["ar"]


@pytest.mark.parametrize("code,language,declared,expect", [
    ("no_derived_bill_sheet", "ar", (), computed_answer.REASONS["ar"]["no_derived_bill_sheet"]),
    ("derived_sheet_has_no_readable_bytes", "en", (),
     computed_answer.REASONS["en"]["derived_sheet_has_no_readable_bytes"]),
    # A declared code with no wording is shown as a code.
    ("quota_window_closed", "ar", ("quota_window_closed",),
     f"{_AR['tool_error']} ({_AR['code']}: quota_window_closed)"),
    ("quota_window_closed", "en", ("quota_window_closed",),
     f"{_EN['tool_error']} ({_EN['code']}: quota_window_closed)"),
    # An undeclared snake_case token (a table or column name) is not a code.
    ("quota_window_closed", "en", (), _EN["tool_error"]),
    ("bill_rows_v2", "ar", ("quota_window_closed",), _AR["tool_error"]),
    # An error equal to a LABELS key is not that label.
    ("count", "en", (), _EN["tool_error"]),
    ("rows", "ar", (), _AR["tool_error"]),
    # Free text never reaches the answer, English included.
    ("the rate column is blank", "ar", (), _AR["tool_error"]),
    ("the rate column is blank", "en", (), _EN["tool_error"]),
])
def test_a_reason_is_worded_in_the_question_language(code, language, declared, expect):
    lab = computed_answer.labels_for(language)
    assert computed_answer.reason_text(code, lab, language, declared) == expect


# ── no free text a tool wrote reaches the answer ────────────────────────────
# A bare table name, a path with spaces (POSIX and Windows), a stack trace, a
# constraint name, an address with a port: each injected through the real
# recovery, under an English and an Arabic question.

LEAKS = [
    ("bare table", "bill_rows_v2", ["bill_rows_v2"]),
    ("bare table", "SELECT qty FROM bill_rows WHERE doc = 4", ["bill_rows", "SELECT"]),
    ("posix path with spaces", "cannot open /srv/data/Project Files/derived bill.xlsx",
     ["Project Files", "derived bill", "/srv/data", "cannot open"]),
    ("posix path with spaces", "missing /var/lib/the fork/sheets/p 17", ["the fork", "p 17", "/var/lib"]),
    ("windows path with spaces", "cannot open C:\\Program Files\\Fork Data\\bill sheet.xlsx",
     ["Program Files", "Fork Data", "bill sheet", "C:\\"]),
    ("windows path with spaces", "D:\\My Bills\\kerb run.xlsx not found",
     ["My Bills", "kerb run", "not found"]),
    ("stack trace", "Traceback (most recent call last):\n  File \"/srv/app/x.py\", line 9\nKeyError: 'rate'",
     ["Traceback", "x.py", "KeyError"]),
    ("constraint and address", 'duplicate key violates unique constraint "uq_bill_doc" at 10.0.4.7:5432',
     ["uq_bill_doc", "10.0.4.7", "5432", "duplicate key"]),
]


@pytest.mark.parametrize("question,language", [
    ("Bank volume of the pit?", "en"),
    ("\u0645\u0627 \u062d\u062c\u0645 \u0627\u0644\u062d\u0641\u0631\u061f", "ar"),
])
@pytest.mark.parametrize("kind,error,gone", LEAKS, ids=[f"{k}-{i}" for i, (k, *_r) in enumerate(LEAKS)])
def test_no_tool_free_text_reaches_the_answer(kind, error, gone, question, language):
    msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "", "tool_calls": [call("construction_calc", {
            "calculation": "excavation_volume", "params": {}}, "e1")]},
        {"role": "tool", "name": "construction_calc", "tool_call_id": "e1",
         "content": json.dumps({"status": "error", "error": error})},
    ]
    out = runtime._recover_answer_from_tool_messages("", msgs)
    lab = computed_answer.LABELS[language]
    assert out.strip() and "unable to generate" not in out.lower()
    assert f"{lab['not_returned']}: {lab['tool_error']}" in out, out
    for piece in gone:
        assert piece not in out, (kind, out)


@pytest.mark.parametrize("reason", ["no_derived_bill_sheet", "derived_sheet_is_not_a_readable_workbook"])
def test_a_declared_reason_code_through_the_real_recovery_reads_as_its_wording(reason):
    msgs = [
        {"role": "user", "content": "How many kerb items?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("bill_rows_aggregate", {"terms": "kerb"}, "e3")]},
        {"role": "tool", "name": "bill_rows_aggregate", "tool_call_id": "e3",
         "content": json.dumps({"status": "error", "reason": reason})},
    ]
    out = runtime._recover_answer_from_tool_messages("", msgs)
    assert computed_answer.REASONS["en"][reason] in out, out
    assert reason not in out, out


# The echoed call arguments are still shown, scrubbed.
ECHOED = [
    ({"document_id": "3f2a9c1e-1b2c-4d5e-8f90-123456789abc", "terms": "kerb"}, "kerb", ["3f2a9c1e"]),
    ({"path": "/srv/data/p17/bill.xlsx", "terms": "duct"}, "duct", ["/srv/data", "bill.xlsx"]),
]


@pytest.mark.parametrize("args,kept,gone", ECHOED)
def test_the_echoed_arguments_are_scrubbed(args, kept, gone):
    msgs = [
        {"role": "user", "content": "How many items?"},
        {"role": "assistant", "content": "", "tool_calls": [call("bill_rows_aggregate", args, "e2")]},
        {"role": "tool", "name": "bill_rows_aggregate", "tool_call_id": "e2",
         "content": json.dumps({"status": "error"})},
    ]
    out = runtime._recover_answer_from_tool_messages("", msgs)
    assert kept in out, out
    for piece in gone:
        assert piece not in out, out


# ── the paths that never see the turn's messages ────────────────────────────


def _driver_chat(monkeypatch, user: str, replies: List[Any], *, timeout_s: str = "60"):
    """Run ``Agent.chat_stream`` (the core loop wrapper a user's turn goes
    through) in driver mode; each reply is a message, an exception to raise,
    or "hang" for a provider that never answers."""
    from app.agents.driver import llm

    monkeypatch.setenv("DRIVER_MODE", "on")
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", timeout_s)
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "0.5")
    from tests.lane5_core_support import no_retrieval

    no_retrieval(monkeypatch)
    queue = list(replies)

    async def fake_call(_agent, messages, offered, model="", timeout=120.0):
        reply = queue.pop(0)
        if reply == "hang":
            await asyncio.Future()
        if isinstance(reply, BaseException):
            raise reply
        return {"status": "success", "message": reply}

    monkeypatch.setattr(llm, "call", fake_call)
    install_driver_stream(monkeypatch)
    probe = agent()

    async def go():
        return [e async for e in core_loop.chat_stream(probe, user_message=user)]

    return asyncio.run(go())


@pytest.mark.parametrize("pit", PITS)
def test_the_wall_clock_deadline_after_a_successful_tool_shows_its_figure(monkeypatch, pit):
    events = _driver_chat(monkeypatch, "Bank volume of the pit {} x {} x {} m?".format(*pit),
                          [{"content": "", "tool_calls": [_calc_call(pit)]}, "hang"], timeout_s="3")
    assert any(e.get("type") == "error" and "timeout" in e.get("message", "").lower() for e in events)
    text = answer_text(events)
    assert FALLBACK not in text, text
    assert f"{_bank(*pit):g}" in text.replace(",", ""), text


@pytest.mark.parametrize("pit", PITS)
def test_the_crash_safety_net_after_a_successful_tool_shows_its_figure(monkeypatch, pit):
    events = _driver_chat(monkeypatch, "Bank volume of the pit {} x {} x {} m?".format(*pit),
                          [{"content": "", "tool_calls": [_calc_call(pit)]},
                           RuntimeError("provider exploded")])
    text = answer_text(events)
    assert any(e.get("type") in ("error", "end") for e in events), events
    assert FALLBACK not in text, text
    assert f"{_bank(*pit):g}" in text.replace(",", ""), text


def _turn_with_calc(pit) -> answer_exit.Turn:
    """A turn record holding one successful calculation, recorded the way the
    tool runner records it."""
    length, width, depth = pit
    result = cf.run_calculation("excavation_volume",
                                {"length_m": length, "width_m": width, "depth_m": depth})
    turn = answer_exit.Turn(question=f"Bank volume of {length} x {width} x {depth}?")
    tc = _calc_call(pit, "t1")
    answer_exit.note_tool_message(tc, {"role": "tool", "name": "construction_calc", "tool_call_id": "t1",
                                       "content": json.dumps({"name": "construction_calc", "ok": True,
                                                              "result": result})}, turn)
    return turn


@pytest.mark.parametrize("pit", PITS)
def test_the_exit_check_recovers_before_the_generic_reply(pit):
    turn = _turn_with_calc(pit)
    out = answer_exit.check_text_or_fallback(FALLBACK, turn=turn)
    assert out != FALLBACK
    assert f"{_bank(*pit):g}" in out.replace(",", ""), out
    # With no tool result the generic reply still stands (nothing invented).
    assert answer_exit.check_text_or_fallback(FALLBACK, turn=answer_exit.Turn()) == FALLBACK


@pytest.mark.parametrize("pit", PITS)
def test_the_end_of_turn_nonblank_guard_recovers_before_the_generic_reply(pit):
    turn = _turn_with_calc(pit)
    msgs = [{"role": "user", "content": turn.question}, *turn.tool_messages]
    out = runtime._nonblank_after_empty_synthesis("", msgs)
    assert FALLBACK not in out
    assert f"{_bank(*pit):g}" in out.replace(",", ""), out
    assert runtime._nonblank_after_empty_synthesis(FALLBACK, msgs) != FALLBACK


@pytest.mark.parametrize("pit", PITS)
def test_a_stream_with_no_terminal_event_recovers_before_the_generic_reply(monkeypatch, pit):
    """The inner stream ran a tool and returned with no token and no end."""
    probe = agent()
    monkeypatch.delenv("DRIVER_MODE", raising=False)
    turn_holder: Dict[str, Any] = {}

    async def impl(**_kwargs):
        turn = answer_exit.current_turn()
        turn_holder["turn"] = turn
        seeded = _turn_with_calc(pit)
        turn.tool_messages.extend(seeded.tool_messages)
        yield {"type": "tool_result", "tool": "construction_calc", "name": "construction_calc", "ok": True}

    monkeypatch.setattr(probe, "_chat_stream_impl", impl)

    async def go():
        return [e async for e in core_loop.chat_stream(probe, user_message="Bank volume?")]

    events = asyncio.run(go())
    text = answer_text(events)
    assert FALLBACK not in text, text
    assert f"{_bank(*pit):g}" in text.replace(",", ""), text
    end = [e for e in events if e.get("type") == "end"]
    assert end and FALLBACK not in (end[-1].get("content") or ""), end


# ── the turn record outside a turn ───────────────────────────────────────────


@pytest.mark.parametrize("message", [
    {"role": "tool", "name": "construction_calc", "tool_call_id": "x1", "content": "{}"},
    {"role": "tool", "name": "bill_rows_aggregate", "tool_call_id": "x2", "content": "{}"},
])
def test_outside_a_turn_nothing_is_recorded_or_recovered_and_nothing_is_logged(caplog, message):
    """With no open turn, recording a tool message is a silent no-op and the
    recovery says nothing (it never raises)."""
    token = answer_exit._TURN.set(None)
    try:
        with caplog.at_level("WARNING", logger=answer_exit._LOG.name):
            answer_exit.note_tool_message(None, message)
            assert answer_exit.recovered_answer() == ""
            assert answer_exit.recovered_answer(None) == ""
        assert not caplog.records, [r.getMessage() for r in caplog.records]
    finally:
        answer_exit._TURN.reset(token)


@pytest.mark.parametrize("not_a_message", [None, "tool said 42"])
def test_a_tool_message_that_is_not_a_dict_is_ignored_silently(caplog, not_a_message):
    turn = answer_exit.Turn(question="How many?")
    with caplog.at_level("WARNING", logger=answer_exit._LOG.name):
        answer_exit.note_tool_message(None, not_a_message, turn)
    assert turn.tool_messages == []
    assert not caplog.records, [r.getMessage() for r in caplog.records]
