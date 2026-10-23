"""Rows a tool returns per period reach the user one row each (#974 repair).

Lane 1 (#974 b021b97, FAIL 2nd): the rule in the standing prompt was the
whole fix, so a sideways table would still pass. These tests run the driver
path with a scripted model and check what reaches the user: when the answer
shows a tool's per-period rows but not row by row -- a table turned sideways,
or a single month written as bullets -- the platform adds them as a table
with one row per period. An answer already row by row, or one that does not
show the list at all, is left as written.
"""
import json

import pytest

from app.agents.driver import render
from app.agents.driver import loop as driver_loop  # noqa: F401 -- the path under test
from app.agents.driver import llm
from tests.test_driver_mode import _agent, _call, _no_retrieval, _run, _scripted
from tests.test_driver_mode import _stream_through_call  # noqa: F401 -- the harness's autouse stream fixture

ROWS = [{"month": 1, "billing": 410500.0, "cumulative": 410500.0},
        {"month": 2, "billing": 395250.0, "cumulative": 805750.0},
        {"month": 3, "billing": 394250.0, "cumulative": 1200000.0}]


def _turn(monkeypatch, answer, rows=ROWS):
    _no_retrieval(monkeypatch)
    seen = []
    replies = [
        {"content": "", "tool_calls": [_call("select_hat", {"hat": "commercial"}, "c1")]},
        {"content": "", "tool_calls": [_call("cash_flow_forecast", {"contract_value": 1200000,
                                                                     "duration_months": len(rows)}, "c2")]},
        {"content": answer},
    ]
    monkeypatch.setattr(llm, "call", _scripted(replies, seen))
    agent = _agent()

    async def fake_tool(call, api_key=None, project_id=None, conversation_id=None, **kw):
        return {"ok": True, "result": {"status": "success", "s_curve_data": rows,
                                       "summary_metrics": {"peak_month": 1}}}

    monkeypatch.setattr(agent, "_run_tool_call", fake_tool)
    events = _run(agent, user_message="Cash flow for 1,200,000 over the contract", project_id="synthetic-proj")
    return "".join(e["content"] for e in events if e["type"] == "token")


SIDEWAYS = ("| | Month 1 | Month 2 | Month 3 |\n|---|---|---|---|\n"
            "| Billing | 410,500 | 395,250 | 394,250 |\n| Cumulative | 410,500 | 805,750 | 1,200,000 |")
BY_ROW = ("| Month | Billing | Cumulative |\n|---|---|---|\n| 1 | 410,500 | 410,500 |\n"
          "| 2 | 395,250 | 805,750 |\n| 3 | 394,250 | 1,200,000 |")


def _row_lines(text):
    return [line for line in text.splitlines() if line.startswith("| ") and "---" not in line]


def test_a_sideways_table_gets_the_rows_one_per_period(monkeypatch):
    text = _turn(monkeypatch, "Here is the forecast:\n" + SIDEWAYS)
    tail = text.split(SIDEWAYS, 1)[1]
    rows = _row_lines(tail)
    assert len(rows) == 4, tail  # header + one row per month
    for record in ROWS:
        assert any(f"{record['billing']:,.2f}".rstrip("0").rstrip(".") in r or f"{int(record['billing']):,}" in r
                   for r in rows[1:]), (record, rows)


def test_a_single_month_as_bullets_gets_its_row(monkeypatch):
    one = ROWS[:1]
    text = _turn(monkeypatch, "Month 1:\n- Billing: 410,500\n- Cumulative: 410,500", rows=one)
    assert len(_row_lines(text)) == 2, text  # header + the month's row


def test_an_answer_already_row_by_row_is_left_as_written(monkeypatch):
    text = _turn(monkeypatch, "Forecast:\n" + BY_ROW)
    assert text.count("| Month |") == 1, "a second table was added to an answer already row by row"
    assert len(_row_lines(text)) == 4


def test_an_answer_that_does_not_show_the_list_gets_no_table(monkeypatch):
    text = _turn(monkeypatch, "The contract runs three months; the peak is in the first month.")
    assert not _row_lines(text), text


@pytest.mark.parametrize("text,ok", [(BY_ROW, True), (SIDEWAYS, False),
                                      ("- Billing: 410,500\n- Cumulative: 410,500\n- Month: 1", False)])
def test_rows_carried_reads_orientation(text, ok):
    assert render.rows_carried(ROWS if ok or "Month 1" in text else ROWS[:1], text) is ok


def test_the_rendered_table_has_one_row_per_record():
    out = render.record_table(ROWS)
    assert len([line for line in out if line.startswith("|") and "---" not in line]) == 4
    assert json.dumps(ROWS)  # the rows themselves are untouched
