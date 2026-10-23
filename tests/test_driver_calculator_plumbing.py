"""Driver-mode calculator plumbing (F-CORE-ROUND-2 steps 3-5), against a
synthetic formula only: the model is offered the registry's labels and units,
reads results as prose, and no registry identifier reaches the user."""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents.driver import llm, loop, render, tools
from app.lib import construction_formulas as cf
from app.lib import formula_registry as fr

NAME = "syn_widget_area"


@pytest.fixture(autouse=True)
def synthetic_formula():
    def syn_widget_area(widget_width_m: float, widget_depth_m: float, finish_layers: int = 1):
        return {"widget_area_m2": widget_width_m * widget_depth_m * finish_layers,
                "note": "Synthetic widget area."}
    fr._REGISTRY.pop(NAME, None)
    fr.formula(owner="base", description="Area of a synthetic widget.",
               display_name="Synthetic widget area",
               inputs={"widget_width_m": fr.Param(unit="m", lo=0.01, hi=50, label="widget width"),
                       "widget_depth_m": fr.Param(unit="m", lo=0.01, hi=50, label="widget depth"),
                       "finish_layers": fr.Param(unit="count", lo=1, hi=10, label="finish layers")},
               outputs={"widget_area_m2": "m2"})(syn_widget_area)
    cf.CALCULATORS[NAME] = syn_widget_area
    render._MAP.clear()
    yield
    fr._REGISTRY.pop(NAME, None)
    cf.CALCULATORS.pop(NAME, None)
    render._MAP.clear()


def test_the_schema_offers_every_input_by_label_unit_range_and_required():
    schema = tools.calculator_schema(None)
    assert NAME in schema["function"]["parameters"]["properties"]["calculation"]["enum"]
    d = schema["function"]["description"]
    line = next(x for x in d.splitlines() if x.startswith(f"- {NAME}:"))
    assert "Synthetic widget area" in line
    assert "widget_width_m = widget width (m, 0.01 to 50)" in line
    assert "finish_layers = finish layers (count, 1 to 10, optional)" in line


def test_a_result_reaches_the_model_as_prose_without_identifiers():
    env = cf.run_calculation(NAME, {"widget_width_m": 2, "widget_depth_m": 3})
    text = render.render_calculation(env, {"widget_width_m": 2, "widget_depth_m": 3})
    assert text.startswith("Synthetic widget area — platform calculator.")
    assert "widget width 2 m" in text and "widget depth 3 m" in text
    assert "widget area 6 m" in text
    for key in (NAME, "widget_width_m", "widget_area_m2", "{", "}"):
        assert key not in text


def test_an_out_of_range_input_is_asked_for_by_its_label():
    env = cf.run_calculation(NAME, {"widget_width_m": 500, "widget_depth_m": 3})
    text = render.render_calculation(env, {"widget_width_m": 500})
    assert "needs input" in text and "widget width" in text
    assert "widget_width_m" not in text


def test_identifiers_in_an_answer_become_the_names_a_user_reads():
    out, n = render.readable(f"I ran {NAME} with widget_width_m = 2 and `widget_depth_m` = 3.")
    assert out == "I ran Synthetic widget area with widget width = 2 and widget depth = 3."
    assert n == 3
    plain, m = render.readable("The span and the load are within limits.")
    assert m == 0 and plain == "The span and the load are within limits."


def test_a_driver_turn_never_shows_a_registry_identifier(monkeypatch):
    from tests.test_driver_mode import _agent, _no_retrieval

    _no_retrieval(monkeypatch)
    seen = []
    replies = [
        {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "construction_calc",
            "arguments": json.dumps({"calculation": NAME, "params": {"widget_width_m": 2, "widget_depth_m": 3}})}}]},
        {"content": f"Using {NAME}, with widget_width_m 2 m and widget_depth_m 3 m, the widget area is 6 m2."},
    ]

    async def fake_stream(agent, messages, offered, model=""):
        seen.append([dict(m) for m in messages])
        reply = replies.pop(0)
        if reply.get("content"):
            yield "content", reply["content"]
        yield "done", {"status": "success", "message": reply}

    monkeypatch.setattr(llm, "stream", fake_stream)

    async def go():
        return [e async for e in loop.stream(_agent(), user_message="widget 2 m by 3 m, what area?", project_id="p")]
    events = asyncio.run(go())
    text = "".join(e["content"] for e in events if e["type"] == "token")
    for key in (NAME, "widget_width_m", "widget_depth_m", "construction_calc"):
        assert key not in text, text
    assert "6 m2" in text
    tool_msg = next(m for m in seen[1] if m.get("role") == "tool")
    assert "widget_width_m" not in tool_msg["content"] and "{" not in tool_msg["content"]
    end = events[-1]
    assert end["exit_check"]["identifiers_replaced"] >= 3


def test_a_rejected_input_ends_the_turn_with_the_calculators_question(monkeypatch):
    """Lane 2 CORE: after a calculator rejected an out-of-range input, the
    driver computed its own figure. The calculator's question is the answer."""
    from tests.test_driver_mode import _agent, _no_retrieval

    _no_retrieval(monkeypatch)
    calls = []
    replies = [
        {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "construction_calc",
            "arguments": json.dumps({"calculation": NAME, "params": {"widget_width_m": 500, "widget_depth_m": 3}})}}]},
        {"content": "Working it out myself: the widget area is 1500 m2."},
    ]

    async def fake_stream(agent, messages, offered, model=""):
        calls.append(1)
        reply = replies.pop(0)
        if reply.get("content"):
            yield "content", reply["content"]
        yield "done", {"status": "success", "message": reply}

    monkeypatch.setattr(llm, "stream", fake_stream)

    async def go():
        return [e async for e in loop.stream(_agent(), user_message="widget 500 m by 3 m, what area?", project_id="p")]
    events = asyncio.run(go())
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert len(calls) == 1, "the model must not get a turn after the calculator asked"
    assert "1500" not in text
    assert "widget width" in text and "?" in text
    assert "widget_width_m" not in text


def test_the_standing_rules_send_computed_figures_to_the_calculator():
    """Lane 4 CORE: a plain-words calculation was answered from general
    knowledge; the rules ranked general knowledge before a calculation."""
    from pathlib import Path

    rules = (Path(__file__).resolve().parents[1] / "config" / "driver_standing_rules.md").read_text(encoding="utf-8")
    assert "A figure a platform calculator computes comes from that calculator" in rules
    assert "then general knowledge, then a calculation" not in rules


def test_count_and_ratio_units_are_dimensionless_for_the_converter():
    """Lane 1 asked: schema_gaps tells a dimensionless input to declare 'count'
    or 'ratio' -- does the converter know them? It treats them as having no
    unit of its own: a figure stands as written, a percentage becomes a
    fraction for a 0..1 input, and nothing is rejected as an unknown unit."""
    count = fr.ParamSpec(name="n", kind="integer", unit="count", lo=1, hi=100,
                         label="number of items", grade="", required=True)
    ratio = fr.ParamSpec(name="r", kind="number", unit="ratio", lo=0, hi=1,
                         label="share", grade="", required=True)
    assert fr.convert_figure(count, 5, "count") == {"value": 5}
    assert fr.convert_figure(count, 5, "pieces") == {"value": 5}
    assert fr.convert_figure(ratio, 0.4, "ratio") == {"value": 0.4}
    assert fr.convert_figure(ratio, 40, "%")["value"] == pytest.approx(0.4)


def test_a_declared_unit_written_as_words_is_not_printed_as_a_unit():
    """An output declared with a note in place of a unit ("same unit as
    specified") shows its figure, not the note."""
    from app.agents.driver import render

    from tests.lane5_core_support import Param, registered

    def fn(span_m: float) -> dict:
        return {"margin": span_m / 2}

    with registered("lane5_unit_words", fn, "Qx margin check", {"span_m": Param("m", lo=0, hi=1e6, label="span")},
                    {"margin": "same unit as specified"}):
        text = render.render_calculation({"status": "success", "calculation": "lane5_unit_words",
                                          "result": {"margin": 2.0}}, {"span_m": 4})
    assert "same unit as specified" not in text
    assert "2" in text


def test_a_calculator_that_ran_on_its_defaults_has_answered():
    """Lane 4 on 17874d1: striking time and guardrail height both ran (live
    trace: ok, nothing missing) and the answers still opened by saying the
    calculator needed an input. An input a calculator filled with its own
    default is not a missing one: the standing rules say so."""
    from pathlib import Path

    rules = (Path(__file__).resolve().parents[1] / "config" / "driver_standing_rules.md").read_text(encoding="utf-8")
    assert "A calculator that ran has answered" in rules
    assert "Ask for an input only when the calculator itself asks for it" in rules


def test_a_stated_zero_is_passed_not_left_to_the_default():
    """Lane 1 on f7d3b94: a payment certificate at 0% retention with 100,000
    previously certified netted 125,000 (verified 150,000): the 0% was left
    out, so the tool's 10% default stood in for it. A stated zero is a
    figure, and the standing rules say to pass it."""
    from pathlib import Path

    rules = (Path(__file__).resolve().parents[1] / "config" / "driver_standing_rules.md").read_text(encoding="utf-8")
    assert "A figure the user states is passed to the tool even when it is zero" in rules


def test_a_one_row_result_is_still_a_table():
    """Lane 1 on 9c1d155: a 12-month cash flow came out as a 12-row table,
    the 1-month one as bullets with no table row (0/3). The standing rules
    say per-period or per-item rows are a table with every row, also one."""
    from pathlib import Path

    rules = (Path(__file__).resolve().parents[1] / "config" / "driver_standing_rules.md").read_text(encoding="utf-8")
    assert "shown as a table with every row, also when there is only one row" in rules


def test_periods_are_rows_and_their_figures_are_columns():
    """Lane 1 on 12e9321 (order 1310Z item 9): a cash flow came out turned
    sideways in 1 run of 3 -- periods as columns, no Month 1 row. The
    standing rules fix the orientation: one row per period or item, its
    figures as the columns."""
    from pathlib import Path

    rules = (Path(__file__).resolve().parents[1] / "config" / "driver_standing_rules.md").read_text(encoding="utf-8")
    assert "one row per period or item, its figures as the columns, never turned sideways" in rules
