"""The ONE-USER UI battery runner reads the route the product TOOK from the
SSE turn, never guesses it. These exercise the two pure pieces -- turn_from_
events (raw SSE -> normalized turn) and classify_route (turn -> route label) --
on SYNTHETIC events only; no battery questions, answers or client figures live
here or in the runner. Full dotted import so the mutation gate selects this.
"""
from __future__ import annotations

import pytest

from scripts.ui_battery_runner import (
    ROUTE_ASKS_INPUTS,
    ROUTE_CALCULATOR,
    ROUTE_GENERAL_RAG,
    ROUTE_PROJECT_RAG,
    ROUTE_TOOL_ON_DOC,
    classify_route,
    turn_from_events,
)


def _turn(tools=(), sources=(), provenance=(), answer=""):
    return {"tools": list(tools), "sources": list(sources),
            "provenance": list(provenance), "answer": answer}


@pytest.mark.parametrize("turn,expected", [
    # a calculator run -> LLM + calculator
    (_turn(tools=["construction_calc"], sources=[{"source_class": "project_corpus"}]),
     ROUTE_CALCULATOR),
    # a tool over a document / deliverable
    (_turn(tools=["generate_wbs"]), ROUTE_TOOL_ON_DOC),
    (_turn(tools=["fetch_document"]), ROUTE_TOOL_ON_DOC),
    # retrieved project corpus, no tool -> project RAG
    (_turn(sources=[{"source_class": "project_corpus"}]), ROUTE_PROJECT_RAG),
    # retrieved general/master corpus -> general RAG
    (_turn(sources=[{"source_class": "master_corpus"}]), ROUTE_GENERAL_RAG),
    # no tool, no retrieved source -> general LLM knowledge
    (_turn(answer="Curing keeps concrete moist for 7 days."), ROUTE_GENERAL_RAG),
    # an input request wins even when a calculator was attempted but came back
    # empty (no credited calc figure in provenance)
    (_turn(tools=["construction_calc"],
           answer="The delay damages calculator needs the rate and contract amount."),
     ROUTE_ASKS_INPUTS),
])
def test_classify_route_reads_the_observed_route(turn, expected):
    assert classify_route(turn) == expected


def test_a_credited_calc_figure_is_not_mistaken_for_an_input_request():
    # the answer merely mentions "needs" phrasing, but a calculator figure was
    # credited -> it is a calculator turn, not an input request.
    turn = _turn(
        tools=["construction_calc"],
        answer="You asked what values I used; the rate needs no change.",
        provenance=[{"figure": "SAR 10000", "source": "calculator",
                     "inputs": [{"name": "rate_percent", "source": "project_data"}]}],
    )
    assert classify_route(turn) == ROUTE_CALCULATOR


@pytest.mark.parametrize("events,exp_tools,exp_classes,exp_answer", [
    (
        [{"type": "tool_call", "tool": "construction_calc", "args_preview": "{}"},
         {"type": "tool_result"},
         {"type": "end", "content": "SAR 10,000/day",
          "sources": [{"source_class": "project_corpus"}],
          "provenance": [{"figure": "SAR 10,000/day", "source": "calculator",
                          "inputs": [{"name": "rate_percent", "value": 0.1,
                                      "unit": "%", "source": "project_data"}]}]}],
        ["construction_calc"], {"project_corpus"}, "SAR 10,000/day",
    ),
    (
        [{"type": "tool_call", "tool": "fetch_document"},
         {"type": "end", "content": "The spec says ...",
          "sources": [{"source_class": "project_document"}], "provenance": []}],
        ["fetch_document"], {"project_document"}, "The spec says ...",
    ),
])
def test_turn_from_events_collects_the_observable_signals(events, exp_tools, exp_classes, exp_answer):
    turn = turn_from_events(events)
    assert turn["tools"] == exp_tools
    assert {s["source_class"] for s in turn["sources"]} == exp_classes
    assert turn["answer"] == exp_answer
