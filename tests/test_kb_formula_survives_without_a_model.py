"""Exam 12 extended to the knowledge base (BUILD 2 last gap, part 2).

A knowledge-base formula the ask names and fully supplies is evaluated by the
deterministic pre-dispatch, before any model is asked. With NO provider
configured the user still gets the figure, the working with the stated values,
and one plain line that the explanation is unavailable. An ask that names an
entry but leaves an input out is not answered with a figure: it gets the
honest "model unavailable" message.

Every entry, title, variable and figure below is invented here and lives in
this test's own knowledge-base file (CONSTRUCTION_KB_FILE). Expected figures
are hand calculations from the stated inputs. The provider boundary is a
recorder that fails the test if anything tries to reach it. Failures name the
stage: select (the entry was not chosen), evaluate (wrong figure), present
(what the user sees).
"""
from __future__ import annotations

import json
import math

import pytest

from app.blocks import _knowledge as kb
from tests.conftest import construction_kit_markers
from tests.test_calculations_survive_without_a_model import _NOTE, _ask, no_model  # noqa: F401

pytestmark = construction_kit_markers


def _entry(eid, title, expr, variables, unit):
    return {
        "id": eid, "type": "formula", "title": title, "expression": expr,
        "statement": title, "domain": "construction.test",
        "variables": variables, "thresholds": {"unit": unit},
        "credibility_tier": 3, "needs_review": False,
        "provenance": {"source": "invented test reference",
                       "confidence": "site_experience",
                       "verified_against_standard": False},
    }


ENTRIES = [
    _entry("t.kiln_soak", "Kiln soak duration", "40 * Z",
           {"Z": {"desc": "Wall thickness of the kiln", "unit": "meter"}}, "hour"),
    _entry("t.barge_trips", "Barge trip count", "ceiling(cargo / hold)",
           {"cargo": {"desc": "Total cargo to move", "unit": "tonne"},
            "hold": {"desc": "Capacity of one barge hold", "unit": "tonne"}}, "trip"),
]


@pytest.fixture(autouse=True)
def _own_kb(tmp_path, monkeypatch):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"schema_version": "1", "kb_version": "exam",
                                "entries": ENTRIES}), encoding="utf-8")
    monkeypatch.setenv("CONSTRUCTION_KB_FILE", str(path))
    monkeypatch.setattr(kb, "_KB_CACHE", None)
    yield


# Two different invented entries, each with every input stated.
KB_ASKS = [
    pytest.param("Calculate the kiln soak duration for a 0.6 m wall.",
                 f"{40 * 0.6:g} hour", "Z = 0.6", id="kiln-soak-by-unit"),
    pytest.param("Calculate the barge trip count: cargo = 950 t, hold = 300 t.",
                 f"{math.ceil(950 / 300)} trip", "cargo = 950", id="barge-trips-by-name"),
]


@pytest.mark.parametrize("message, figure, working", KB_ASKS)
def test_a_kb_formula_answers_with_no_model(no_model, message, figure, working):
    shown, errors = _ask(message)
    assert not errors, ("stage=present", shown, errors)
    assert figure in shown, ("stage=evaluate", shown)
    assert working in shown, ("stage=present: working line missing", shown)
    assert _NOTE in shown.lower(), ("stage=present: no unreached note", shown)
    assert shown.lower().index(_NOTE) > shown.index(figure), ("stage=present", shown)
    assert '"status"' not in shown and "PRE-DISPATCH" not in shown, ("stage=present", shown)
    assert no_model == [], no_model


@pytest.mark.parametrize("message", [
    # The entry is named but its one input is not given.
    "Calculate the kiln soak duration for the kiln.",
    # Two figures of the same unit and no names: which is the cargo is unknown.
    "Calculate the barge trip count for 950 t and 300 t.",
])
def test_an_incomplete_kb_ask_gets_no_invented_figure(no_model, message):
    from app.routers.chat_watchdog import PROVIDER_FAILURE_MESSAGE

    shown, errors = _ask(message)
    said = (shown + " " + " ".join(errors)).strip()
    assert PROVIDER_FAILURE_MESSAGE in said, ("stage=select: should not answer", said)
    for figure in (f"{40:g} hour", f"{math.ceil(950 / 300)} trip"):
        assert figure not in said, ("stage=select: figure invented", said)
