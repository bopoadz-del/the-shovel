"""Weighted records an ask states bind to the records-and-weights formula.

Every ask here is invented. Each decision is shown from both sides: the
input that binds and the nearest input that must not.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import runtime
from app.lib import stated_records
from app.lib.stated_records import (
    records_completed_by_user_text,
    stated_weights,
)
from app.lib.stated_records import (
    stated_records as read_records,
)

_BIDS = (
    "Four bids: P 3.20 M quality 75, Q 2.90 M quality 60, R 3.60 M quality 92, "
    "S 3.05 M quality 81. Weights cost 60 / quality 40, cost score = lowest/cost*100. "
    "Which bid wins?"
)
_OFFERS = (
    "Two offers: North 1.2 M technical 88, South 1.0 M technical 64. "
    "Weights price 50 / technical 50. Who wins?"
)


# -- weights ---------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Weights cost 60 / quality 40.", {"cost": 60.0, "quality": 40.0}),
    ("weighting: 25% safety, 75% durability", {"safety": 25.0, "durability": 75.0}),
])
def test_weights_are_read_after_the_weight_word(text, expected):
    assert stated_weights(text) == expected


@pytest.mark.parametrize("text", [
    "Cost 60 / quality 40.",                 # no weight word
    "Weights cost 60.",                      # one criterion is no weighting
    "Weights cost 60 / cost 40.",            # one label twice
    "Weights cost 60 / about forty quality", # an item that is no label-and-figure
])
def test_no_weights_without_two_labelled_figures_after_the_weight_word(text):
    assert stated_weights(text) is None


# -- records ---------------------------------------------------------------

def test_a_bare_figure_goes_to_the_one_criterion_no_record_labels():
    rows = read_records(_OFFERS, ["price", "technical"])
    assert rows == [
        {"name": "North", "technical": 88.0, "price": 1_200_000.0},
        {"name": "South", "technical": 64.0, "price": 1_000_000.0},
    ]


def test_a_bare_figure_binds_nowhere_when_two_criteria_are_unlabelled():
    assert read_records(_OFFERS, ["price", "technical", "safety"]) is None


@pytest.mark.parametrize("text, expected_price", [
    ("Two offers: North 1.2 M technical 88, South 900 k technical 64.", 900_000.0),
    ("Two offers: North 1.2 M technical 88, South 1.0 million technical 64.", 1_000_000.0),
])
def test_a_scale_word_scales_the_amount(text, expected_price):
    rows = read_records(text, ["price", "technical"])
    assert rows is not None and rows[1]["price"] == expected_price


def test_a_lower_case_m_is_a_metre_not_a_million():
    text = "Two runs: North 1.2 m technical 88, South 1.0 m technical 64."
    assert read_records(text, ["price", "technical"]) is None


@pytest.mark.parametrize("text", [
    "One offer: North 1.2 M technical 88.",                               # one record
    "Two offers: North 1.2 M technical 88, North 1.0 M technical 64.",    # same name twice
    "Two offers: North 1.2 M technical 88, South 1.0 M.",                 # a criterion missing
    "Two offers: North 1.2 M technical 88, South roughly 1.0 M technical 64.",  # a stray word
    "Two offers: technical 1.2 M technical 88, South 1.0 M technical 64.",      # opens on a criterion
    "Two offers: North 1.2 M 0.9 M technical 88, South 1.0 M technical 64.",    # two bare figures
])
def test_no_records_when_any_clause_is_not_a_full_record(text):
    assert read_records(text, ["price", "technical"]) is None


def test_a_criterion_label_matches_its_plural():
    text = "Two offers: North 1.2 M points 88, South 1.0 M points 64."
    rows = read_records(text, ["price", "point"])
    assert rows is not None and rows[0]["point"] == 88.0


# -- the stated scoring rule ------------------------------------------------

@pytest.mark.parametrize("rule, binds", [
    ("price score = lowest/price*100", True),
    ("price score = lowest price / price x 100", True),
    ("price score = 100 - percent above lowest", False),
    ("price score = price/highest*100", False),
])
def test_a_stated_amount_rule_other_than_lowest_over_amount_binds_nothing(rule, binds):
    text = _OFFERS.replace(". Who wins?", f", {rule}. Who wins?")
    assert (records_completed_by_user_text(text) is not None) is binds


# -- selection by signature --------------------------------------------------

def _records_and_weights(tenderers: list[dict], weights: dict | None = None) -> dict:
    return {}


def _records_only(tenderers: list[dict], factor: float = 1.0) -> dict:
    return {}


def _two_record_lists(a: list[dict], b: list[dict], weights: dict | None = None) -> dict:
    return {}


def _weights_and_figures(values: list[float], weights: dict | None = None) -> dict:
    return {}


@pytest.mark.parametrize("fn, expected", [
    (_records_and_weights, "tenderers"),
    (_records_only, None),
    (_two_record_lists, None),
    (_weights_and_figures, None),
])
def test_the_records_parameter_is_found_from_the_signature(fn, expected):
    assert stated_records._takes_records_and_weights(fn) == expected


def _unresolved_hint(tenderers: NoSuchRecordType, weights: dict | None = None) -> dict:  # noqa: F821
    return {}


@pytest.mark.parametrize("fn, readable", [(_unresolved_hint, False), (_records_and_weights, True)])
def test_an_unreadable_signature_is_logged_and_takes_nothing(fn, readable, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="app.lib.stated_records"):
        out = stated_records._takes_records_and_weights(fn)
    logged = "signature unreadable" in caplog.text
    assert logged is (not readable)
    assert (out is not None) is readable


def test_the_one_records_and_weights_formula_is_selected_with_both_inputs():
    name, bound = records_completed_by_user_text(_BIDS)
    assert name == "evaluate_tender"
    assert bound["weights"] == {"cost": 60.0, "quality": 40.0}
    assert [r["name"] for r in bound["tenderers"]] == ["P", "Q", "R", "S"]
    assert bound["tenderers"][3] == {"name": "S", "quality": 81.0, "cost": 3_050_000.0}


def test_no_selection_when_two_formulas_take_records_and_weights(monkeypatch):
    from app.lib import formula_registry

    real = list(formula_registry.all_specs())
    twin = next(s for s in real if stated_records._takes_records_and_weights(s.fn))
    monkeypatch.setattr(formula_registry, "all_specs", lambda: real + [twin])
    assert records_completed_by_user_text(_BIDS) is None


# -- routing: the ask is a supplied calculation ------------------------------

@pytest.mark.parametrize("text, supplied", [
    (_BIDS, True),
    (_BIDS.replace("Weights cost 60 / quality 40, ", ""), False),
])
def test_stated_weighted_records_make_a_supplied_calculation(text, supplied):
    assert bool(runtime._user_supplied_registered_calculation(text)) is supplied


def test_stated_weighted_records_are_a_formula_style_ask():
    assert runtime._message_is_formula_style_ask(_BIDS) is True


def test_a_scalar_formula_the_ask_names_still_wins_over_records(monkeypatch):
    from app.lib import construction_formulas

    monkeypatch.setattr(
        construction_formulas, "formula_completed_by_user_text",
        lambda raw: ("scalar_formula", {"x": 1.0}),
    )
    assert runtime._user_supplied_registered_calculation(_BIDS) == ("scalar_formula", {"x": 1.0})


class _Agent:
    allowed_blocks = ("construction",)

    def __init__(self):
        self.calls: list[dict] = []

    async def _run_tool_call(self, tc):
        self.calls.append(json.loads(tc["function"]["arguments"]))
        from app.lib.formula_registry import get

        args = json.loads(tc["function"]["arguments"])
        params = {k: v for k, v in args["params"].items() if k != "text"}
        return {"result": {"status": "success", "calculation": args["calculation"],
                           "result": get(args["calculation"]).fn(**params)}}


@pytest.mark.parametrize("text, ranked_first", [
    (_BIDS, "S"),
    (_OFFERS, "North"),
])
def test_predispatch_runs_the_scoring_on_the_stated_records(text, ranked_first):
    agent = _Agent()
    messages = [{"role": "user", "content": text}]
    asyncio.run(runtime._predispatch_formula_calc(agent, messages, "p1", operator_text=text))
    assert agent.calls and agent.calls[0]["calculation"] == "evaluate_tender"
    tool_turns = [m for m in messages if m.get("role") == "tool"]
    assert tool_turns
    body = json.loads(tool_turns[0]["content"])
    assert body["result"]["recommended"]["name"] == ranked_first
