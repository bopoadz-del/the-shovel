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
    "Quality scores are out of 100 and higher is better. Which bid wins?"
)
_OFFERS = (
    "Two offers: North 1.2 M technical score 88, South 1.0 M technical score 64. "
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

@pytest.mark.parametrize("text, criteria, expected", [
    (_OFFERS, ["price", "technical"], [
        {"name": "North", "technical": 88.0, "price": 1_200_000.0},
        {"name": "South", "technical": 64.0, "price": 1_000_000.0},
    ]),
    ("Two quotes: East quality 70 2.4 M, West quality 55 2.0 M.", ["sum", "quality"], [
        {"name": "East", "quality": 70.0, "sum": 2_400_000.0},
        {"name": "West", "quality": 55.0, "sum": 2_000_000.0},
    ]),
])
def test_a_bare_figure_goes_to_the_one_criterion_no_record_labels(text, criteria, expected):
    assert read_records(text, criteria) == expected


def test_a_bare_figure_binds_nowhere_when_two_criteria_are_unlabelled():
    assert read_records(_OFFERS, ["price", "technical", "safety"]) is None


@pytest.mark.parametrize("text, criteria", [
    ("Two offers: North price 1.2 M technical 88, South price 1.0 M 64.", ["price", "technical"]),
    ("Two quotes: East sum 2.4 M quality 70, West quality 55 2.0 M.", ["sum", "quality"]),
])
def test_a_bare_figure_binds_nowhere_when_every_criterion_is_labelled(text, criteria):
    # Every criterion carries a label somewhere, so no criterion is left
    # for the bare figure; the run is no set of records.
    assert read_records(text, criteria) is None


@pytest.mark.parametrize("text, criteria", [
    # The clause would otherwise be a full record: the name, one labelled
    # figure and one bare figure for the single unlabelled criterion.
    ("Two offers: Price technical 88 1.2 M, South technical 64 1.0 M.", ["price", "technical"]),
    ("Two quotes: Quality sum 2.4 M 70, West sum 2.0 M 55.", ["sum", "quality"]),
])
def test_a_capitalised_criterion_word_is_never_a_record_name(text, criteria):
    assert read_records(text, criteria) is None


@pytest.mark.parametrize("text, criteria, name", [
    ("Two offers: Pricey technical 88 1.2 M, South technical 64 1.0 M.", ["price", "technical"], "Pricey"),
    ("Two quotes: Qualia sum 2.4 M 70, West sum 2.0 M 55.", ["sum", "quality"], "Qualia"),
])
def test_a_name_that_only_starts_like_a_criterion_is_a_record_name(text, criteria, name):
    rows = read_records(text, criteria)
    assert rows is not None and rows[0]["name"] == name


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


def _optional_record(tenderers: dict | None = None, weights: dict | None = None) -> dict:
    return {}


def _weights_and_figures(values: list[float], weights: dict | None = None) -> dict:
    return {}


@pytest.mark.parametrize("fn, expected", [
    (_records_and_weights, "tenderers"),
    (_records_only, None),
    (_two_record_lists, None),
    (_weights_and_figures, None),
    (_optional_record, None),
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


@pytest.mark.parametrize("text, weights, names, last", [
    (_BIDS, {"cost": 60.0, "quality": 40.0}, ["P", "Q", "R", "S"],
     {"name": "S", "quality": 81.0, "cost": 3_050_000.0}),
    (_OFFERS, {"price": 50.0, "technical": 50.0}, ["North", "South"],
     {"name": "South", "technical": 64.0, "price": 1_000_000.0}),
])
def test_the_one_records_and_weights_formula_is_selected_with_both_inputs(text, weights, names, last):
    name, bound = records_completed_by_user_text(text)
    assert name == "evaluate_tender"
    assert bound["weights"] == weights
    assert [r["name"] for r in bound["tenderers"]] == names
    assert bound["tenderers"][-1] == last


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


@pytest.mark.parametrize("text", [_BIDS, _OFFERS])
def test_a_scalar_formula_the_ask_names_still_wins_over_records(monkeypatch, text):
    from app.lib import construction_formulas

    monkeypatch.setattr(
        construction_formulas, "formula_completed_by_user_text",
        lambda raw: ("scalar_formula", {"x": 1.0}),
    )
    assert runtime._user_supplied_registered_calculation(text) == ("scalar_formula", {"x": 1.0})


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


@pytest.mark.parametrize("rule, binds", [
    ("handling score = 3 points", True),     # not a weighted criterion: no rule of the scorer's
    ("price score = 3 points", False),       # a weighted amount criterion: a different rule
])
def test_a_rule_for_a_criterion_the_weights_do_not_name_is_not_read(rule, binds):
    text = _OFFERS.replace(". Who wins?", f", {rule}. Who wins?")
    assert (records_completed_by_user_text(text) is not None) is binds


@pytest.mark.parametrize("text, binds", [
    (_OFFERS, True),
    (_OFFERS.replace("Weights price 50 / technical 50", "Prices as stated"), False),
])
def test_no_selection_without_stated_weights(text, binds):
    assert (records_completed_by_user_text(text) is not None) is binds


@pytest.mark.parametrize("vetoed, supplied", [(True, False), (False, True)])
def test_a_vetoed_deliverable_is_no_supplied_calculation_even_with_records(monkeypatch, vetoed, supplied):
    monkeypatch.setattr(runtime, "_not_a_supplied_calculation", lambda raw: vetoed)
    assert bool(runtime._user_supplied_registered_calculation(_OFFERS)) is supplied


# -- the scorer's contract: only inputs it can take as written bind -----------

_TWO = "Two suppliers: Acme 1.2 M {c} {a}, Beta 1.2 M {c} {b}. Weights price 50 / {w} 50."


@pytest.mark.parametrize("crit, a, b", [
    ("delivery", 30, 90),       # days: lower is better, never called a score
    ("programme", 40, 20),      # weeks
])
def test_a_figure_the_ask_never_calls_a_score_binds_nothing(crit, a, b):
    text = _TWO.format(c=crit, a=a, b=b, w=crit)
    assert records_completed_by_user_text(text) is None


@pytest.mark.parametrize("crit, a, b, first", [
    ("delivery score", 30, 90, "Beta"),
    ("programme score", 40, 20, "Acme"),
])
def test_the_same_figure_called_a_score_and_said_better_higher_binds_and_ranks_higher_first(crit, a, b, first):
    word = crit.split()[0]
    text = _TWO.format(c=crit, a=a, b=b, w=word)
    assert records_completed_by_user_text(text) is None   # a score word alone: direction unknown
    text += " Higher scores are better."
    name, bound = records_completed_by_user_text(text)
    from app.lib.formula_registry import get

    out = get(name).fn(**bound)
    assert out["recommended"]["name"] == first


@pytest.mark.parametrize("text, binds", [
    ("Two suppliers: Acme 1.2 M quality 70, Beta 1.0 M quality 60. "
     "Weights price 50 / quality 50. Quality marks are from the panel. Higher marks are better.", True),
    ("Two suppliers: Acme 1.2 M quality 70, Beta 1.0 M quality 60. "
     "Weights price 50 / quality 50. Ratings for quality are from the panel. Higher marks are better.", True),
    ("Two suppliers: Acme 1.2 M quality 70, Beta 1.0 M quality 60. "
     "Weights price 50 / quality 50. Delivery marks are from the panel. Higher marks are better.", False),
])
def test_a_score_word_must_name_the_criterion_it_scores(text, binds):
    assert (records_completed_by_user_text(text) is not None) is binds


@pytest.mark.parametrize("sentence", [
    "Price is scored linearly, lowest 100 and highest 0.",
    "Scoring of cost is by rank.",
])
def test_a_prose_amount_rule_other_than_the_scorers_binds_nothing(sentence):
    crit = "cost" if "cost" in sentence else "price"
    text = (f"Two offers: North 2.0 M technical score 80, South 2.2 M technical score 90. "
            f"Weights {crit} 50 / technical 50. {sentence} Who wins?")
    control = text.replace(sentence + " ", "")
    assert records_completed_by_user_text(control) is not None
    assert records_completed_by_user_text(text) is None


@pytest.mark.parametrize("sentence", [
    "Price is scored as price score = lowest/price*100.",
    "Price scoring: price score = lowest price / price x 100.",
])
def test_a_prose_sentence_that_writes_the_scorers_own_rule_still_binds(sentence):
    text = ("Two offers: North 2.0 M technical score 80, South 2.2 M technical score 90. "
            f"Weights price 50 / technical 50. {sentence} Who wins?")
    assert records_completed_by_user_text(text) is not None


@pytest.mark.parametrize("text", [
    "Two mixes: M1 cement 350 water 175, M2 cement 400 water 160. Weights cement 50 / water 50.",
    "Two blends: B1 sand 60 gravel 40, B2 sand 55 gravel 45. Weights sand 70 / gravel 30.",
])
def test_records_and_weights_that_are_not_scores_or_amounts_bind_nothing(text):
    assert records_completed_by_user_text(text) is None


@pytest.mark.parametrize("weights, binds", [
    ("price 30 / technical 30", False),    # sums to 60: neither fractions nor percent
    ("price 0 / technical 0", False),      # nothing to weigh
    ("price 0.5 / technical 0.5", True),   # fractions of one
    ("price 50 / technical 50", True),     # percentages
    ("price 50.4 / technical 50", True),   # inside the scorer's own percent tolerance
    ("price 51 / technical 50", False),    # outside it
])
def test_only_weights_the_scorer_can_apply_bind(weights, binds):
    text = _OFFERS.replace("price 50 / technical 50", weights)
    assert (records_completed_by_user_text(text) is not None) is binds


@pytest.mark.parametrize("north, binds", [
    ("0 technical score 88", False),       # a zero amount: the scorer divides by it
    ("1.2 M technical score 880", False),  # over the score scale
    ("1.2 M technical score 100", True),   # the top of the scale
    ("1.2 M technical score 0", True),     # the bottom of the scale
])
def test_figures_outside_the_scorers_scale_bind_nothing(north, binds):
    text = _OFFERS.replace("1.2 M technical score 88", north)
    assert (records_completed_by_user_text(text) is not None) is binds


@pytest.mark.parametrize("bad", [
    "North 1.2 M ~ technical 88, South 1.0 M ~ technical 64.",
    "North 1.2 M + technical 88, South 1.0 M + technical 64.",
])
def test_stray_text_between_figures_makes_no_record(bad):
    good = bad.replace(" ~", "").replace(" +", "")
    assert read_records(good, ["price", "technical"]) is not None
    assert read_records(bad, ["price", "technical"]) is None


@pytest.mark.parametrize("text", [
    "Two offers: North 1.2 M technical 88 technical 70, South 1.0 M technical 64.",
    "Two quotes: East quality 70 quality 50 2.4 M, West quality 55 2.0 M.",
])
def test_one_criterion_twice_in_one_record_makes_no_record(text):
    crit = "technical" if "technical" in text else "quality"
    amount = "price" if crit == "technical" else "sum"
    assert read_records(text, [amount, crit]) is None


@pytest.mark.parametrize("label, key", [
    ("gas", "gas"),          # a three-letter word ending in s keeps its s
    ("bus fares", "bus_fare"),
    ("costs", "cost"),
])
def test_only_words_longer_than_three_letters_lose_a_plural_s(label, key):
    assert stated_records._key(label) == key


@pytest.mark.parametrize("rule", [
    "technical score = rating x 10",       # the figures are ratings, not the scores the scorer takes
    "technical score: points / 2",
])
def test_a_rule_that_rescales_a_weighted_score_criterion_binds_nothing(rule):
    text = ("Two offers: North 1.2 M technical score 8, South 1.0 M technical score 6. "
            "Weights price 50 / technical 50. Who wins?")
    assert records_completed_by_user_text(text) is not None
    ruled = text.replace(". Who wins?", f", {rule}. Who wins?")
    assert records_completed_by_user_text(ruled) is None


# -- which way a score is better ---------------------------------------------

_RANKED = "Two bids: North 1.2 M {c} {a}, South 1.2 M {c} {b}. Weights price 50 / {w} 50.{tail}"


@pytest.mark.parametrize("crit, w", [
    ("risk score", "risk"),       # a higher risk score can be the worse bid
    ("defect points", "defect"),
    ("hazard rating", "hazard"),
])
def test_a_score_of_unknown_direction_binds_nothing(crit, w):
    text = _RANKED.format(c=crit, w=w, a=20, b=80, tail="")
    assert records_completed_by_user_text(text) is None


@pytest.mark.parametrize("crit, w, tail, binds", [
    ("risk score", "risk", " Higher scores are better.", True),
    ("risk score", "risk", " Lower risk scores are better.", False),
    ("defect points", "defect", " The highest defect points win.", True),
    ("defect points", "defect", " Lowest defect points are preferred.", False),
    ("risk score", "risk", " Higher scores are better. Lower risk is better.", False),   # both ways
])
def test_the_ask_says_which_way_its_score_is_better(crit, w, tail, binds):
    text = _RANKED.format(c=crit, w=w, a=20, b=80, tail=tail)
    out = records_completed_by_user_text(text)
    assert (out is not None) is binds
    if binds:
        from app.lib.formula_registry import get

        assert get(out[0]).fn(**out[1])["recommended"]["name"] == "South"


@pytest.mark.parametrize("crit, a, b, first", [
    ("technical", 80, 60, "North"),     # the scorer's own criteria are scores, higher better
    ("commercial", 55, 75, "South"),
])
def test_the_scorers_own_criterion_binds_without_a_score_word(crit, a, b, first):
    text = _RANKED.format(c=crit, w=crit, a=a, b=b, tail=" Who wins?")
    name, bound = records_completed_by_user_text(text)
    from app.lib.formula_registry import get

    assert get(name).fn(**bound)["recommended"]["name"] == first
    lower = text.replace(" Who wins?", f" Lower {crit} marks are better.")  # no 'score is' rule form
    assert records_completed_by_user_text(lower) is None


@pytest.mark.parametrize("crit", ["quality", "delivery"])
def test_a_criterion_not_the_scorers_needs_a_score_word_even_with_a_direction(crit):
    text = _RANKED.format(c=crit, w=crit, a=60, b=80, tail=" Higher is better.")
    assert records_completed_by_user_text(text) is None


@pytest.mark.parametrize("sentence", ["A lower price wins.", "The lowest cost is preferred."])
def test_a_direction_said_of_another_criterion_does_not_apply(sentence):
    amount = "cost" if "cost" in sentence else "price"
    base = (f"Two bids: North 1.2 M quality score 70, South 1.0 M quality score 60. "
            f"Weights {amount} 50 / quality 50.")
    assert records_completed_by_user_text(f"{base} {sentence}") is None   # quality's way still unknown
    assert records_completed_by_user_text(f"{base} Higher quality scores are better. {sentence}") is not None


@pytest.mark.parametrize("sentence", ["The lowest bid wins.", "The lowest tender is preferred."])
def test_a_direction_sentence_naming_no_criterion_and_no_score_does_not_apply(sentence):
    text = ("Two bids: North 1.2 M quality score 70, South 1.0 M quality score 60. "
            f"Weights price 50 / quality 50. Higher quality scores are better. {sentence}")
    assert records_completed_by_user_text(text) is not None


@pytest.mark.parametrize("crit", ["quality", "delivery"])
def test_a_criterion_is_scored_counts_as_calling_it_a_score(crit):
    text = _RANKED.format(c=crit, w=crit, a=60, b=80,
                          tail=f" {crit.capitalize()} is scored out of 100, higher is better.")
    assert records_completed_by_user_text(text) is not None


# -- an amount the ask has already scored is not an amount -------------------

@pytest.mark.parametrize("word", ["marks", "rating", "points"])
def test_an_amount_criterion_the_ask_scores_with_any_score_word_binds_nothing(word):
    text = (f"Two bids: North price {word} 90 technical 80, South price {word} 70 technical 85. "
            "Weights price 50 / technical 50.")
    assert records_completed_by_user_text(text) is None
    control = text.replace(f"price {word} 90", "1.2 M").replace(f"price {word} 70", "1.0 M")
    assert records_completed_by_user_text(control) is not None


@pytest.mark.parametrize("sentence, crit", [
    ("Prices are scored by rank.", "price"),
    ("Costs are scored by rank.", "cost"),
    ("Scoring on price is by rank.", "price"),
    ("Scoring on the cost is by rank.", "cost"),
])
def test_a_plural_or_scoring_on_amount_rule_binds_nothing(sentence, crit):
    text = (f"Two offers: North 2.0 M technical 80, South 2.2 M technical 90. "
            f"Weights {crit} 50 / technical 50. {sentence}")
    assert records_completed_by_user_text(text.replace(" " + sentence, "")) is not None
    assert records_completed_by_user_text(text) is None


@pytest.mark.parametrize("rule", ["price score = lowest/price", "cost score = lowest cost / cost"])
def test_the_scorers_rule_written_without_x100_binds(rule):
    crit = rule.split()[0]
    text = (f"Two offers: North 2.0 M technical 80, South 2.2 M technical 90. "
            f"Weights {crit} 50 / technical 50, {rule}. Who wins?")
    assert records_completed_by_user_text(text) is not None


@pytest.mark.parametrize("weights, binds", [
    ("price 0.5 / technical 0.504", True),    # inside the fraction tolerance
    ("price 0.5 / technical 0.53", False),    # outside it
    ("price 0.5 / technical 0.47", False),
])
def test_fraction_weights_hold_the_scorers_tolerance(weights, binds):
    text = _OFFERS.replace("price 50 / technical 50", weights)
    assert (records_completed_by_user_text(text) is not None) is binds


@pytest.mark.parametrize("second", ["price score = rank order", "price score = 100 - rank"])
def test_two_rules_for_one_amount_one_not_the_scorers_binds_nothing(second):
    text = ("Two offers: North 2.0 M technical 80, South 2.2 M technical 90. Weights price 50 / technical 50. "
            f"Price is scored with price score = lowest/price x 100, {second}.")
    assert records_completed_by_user_text(text) is None
