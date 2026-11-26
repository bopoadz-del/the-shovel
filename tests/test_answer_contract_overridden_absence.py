"""Rule 5: an answer never says a thing is absent and then states it.

An absence sentence goes only when a later sentence of the same answer states
a figure, carries one of the turn's own citation markers, and names exactly
the thing said to be absent (no word missing, no word added). Every input is
written here; each decision is checked on at least two inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.answer_contract import _drop_overridden_absence

#: The turn's citation markers for the direct-function tests.
CITED = frozenset(f"[{n}]" for n in range(1, 10))


def drop(text: str, cited=CITED) -> str:
    return _drop_overridden_absence(text, cited)


# ── GO: a later cited figure states exactly the thing said to be absent ────

@pytest.mark.parametrize(
    "text, gone, kept",
    [
        (
            ("No price on file for blockwork — upload a priced schedule.\n\n"
            "The blockwork price is 12 per block [1]."),
            "No price on file",
            "12 per block [1]",
        ),
        (
            ("**There is no hire charge for the crawler crane in the documents.**\n"
            "The crawler crane hire charge is 4,200 per week [2]."),
            "There is no hire charge",
            "4,200 per week [2]",
        ),
        (
            ("No lead time is listed for glazing units.\n"
            "- Glazing units lead time: 14 weeks from approval [3]."),
            "No lead time is listed",
            "14 weeks from approval [3]",
        ),
    ],
)
def test_an_absence_a_later_cited_figure_states_goes(text, gone, kept):
    out = drop(text)
    assert gone not in out
    assert kept in out


@pytest.mark.parametrize(
    "text, gone",
    [
        ("The crane hire charge is not stated. The crane hire charge is 4,200 per week [1].",
         "is not stated"),
        ("Lead time for glazing: not listed. Glazing lead time is 14 weeks [2].",
         "not listed"),
    ],
)
def test_a_negation_after_the_thing_with_only_filler_is_an_absence(text, gone):
    assert gone not in drop(text)


@pytest.mark.parametrize(
    "text, gone",
    [
        ("No price on file for blockwork — ask the supplier for one. The blockwork price is 12 per block [1].",
         "No price on file"),
        ("No pump duty is given; check with the supplier. The pump duty is 45 l/s [2].",
         "No pump duty"),
    ],
)
def test_the_absent_thing_is_the_first_clause_only(text, gone):
    # The remedy after the dash or semicolon names other words; they are not
    # part of the absent thing, so the cited figure still overrides it.
    assert gone not in drop(text)


@pytest.mark.parametrize(
    "text, gone",
    [
        (
            "Nothing on the pump duty was found. The pump duty is 45 l/s at 30 m head (pumps_schedule.pdf, p. 4).",
            "Nothing on the pump duty",
        ),
        (
            ("I could not find a cover depth for the drainage run. "
            "The drainage run cover depth is 900 mm (drainage_notes.pdf, p. 2)."),
            "could not find a cover depth",
        ),
    ],
)
def test_through_apply_the_overridden_absence_goes(text, gone):
    res = answer_contract.apply(text)
    assert gone not in res.text
    assert "[1]" in res.text
    assert res.citations


@pytest.mark.parametrize(
    "text, expected",
    [
        ("No lead time is listed for glazing units.\nGlazing units lead time: 14 weeks [3].",
         "Glazing units lead time: 14 weeks [3]."),
        ("There is no pump duty.\n\nThe pump duty is 45 l/s [1].",
         "\nThe pump duty is 45 l/s [1]."),
    ],
)
def test_a_line_that_was_only_the_absence_goes_whole(text, expected):
    assert drop(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("- Glazing:\n  - No lead time is listed for glazing units. Glazing units lead time: 14 weeks [3].",
         "- Glazing:\n  - Glazing units lead time: 14 weeks [3]."),
        ("1. No pump duty is given. The pump duty is 45 l/s [1].",
         "1. The pump duty is 45 l/s [1]."),
    ],
)
def test_a_list_item_keeps_its_indent_and_bullet(text, expected):
    assert drop(text) == expected


# ── STAY ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text",
    [
        "No price on file for that — upload a priced schedule, or get a supplier quote.",
        "There is no hire charge for the crawler crane in the documents.\n\nAsk the plant supplier for a quote.",
    ],
)
def test_an_absence_with_no_later_figure_stays(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "No price on file for blockwork.\nThe blockwork price runs 12 per block.",
        "There is no hire charge for the crawler crane. A crawler crane hire charge is usually 4,200 per week.",
    ],
)
def test_an_uncited_figure_overrides_nothing(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text, cited",
    [
        ("No steel rate is given. The steel rate is 2,400 per tonne [3].", frozenset({"[1]"})),
        ("There is no pump duty. The pump duty is 45 l/s [2].", frozenset()),
    ],
)
def test_a_bracket_that_is_not_one_of_the_turns_citations_overrides_nothing(text, cited):
    assert drop(text, cited) == text
    # The same answer with that bracket among the turn's citations does drop.
    assert drop(text) != text


@pytest.mark.parametrize(
    "text",
    [
        "No steel rate is given. Section [3] gives the steel rate as 2,400.",
        "There is no pump duty. Clause [2] puts the pump duty at 45 l/s.",
    ],
)
def test_through_apply_a_bare_bracket_with_no_sources_overrides_nothing(text):
    assert answer_contract.apply(text).text == text


@pytest.mark.parametrize(
    "text",
    [
        "No price on file for blockwork.\nThe reinforcement price is 2,900 QAR per tonne [1].",
        "There is no hire charge for the crawler crane. The tower crane hire charge is 6,000 per week [2].",
    ],
)
def test_a_cited_figure_about_something_else_overrides_nothing(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "There is no rate for excavation. The rate for rock excavation is 45 per m3 [1].",
        "| Item | Rate |\n|---|---|\n| Rebar | not listed |\n| Rebar couplers | 15 each [1] |",
        "No price on file for that — upload a priced schedule.\n\nIndicative prices: 2,750 to 3,100 QAR per tonne [1].",
    ],
)
def test_a_cited_figure_about_a_narrower_thing_overrides_nothing(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "No rate for rock excavation is given. The excavation rate is 30 per m3 [1].",
        "There is no hire charge for the crawler crane. The crane hire charge is 4,000 per week [2].",
    ],
)
def test_a_cited_figure_about_a_wider_thing_overrides_nothing(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "The steel rate does not include VAT. The steel rate including VAT is 2,880 per tonne [1].",
        "The crane hire charge does not cover the operator. The crane hire charge with operator is 5,000 per week [1].",
    ],
)
def test_a_negation_between_two_named_things_is_not_an_absence(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "The pump duty is 45 l/s [1]. No pump duty is given for the standby set.",
        "Cover depth is 900 mm [2].\nThe notes do not give a cover depth for the outfall.",
        "The blockwork price is 12 per block [1]. No blockwork price is given.",
        "The pump duty is 45 l/s [2].\nNothing on the pump duty was found.",
    ],
)
def test_an_absence_after_the_cited_figure_stays(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "No blockwork price is given. The blockwork price is listed [1].",
        "There is no pump duty. The pump duty is shown [2].",
    ],
)
def test_a_cited_sentence_with_no_figure_overrides_nothing(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text, gone, kept",
    [
        (
            "No steel rate is given. The steel rate is given as 2,400 per tonne [1].",
            "No steel rate",
            "given as 2,400 per tonne [1]",
        ),
        (
            "The pump duty is not stated. The pump duty is shown as 45 l/s [2].",
            "is not stated",
            "shown as 45 l/s [2]",
        ),
    ],
)
def test_a_figure_stated_as_a_value_names_the_same_thing(text, gone, kept):
    out = drop(text)
    assert gone not in out
    assert kept in out


@pytest.mark.parametrize(
    "text",
    [
        "Crane hire is not 4,000 per week.\nCrane hire is 4,200 per week [1].",
        "There is no crane hire charge [1]. The crane hire charge is 4,200 per week [2].",
    ],
)
def test_a_sentence_with_its_own_figure_or_marker_is_not_an_absence(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Nothing was found. The slab is 250 mm thick [1].",
        "None of it is there. Concrete grade C40 applies [2].",
    ],
)
def test_an_absence_that_names_nothing_stays(text):
    assert drop(text) == text


@pytest.mark.parametrize(
    "text, absence",
    [
        ("No price on file for blockwork. The blockwork price is 12 per block (boq.pdf, p. 4).",
         "No price on file for blockwork."),
        ("There is no hire charge for the crawler crane. The crawler crane hire charge is 4,200 per week (plant.pdf, p. 2).",
         "There is no hire charge for the crawler crane."),
    ],
)
def test_an_asking_turn_is_left_as_asked(text, absence):
    assert absence in answer_contract.apply(text, asking=True).text
    # The same answer on a turn that is not asking loses the absence line.
    assert absence not in answer_contract.apply(text).text
