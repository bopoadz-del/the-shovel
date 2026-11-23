"""Rule 5: an answer never says a thing is absent and then states it.

An absence sentence goes only when a later sentence of the same answer states
a cited figure naming every content word of what was said to be absent. Every
input is written here; each decision is checked on at least two inputs.
"""

from __future__ import annotations

import pytest

from app.agents import answer_contract
from app.agents.answer_contract import _drop_overridden_absence

# ── GO: a later cited figure states the thing said to be absent ────────────

@pytest.mark.parametrize(
    "text, gone, kept",
    [
        (
            ("No price on file for that — upload a priced schedule.\n\n"
            "Indicative prices: 2,750 to 3,100 QAR per tonne [1]."),
            "No price on file",
            "2,750 to 3,100 QAR per tonne [1]",
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
    out = _drop_overridden_absence(text)
    assert gone not in out
    assert kept in out


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


# ── STAY ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text",
    [
        "No price on file for that — upload a priced schedule, or get a supplier quote.",
        "There is no hire charge for the crawler crane in the documents.\n\nAsk the plant supplier for a quote.",
    ],
)
def test_an_absence_with_no_later_figure_stays(text):
    assert _drop_overridden_absence(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "No price on file for that — upload a priced schedule.\nIndicative prices run 2,750 to 3,100 QAR per tonne.",
        "There is no hire charge for the crawler crane. A crawler crane hire charge is usually 4,200 per week.",
    ],
)
def test_an_uncited_figure_overrides_nothing(text):
    assert _drop_overridden_absence(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "No price on file for blockwork.\nThe reinforcement price is 2,900 QAR per tonne [1].",
        "There is no hire charge for the crawler crane. The tower crane hire charge is 6,000 per week [2].",
    ],
)
def test_a_cited_figure_about_something_else_overrides_nothing(text):
    assert _drop_overridden_absence(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "The pump duty is 45 l/s [1]. No pump duty is given for the standby set.",
        "Cover depth is 900 mm [2].\nThe notes do not give a cover depth for the outfall.",
    ],
)
def test_an_absence_after_the_cited_figure_stays(text):
    assert _drop_overridden_absence(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Nothing was found. The slab is 250 mm thick [1].",
        "None of it is there. Concrete grade C40 applies [2].",
    ],
)
def test_an_absence_that_names_nothing_stays(text):
    assert _drop_overridden_absence(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "No price on file for that — upload a priced schedule. Indicative prices: 2,750 QAR per tonne [1].",
        "There is no hire charge for the crawler crane. The crawler crane hire charge is 4,200 per week [2].",
    ],
)
def test_an_asking_turn_is_left_as_asked(text):
    res = answer_contract.apply(text, asking=True)
    assert res.text == text
