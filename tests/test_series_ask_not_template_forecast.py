"""A cash-flow / S-curve ask that writes out its own period figures is not
handed to the template forecaster.

``cash_flow_forecast`` draws a template curve from ONE money total and a
duration. When the user typed the figures period by period, the template can
neither read them (no single money total -> "I need the contract value") nor
honour them (it prints its own curve). The keyword alone used to force it.

Every input here is invented for this file; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.agents import runtime as rt
from tests.conftest import requires_construction_kit

# Cash-flow / S-curve wording AND the user's own period figures.
SERIES_ASKS = (
    "Monthly valuations 4.5, 6.0, 7.5, 3.0 M; costs 80 % of value, paid "
    "the same month. Advance 15 %, retention 10 %. Show the cash flow and "
    "the S-curve.",
    "Weekly labour hours planned 320, 410, 515, 290, 160 and spent 300, "
    "430, 500 so far. Draw the cumulative S-curve for both.",
)
# Cash-flow / S-curve wording and ONE total + a duration: the template's
# own contract.
TOTAL_ASKS = (
    "Give me a 14-month cash flow forecast for a USD 36,000,000 contract.",
    "S-curve for an AED 9.5M fit-out over 10 months please.",
)
# Lists of figures with no cash-flow wording at all.
PLAIN_LISTS = (
    "Pile lengths 18.5, 21.0, 19.75 and 22.25 m - what is the average?",
    "Bay widths 7,200, 8,400, 7,200 mm along grid B.",
)

AVAILABLE = {"cash_flow_forecast", "construction_calc", "generate_wbs"}


def _turn(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# -- the series reader ------------------------------------------------------

@pytest.mark.parametrize("text", SERIES_ASKS + PLAIN_LISTS)
def test_a_written_run_of_figures_is_a_series(text):
    assert rt._message_supplies_period_series(text) is True


@pytest.mark.parametrize("text", TOTAL_ASKS + (
    "Advance 10%, retention 5%, paid 30 days after the certificate.",
    "Contract 2,400,000 over 12 months, then 3 months maintenance.",
))
def test_scattered_figures_are_not_a_series(text):
    assert rt._message_supplies_period_series(text) is False


@pytest.mark.parametrize("text,expected", [
    ("1,250, 2,500, 3,750", True),     # thousands separators inside items
    ("9; 11; 13", True),               # semicolon-separated
    ("starting 01/03/2027", False),    # a date is not three figures
    ("1,250,000 and 2,500", False),    # one grouped figure is one figure
    ("2.5, 3.5", False),               # two figures are not a run
])
def test_separator_reading(text, expected):
    assert rt._message_supplies_period_series(text) is expected


# -- the forecaster's gate --------------------------------------------------

@pytest.mark.parametrize("text", SERIES_ASKS)
def test_series_ask_is_cash_flow_but_not_for_the_template(text):
    assert rt._message_wants_cash_flow(text) is True
    assert rt._message_wants_cash_flow_forecast(text) is False


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_total_ask_still_goes_to_the_template(text):
    assert rt._message_wants_cash_flow_forecast(text) is True


@pytest.mark.parametrize("text", PLAIN_LISTS)
def test_list_without_cash_flow_wording_is_not_cash_flow(text):
    assert rt._message_wants_cash_flow_forecast(text) is False


# -- the forced tool --------------------------------------------------------

@pytest.mark.parametrize("text", SERIES_ASKS)
def test_series_ask_does_not_force_the_forecaster(text):
    assert rt._forced_specific_tool(_turn(text), AVAILABLE) != "cash_flow_forecast"


@pytest.mark.parametrize("text", TOTAL_ASKS)
def test_total_ask_forces_the_forecaster(text):
    assert rt._forced_specific_tool(_turn(text), AVAILABLE) == "cash_flow_forecast"


# -- the deterministic pre-dispatch ----------------------------------------

class _Agent:
    allowed_blocks = ["construction"]
    name = "project-assistant"


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("text", SERIES_ASKS)
async def test_series_ask_is_not_predispatched_to_the_forecaster(text):
    out = await rt._predispatch_remaining_deliverables(_Agent(), _turn(text), "proj")
    assert out is None or out.get("name") != "cash_flow_forecast"


@requires_construction_kit
@pytest.mark.asyncio
@pytest.mark.parametrize("text", TOTAL_ASKS)
async def test_total_ask_is_predispatched_to_the_forecaster(text):
    out = await rt._predispatch_remaining_deliverables(_Agent(), _turn(text), "proj")
    assert out is not None
    assert out["name"] == "cash_flow_forecast"
    assert out["ok"] is True
