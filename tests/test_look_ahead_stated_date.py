"""A stated today is the look-ahead window, not the clock.

Live D3, 2026-09-27: "Today is 21 September 2026" still opened
27 Sep–10 Oct, so Backfill was in and Blinding was out. The window is
the stated date plus 14 days: 21 Sep–4 Oct.
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import date

import pytest

import app.containers.construction.schedule as schedule_mod
import app.lib.pm_computations as pm

CONTAINER = next(
    obj for obj in vars(schedule_mod).values()
    if inspect.isclass(obj) and hasattr(obj, "look_ahead")
)

D3 = (
    "Today is 21 September 2026. Give me a two-week look-ahead from these "
    "activities: Blinding 15 Sep to 25 Sep 2026; Rebar fixing 28 Sep to "
    "2 Oct 2026; Formwork 22 Sep to 20 Oct 2026; Backfill 6 Oct to 10 Oct 2026; "
    "Survey 1 Sep to 10 Sep 2026 (complete)."
)
# Same programme, no stated today. The clock decides the window.
D3_NO_DATE = (
    "Give me a two-week look-ahead from these activities: "
    "Blinding 15 Sep to 25 Sep 2026; Rebar fixing 28 Sep to 2 Oct 2026; "
    "Formwork 22 Sep to 20 Oct 2026; Backfill 6 Oct to 10 Oct 2026; "
    "Survey 1 Sep to 10 Sep 2026 (complete)."
)


def _freeze(monkeypatch, day: date):
    monkeypatch.setattr(pm, "_clock_today", lambda: day)


def _look_ahead(params):
    return asyncio.run(CONTAINER().look_ahead({}, params))


def test_stated_date_wins_over_a_frozen_clock(monkeypatch):
    """Clock is 27 Sep (the live run). The ask says 21 September."""
    _freeze(monkeypatch, date(2026, 9, 27))
    out = _look_ahead({"user_message": D3})
    assert out["status"] == "success"
    assert out["as_of"] == "2026-09-21"
    assert out["window_days"] == 14
    assert out["window_end"] == "2026-10-04"
    assert {a["name"] for a in out["activities"]} == {
        "Blinding", "Rebar fixing", "Formwork",
    }


def test_a_missing_date_uses_the_frozen_clock(monkeypatch):
    _freeze(monkeypatch, date(2026, 9, 27))
    out = _look_ahead({"user_message": D3_NO_DATE})
    assert out["as_of"] == "2026-09-27"
    assert out["window_end"] == "2026-10-10"
    assert {a["name"] for a in out["activities"]} == {
        "Rebar fixing", "Formwork", "Backfill",
    }


@pytest.mark.parametrize("prefix", [
    "Today is 21 September 2026.",
    "Today is 21 Sep 2026.",
    "Today's date is 21 September 2026.",
    "Current date is 21 September 2026.",
    "Today is 21/09/2026.",
    "Today is 2026-09-21.",
    "as of 21 September 2026.",
])
def test_stated_date_formats(prefix, monkeypatch):
    _freeze(monkeypatch, date(2027, 1, 1))
    out = _look_ahead({"message": f"{prefix} Give me a two-week look-ahead: "
                                  "Blinding 15 Sep to 25 Sep 2026."})
    assert out["as_of"] == "2026-09-21"
    assert out["window_end"] == "2026-10-04"


def test_a_yearless_today_takes_the_programme_year(monkeypatch):
    """'21 September' with activities in 2026, clock in 2027, is 2026."""
    _freeze(monkeypatch, date(2027, 3, 1))
    out = _look_ahead({
        "user_message": (
            "Today is 21 September. Two-week look-ahead: "
            "Blinding 15 Sep to 25 Sep 2026."
        ),
    })
    assert out["as_of"] == "2026-09-21"


def test_a_yearless_slash_date_uses_the_programme_year(monkeypatch):
    _freeze(monkeypatch, date(2027, 3, 1))
    out = _look_ahead({
        "user_message": (
            "Today is 21/09. Two-week look-ahead: "
            "Blinding 15 Sep to 25 Sep 2026."
        ),
    })
    assert out["as_of"] == "2026-09-21"


def test_a_yearless_today_with_no_programme_year_uses_the_clock_year(monkeypatch):
    _freeze(monkeypatch, date(2027, 3, 1))
    assert schedule_mod._stated_look_ahead_date("Today is 21 September") == "2027-09-21"
    assert schedule_mod._stated_look_ahead_date("Today is 21/09") == "2027-09-21"


def test_an_explicit_as_of_still_wins_over_the_prose(monkeypatch):
    """A model as_of equal to the clock is the clock, not an explicit date.

    This used to assert the opposite: as_of 2026-09-27 beat "Today is
    21 September" while the clock was also 27 September. That echo is
    how a stated today lost. The stated date now wins. A genuine as_of
    that is not the clock still wins; that case is
    test_a_genuine_as_of_that_is_not_the_clock_still_wins.
    """
    _freeze(monkeypatch, date(2026, 9, 27))
    out = _look_ahead({
        "user_message": D3,
        "as_of": "2026-09-27",
    })
    assert out["as_of"] == "2026-09-21"


def test_understand_intent_passes_the_stated_date():
    from app.core import dynamic_reasoning as dr

    out = asyncio.run(dr.understand_intent(D3))
    assert out["action"] == "look_ahead"
    assert out["params"]["as_of"] == "2026-09-21"


def test_understand_intent_omits_as_of_when_no_date_is_stated():
    from app.core import dynamic_reasoning as dr

    out = asyncio.run(dr.understand_intent(D3_NO_DATE))
    assert out["action"] == "look_ahead"
    assert "as_of" not in out["params"]


def test_look_ahead_tool_schema_names_the_stated_date():
    from app.agents.runtime import Agent

    agent = Agent(
        name="stated-date",
        description="schema",
        system_prompt="schema",
        allowed_blocks=["construction"],
        can_delegate=False,
    )
    tools = agent.tool_definitions(project_id="p")
    look = next(t for t in tools if t["function"]["name"] == "look_ahead")
    as_of = look["function"]["parameters"]["properties"]["as_of"]
    assert "Today is 21 September" in as_of["description"]
    assert "21/09" in as_of["description"]
