"""Next-N-day questions that reword 'today' must not become a WBS.

Live D3, 2026-09-28 (2/2): 'As of today, 21 September 2026, what falls in
the next 14 days?' was routed to generate_wbs. 'Today is …' already
opens the look-ahead on the stated date. These phrasings are the same
ask: a rolling window from a stated today, not a WBS generate.

Synthetic programme only: FIXTURE-d-20260928-lookahead.
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import date

import pytest

import app.containers.construction.schedule as schedule_mod
import app.lib.pm_computations as pm
from app.agents.runtime import _forced_specific_tool
from app.containers.construction import ConstructionContainer
from app.core.action_router import best_action, message_wants_look_ahead
from tests.conftest import is_construction_kit_enabled, requires_construction_kit

CONTAINER = next(
    obj for obj in vars(schedule_mod).values()
    if inspect.isclass(obj) and hasattr(obj, "look_ahead")
)

# Three phrasings of the live miss, plus one sibling horizon.
AS_OF_TODAY = (
    "As of today, 21 September 2026, what falls in the next 14 days?"
)
TODAYS_DATE_IS = (
    "Today's date is 21 September 2026, what falls in the next 14 days?"
)
FROM_DATE = (
    "From 21 September 2026, what is due in the next 14 days?"
)
NEXT_THREE_WEEKS = (
    "What's coming up over the next 3 weeks as at 21 September 2026?"
)

HORIZON_ASKS = [
    pytest.param(AS_OF_TODAY, id="as-of-today-next-14-days"),
    pytest.param(TODAYS_DATE_IS, id="todays-date-is-next-14-days"),
    pytest.param(FROM_DATE, id="from-date-due-next-14-days"),
    pytest.param(NEXT_THREE_WEEKS, id="coming-up-next-3-weeks"),
]
# Sibling horizon is 3 weeks (21 days inclusive through 11 Oct). The
# three 14-day phrasings close on 4 Oct.
HORIZON_WINDOWS = [
    pytest.param(AS_OF_TODAY, 14, "2026-10-04", id="as-of-today-next-14-days"),
    pytest.param(TODAYS_DATE_IS, 14, "2026-10-04", id="todays-date-is-next-14-days"),
    pytest.param(FROM_DATE, 14, "2026-10-04", id="from-date-due-next-14-days"),
    pytest.param(NEXT_THREE_WEEKS, 21, "2026-10-11", id="coming-up-next-3-weeks"),
]

GENUINE_WBS = "Generate a WBS for a 10-floor tower"

# FIXTURE-d-20260928-lookahead. Blinding overlaps a 21 Sep window and
# misses a 27 Sep window. Backfill starts 6 Oct: inside 3 weeks, outside
# 14 days. Survey is already finished on 21 Sep.
PROGRAMME = (
    "FIXTURE-d-20260928-lookahead: "
    "Blinding 15 Sep to 25 Sep 2026; "
    "Rebar fixing 28 Sep to 2 Oct 2026; "
    "Backfill 6 Oct to 10 Oct 2026; "
    "Survey 1 Sep to 10 Sep 2026."
)

_AVAILABLE = {"generate_wbs", "look_ahead", "primavera_parser", "construction_calc"}


def _run(coro):
    return asyncio.run(coro)


def _freeze(monkeypatch, day: date):
    monkeypatch.setattr(pm, "_clock_today", lambda: day)


def _look_ahead(params):
    return asyncio.run(CONTAINER().look_ahead({}, params))


@pytest.mark.parametrize("message", HORIZON_ASKS)
def test_forced_tool_routes_reworded_horizon_to_look_ahead(message):
    msgs = [{"role": "user", "content": message}]
    assert _forced_specific_tool(msgs, _AVAILABLE) == "look_ahead"
    assert _forced_specific_tool(msgs, {"generate_wbs"}) != "generate_wbs"


@pytest.mark.parametrize("message", HORIZON_ASKS)
def test_understand_intent_routes_reworded_horizon_to_look_ahead(
    message, monkeypatch,
):
    """A schedule-shaped LLM read must not win. The stated date is today."""
    from app.core import dynamic_reasoning as dr

    called = {"n": 0}

    async def fake_schedule(*a, **k):
        called["n"] += 1
        return {
            "workflow": "schedule",
            "mode": "produce",
            "params": {"target_count": 200},
        }

    monkeypatch.setattr(dr, "complete_json", fake_schedule)
    out = _run(dr.understand_intent(message))
    assert out["action"] == "look_ahead", out
    assert out["action"] != "generate_wbs"
    assert out["params"]["as_of"] == "2026-09-21"
    assert called["n"] == 0


@pytest.mark.parametrize("message,window_days,window_end", HORIZON_WINDOWS)
def test_reworded_horizon_uses_the_stated_date_as_today(
    message, window_days, window_end, monkeypatch,
):
    """Clock is 27 Sep. The ask names 21 September as today."""
    _freeze(monkeypatch, date(2026, 9, 27))
    out = _look_ahead({"user_message": f"{message} {PROGRAMME}"})
    assert out["status"] == "success", out
    assert out["action"] == "look_ahead"
    assert out["as_of"] == "2026-09-21"
    assert out["window_days"] == window_days
    assert out["window_end"] == window_end
    names = {a["name"] for a in out["activities"]}
    assert "Blinding" in names
    assert "Survey" not in names
    if window_days == 14:
        assert "Backfill" not in names
    else:
        assert "Backfill" in names


@pytest.mark.asyncio
@pytest.mark.parametrize("message", HORIZON_ASKS)
async def test_generate_wbs_refuses_reworded_horizon(message):
    result = await ConstructionContainer().generate_wbs(
        {},
        {"brief": message, "user_message": message, "target_count": 40},
    )
    assert result.get("status") == "error", result
    assert result.get("action") == "generate_wbs"
    err = (result.get("error") or "").lower()
    assert "look-ahead" in err or "look ahead" in err or "lookahead" in err
    assert not result.get("activities")


@requires_construction_kit
@pytest.mark.parametrize("message", HORIZON_ASKS)
def test_orchestrator_routes_reworded_horizon_to_look_ahead(message):
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    result = _run(SmartOrchestratorBlock().process({"user_message": message}))
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    assert "generate_wbs" not in actions, matched
    action, confidence = best_action(result)
    assert action == "look_ahead", (action, confidence, matched)


def test_generate_a_wbs_still_routes_to_generate_wbs(monkeypatch):
    """A genuine WBS ask is not a next-N-days look-ahead."""
    from app.core import dynamic_reasoning as dr

    assert message_wants_look_ahead(GENUINE_WBS) is False
    msgs = [{"role": "user", "content": GENUINE_WBS}]
    assert _forced_specific_tool(msgs, _AVAILABLE) != "look_ahead"

    async def fake_schedule(*a, **k):
        return {
            "workflow": "schedule",
            "mode": "produce",
            "params": {"target_count": 40},
        }

    monkeypatch.setattr(dr, "complete_json", fake_schedule)
    out = _run(dr.understand_intent(GENUINE_WBS))
    assert out["action"] == "generate_wbs"
    assert out["deliverable"] is True

    if is_construction_kit_enabled():
        from app.blocks.smart_orchestrator import SmartOrchestratorBlock

        result = _run(SmartOrchestratorBlock().process({"user_message": GENUINE_WBS}))
        action, _confidence = best_action(result)
        assert action == "generate_wbs", result.get("matched_actions")


@pytest.mark.asyncio
async def test_generate_wbs_still_builds_a_genuine_wbs():
    result = await ConstructionContainer().generate_wbs(
        {},
        {
            "brief": GENUINE_WBS,
            "user_message": GENUINE_WBS,
            "target_count": 40,
        },
    )
    assert result.get("status") == "success"
    assert result.get("activities")
