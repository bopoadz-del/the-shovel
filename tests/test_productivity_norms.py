"""S-1b (ORDERS_28OCT_0900Z): generic, sourced productivity norms.

When a brief states a *quantity* for an activity but no productivity *rate*,
the schedule estimator consults a GENERIC per-trade norm table so the activity
gets a kernel-computed duration instead of UNKNOWN. The norm figures are
generic planning values (no client/project data), each sourced, and a rate
stated in the brief always wins over a norm.

These pins (complete_json monkeypatched, construction kernel live):
  * two quantities of one trade -> two input-derived durations (the norm is a
    rate, not a canned answer);
  * mutate the norm rate -> the duration moves (kernel-detection);
  * a STATED rate is used and the norm is NOT consulted;
  * a trade the table does not cover -> UNKNOWN (never a default);
  * the derived duration is labelled with the norm source;
  * every norm row is generic + sourced.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer
from app.core import llm_client
from app.lib import productivity_norms
from app.lib.productivity_norms import norm_rate_for


def _patch_llm(monkeypatch, mapping):
    async def fake(system, user, **kwargs):
        low = (user or "").lower()
        for key, acts in mapping.items():
            if key in low:
                return {"activities": acts}
        return {"activities": []}
    monkeypatch.setattr(llm_client, "complete_json", fake)


async def _wbs(brief):
    return await ConstructionContainer().generate_wbs({"brief": brief}, {"target_count": 40})


def _act(result, name):
    for a in result.get("activities", []):
        if a["name"] == name:
            return a
    for a in result.get("unscheduled_activities", []) or []:
        if a["name"] == name:
            return a
    return None


# ---- the norm lookup itself ------------------------------------------------

def test_norm_rate_for_matches_generic_trades_and_is_sourced():
    exc = norm_rate_for("Bulk excavation to reduced level")
    assert exc is not None
    assert "per gang-day" in exc["rate"]
    assert exc["source"]
    assert norm_rate_for("General site clearance")["rate"] == "500 m2 per gang-day"
    # An uncovered / bespoke trade has no generic norm.
    assert norm_rate_for("Install the proprietary widget assembly") is None
    assert norm_rate_for("") is None


def test_every_norm_row_is_generic_and_sourced():
    for row in productivity_norms._NORMS:
        assert row["keywords"] and all(isinstance(k, str) and k for k in row["keywords"])
        assert "per gang-day" in str(row["rate"])
        assert str(row["source"]).strip()
        # generic: the source declares it is not project-specific
        assert "not project-specific" in str(row["source"]).lower()


# ---- integration through generate_wbs --------------------------------------

@pytest.mark.asyncio
async def test_two_quantities_one_trade_give_two_input_derived_durations(monkeypatch):
    _patch_llm(monkeypatch, {
        "small": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 200, "unit": "m3"}],
        "large": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 800, "unit": "m3"}],
    })
    small = await _wbs("A small job: 200 m3 of excavation.")
    large = await _wbs("A large job: 800 m3 of excavation.")
    a_small = _act(small, "Bulk excavation")
    a_large = _act(large, "Bulk excavation")
    assert a_small and a_large
    # norm 200 m3/gang-day -> 1 day vs 4 days; derived from the quantity.
    assert a_small["duration_days"] == 1
    assert a_large["duration_days"] == 4
    assert a_large["duration_days"] > a_small["duration_days"]


@pytest.mark.asyncio
async def test_mutating_the_norm_moves_the_duration(monkeypatch):
    """Kernel-detection: the duration tracks the norm, it is not canned."""
    _patch_llm(monkeypatch, {
        "job": [{"name": "Bulk excavation", "source_span": "excavation", "quantity": 400, "unit": "m3"}],
    })

    monkeypatch.setattr(
        productivity_norms, "norm_rate_for",
        lambda _n: {"rate": "100 m3 per gang-day", "source": "test norm A (not project-specific)"},
    )
    slow = await _wbs("The job: 400 m3 of excavation.")
    monkeypatch.setattr(
        productivity_norms, "norm_rate_for",
        lambda _n: {"rate": "200 m3 per gang-day", "source": "test norm B (not project-specific)"},
    )
    fast = await _wbs("The job: 400 m3 of excavation.")
    assert _act(slow, "Bulk excavation")["duration_days"] == 4   # 400/100
    assert _act(fast, "Bulk excavation")["duration_days"] == 2   # 400/200


@pytest.mark.asyncio
async def test_stated_rate_wins_over_the_norm(monkeypatch):
    """A rate in the brief is used verbatim; the norm is not consulted."""
    _patch_llm(monkeypatch, {
        "job": [{
            "name": "Bulk excavation", "source_span": "excavation",
            "quantity": 200, "unit": "m3",
            "productivity_rate": "100 m3 per gang-day",  # half the norm's 200
        }],
    })
    r = await _wbs("The job: 200 m3 of excavation at 100 m3 per gang-day.")
    a = _act(r, "Bulk excavation")
    assert a["duration_days"] == 2   # 200/100 from the STATED rate, not 1 from the norm
    assert "basis" in a["duration_basis"]
    assert "norm" not in a["duration_basis"]


@pytest.mark.asyncio
async def test_uncovered_trade_with_quantity_stays_unknown(monkeypatch):
    _patch_llm(monkeypatch, {
        "job": [{"name": "Install proprietary widget", "source_span": "widget", "quantity": 50, "unit": "nr"}],
    })
    r = await _wbs("The job: install the widget, 50 nr.")
    a = _act(r, "Install proprietary widget")
    assert a is not None
    assert a["duration_days"] is None           # UNKNOWN, never a default
    assert r.get("schedule_incomplete") is True


@pytest.mark.asyncio
async def test_norm_derived_duration_is_labelled_with_the_source(monkeypatch):
    _patch_llm(monkeypatch, {
        "job": [{"name": "Blockwork to walls", "source_span": "blockwork", "quantity": 150, "unit": "m2"}],
    })
    r = await _wbs("The job: 150 m2 of blockwork.")
    a = _act(r, "Blockwork to walls")
    assert a["duration_days"] == 10             # 150 / 15 m2 gang-day
    assert "norm" in a["duration_basis"]
    assert "not project-specific" in a["duration_basis"].lower()
