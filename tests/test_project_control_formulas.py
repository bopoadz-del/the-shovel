"""A4 crew sizing / productivity (SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_RATES).

Generic synthetic inputs; no project values.
"""
from __future__ import annotations

import pytest

from app.agents.hats.planning.formulas.project_control import crew_sizing


@pytest.mark.parametrize("qty,days,rate,hours,want_crews,up,crew_days,mh", [
    # blockwork: 3,000 m2 in 15 days at 12 m2/crew-day, 10-hour day
    (3000.0, 15.0, 12.0, 10.0, 16.667, 17, 250.0, 2500.0),
    # plaster: 350 m2 in 10 days at 20 m2/crew-day, 8-hour day
    (350.0, 10.0, 20.0, 8.0, 1.75, 2, 17.5, 140.0),
])
def test_crew_sizing_agrees(qty, days, rate, hours, want_crews, up, crew_days, mh):
    out = crew_sizing(qty, days, rate, hours)
    assert out["crews_required"] == pytest.approx(want_crews, abs=0.001)
    assert out["crews_required_rounded_up"] == up
    assert out["crew_days"] == pytest.approx(crew_days)
    assert out["man_hours"] == pytest.approx(mh)


@pytest.mark.parametrize("qty,rate,crew,days", [
    (3000.0, 12.0, 17.0, 14.71),
    (350.0, 20.0, 2.0, 8.75),
])
def test_crew_sizing_duration_for_crews(qty, rate, crew, days):
    out = crew_sizing(qty, 15.0, rate, 10.0, crew_size=crew)
    assert out["duration_for_crews"] == pytest.approx(days, abs=0.01)
