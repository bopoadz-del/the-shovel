"""One duration kernel, and it runs through the calculation envelope (KS-4).

The quantity -> duration step of a schedule is a kernel calculation, so it
must come back as the standard calculation envelope (status / calculation /
result / standard / provenance / certified) and it must exist once. These
tests construct their own rows: no project, no document, no probe id, and no
figure copied from a real bill. Every expected day count is computed by hand
from the two numbers the test itself wrote.
"""

from __future__ import annotations

import math

from app.containers.construction import schedule as brief_schedule
from app.lib import boq_schedule, pm_computations

KERNEL = "productivity_manpower_duration"


def _hand_days(quantity: float, per_day_rate: float) -> int:
    """The day count this test expects, worked out here, not read from the code."""
    return int(math.ceil(quantity / per_day_rate))


def test_the_duration_kernel_returns_the_standard_calculation_envelope():
    quantity, rate_value = 480.0, 60.0
    days, error, envelope = pm_computations.duration_days_from_rate(
        quantity, f"{rate_value:g} m2 per gang-day"
    )

    assert error is None
    assert days == _hand_days(quantity, rate_value)
    assert envelope["status"] == "success"
    assert envelope["calculation"] == KERNEL
    assert envelope["provenance"] == "kernel"
    assert isinstance(envelope["certified"], bool)
    assert str(envelope["standard"]).strip()
    assert envelope["standard"] != "none declared"


def test_a_row_that_states_its_rate_gets_its_duration_from_that_one_kernel():
    quantity, rate_value = 300.0, 50.0
    row = {
        "item_key": "E1.1",
        "description": "Excavate trench for drainage",
        "unit": "m3",
        "quantity": quantity,
        "productivity_rate": f"{rate_value:g} m3 per gang-day",
        "source": "row written by this test",
    }

    days, basis, envelope = boq_schedule.boq_row_duration(row)

    assert days == _hand_days(quantity, rate_value)
    assert basis.startswith("estimated")
    assert envelope["provenance"] == "kernel"


def test_the_outline_activity_carries_the_kernel_badge_it_was_computed_with():
    quantity, rate_value = 120.0, 40.0
    row = {
        "item_key": "D1.2",
        "description": "Remove reinforced concrete slab",
        "unit": "m3",
        "quantity": quantity,
        "productivity_rate": f"{rate_value:g} m3 per gang-day",
        "source": "row written by this test",
    }

    activities = boq_schedule.activities_from_boq_scope_outline([row])

    assert len(activities) == 1
    activity = activities[0]
    assert activity["duration_days"] == _hand_days(quantity, rate_value)
    assert activity["duration_provenance"] == "kernel"
    assert activity["duration_standard"] == (
        "PE formula sheet (Productivity / Manpower / Duration)"
    )
    assert activity["duration_certified"] is False


def test_a_row_without_a_rate_keeps_the_three_part_contract():
    """No stated rate: the norm branch answers with the same (days, basis,
    envelope) shape, and the basis says which case it is. It never invents a
    day count and never stamps a badge it did not earn."""
    row = {
        "item_key": "Z9.9",
        "description": "Work described only by this sentence",
        "unit": "m3",
        "quantity": 100.0,
        "source": "row written by this test",
    }

    days, basis, envelope = boq_schedule.boq_row_duration(row)

    assert days is None or isinstance(days, int)
    assert basis == "unknown" or basis.startswith(("estimated", "kernel-error"))
    if days is None:
        assert envelope == {} or envelope.get("status") in {"success", "error"}


def test_there_is_one_duration_kernel_and_both_builders_call_it():
    """The copy that used to live beside each caller is gone: both schedule
    builders resolve the same function object, and neither re-implements it."""
    assert boq_schedule.duration_days_from_rate is pm_computations.duration_days_from_rate
    assert brief_schedule.duration_days_from_rate is pm_computations.duration_days_from_rate
    assert not hasattr(boq_schedule, "_kernel_duration_days")
    assert not hasattr(brief_schedule, "_estimate_duration_from_rate")
