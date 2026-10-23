"""Man-hours are already one person's hours. A crew size does not divide them again.

Hand calculation, generic inputs: 100 units of output in 320 man-hours is
100/320 = 0.3125 units per man-hour. A crew of 4 does not make that
0.3125/4 = 0.078125. The same rule on a second input: 48 units in 16
man-hours is 3, whether the crew is 8 or 1.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation


def _per_worker(output: float, man_hours: float, crew: int) -> float:
    env = run_calculation(
        "productivity_rate",
        {"output_quantity": output, "labor_hours": man_hours, "crew_size": crew},
    )
    assert env["status"] == "success", env
    return env["result"]["rate_per_worker_hour"]


def test_man_hours_are_not_divided_by_crew_a_second_time():
    # 100 / 320 = 0.3125. Dividing by the crew of 4 again is 0.078125.
    output, hours, crew = 100.0, 320.0, 4
    got = _per_worker(output, hours, crew)
    assert got == pytest.approx(output / hours, abs=1e-4)
    assert got != pytest.approx(output / hours / crew, abs=1e-3)


def test_a_second_output_follows_the_same_man_hour_rule():
    # 48 / 16 = 3. A crew of 8 must not turn that into 0.375.
    output, hours, crew = 48.0, 16.0, 8
    got = _per_worker(output, hours, crew)
    assert got == pytest.approx(3.0, abs=1e-4)
    assert got == pytest.approx(_per_worker(output, hours, 1), abs=1e-4)
