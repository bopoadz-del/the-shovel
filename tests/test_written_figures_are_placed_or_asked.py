"""A figure the ask writes with a unit reaches the calculator in that unit.
Synthetic calculator."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import Param, registered


def _area(floor_m2: float) -> dict:
    return {"value": floor_m2}


def test_a_bare_figure_the_ask_wrote_in_sq_ft_is_converted():
    """The model passes the number; the ask wrote it with "sq ft"."""
    from app.lib.construction_formulas import run_calculation

    with registered("lane5_sqft_area", _area, "Qx floor check",
                    {"floor_m2": Param("m2", lo=0, hi=1e9, label="floor area")}, {"value": "m2"}):
        env = run_calculation("lane5_sqft_area", {"text": "qx floor check for 21,528 sq ft",
                                                  "floor_m2": 21528})
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(2000.0166, rel=1e-4)
