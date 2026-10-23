"""As-built deviation: a single measured/design pair, numbers or labelled
measurements, compared in millimetres against a tolerance in millimetres.
Synthetic figures; construction kit only."""
from __future__ import annotations

import asyncio

import pytest

from tests.conftest import construction_kit_markers

pytestmark = construction_kit_markers


def _run(params):
    from app.dependencies import get_block_instance

    con = get_block_instance("construction")
    return asyncio.run(con.as_built_deviation_report(params, params))


def test_a_single_pair_within_tolerance_is_approved():
    out = _run({"measurements": 2410, "design_measurements": 2400, "tolerance_mm": 10})
    assert out["status"] == "success", out
    assert out["deviation_summary"]["total_deviations"] == 0
    assert out["sign_off_status"] == "APPROVED"


def test_a_pair_outside_tolerance_is_a_deviation_in_mm():
    out = _run({"measurements": [{"type": "wall", "value": 2.43, "unit": "m"}],
                "design_measurements": [{"type": "wall", "value": 2400, "unit": "mm"}],
                "tolerance_mm": 10})
    assert out["status"] == "success", out
    [dev] = out["deviations"]
    assert dev["deviation_mm"] == pytest.approx(30)
    assert dev["severity"] == "major"   # more than twice the tolerance


def test_numbers_pair_by_position():
    out = _run({"as_built_measurements": [300, 512], "design_measurements": [300, 500], "tolerance_mm": 5})
    assert out["status"] == "success", out
    assert [d["deviation_mm"] for d in out["deviations"]] == [pytest.approx(12)]
