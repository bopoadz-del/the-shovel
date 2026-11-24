"""The design hat's uplift calculator takes its default factor of safety from
the knowledge entry it declares, not from a literal in its signature.

Expected floor counts are worked by hand from the factor the entry states;
the inputs are invented.
"""

from __future__ import annotations

import math

import pytest

from app.agents.hats.design.formulas import construction_formulas as design
from app.agents.hats.design.formulas.construction_formulas import dewatering_uplift_check
from app.blocks import _knowledge


def _stated_factor():
    declared = design._UPLIFT_K.declared()["required_fos"]
    entry_id = declared.source.split()[2]  # "knowledge entry <id> (...)"
    return float(_knowledge.shipped_entry(entry_id)["thresholds"]["min_fos"])


@pytest.mark.parametrize(
    "head, raft, floors_built, floor_t",
    [
        (23.0, 4.0, 0, 0.6),  # invented
        (12.0, 1.5, 3, 0.25),  # invented
    ],
)
def test_unstated_factor_is_the_entrys(head, raft, floors_built, floor_t):
    fos = _stated_factor()
    r = dewatering_uplift_check(
        water_depth=head, raft_thickness=raft, floor_count=floors_built,
        floor_thickness=floor_t, concrete_unit_weight=2.5, water_unit_weight=1.0,
    )
    # hand: floors = ceil((fos x head - raft x 2.5) / (floor_t x 2.5))
    want = max(0, math.ceil((fos * head - raft * 2.5) / (floor_t * 2.5)))
    assert r.min_floors_for_stop == want
    assert r.can_stop is (r.fos >= fos)
    assert f"(required: {fos})" in r.notes[2]


@pytest.mark.parametrize(
    "stated, floors",
    [
        (2.0, 24),  # hand: head 10, raft 1 m x 2.5 = 2.5, floor 0.3 x 2.5 = 0.75; ceil((20 - 2.5) / 0.75) = ceil(23.33) = 24
        (1.0, 10),  # hand: ceil((10 - 2.5) / 0.75) = 10
    ],
)
def test_a_stated_factor_still_wins(stated, floors):
    r = dewatering_uplift_check(
        water_depth=10.0, raft_thickness=1.0, floor_count=0, floor_thickness=0.3,
        required_fos=stated,
    )
    assert r.min_floors_for_stop == floors


@pytest.mark.parametrize(
    "kwargs",
    [
        {"water_depth": -1.0, "raft_thickness": 1.0, "floor_count": 2},
        {"water_depth": -0.5, "raft_thickness": 2.0, "floor_count": 0},
    ],
)
def test_negative_head_is_refused(kwargs):
    with pytest.raises(ValueError, match="water_depth"):
        dewatering_uplift_check(**kwargs)
