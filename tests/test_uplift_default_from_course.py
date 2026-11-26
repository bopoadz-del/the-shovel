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
    return float(_knowledge.shipped_entry(declared.entry_id)["thresholds"]["min_fos"])


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


def _entry_id():
    return design._UPLIFT_K.declared()["required_fos"].entry_id


@pytest.mark.parametrize("head, raft", [(23.0, 4.0), (12.0, 1.5)])  # invented
def test_the_public_door_records_the_entrys_factor(head, raft):
    # Through the calculator door with no factor given, the run's constants
    # record names the factor applied and the knowledge entry that states it;
    # it is not disclosed as a signature default.
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(
        "dewatering_uplift_check", {"water_depth": head, "raft_thickness": raft, "floor_count": 0}
    )
    assert env["status"] == "success", env
    rows = env["result"]["constants_used"]
    assert [r["name"] for r in rows] == ["required_fos"]
    assert rows[0]["value"] == _stated_factor()
    assert rows[0]["source"].startswith("knowledge entry " + _entry_id())
    assert all(d["parameter"] != "required_fos" for d in env["result"].get("defaults_used", []))


@pytest.mark.parametrize("given", [1.5, 2.0])
def test_a_given_factor_is_not_recorded_as_a_constant(given):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(
        "dewatering_uplift_check",
        {"water_depth": 10.0, "raft_thickness": 1.0, "floor_count": 0, "required_fos": given},
    )
    assert env["status"] == "success", env
    assert "constants_used" not in env["result"]
    assert f"(required: {given})" in env["result"]["notes"][2]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_entry_breaks_the_run_and_says_so(tmp_path, monkeypatch, bad):
    # The shipped entry turned non-finite: the door returns a raised error
    # envelope, never a figure computed on NaN or infinity.
    import json

    from app.lib.construction_formulas import run_calculation

    with open(_knowledge._KB_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for e in data["entries"]:
        if e.get("id") == _entry_id():
            e["thresholds"]["min_fos"] = bad
    kb = tmp_path / "kb.json"
    kb.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(_knowledge, "_KB_PATH", str(kb))
    env = run_calculation(
        "dewatering_uplift_check", {"water_depth": 10.0, "raft_thickness": 1.0, "floor_count": 0}
    )
    assert env["status"] == "error" and env.get("raised") is True, env
    assert "not a finite number" in env["error"]


@pytest.mark.parametrize(
    "head, raft, floors_built, floor_t",
    [
        (10.0, 1.8, 10, 0.3),  # hand: uplift 10; counter 1.8 x 2.5 + 10 x 0.3 x 2.5 = 4.5 + 7.5 = 12.0; FOS 1.2
        (20.0, 4.0, 7, 0.8),  # hand: uplift 20; counter 4 x 2.5 + 7 x 0.8 x 2.5 = 10 + 14 = 24.0; FOS 1.2
    ],
)
def test_fos_exactly_at_the_stated_factor_can_stop(head, raft, floors_built, floor_t):
    # The line itself passes: FOS equal to the required factor is enough.
    assert _stated_factor() == 1.2  # the hand values above assume the entry's 1.2
    r = dewatering_uplift_check(
        water_depth=head, raft_thickness=raft, floor_count=floors_built, floor_thickness=floor_t,
    )
    assert r.fos == 1.2
    assert r.can_stop is True and r.needs_tension_piles is False
    assert r.min_floors_for_stop == floors_built
