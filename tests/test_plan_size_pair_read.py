"""A plan size written as two figures, with the third dimension written apart.

"24 m by 12 m, 250 mm thick" is a length, a width and a thickness. The size
reader took only a three-figure chain, so the calculator asked for its length
although the ask wrote it. The pair is read only when the ask writes exactly
one plan size and exactly one thickness / depth / height; otherwise nothing is
read and the calculator still asks.

Every ask here is invented for the test; expected values are worked by hand.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation
from app.lib.construction_formulas_quantities import (
    plan_size_pairs,
    third_dimension_metres,
)


@pytest.mark.parametrize(
    "ask, expected",
    [
        ("Ground slab 18 m by 9 m, 200 mm thick", (18.0, 9.0)),
        ("Pad 1,500 x 800 mm", (1.5, 0.8)),
    ],
)
def test_one_plan_size_is_read_in_metres(ask, expected):
    assert plan_size_pairs(ask) == [expected]


@pytest.mark.parametrize(
    "ask",
    [
        # The pair is the head of a three-figure chain, not a plan size.
        "Raft 18 x 9 x 0.2 m",
        # Two bare figures with no length unit are not a size.
        "3 x 4 gangs",
    ],
)
def test_no_plan_size_outside_a_two_figure_length_pair(ask):
    assert plan_size_pairs(ask) == []


@pytest.mark.parametrize(
    "ask, expected",
    [
        ("slab 18 m by 9 m, 200 mm thick", [0.2]),
        ("trench 35 m by 0.7 m, depth 1.4 m", [1.4]),
    ],
)
def test_third_dimension_read_by_its_word(ask, expected):
    assert third_dimension_metres(ask) == expected


@pytest.mark.parametrize(
    "name, ask, key, expected",
    [
        # 18 x 9 x 0.2 = 32.4 m3 net.
        ("concrete_volume", "Ground slab 18 m by 9 m, 200 mm thick. Volume?",
         "net_volume_m3", 32.4),
        # 35 x 0.7 x 1.4 = 34.3 m3 bank.
        ("excavation_volume", "Trench 35 m by 0.7 m, 1.4 m deep. Bank volume?",
         "bank_volume_m3", 34.3),
    ],
)
def test_pair_and_third_dimension_complete_the_calculator(name, ask, key, expected):
    out = run_calculation(name, {"text": ask})
    assert out["status"] == "success", out
    assert out["result"][key] == pytest.approx(expected)


@pytest.mark.parametrize(
    "ask",
    [
        # No thickness written: the volume is not run on a default depth.
        "Ground slab 18 m by 9 m. Volume?",
        # Two plan sizes: neither is the slab.
        "Bay 7 m by 5 m and bay 6 m by 4 m, 200 mm thick. Volume?",
        # Two thicknesses: the slab's is not known.
        "Ground slab 18 m by 9 m, 200 mm thick, topping 50 mm thick. Volume?",
    ],
)
def test_no_single_reading_asks_for_the_sizes(ask):
    out = run_calculation("concrete_volume", {"text": ask})
    assert out["status"] == "needs_input", out
    assert "length_m" in (out.get("missing") or [])


@pytest.mark.parametrize(
    "given, expected",
    [
        # All three sizes already given (the third as thickness_m): the ask's
        # text is not read, so no depth_m is added beside the typed thickness.
        ({"length_m": 5.0, "width_m": 4.0, "thickness_m": 0.3},
         {"length_m": 5.0, "width_m": 4.0, "thickness_m": 0.3}),
        # One size missing: the text is read, and setdefault keeps the typed
        # length and width while the written 200 mm fills depth_m.
        ({"length_m": 5.0, "width_m": 4.0},
         {"length_m": 5.0, "width_m": 4.0, "depth_m": 0.2}),
    ],
)
def test_text_is_read_only_while_a_size_is_missing(given, expected):
    from app.lib.construction_formulas import _fill_lwd_from_text

    out = dict(given)
    _fill_lwd_from_text("Ground slab 18 m by 9 m, 200 mm thick", out)
    assert out == expected
