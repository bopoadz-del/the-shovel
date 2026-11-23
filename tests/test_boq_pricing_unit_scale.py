"""A pooled rate is restated per the priced line's unit before the median.

Units of one class are not one unit: a rate per tonne is 1,000 times a rate per
kilogram. The fallback tiers of ``_lookup`` (category median, asset median)
restate each pooled rate per the line's unit when both units have a known size,
and leave out a rate whose unit has no known size when the line's unit has one
(or the reverse). Classes with no sized units keep the earlier same-class
pooling. Every pool below is built here from invented numbers; the expected
rates are worked by hand beside each row.
"""
from __future__ import annotations

import pytest

from app.lib.boq_pricing import _lookup


def _category_only(pool):
    """A lookup where only the category tier holds rates (no exact row)."""
    return {}, {"Cat": list(pool)}, []


def _asset_only(pool):
    """A lookup where only the asset tier holds rates."""
    return {}, {}, list(pool)


@pytest.mark.parametrize("pool, line_unit, want", [
    ([(3000.0, 2, "t")], "kg", 3.0),      # 3,000 per t / 1,000 = 3 per kg
    ([(8000.0, 1, "mt")], "kg", 8.0),     # 8,000 per mt / 1,000 = 8 per kg
])
def test_a_per_tonne_rate_prices_a_kg_line_per_kilogram(pool, line_unit, want):
    rate, _, source = _lookup("Cat", line_unit, *_category_only(pool))
    assert rate == pytest.approx(want)
    assert source == "fallback (category median)"


@pytest.mark.parametrize("pool, line_unit, want", [
    ([(2.5, 3, "kg")], "t", 2500.0),       # 2.5 per kg x 1,000 = 2,500 per t
    ([(4.0, 1, "kg")], "tonne", 4000.0),   # 4 per kg x 1,000 = 4,000 per tonne
])
def test_a_per_kg_rate_prices_a_tonne_line_per_tonne(pool, line_unit, want):
    rate, _, source = _lookup("Cat", line_unit, *_asset_only(pool))
    assert rate == pytest.approx(want)
    assert source == "weak (asset median)"


@pytest.mark.parametrize("pool, line_unit, want", [
    # per kg: 3,000/t -> 3; 2 and 4 stay; median of 2, 3, 4 = 3
    ([(3000.0, 1, "t"), (2.0, 1, "kg"), (4.0, 1, "kg")], "kg", 3.0),
    # per t: 5 per kg -> 5,000; 1,000 and 9,000 stay; median = 5,000
    ([(5.0, 1, "kg"), (1000.0, 1, "t"), (9000.0, 1, "mt")], "t", 5000.0),
])
def test_the_median_is_taken_after_every_rate_is_restated(pool, line_unit, want):
    rate, _, _ = _lookup("Cat", line_unit, *_category_only(pool))
    assert rate == pytest.approx(want)


@pytest.mark.parametrize("pool, line_unit", [
    ([(2500.0, 2, "ton")], "kg"),    # 'ton' has no known size: never converted
    ([(3000.0, 2, "t")], "ton"),     # nor converted into
])
def test_a_unit_of_unknown_size_is_never_converted(pool, line_unit):
    assert _lookup("Cat", line_unit, *_category_only(pool)) == (None, 0, "NO RATE")


@pytest.mark.parametrize("pool, line_unit, want", [
    ([(2500.0, 2, "ton")], "ton", 2500.0),   # same unknown-size unit: as is
    ([(700.0, 1, "kg")], "kg", 700.0),        # same sized unit: as is
])
def test_a_rate_in_the_line_unit_is_kept_as_is(pool, line_unit, want):
    rate, _, _ = _lookup("Cat", line_unit, *_category_only(pool))
    assert rate == pytest.approx(want)


@pytest.mark.parametrize("pool, line_unit, want", [
    ([(120.0, 1, "no")], "set", 120.0),   # count class has no sizes: pooled as before
    ([(45.0, 2, "pair")], "no", 45.0),
])
def test_a_class_without_sizes_keeps_same_class_pooling(pool, line_unit, want):
    rate, _, _ = _lookup("Cat", line_unit, *_category_only(pool))
    assert rate == pytest.approx(want)


@pytest.mark.parametrize("pool, line_unit", [
    ([(85.0, 1, "m2")], "kg"),   # an area rate never prices a mass line
    ([(3000.0, 1, "t")], "m3"),  # a mass rate never prices a volume line
])
def test_a_rate_of_another_class_is_left_out(pool, line_unit):
    assert _lookup("Cat", line_unit, *_category_only(pool)) == (None, 0, "NO RATE")
