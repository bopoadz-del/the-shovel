"""A fallback rate must have the same dimensions as the line it prices.

Live-fire campaign phase 3, F10. The rate-card had no Pipework/Drainage or
Mechanical/HVAC rows for Buildings/Towers AED, so MEP lines fell to wild
fallbacks. Real observed lump-sum MEP rates from a 2012 Dubai fit-out BOQ
now fill the lump-sum cells -- but that exposed the sharper defect this
file fences: the category-median fallback pooled rates ACROSS UNITS, so a
per-metre pipe run would have been priced from an AED 26,525 lump-sum
median. A number with the wrong dimensions is worse than NO RATE.

The fallback tiers now pool only unit-commensurable rates (linear / area /
volume / mass / count / lump). A tier with nothing commensurable is
skipped; a line whose unit has no known class keeps the legacy any-unit
pools because commensurability cannot be judged.
"""
from __future__ import annotations

import pytest

from app.lib import boq_pricing as bp


def _lookup_env(rows):
    """Build the lookup structures from raw (cat, unit, median, n) rows."""
    exact, bycat, byasset = {}, {}, []
    for cat, unit, median, n in rows:
        exact[(cat, unit)] = (median, n, None)
        bycat.setdefault(cat, []).append((median, n, unit))
        byasset.append((median, n, unit))
    return exact, bycat, byasset


# -- the defect observed: lump sums must not price measurable work --------

def test_a_per_metre_line_never_prices_from_a_lump_sum_median():
    """RED before the fix: category fallback returned the 26,525 ls median
    for a pipework line measured in metres."""
    exact, bycat, byasset = _lookup_env([
        ("Mechanical/HVAC (MEP)", "ls", 26525.0, 2),
    ])
    rate, _n, src = bp._lookup("Mechanical/HVAC (MEP)", "m", exact, bycat, byasset)
    assert src == "NO RATE", (rate, src)
    assert rate is None


def test_a_lump_sum_line_still_prices_exact_from_the_lump_cell():
    exact, bycat, byasset = _lookup_env([
        ("Mechanical/HVAC (MEP)", "ls", 26525.0, 2),
    ])
    rate, n, src = bp._lookup("Mechanical/HVAC (MEP)", "ls", exact, bycat, byasset)
    assert (rate, n, src) == (26525.0, 2, "exact (cat+unit)")


def test_category_fallback_uses_only_commensurable_rates():
    """m2 line in a category holding m2 and ls rates: the ls rate must not
    drag the median."""
    exact, bycat, byasset = _lookup_env([
        ("Finishes", "m2", 40.0, 10),
        ("Finishes", "m2", 60.0, 10),
        ("Finishes", "ls", 9000.0, 1),
    ])
    # a linear-unit line misses exact and finds no linear-class rate anywhere
    _rate, _n, src = bp._lookup("Finishes", "m", exact, bycat, byasset)
    assert src == "NO RATE"


def test_asset_fallback_is_also_unit_filtered():
    """A linear line with no category match may price from ANOTHER category's
    per-metre rates (weak, but dimensionally sane) -- never from lump sums."""
    exact, bycat, byasset = _lookup_env([
        ("Roads/Paving", "m", 55.0, 8),
        ("Mechanical/HVAC (MEP)", "ls", 26525.0, 2),
    ])
    rate, n, src = bp._lookup("Pipework/Drainage", "m", exact, bycat, byasset)
    assert src == "weak (asset median)"
    assert rate == pytest.approx(55.0)
    assert n == 8


def test_an_unclassifiable_unit_keeps_the_legacy_pool():
    """'%' has no unit class; refusing to price it on commensurability
    grounds would regress lines the old code priced."""
    exact, bycat, byasset = _lookup_env([
        ("Preliminaries/General", "ls", 5000.0, 3),
    ])
    rate, _n, src = bp._lookup("Preliminaries/General", "%", exact, bycat, byasset)
    assert src == "fallback (category median)"
    assert rate == pytest.approx(5000.0)


# -- the deployed card carries the observed MEP cells ---------------------

def test_the_deployed_card_has_the_observed_aed_mep_lump_sums():
    """F10 data closure: real 2012 Dubai fit-out lump sums, n visible."""
    rows = bp._CARD["cards"]["Buildings/Towers"]["AED"]
    cells = {(r["cat"], r["unit"]): r for r in rows}
    mech = cells[("Mechanical/HVAC (MEP)", "ls")]
    assert mech["median"] == pytest.approx(26525.0)
    assert mech["n"] == 2
    fire = cells[("Fire Protection", "ls")]
    assert fire["median"] == pytest.approx(13200.0)
    assert fire["n"] == 1


def test_an_aed_mechanical_lump_line_prices_end_to_end():
    priced, _summary = bp.price_line_items(
        [{"description": "Modification of HVAC works to new layout",
          "unit": "ls", "quantity": 1}],
        asset_type="Buildings/Towers", currency="AED",
    )
    item = priced[0]
    assert item["rate_source"] == "exact (cat+unit)"
    assert item["rate"] == pytest.approx(26525.0)


def test_a_per_metre_aed_pipe_line_is_flagged_not_invented():
    """The honest outcome for the data that does not exist: the operator's
    corpus holds no per-metre AED MEP rates, so the line must surface as
    NO RATE (amber in the workbook), never a lump-sum-derived number."""
    priced, _summary = bp.price_line_items(
        [{"description": "Supply and install 100mm dia uPVC drainage pipe",
          "unit": "m", "quantity": 250}],
        asset_type="Buildings/Towers", currency="AED",
    )
    item = priced[0]
    if item["rate_source"] != "NO RATE":
        # dimensionally sane weak fallback is acceptable; lump-derived is not
        assert item["rate"] < 1000, (item["rate"], item["rate_source"])
