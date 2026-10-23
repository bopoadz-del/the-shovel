"""MEP lines price from CITED published references, never masquerading.

Operator directive: kill the MEP rate gap. No observed per-unit MEP rates
exist in the archives (measured -- the contract bill's MEP sections carry
quantities without surviving rates), so the honest fill is the published
tables the curated KB already carries, transcribed to the rate card with a
CELL-LEVEL source field. The pricing path surfaces that provenance all the
way into the workbook's Rate Basis column as "published reference (...)" --
distinct from observed "exact (cat+unit)" cells, so a reviewer can never
mistake a handbook figure for a project-observed rate.

The probe that built this also caught a classifier gap: "fire detection and
alarm" missed Fire Protection entirely (the pattern knew extinguishers and
alarms but not detection) and fell to an uncategorised 120,000 lump cell.
"""
from __future__ import annotations

import pytest

from app.lib import boq_pricing as bp


def price_one(desc, unit, qty=100):
    priced, _ = bp.price_line_items(
        [{"description": desc, "unit": unit, "quantity": qty}],
        "Buildings/Towers", "SAR")
    return priced[0]


def test_hvac_prices_from_the_published_tt_cell():
    it = price_one("Supply and install HVAC ductwork complete", "m2")
    assert it["work_category"] == "Mechanical/HVAC (MEP)"
    assert it["rate"] == pytest.approx(1200.0)
    assert it["rate_source"].startswith("published reference (Turner & Townsend")


def test_electrical_containment_prices_per_metre_from_aecom():
    it = price_one("Electrical containment and cable trays", "m")
    assert it["work_category"] == "Electrical (MEP)"
    assert it["rate"] == pytest.approx(85.0)
    assert it["rate_source"].startswith("published reference (AECOM")


def test_fire_detection_classifies_and_prices_published():
    """RED before the fix: category Other/Uncategorized, rate 120,000 from a
    lump cell -- a detection POINT priced as a whole lump sum."""
    it = price_one("Fire detection and alarm point", "no", qty=60)
    assert it["work_category"] == "Fire Protection"
    assert it["rate"] == pytest.approx(450.0)
    assert "published reference" in it["rate_source"]


def test_sanitary_plumbing_prices_published_per_built_area():
    it = price_one("Sanitary plumbing installation to apartments", "m2", qty=400)
    assert it["rate"] == pytest.approx(650.0)
    assert "published reference" in it["rate_source"]
    assert "per m2 built area" in it["rate_source"]


def test_observed_cells_keep_their_label():
    """Provenance separation both ways: an observed cell must NOT read as a
    published reference."""
    it = price_one("Reinforced concrete C40 to columns", "m3")
    assert it["rate_source"] == "exact (cat+unit)"


def test_published_cells_declare_zero_observations():
    rows = bp._CARD["cards"]["Buildings/Towers"]["SAR"]
    for r in rows:
        if r.get("source"):
            assert r["n"] == 0, (
                f"published cell {r['cat']}|{r['unit']} claims n={r['n']} "
                "observations -- provenance lie")
