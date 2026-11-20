"""Course program, file C, asphalt formula batch.

Theoretical and budget bitumen, material over-consumption and stock from the
monthly asphalt material sheet, haulage ton-km from the monthly operations
summary, Marshall stiffness and paver hours. Every entry is a ``type=formula``
knowledge entry evaluated by ``app.blocks._knowledge.evaluate``. Values come
from the owner-supplied course material (roads and earthworks section 3,
blocks T17, T22, T25, T26 and W9, all CONFIRMED in its verify file) or are
invented; each case states its hand calculation.
"""
from __future__ import annotations

import pint
import pytest
import sympy

from app.blocks import _knowledge as kb

NEW_IDS = [
    "asphalt.theoretical_bitumen",
    "asphalt.bitumen_budget",
    "asphalt.material_overconsumption",
    "asphalt.material_stock",
    "asphalt.haul_ton_km",
    "asphalt.marshall_stiffness",
    "asphalt.paving_hours",
]

_UREG = pint.UnitRegistry()


def ev(rule_id, **values):
    return kb.evaluate(rule_id, **values)


@pytest.mark.parametrize("rule_id", NEW_IDS)
def test_entry_carries_course_source_and_declared_units(rule_id):
    e = kb.get_rule(rule_id)
    assert e is not None and e["type"] == "formula"
    assert e["domain"] == "construction.roads"
    assert e["provenance"]["course_reference"].startswith("owner-supplied course material, ")
    assert e["provenance"]["project"] is None
    assert e["applicability"]["project_specific"] is False
    assert "not a code or standard requirement" in e["statement"]
    symbols = {str(s) for s in sympy.sympify(e["expression"]).free_symbols}
    assert symbols == set(e["variables"]), rule_id
    for v in e["variables"].values():
        _UREG(v["unit"])  # every declared unit parses (catches typos)
    _UREG(e["thresholds"]["unit"])


@pytest.mark.parametrize(
    "rule_id, partial",
    [
        ("asphalt.theoretical_bitumen", {"tonnage_laid": 1000}),
        ("asphalt.bitumen_budget", {"budget_allowance": 4.5}),
        ("asphalt.material_overconsumption", {"actual_consumption": 40}),
        ("asphalt.material_stock", {"cumulative_received": 500}),
        ("asphalt.haul_ton_km", {"quantity_laid": 2000}),
        ("asphalt.marshall_stiffness", {"marshall_stability": 1633}),
        ("asphalt.paving_hours", {"paver_rate": 320}),
    ],
)
def test_missing_required_input_raises_not_defaulted(rule_id, partial):
    with pytest.raises(ValueError):
        ev(rule_id, **partial)


@pytest.mark.parametrize(
    "tonnage, pct, bitumen",
    [
        (1000, 4.45, 44.5),    # invented tonnage at the course's wearing-course 4.45 %: 1000 x 4.45 / 100 = 44.5 t
        (2500, 3.6, 90.0),     # invented tonnage at the course's base-course 3.6 % (p35): 2500 x 3.6 / 100 = 90 t
        (1200, 4.15, 49.8),    # invented tonnage at the course's binder-course 4.15 %: 1200 x 4.15 / 100 = 49.8 t
    ],
)
def test_theoretical_bitumen(tonnage, pct, bitumen):
    out = ev("asphalt.theoretical_bitumen", tonnage_laid=tonnage, bitumen_content=pct)
    assert out["result"] == pytest.approx(bitumen)
    assert out["unit"] == "tonne"


@pytest.mark.parametrize(
    "tonnage, allowance, budget",
    [
        (1000, 4.5, 45.0),    # course 60/70 allowance 4.5 %: 1000 x 4.5 / 100 = 45 t
        (2000, 4.8, 96.0),    # course 40/50 allowance 4.8 % (p35 typed): 2000 x 4.8 / 100 = 96 t
    ],
)
def test_bitumen_budget(tonnage, allowance, budget):
    out = ev("asphalt.bitumen_budget", tonnage_laid=tonnage, budget_allowance=allowance)
    assert out["result"] == pytest.approx(budget)


@pytest.mark.parametrize(
    "actual, theoretical, over",
    [
        (47.0, 44.5, 2.5),    # invented: 47 - 44.5 = 2.5 t used beyond theory
        (88.0, 90.0, -2.0),   # invented: 88 - 90 = -2 t, actual below theory (to be checked)
    ],
)
def test_material_overconsumption_sign_is_kept(actual, theoretical, over):
    out = ev(
        "asphalt.material_overconsumption",
        actual_consumption=actual,
        theoretical_consumption=theoretical,
    )
    assert out["result"] == pytest.approx(over)


@pytest.mark.parametrize(
    "received, consumed, stock",
    [
        (500, 430, 70),       # invented: 500 - 430 = 70 t in stock
        (1200, 1185, 15),     # invented: 1200 - 1185 = 15 t in stock
    ],
)
def test_material_stock(received, consumed, stock):
    out = ev(
        "asphalt.material_stock",
        cumulative_received=received,
        cumulative_consumption=consumed,
    )
    assert out["result"] == pytest.approx(stock)


@pytest.mark.parametrize(
    "tonnes, ahd, tkm",
    [
        (2000, 12.5, 25000),  # invented: 2000 t x 12.5 km = 25,000 t-km
        (850, 8, 6800),       # invented: 850 t x 8 km = 6,800 t-km
    ],
)
def test_haul_ton_km(tonnes, ahd, tkm):
    out = ev("asphalt.haul_ton_km", quantity_laid=tonnes, average_haul_distance=ahd)
    assert out["result"] == pytest.approx(tkm)
    assert _UREG(out["unit"]).dimensionality == _UREG("tonne*kilometer").dimensionality


@pytest.mark.parametrize(
    "stability, flow, stiffness, meets_500",
    [
        # course W9 (p13): 1633 / 2.6 = 628.08 kg/mm, above the 500 minimum of T17 (p12)
        (1633, 2.6, 1633 / 2.6, True),
        # invented: 1000 / 4.0 = 250 kg/mm, stability and flow at their limits but stiffness below 500
        (1000, 4.0, 250.0, False),
    ],
)
def test_marshall_stiffness(stability, flow, stiffness, meets_500):
    out = ev("asphalt.marshall_stiffness", marshall_stability=stability, marshall_flow=flow)
    assert out["result"] == pytest.approx(stiffness)
    assert (out["result"] >= 500) is meets_500


@pytest.mark.parametrize(
    "tonnes, rate, hours",
    [
        (1920, 240, 8.0),     # invented tonnage at the course's wearing-course norm 240 t/hr (p23): 1920 / 240 = 8 hr
        (4000, 320, 12.5),    # invented tonnage at the course's base-course norm 320 t/hr (p23): 4000 / 320 = 12.5 hr
    ],
)
def test_paving_hours(tonnes, rate, hours):
    out = ev("asphalt.paving_hours", tonnage_to_lay=tonnes, paver_rate=rate)
    assert out["result"] == pytest.approx(hours)
    assert out["unit"] == "hour"
