"""Course program, file C, uplift and grading formula batch.

Uplift pressure and floor counterweight from the course's dewatering uplift
check, and grader volume rate and grading area rate from its two earthmoving
fleet studies. Every entry is a ``type=formula`` knowledge entry evaluated by
``app.blocks._knowledge.evaluate``. Values come from the owner-supplied course
material (roads and earthworks section 1 block W1, section 2 blocks W7 and W8,
all CONFIRMED in its verify file) or are invented; each case states its hand
calculation.
"""
from __future__ import annotations

import pint
import pytest
import sympy

from app.blocks import _knowledge as kb

NEW_IDS = {
    "foundation.uplift_pressure": "construction.buildings",
    "foundation.floor_counterweight": "construction.buildings",
    "earthworks.grader_volume_rate": "construction.roads",
    "earthworks.grading_area_rate": "construction.roads",
}

_UREG = pint.UnitRegistry()


def ev(rule_id, **values):
    return kb.evaluate(rule_id, **values)


@pytest.mark.parametrize("rule_id, domain", sorted(NEW_IDS.items()))
def test_entry_carries_course_source_and_declared_units(rule_id, domain):
    e = kb.get_rule(rule_id)
    assert e is not None and e["type"] == "formula"
    assert e["domain"] == domain
    assert e["provenance"]["course_reference"].startswith("owner-supplied course material, ")
    assert e["provenance"]["project"] is None
    assert e["applicability"]["project_specific"] is False
    assert "not a code or standard requirement" in e["statement"]
    symbols = {str(s) for s in sympy.sympify(e["expression"]).free_symbols}
    assert symbols == set(e["variables"]), rule_id
    for v in e["variables"].values():
        _UREG(v["unit"])  # every declared unit parses (catches typos)
    _UREG(e["thresholds"]["unit"])


@pytest.mark.parametrize("rule_id", sorted(NEW_IDS))
def test_declared_result_unit_matches_the_variables(rule_id):
    """The result unit is what the expression makes of the variable units."""
    e = kb.get_rule(rule_id)
    names = sorted(e["variables"])
    # Evaluate the expression on pint quantities of magnitude 1, one per variable.
    func = sympy.lambdify([sympy.Symbol(n) for n in names], sympy.sympify(e["expression"]))
    result = func(*[_UREG.Quantity(1, e["variables"][n]["unit"]) for n in names])
    declared = _UREG(e["thresholds"]["unit"])
    assert result.dimensionality == declared.dimensionality, rule_id


@pytest.mark.parametrize(
    "rule_id, partial",
    [
        ("foundation.uplift_pressure", {"water_head": 23}),
        ("foundation.floor_counterweight", {"concrete_unit_weight": 2.5, "raft_thickness": 0.9}),
        ("earthworks.grader_volume_rate", {"grader_area_rate": 750}),
        ("earthworks.grading_area_rate", {"layer_count": 19, "fill_area": 2_000_000}),
    ],
)
def test_missing_required_input_raises_not_defaulted(rule_id, partial):
    with pytest.raises(ValueError):
        ev(rule_id, **partial)


@pytest.mark.parametrize(
    "gamma_w, head, uplift",
    [
        (1.0, 23.0, 23.0),     # course W1 (p10): 1 x 23 = 23 t/m2
        (1.025, 12.0, 12.3),   # invented saline groundwater: 1.025 x 12 = 12.3 t/m2
    ],
)
def test_uplift_pressure(gamma_w, head, uplift):
    out = ev("foundation.uplift_pressure", water_unit_weight=gamma_w, water_head=head)
    assert out["result"] == pytest.approx(uplift)
    assert out["unit"] == "tonne/meter**2"


@pytest.mark.parametrize(
    "gamma_c, raft, slabs, slab_t, counterweight",
    [
        (2.5, 0.9, 5, 0.3, 6.0),      # course W1 (p10): 0.9 x 2.5 + 5 x 0.3 x 2.5 = 2.25 + 3.75 = 6 t/m2
        (2.4, 1.5, 10, 0.25, 9.6),    # invented: 2.4 x (1.5 + 10 x 0.25) = 2.4 x 4.0 = 9.6 t/m2
        (2.5, 2.0, 0, 0.3, 5.0),      # invented, raft alone before any slab: 2.5 x 2.0 = 5 t/m2
    ],
)
def test_floor_counterweight(gamma_c, raft, slabs, slab_t, counterweight):
    out = ev(
        "foundation.floor_counterweight",
        concrete_unit_weight=gamma_c,
        raft_thickness=raft,
        slab_count=slabs,
        slab_thickness=slab_t,
    )
    assert out["result"] == pytest.approx(counterweight)
    assert out["unit"] == "tonne/meter**2"


@pytest.mark.parametrize(
    "raft, slabs, head, passes_course_1_2",
    [
        # course W1 (p10): 6 / 23 = 0.26 < 1.2 -> tension piles
        (0.9, 5, 23.0, False),
        # invented: 2.5 x (1.2 + 20 x 0.3) = 18 t/m2 against 1 x 10 = 10 t/m2 -> 1.8 >= 1.2
        (1.2, 20, 10.0, True),
    ],
)
def test_terms_chain_into_the_existing_uplift_factor(raft, slabs, head, passes_course_1_2):
    cw = ev(
        "foundation.floor_counterweight",
        concrete_unit_weight=2.5,
        raft_thickness=raft,
        slab_count=slabs,
        slab_thickness=0.3,
    )["result"]
    up = ev("foundation.uplift_pressure", water_unit_weight=1.0, water_head=head)["result"]
    fos = ev("foundation.uplift_fos", counterweight=cw, uplift=up)["result"]
    assert fos == pytest.approx(cw / up)
    assert (fos >= 1.2) is passes_course_1_2


@pytest.mark.parametrize(
    "area_rate, layer, volume_rate",
    [
        (750, 0.25, 187.5),   # course W7 (p27): 750 m2/hr at 25 cm = 187.5 (printed 187) m3/hr
        (700, 0.25, 175.0),   # course W8 (p29): 700 m2/hr at 25 cm = 175 m3/hr
        (800, 0.15, 120.0),   # invented: 800 x 0.15 = 120 m3/hr
    ],
)
def test_grader_volume_rate(area_rate, layer, volume_rate):
    out = ev("earthworks.grader_volume_rate", grader_area_rate=area_rate, layer_thickness=layer)
    assert out["result"] == pytest.approx(volume_rate)
    assert out["unit"] == "meter**3/hour"


@pytest.mark.parametrize(
    "layers, area, days, hours, rate",
    [
        (19, 2_000_000, 210, 20, 38_000_000 / 4200),  # course W7 (p27): 9,047.6 (printed 9,048) m2/hr
        (20, 2_000_000, 210, 20, 40_000_000 / 4200),  # course W8 (p29): 9,523.8 (printed 9,524) m2/hr
        (4, 50_000, 25, 10, 800.0),                   # invented: 4 x 50,000 / (25 x 10) = 800 m2/hr
    ],
)
def test_grading_area_rate(layers, area, days, hours, rate):
    out = ev(
        "earthworks.grading_area_rate",
        layer_count=layers,
        fill_area=area,
        working_days=days,
        hours_per_day=hours,
    )
    assert out["result"] == pytest.approx(rate)
    assert out["unit"] == "meter**2/hour"


@pytest.mark.parametrize(
    "layers, area, days, hours, grader_rate, graders",
    [
        (20, 2_000_000, 210, 20, 700, 14),  # course W8 (p29): 9,523.8 / 700 = 13.6 -> 14 graders
        (4, 50_000, 25, 10, 750, 2),        # invented: 800 / 750 = 1.07 -> 2 graders
    ],
)
def test_area_rate_sizes_graders_with_the_existing_machine_count(
    layers, area, days, hours, grader_rate, graders
):
    rate = ev(
        "earthworks.grading_area_rate",
        layer_count=layers,
        fill_area=area,
        working_days=days,
        hours_per_day=hours,
    )["result"]
    out = ev("equipment.machines_required", required_rate=rate, production_per_machine=grader_rate)
    assert out["result"] == graders
