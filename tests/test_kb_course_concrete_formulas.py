"""Course program, file A, first formula batch: concrete technology.

Target mean strength, workability water, elastic axial shortening, the
temperature drop equivalent to a strain, the modulus of rupture from a beam
test and the splitting tensile strength. Every entry is a ``type=formula``
knowledge entry evaluated by ``app.blocks._knowledge.evaluate``. Values come
from the owner-supplied course material (concrete technology part 1, all
blocks CONFIRMED in its verify file) or are invented; each case states its
hand calculation.
"""
from __future__ import annotations

import math

import pint
import pytest
import sympy

from app.blocks import _knowledge as kb

NEW_IDS = [
    "concrete.target_mean_strength",
    "concrete.workability_water",
    "structure.axial_shortening",
    "concrete.equivalent_temperature_drop",
    "concrete.modulus_of_rupture_beam",
    "concrete.split_cylinder_tensile",
]

_UREG = pint.UnitRegistry()


def ev(rule_id, **values):
    return kb.evaluate(rule_id, **values)


@pytest.mark.parametrize("rule_id", NEW_IDS)
def test_entry_carries_course_source_and_declared_units(rule_id):
    e = kb.get_rule(rule_id)
    assert e is not None and e["type"] == "formula"
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
        ("concrete.target_mean_strength", {"characteristic_strength": 30}),
        ("concrete.workability_water", {"water_cement_ratio": 0.4, "cement_content": 400}),
        ("structure.axial_shortening", {"axial_load": 3e6, "member_length": 3500, "section_area": 640000}),
        ("concrete.split_cylinder_tensile", {"failure_load": 20000, "cylinder_diameter": 15}),
    ],
)
def test_missing_required_input_raises_not_defaulted(rule_id, partial):
    with pytest.raises(ValueError):
        ev(rule_id, **partial)


@pytest.mark.parametrize(
    "fk, s, fm",
    [
        (30, 7.5, 42.3),   # course p28: 30 + 1.64 x 7.5 = 42.3 N/mm2
        (40, 5.0, 48.2),   # invented: 40 + 1.64 x 5 = 48.2 N/mm2
    ],
)
def test_target_mean_strength(fk, s, fm):
    out = ev("concrete.target_mean_strength", characteristic_strength=fk, standard_deviation=s)
    assert out["result"] == pytest.approx(fm)
    assert out["unit"] == "N/mm**2"


@pytest.mark.parametrize(
    "wc, cement, fraction, extra",
    [
        (0.4, 400, 0.20, 80.0),   # course pp25-26: 160 - 0.20 x 400 = 160 - 80 = 80 L
        (0.4, 400, 0.25, 60.0),   # course pp25-26: 160 - 0.25 x 400 = 160 - 100 = 60 L
        (0.5, 350, 0.22, 98.0),   # invented: 175 - 77 = 98 L
    ],
)
def test_workability_water_fraction_is_an_input(wc, cement, fraction, extra):
    out = ev(
        "concrete.workability_water",
        water_cement_ratio=wc,
        cement_content=cement,
        hydration_fraction=fraction,
    )
    assert out["result"] == pytest.approx(extra)


@pytest.mark.parametrize(
    "p, length, area, e_mod, dl",
    [
        # invented: 3,000,000 x 3,500 / (640,000 x 30,000) = 1.05e10 / 1.92e10 = 0.546875 mm
        (3_000_000, 3500, 640_000, 30_000, 0.546875),
        # invented: 1,500,000 x 4,000 / (250,000 x 25,000) = 6e9 / 6.25e9 = 0.96 mm
        (1_500_000, 4000, 250_000, 25_000, 0.96),
    ],
)
def test_axial_shortening(p, length, area, e_mod, dl):
    out = ev(
        "structure.axial_shortening",
        axial_load=p,
        member_length=length,
        section_area=area,
        elastic_modulus=e_mod,
    )
    assert out["result"] == pytest.approx(dl)
    assert out["unit"] == "millimeter"


@pytest.mark.parametrize(
    "strain, alpha, dt",
    [
        (0.0002, 10e-6, 20.0),    # course p35: 0.0002 / 10e-6 = 20 C
        (0.0003, 12e-6, 25.0),    # invented: 0.0003 / 12e-6 = 25 C
    ],
)
def test_equivalent_temperature_drop(strain, alpha, dt):
    out = ev("concrete.equivalent_temperature_drop", strain=strain, thermal_coefficient=alpha)
    assert out["result"] == pytest.approx(dt)


@pytest.mark.parametrize(
    "p, span, b, d, fr",
    [
        # invented load on the course's standard beam (15 x 15, span 55):
        # M = 2000 x 55 / 4 = 27,500; Z = 15 x 15^2 / 6 = 562.5 (course p36); fr = 48.889 kgf/cm2
        (2000, 55, 15, 15, 27500 / 562.5),
        # invented: M = 1500 x 45 / 4 = 16,875; Z = 10 x 100 / 6 = 166.667; fr = 101.25 kgf/cm2
        (1500, 45, 10, 10, 101.25),
    ],
)
def test_modulus_of_rupture_beam(p, span, b, d, fr):
    out = ev(
        "concrete.modulus_of_rupture_beam",
        failure_load=p,
        span=span,
        beam_width=b,
        beam_depth=d,
    )
    assert out["result"] == pytest.approx(fr)


@pytest.mark.parametrize(
    "p, dia, length, f",
    [
        # invented, course cylinder 15 x 30: 2 x 20,000 / (pi x 15 x 30) = 40,000 / 1,413.72 = 28.294
        (20000, 15, 30, 28.2942),
        # invented: 2 x 15,000 / (pi x 10 x 20) = 30,000 / 628.32 = 47.746
        (15000, 10, 20, 47.7465),
    ],
)
def test_split_cylinder_tensile(p, dia, length, f):
    out = ev(
        "concrete.split_cylinder_tensile",
        failure_load=p,
        cylinder_diameter=dia,
        cylinder_length=length,
    )
    assert out["result"] == pytest.approx(f, rel=1e-5)
    assert out["result"] == pytest.approx(2 * p / (math.pi * dia * length))
