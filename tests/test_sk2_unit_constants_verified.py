"""S-K2 item 1: every unit conversion constant, verified from its definition.

The audit of 13 Nov (S-K2, first entry) asks for "unit conversion constants
(verified)". ``_FAMILIES`` in the base unit converter is the one table of
conversion factors in the calculator lane, so this file rebuilds every factor
in it from the definitions those units are legally defined by, and checks the
table and the converter against them:

* SI prefixes, and the litre = 1 dm3, hectare = (100 m)2, bar = 1e5 Pa,
  standard atmosphere = 101 325 Pa;
* the international yard and pound agreement of 1959: inch = 0.0254 m
  exactly, pound = 0.453 592 37 kg exactly; foot = 12 in, yard = 3 ft,
  mile = 5280 ft, acre = 43 560 ft2;
* standard acceleration of gravity g_n = 9.806 65 m/s2 exactly (CGPM 1901),
  which defines kgf, tonne-force and lbf;
* a pressure is a force over an area (psi = lbf/in2, ksc = kgf/cm2, ...);
* derived area, volume and second-moment units are the powers of a length.

Nothing in the table is taken on trust: a unit added to ``_FAMILIES`` without
a definition here fails ``test_every_unit_in_the_table_has_a_definition``, so
every new constant is verified before it can convert a figure.
"""
from __future__ import annotations

import math

import pytest

from app.agents.base.formulas.construction_formulas_planning import (
    _FAMILIES,
    convert_units,
)

# ── Definitions (exact unless the definition itself is irrational) ─────────
M = 1.0
MM, CM, DM, KM = 1e-3, 1e-2, 1e-1, 1e3
INCH = 0.0254                 # 1959 agreement, exact
FT = 12 * INCH
YD = 3 * FT
MI = 5280 * FT
KG = 1.0
LB = 0.45359237               # 1959 agreement, exact
G_N = 9.80665                 # standard gravity, exact
N = 1.0
KGF = KG * G_N
LBF = LB * G_N
PA_IN_KPA = 1e-3              # the pressure family is in kPa
HOUR = 1.0                    # the time family is in hours


def _pressure(force_n: float, area_m2: float) -> float:
    return force_n / area_m2 * PA_IN_KPA


_LENGTH = {"m": M, "mm": MM, "cm": CM, "dm": DM, "km": KM,
           "in": INCH, "ft": FT, "yd": YD, "mi": MI}
_AREA = {"m2": M ** 2, "mm2": MM ** 2, "cm2": CM ** 2, "km2": KM ** 2,
         "ha": (100 * M) ** 2, "in2": INCH ** 2, "ft2": FT ** 2, "yd2": YD ** 2,
         "acre": 43560 * FT ** 2}
_VOLUME = {"m3": M ** 3, "mm3": MM ** 3, "cm3": CM ** 3, "l": DM ** 3,
           "in3": INCH ** 3, "ft3": FT ** 3, "yd3": YD ** 3}
_FORCE = {"n": N, "kn": 1e3 * N, "mn": 1e6 * N, "kgf": KGF, "tf": 1000 * KGF,
          "lbf": LBF, "kip": 1000 * LBF}
_PRESSURE = {
    "kpa": 1.0, "pa": PA_IN_KPA, "n/m2": _pressure(N, M ** 2), "kn/m2": _pressure(1e3, M ** 2),
    "mpa": 1e6 * PA_IN_KPA, "n/mm2": _pressure(N, MM ** 2), "mn/m2": _pressure(1e6, M ** 2),
    "gpa": 1e9 * PA_IN_KPA, "kn/mm2": _pressure(1e3, MM ** 2),
    "bar": 1e5 * PA_IN_KPA, "mbar": 1e2 * PA_IN_KPA, "atm": 101325 * PA_IN_KPA,
    "kg/cm2": _pressure(KGF, CM ** 2), "kgf/cm2": _pressure(KGF, CM ** 2),
    "ksc": _pressure(KGF, CM ** 2), "t/m2": _pressure(1000 * KGF, M ** 2),
    "psi": _pressure(LBF, INCH ** 2), "lb/in2": _pressure(LBF, INCH ** 2),
    "psf": _pressure(LBF, FT ** 2), "lb/ft2": _pressure(LBF, FT ** 2),
    "ksi": _pressure(1000 * LBF, INCH ** 2), "kip/in2": _pressure(1000 * LBF, INCH ** 2),
}
_MASS = {"kg": KG, "g": 1e-3 * KG, "t": 1e3 * KG, "lb": LB}
_TIME = {"h": HOUR, "s": HOUR / 3600, "min": HOUR / 60, "d": 24 * HOUR, "wk": 7 * 24 * HOUR}
_CALENDAR = {"month": 1.0, "year": 12.0}
_ANGLE = {"deg": 1.0, "rad": 180 / math.pi}
_FRACTION = {"fraction": 1.0, "%": 1e-2, "‰": 1e-3}

# Spellings of one unit share its definition.
_SAME_AS = {
    "meter": "m", "metre": "m", "meters": "m", "metres": "m",
    "millimetre": "mm", "millimeter": "mm", "millimetres": "mm", "millimeters": "mm",
    "inch": "in", "inches": "in", "feet": "ft", "foot": "ft",
    "yard": "yd", "yards": "yd", "mile": "mi", "miles": "mi",
    "sqm": "m2", "hectare": "ha", "hectares": "ha", "sqft": "ft2", "sf": "ft2", "acres": "acre",
    "cum": "m3", "cc": "cm3", "lit": "l", "litre": "l", "liter": "l", "litres": "l", "liters": "l",
    "cuft": "ft3", "cf": "ft3", "cuyd": "yd3", "kips": "kip",
    "tonne": "t", "tonnes": "t", "lbs": "lb",
    "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "sec": "s", "second": "s", "seconds": "s", "mins": "min", "minute": "min", "minutes": "min",
    "day": "d", "days": "d", "calendar day": "d", "calendar days": "d",
    "week": "wk", "weeks": "wk", "calendar week": "wk", "calendar weeks": "wk",
    "months": "month", "mo": "month", "years": "year", "yr": "year", "yrs": "year",
    "degree": "deg", "degrees": "deg", "radian": "rad", "radians": "rad",
    "percent": "%", "pct": "%",
}

_DEFINED = {
    "length": _LENGTH, "area": _AREA, "volume": _VOLUME,
    "second moment of area": {"m4": M ** 4, "mm4": MM ** 4, "cm4": CM ** 4, "in4": INCH ** 4},
    "force": _FORCE, "pressure": _PRESSURE, "mass": _MASS, "time": _TIME,
    "calendar period": _CALENDAR, "angle": _ANGLE, "fraction": _FRACTION,
    "count": {"nr": 1.0, "count": 1.0}, "ratio": {"ratio": 1.0},
}

# Labour effort and working time carry their hours from the context, so
# they are defined by the hours they come to under a stated working pattern.
_CONTEXT_DEFINED = {
    "labour effort": {"man-hour": lambda h, d: 1.0, "man-day": lambda h, d: h,
                      "man-week": lambda h, d: h * d, "crew-day": lambda h, d: h,
                      "crew-week": lambda h, d: h * d},
    "working time": {"working hour": lambda h, d: 1.0, "working day": lambda h, d: h,
                     "working week": lambda h, d: h * d},
}
_CONTEXT_SAME_AS = {
    "man-hours": "man-hour", "manhour": "man-hour", "manhours": "man-hour", "mh": "man-hour",
    "mhr": "man-hour", "mhrs": "man-hour", "man-hr": "man-hour", "man-hrs": "man-hour",
    "man-days": "man-day", "md": "man-day", "man-weeks": "man-week",
    "crew-days": "crew-day", "crew-weeks": "crew-week",
    "working hours": "working hour", "working days": "working day", "workday": "working day",
    "workdays": "working day", "working weeks": "working week",
}

_BASE = {family: next(iter(table)) for family, table in _FAMILIES.items()}


def _definition(family: str, unit: str) -> float:
    table = _DEFINED[family]
    return table[_SAME_AS.get(unit, unit)]


def _plain_units():
    for family, table in _FAMILIES.items():
        if family in _CONTEXT_DEFINED:
            continue
        for unit in table:
            yield family, unit


def _context_units():
    for family in _CONTEXT_DEFINED:
        for unit in _FAMILIES[family]:
            yield family, unit


def test_every_unit_in_the_table_has_a_definition():
    undefined = []
    for family, table in _FAMILIES.items():
        for unit in table:
            if family in _CONTEXT_DEFINED:
                known = _CONTEXT_DEFINED[family]
                if _CONTEXT_SAME_AS.get(unit, unit) not in known:
                    undefined.append((family, unit))
            elif family not in _DEFINED or _SAME_AS.get(unit, unit) not in _DEFINED[family]:
                undefined.append((family, unit))
    assert undefined == [], f"units without a verified definition: {undefined}"


@pytest.mark.parametrize("family,unit", list(_plain_units()))
def test_table_factor_equals_its_definition(family, unit):
    assert _FAMILIES[family][unit] == pytest.approx(_definition(family, unit), rel=1e-12)


@pytest.mark.parametrize("family,unit", list(_plain_units()))
@pytest.mark.parametrize("value", [1.0, 37.5])
def test_converter_moves_a_figure_by_the_definition(family, unit, value):
    # Two figures per unit: a converter that adds instead of scaling, or
    # drops the figure, cannot match both.
    base = _BASE[family]
    out = convert_units(value, unit, base)
    assert "error" not in out, out
    assert out["value_out"] == pytest.approx(value * _definition(family, unit), rel=1e-9)
    back = convert_units(value * _definition(family, unit), base, unit)
    assert back["value_out"] == pytest.approx(value, rel=1e-9)


@pytest.mark.parametrize("family,unit", list(_context_units()))
@pytest.mark.parametrize("hours_per_day,days_per_week", [(8.0, 6.0), (10.0, 5.0)])
def test_context_units_come_to_their_defined_hours(family, unit, hours_per_day, days_per_week):
    expected = _CONTEXT_DEFINED[family][_CONTEXT_SAME_AS.get(unit, unit)](hours_per_day, days_per_week)
    scale = _FAMILIES[family][unit]
    factor, needs = scale if isinstance(scale, tuple) else (scale, ())
    context = {"hours_per_day": hours_per_day, "days_per_week": days_per_week}
    got = factor
    for name in needs:
        got *= context[name]
    assert got == pytest.approx(expected, rel=1e-12)
    out = convert_units(3.0, unit, _BASE[family], crew_size=1.0, **context)
    assert "error" not in out and "needs_input" not in out, out
    assert out["value_out"] == pytest.approx(3.0 * expected, rel=1e-9)


@pytest.mark.parametrize("celsius", [-40.0, 0.0, 20.0, 100.0])
def test_temperature_scales_follow_their_definitions(celsius):
    # degF = degC x 9/5 + 32 and K = degC + 273.15, by definition. -40 is
    # the one figure both Celsius and Fahrenheit write the same.
    fahrenheit = celsius * 9 / 5 + 32
    kelvin = celsius + 273.15
    assert convert_units(fahrenheit, "degF", "degC")["value_out"] == pytest.approx(celsius, abs=1e-9)
    assert convert_units(kelvin, "kelvin", "degC")["value_out"] == pytest.approx(celsius, abs=1e-9)
    assert convert_units(celsius, "degC", "degF")["value_out"] == pytest.approx(fahrenheit, abs=1e-9)


@pytest.mark.parametrize("ratio,base,definition", [
    ("kN/m", "N/m", 1e3),
    ("lbf/ft", "N/m", LBF / FT),
    ("kg/m3", "lb/ft3", KG / LB * FT ** 3),
    ("lb/ft3", "kg/m3", LB / FT ** 3),
])
def test_ratio_of_units_is_the_ratio_of_definitions(ratio, base, definition):
    out = convert_units(2.0, ratio, base)
    assert out["value_out"] == pytest.approx(2.0 * definition, rel=1e-9)
