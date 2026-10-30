"""Formulas from app.lib.construction_formulas_planning owned by the base package (every hat).

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).

``pe_unit_convert`` is the platform's one unit converter. Every calculator
input written in a unit other than the one it declares is converted through
it (``app.lib.formula_registry.convert_written_units``), so the tables below
are the only conversion factors in the calculator lane.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.lib.formula_registry import Param, formula

#: Values a conversion may need that no unit carries. Asked for by label.
CONVERSION_CONTEXT = {
    "crew_size": "crew size",
    "hours_per_day": "working hours per day",
    "days_per_week": "working days per week",
}

_HPD = ("hours_per_day",)
_HPD_DPW = ("hours_per_day", "days_per_week")

# Each family converts through its first unit. A factor is "how many base
# units in one of this unit"; a tuple after it names context values the
# factor is multiplied by. Keys are written as people write them; lookups
# normalise case, spaces, hyphens, dots and superscripts.
_FAMILIES: Dict[str, Dict[str, Any]] = {
    "length": {
        "m": 1.0, "meter": 1.0, "metre": 1.0, "meters": 1.0, "metres": 1.0,
        "mm": 1e-3, "millimetre": 1e-3, "millimeter": 1e-3, "millimetres": 1e-3, "millimeters": 1e-3,
        "cm": 1e-2, "dm": 0.1, "km": 1e3,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
        "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
        "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    },
    "area": {
        "m2": 1.0, "sqm": 1.0, "mm2": 1e-6, "cm2": 1e-4, "km2": 1e6,
        "ha": 1e4, "hectare": 1e4, "hectares": 1e4,
        "in2": 0.00064516, "ft2": 0.09290304, "sqft": 0.09290304, "sf": 0.09290304,
        "yd2": 0.83612736, "acre": 4046.8564224, "acres": 4046.8564224,
    },
    "volume": {
        "m3": 1.0, "cum": 1.0, "mm3": 1e-9, "cm3": 1e-6, "cc": 1e-6,
        "l": 1e-3, "lit": 1e-3, "litre": 1e-3, "liter": 1e-3, "litres": 1e-3, "liters": 1e-3,
        "in3": 1.6387064e-5, "ft3": 0.028316846592, "cuft": 0.028316846592, "cf": 0.028316846592,
        "yd3": 0.764554857984, "cuyd": 0.764554857984,
    },
    "second moment of area": {
        "m4": 1.0, "mm4": 1e-12, "cm4": 1e-8, "in4": 0.0254 ** 4,
    },
    "force": {
        "n": 1.0, "kn": 1e3, "mn": 1e6, "kgf": 9.80665, "tf": 9806.65,
        "lbf": 4.4482216152605, "kip": 4448.2216152605, "kips": 4448.2216152605,
    },
    # Pressure and stress, in kPa. Construction asks for these constantly --
    # concrete grades in MPa and psi, bearing capacity in kg/cm2, a
    # specification in N/mm2. kg/cm2 is 0.0980665 MPa, NOT 0.1: dividing by
    # ten is a 2% error that reads as rounding.
    "pressure": {
        "kpa": 1.0, "pa": 0.001, "n/m2": 0.001, "kn/m2": 1.0,
        "mpa": 1000.0, "n/mm2": 1000.0, "mn/m2": 1000.0,
        "gpa": 1_000_000.0, "kn/mm2": 1_000_000.0,
        "bar": 100.0, "mbar": 0.1, "atm": 101.325,
        "kg/cm2": 98.0665, "kgf/cm2": 98.0665, "ksc": 98.0665, "t/m2": 9.80665,
        "psi": 6.894757293168361, "lb/in2": 6.894757293168361,
        "psf": 0.04788025898033584, "lb/ft2": 0.04788025898033584,
        "ksi": 6894.757293168361, "kip/in2": 6894.757293168361,
    },
    "mass": {
        "kg": 1.0, "g": 1e-3, "t": 1e3, "tonne": 1e3, "tonnes": 1e3,
        "lb": 0.45359237, "lbs": 0.45359237,
    },
    "time": {
        "h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
        "s": 1 / 3600, "sec": 1 / 3600, "second": 1 / 3600, "seconds": 1 / 3600,
        "min": 1 / 60, "mins": 1 / 60, "minute": 1 / 60, "minutes": 1 / 60,
        "d": 24.0, "day": 24.0, "days": 24.0,
        "calendar day": 24.0, "calendar days": 24.0,
        "wk": 168.0, "week": 168.0, "weeks": 168.0,
        "calendar week": 168.0, "calendar weeks": 168.0,
    },
    # A month has no fixed number of days, so months convert only to years.
    "calendar period": {
        "month": 1.0, "months": 1.0, "mo": 1.0, "year": 12.0, "years": 12.0, "yr": 12.0, "yrs": 12.0,
    },
    "angle": {
        "deg": 1.0, "degree": 1.0, "degrees": 1.0,
        "rad": 180 / math.pi, "radian": 180 / math.pi, "radians": 180 / math.pi,
    },
    "fraction": {
        "fraction": 1.0, "%": 0.01, "percent": 0.01, "pct": 0.01, "‰": 0.001,
    },
    # A count of things. "nr" is the mark on a quantity that is not a length,
    # an area or a mass; "count" is the same unit written as a word.
    "count": {"nr": 1.0, "count": 1.0},
    # A pure number whose dimension is named by another input.
    "ratio": {"ratio": 1.0},
    "labour effort": {
        "man-hour": 1.0, "man-hours": 1.0, "manhour": 1.0, "manhours": 1.0,
        "mh": 1.0, "mhr": 1.0, "mhrs": 1.0, "man-hr": 1.0, "man-hrs": 1.0,
        "man-day": (1.0, _HPD), "man-days": (1.0, _HPD), "md": (1.0, _HPD),
        "man-week": (1.0, _HPD_DPW), "man-weeks": (1.0, _HPD_DPW),
    },
    "working time": {
        "working hour": 1.0, "working hours": 1.0,
        "working day": (1.0, _HPD), "working days": (1.0, _HPD),
        "workday": (1.0, _HPD), "workdays": (1.0, _HPD),
        "working week": (1.0, _HPD_DPW), "working weeks": (1.0, _HPD_DPW),
    },
}

# Temperature has an offset: base = value * scale + offset, in degC.
_TEMPERATURE: Dict[str, Tuple[float, float]] = {
    "degc": (1.0, 0.0), "celsius": (1.0, 0.0), "c": (1.0, 0.0),
    "degf": (5 / 9, -160 / 9), "fahrenheit": (5 / 9, -160 / 9),
    "kelvin": (1.0, -273.15),
}

_ALIASES = {"mph": "mi/h", "kph": "km/h", "kmh": "km/h", "fps": "ft/s", "pcf": "lb/ft3",
            "plf": "lbf/ft", "klf": "kip/ft"}

# Calendar days and weeks read as working days and weeks when the value is
# labour or working time; a crew does not work 24 hours a day.
_WORKING_EQUIVALENT = {24.0: "working day", 168.0: "working week"}

_OPAQUE = "currency"
_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")


def unit_key(token: Any) -> str:
    """The lookup form of a unit as written: case, spaces, hyphens, dots and
    superscripts do not distinguish units."""
    s = str(token or "").strip()
    s = s.replace("²", "2").replace("³", "3").replace("⁴", "4").replace("^", "")
    s = s.replace("°", "deg").replace("·", "").replace("µ", "u").replace("μ", "u")
    s = s.lower()
    return re.sub(r"[\s_\-.]", "", s)


_INDEX: Dict[str, Tuple[str, float, Tuple[str, ...], str]] = {}
for _family, _table in _FAMILIES.items():
    for _raw, _factor in _table.items():
        _scale, _needs = _factor if isinstance(_factor, tuple) else (_factor, ())
        _key = unit_key(_raw)
        if _key in _INDEX and _INDEX[_key][0] != _family:
            raise ValueError(f"unit {_raw!r} is in two families")
        _INDEX[_key] = (_family, float(_scale), tuple(_needs), _raw)


# A rate written as one word ("daily") is "per" its unit.
_RATE_WORDS = {
    "hourly": "hour", "daily": "day", "weekly": "week",
    "monthly": "month", "yearly": "year", "annually": "year",
}
_RATE_WORD_RE = re.compile(r"\b(" + "|".join(_RATE_WORDS) + r")\b", re.IGNORECASE)
# "per" as a word, then any separator: "per day", "Per Day", "per-day", "per/day".
_PER_WORD_RE = re.compile(r"\s*\bper\b[\s\-/]*", re.IGNORECASE)
# "per" run into a known unit with no separator: "perday", "perWeek".
_PER_JOINED_RE = re.compile(r"\bper([A-Za-z][\w\-]*)", re.IGNORECASE)


def normalise_rate(text: Any) -> str:
    """One spelling for "X per Y": every way of writing the "per" becomes
    "/". Case, separators (space, hyphen, slash or none) and the one-word
    rates ("daily") all normalise; "SAR 5,000 Per Day", "perday", "per-day",
    "per/day" and "daily" all read as "/day". "per" run into a word is split
    off only when the rest is a unit the table knows ("perch" and "percent"
    stay as written)."""
    s = str(text or "").strip()
    s = _RATE_WORD_RE.sub(lambda m: " per " + _RATE_WORDS[m.group(1).lower()], s)
    s = _PER_JOINED_RE.sub(
        lambda m: "per " + m.group(1) if unit_key(m.group(1)) in _INDEX else m.group(0), s)
    return _PER_WORD_RE.sub("/", s)


@dataclass(frozen=True)
class Unit:
    """A unit resolved to its dimension and its factor onto the dimension's base."""
    dimension: str
    scale: float
    offset: float = 0.0
    needs: Tuple[str, ...] = ()
    written: str = ""


def _atom(token: str, in_ratio: bool) -> Optional[Unit]:
    key = unit_key(_ALIASES.get(unit_key(token), token))
    if key in _INDEX:
        family, scale, needs, raw = _INDEX[key]
        if in_ratio and family == "labour effort":
            family = "time"
        return Unit(family, scale, 0.0, needs, raw)
    if key in _TEMPERATURE:
        scale, offset = _TEMPERATURE[key]
        return Unit("temperature", scale, 0.0 if in_ratio else offset, (), token)
    if key.endswith("co2e") and unit_key(key[:-4]) in _INDEX and _INDEX[unit_key(key[:-4])][0] == "mass":
        return Unit("mass of CO2e", _INDEX[unit_key(key[:-4])][1], 0.0, (), token)
    if key == _OPAQUE or _CURRENCY_CODE_RE.match(str(token).strip()):
        return Unit(_OPAQUE, 1.0, 0.0, (), token)
    if in_ratio and key == "1":
        return Unit("1", 1.0, 0.0, (), token)
    return None


def parse_unit(token: Any) -> Optional[Unit]:
    """``token`` resolved to a dimension and factor, or None when the
    converter does not know it. A ratio ("kN/m", "currency/t", "h/m2") is
    resolved from its parts; force over area is a pressure."""
    raw = normalise_rate(token)
    if not raw or raw in ("-",):
        return None
    key = unit_key(_ALIASES.get(unit_key(raw), raw))
    whole = _atom(raw, in_ratio=False)
    if whole is not None or "/" not in key:
        return whole
    parts = [p.strip() for p in _ALIASES.get(unit_key(raw), raw).split("/")]
    if any(not p for p in parts):
        return None
    atoms = [_atom(p, in_ratio=True) for p in parts]
    if any(a is None for a in atoms):
        return None
    head, *tail = atoms
    scale = head.scale
    for a in tail:
        scale /= a.scale
    needs = tuple(n for a in atoms for n in a.needs)
    if head.dimension == "force" and [a.dimension for a in tail] == ["area"]:
        return Unit("pressure", scale / 1000.0, 0.0, needs, raw)
    dimension = "/".join([head.dimension] + [a.dimension for a in tail])
    return Unit(dimension, scale, 0.0, needs, raw)


def unit_dimension(token: Any) -> Optional[str]:
    unit = parse_unit(token)
    return unit.dimension if unit else None


def _context_factor(needs: Tuple[str, ...], context: Dict[str, Any]) -> float:
    factor = 1.0
    for name in needs:
        factor *= float(context[name])
    return factor


def _missing_context(needs: Tuple[str, ...], context: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for name in needs:
        value = context.get(name)
        if (value is None or not isinstance(value, (int, float)) or isinstance(value, bool)
                or value <= 0) and name not in out:
            out.append(name)
    return out


def _as_working(unit: Unit) -> Unit:
    if unit.dimension == "time" and unit.scale in _WORKING_EQUIVALENT:
        family, scale, needs, raw = _INDEX[unit_key(_WORKING_EQUIVALENT[unit.scale])]
        return Unit(family, scale, 0.0, needs, raw)
    return unit


def convert_units(value: Any, from_unit: Any, to_unit: Any,
                  **context: Any) -> Dict[str, Any]:
    """``value`` in ``from_unit`` expressed in ``to_unit``.

    Returns ``value_out`` and ``dimension``; ``needs_input`` naming the
    context values a conversion needs and was not given; or ``error`` when
    the two units are not the same kind of quantity. Clock hours stand as
    working hours or man-hours; man-days and working days need the hours
    per day, and man-hours to time needs the crew size.
    """
    src, dst = parse_unit(from_unit), parse_unit(to_unit)
    if src is None or dst is None:
        unknown = from_unit if src is None else to_unit
        return {"error": f"Unknown unit {unknown!r}.", "unknown_unit": str(unknown)}
    number = float(value)
    route = src.dimension
    crew: Tuple[str, ...] = ()
    if src.dimension == dst.dimension:
        pass
    elif src.dimension in ("labour effort", "working time") and dst.dimension in ("working time", "time"):
        dst = _as_working(dst)
        if src.dimension == "labour effort":
            crew = ("crew_size",)
        route = f"{src.dimension} to {dst.dimension}"
    elif src.dimension == "working time" and dst.dimension == "labour effort":
        crew = ("crew_size",)
        route = f"{src.dimension} to {dst.dimension}"
    elif src.dimension == "time" and dst.dimension in ("working time", "labour effort") and src.scale <= 1.0:
        route = f"{src.dimension} to {dst.dimension}"
    else:
        return {"error": (f"{from_unit} measures {src.dimension} and {to_unit} measures "
                          f"{dst.dimension}; they do not convert."),
                "dimensions": [src.dimension, dst.dimension]}
    needs = src.needs + dst.needs + crew
    missing = _missing_context(needs, context)
    if missing:
        return {"error": (f"Converting {from_unit} to {to_unit} needs the "
                          + " and ".join(CONVERSION_CONTEXT[m] for m in missing) + "."),
                "needs_input": missing}
    base = number * src.scale * _context_factor(src.needs, context) + src.offset
    if crew and src.dimension == "labour effort":
        base /= float(context["crew_size"])
    elif crew:
        base *= float(context["crew_size"])
    out = (base - dst.offset) / (dst.scale * _context_factor(dst.needs, context))
    used = {name: context[name] for name in needs}
    read_as = dst.written if dst.dimension == "working time" else str(to_unit)
    return {"value_out": float(f"{out:.12g}"), "dimension": route, "context": used,
            "to_unit_read_as": read_as}


def text_unit_tokens() -> List[str]:
    """Units as they are written in a sentence after a figure. Tokens that
    are also ordinary words or abbreviations ("in", "d", "s", "l", "mo",
    "c", "cc", "md", "mh", "sf", "cf") are left out: in prose they are not units."""
    skip = {"in", "d", "s", "l", "mo", "c", "cc", "md", "mh", "sf", "cf", "n", "g", "pct",
            "fraction", "count", "ratio", "hr", "sec", "min", "mins", "second", "seconds",
            "minute", "minutes"}
    tokens = [raw for _f, table in _FAMILIES.items() for raw in table if unit_key(raw) not in skip]
    tokens += ["°c", "°f", "degc", "degf", "min", "hr"]
    tokens += _co2e_forms()
    tokens += list(_ALIASES)
    return sorted(set(tokens), key=len, reverse=True)


def _co2e_forms() -> List[str]:
    """A mass of CO2e as written: "kgCO2e", "tCO2e"."""
    return [f"{raw}co2e" for raw in _FAMILIES["mass"] if raw.isalpha() and len(raw) <= 2]


def _written_forms(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for token in tokens:
        out.append(token)
        if len(token) > 1 and token[-1] in "234" and token[-2].isalpha():
            out.append(token[:-1] + {"2": "²", "3": "³", "4": "⁴"}[token[-1]])
            out.append(token[:-1] + "^" + token[-1])
    return sorted(set(out), key=len, reverse=True)


def text_unit_pattern() -> str:
    """A regex for a unit written after a figure, ratios included ("kN/m",
    "SAR/m3", "m2/h"), ending at a word boundary. Matched case-insensitively."""
    lead = _written_forms(text_unit_tokens())
    every = _written_forms(sorted({raw for table in _FAMILIES.values() for raw in table}
                                  | set(_TEMPERATURE) | set(_co2e_forms()), key=len, reverse=True))
    head = "|".join(re.escape(t) for t in lead)
    tail = "|".join(re.escape(t) for t in every)
    ratio = rf"(?:(?-i:[A-Z]{{3}})|currency|{tail})(?:\s*/\s*(?:{tail}))+"
    return rf"(?:{ratio}|(?:{head}))(?![A-Za-z0-9²³⁴])"


@formula(
    owner='base',
    display_name='Unit conversion',
    description=('Converts a value between engineering units: length, area, volume, '
                 'second moment of area, force, pressure and stress, mass, time, angle, '
                 'temperature, percentages, ratios of these (kN/m, kg/m3, price per t), '
                 'and labour effort to working time given the crew size and working hours.'),
    inputs={
        'value': Param('ratio', -1e13, 1e13, label='value to convert'),
        'from_unit': Param('-', label='unit to convert from'),
        'to_unit': Param('-', label='unit to convert to'),
        'crew_size': Param('nr', 1, 100000, label=CONVERSION_CONTEXT['crew_size']),
        'hours_per_day': Param('h', 1, 24, label=CONVERSION_CONTEXT['hours_per_day']),
        'days_per_week': Param('nr', 1, 7, label=CONVERSION_CONTEXT['days_per_week']),
    },
    outputs={'value_in': '-', 'from_unit': '-', 'to_unit': '-', 'value_out': '-', 'dimension': '-', 'formula_used': '-'},
)
def pe_unit_convert(value: float, from_unit: str, to_unit: str,
                    crew_size: float | None = None, hours_per_day: float | None = None,
                    days_per_week: float | None = None) -> dict:
    """Convert ``value`` from ``from_unit`` to ``to_unit`` through the SI base
    of their dimension (m, m2, m3, m4, N, kPa, kg, h, deg, degC, fraction).

    Labour effort (man-hours, man-days) becomes working time with the crew
    size and working hours per day; a missing one is asked for, never assumed.
    """
    if value is None:
        return {"error": "pe_unit_convert requires value.", "required": ["value", "from_unit", "to_unit"]}
    src = (from_unit or "").strip()
    dst = (to_unit or "").strip()
    if not src or not dst:
        return {
            "error": "pe_unit_convert requires from_unit and to_unit.",
            "required": ["value", "from_unit", "to_unit"],
            "supported": sorted(_FAMILIES),
        }
    out = convert_units(value, src, dst, crew_size=crew_size,
                        hours_per_day=hours_per_day, days_per_week=days_per_week)
    if "needs_input" in out:
        return out
    if "error" in out:
        supported = sorted(_FAMILIES) + ["temperature"]
        failed = {"error": (f"Unsupported conversion {from_unit!r} → {to_unit!r}: {out['error']} "
                            f"Supported: {', '.join(supported)}."),
                  "supported": supported}
        failed.update({k: out[k] for k in ("dimensions", "unknown_unit") if k in out})
        return failed
    converted = out["value_out"]
    context = ", ".join(f"{CONVERSION_CONTEXT[k]} {v:g}" for k, v in out["context"].items())
    return {
        "value_in": float(value),
        "from_unit": src.lower(),
        "to_unit": dst.lower(),
        "value_out": converted,
        "dimension": out["dimension"],
        "standard": "PE formula sheet (Conversion Formula)",
        "note": f"{value} {src} = {converted:.8g} {out['to_unit_read_as']}" + (f" ({context})" if context else ""),
        "formula_used": f"convert via SI base ({out['dimension']})",
    }
