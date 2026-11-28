"""The ``arithmetic_step`` formula: one step of plain arithmetic on two values.

Every in-between step of a chained answer -- a sum, a difference, a product,
even a single division -- is a calculator call, never arithmetic written in
the model's text. The values carry their units: two values are added or
subtracted only in the same unit, a product or quotient composes the units,
and the written method is returned with the result. The units are read
through the one unit table (``parse_unit``): in a product or quotient a
fraction (a percent) is its number times the table's scale, a count is a
plain number in a product, and a length unit raised to a power the table
names is written as that unit (mm by mm is mm2, m3 over m is m2).

The result's unit depends on the inputs, so it is returned beside the
result as ``result_unit`` (an output ``<name>`` whose unit is not fixed is
written next to it as ``<name>_unit``).
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Optional, Tuple

from app.lib.formula_registry import Param, formula

_LOG = logging.getLogger(__name__)

OPERATORS = ("add", "subtract", "multiply", "divide")

#: A value's declared unit: a bare number. Its real unit is given beside it
#: (``a_unit`` / ``b_unit``); a value written with a unit is refused and asked
#: for by its label, so the unit goes where the arithmetic reads it.
BARE_NUMBER = "ratio"

#: The magnitude either value may have. Wide on purpose: this is plain
#: arithmetic on figures that already passed their own formula's checks.
_LIMIT = 1e15


def finite_number(value: Any) -> Optional[float]:
    """``value`` as a finite float -- a number, or text such as "3,000" --
    or None for anything else (a flag, a word, None, NaN, an infinity)."""
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace(",", "").strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        _LOG.debug("not a number: %r", value)
        return None
    return number if math.isfinite(number) else None


def _unit(text: Any) -> str:
    return str(text or "").strip() or "-"


def _same_unit(a: str, b: str) -> bool:
    return a.lower().replace(" ", "") == b.lower().replace(" ", "")


_POWERED_RE = re.compile(r"([A-Za-z]+)([1-4]?)")


def _table_unit(text: str) -> Any:
    from app.agents.base.formulas.construction_formulas_planning import parse_unit

    return parse_unit(text) if text != "-" else None


def _length_power(unit: str) -> Optional[Tuple[str, int]]:
    """``unit`` as a length unit and its power when the unit table names it
    so ("mm2" -> ("mm", 2), "m" -> ("m", 1)), else None. The power is taken
    only when the base is a length and the table knows the whole name, so a
    name that merely ends in a digit is not one (the table holds each power
    of a length under its own dimension: area, volume, second moment)."""
    match = _POWERED_RE.fullmatch(unit.strip())
    if match is None:
        return None
    base, power = match.group(1), int(match.group(2) or 1)
    base_unit, whole = _table_unit(base), _table_unit(unit.strip())
    if base_unit is None or whole is None or base_unit.dimension != "length":
        return None
    return base, power


def _powered_name(base: str, power: int) -> Optional[str]:
    """The unit the table names for ``base`` to ``power``, else None (a
    power of zero or below, or one past the table)."""
    name = base if power == 1 else f"{base}{power}"
    return name if _length_power(name) is not None else None


def _as_plain(value: float, unit: str, operator: str) -> Tuple[float, str, str]:
    """A value about to be multiplied or divided, with a fraction unit (a
    percent) turned into its plain number by the table's scale, and a count
    unit dropped from a product (a count times a length is a length). Returns
    the value, the unit that composes, and how it was read ("" when as given)."""
    table = _table_unit(unit)
    if table is not None and table.dimension == "fraction":
        plain = value * table.scale
        return plain, "-", f"{value:g} {unit} = {plain:g}"
    if table is not None and table.dimension == "count" and operator == "multiply":
        return value, "-", ""
    return value, unit, ""


def _compose(a: str, b: str, operator: str) -> str:
    """The unit of a product or quotient of a value in ``a`` and one in ``b``."""
    pa, pb = _length_power(a), _length_power(b)
    if pa is not None and pb is not None and pa[0].lower() == pb[0].lower():
        power = pa[1] + pb[1] if operator == "multiply" else pa[1] - pb[1]
        named = _powered_name(pa[0], power)
        if named is not None:
            return named
    if operator == "multiply":
        if a == "-":
            return b
        return a if b == "-" else f"{a}*{b}"
    if b == "-":
        return a
    if _same_unit(a, b):
        return "-"
    return f"1/{b}" if a == "-" else f"{a}/{b}"


@formula(
    owner="base",
    display_name="Arithmetic step",
    description=("One arithmetic step on two values with their units: add, subtract, multiply "
                 "or divide. Use it for every in-between step of a worked answer, even a single "
                 "division. Values in different units are added or subtracted only after one is "
                 "converted to the other's unit."),
    inputs={
        "operator": Param("-", label="operator (add, subtract, multiply or divide)"),
        "a": Param(BARE_NUMBER, -_LIMIT, _LIMIT, label="first value"),
        "b": Param(BARE_NUMBER, -_LIMIT, _LIMIT, label="second value"),
        "a_unit": Param("-", label="unit of the first value"),
        "b_unit": Param("-", label="unit of the second value"),
    },
    outputs={"result": "-", "result_unit": "-"},
    main_output="result",
    standard="arithmetic",
)
def arithmetic_step(operator: str, a: float, b: float, a_unit: str = "-", b_unit: str = "-") -> Dict[str, Any]:
    """``a <operator> b`` with units. An unknown operator, a value that is
    not a finite number, a sum or difference of two different units, or a
    division by zero is refused with the reason -- never a guessed result."""
    op = str(operator or "").strip().lower()
    if op not in OPERATORS:
        return {"error": f"operator must be one of {', '.join(OPERATORS)}"}
    x, y = finite_number(a), finite_number(b)
    if x is None or y is None:
        return {"error": "both values must be finite numbers"}
    au, bu = _unit(a_unit), _unit(b_unit)
    if op in ("add", "subtract") and not _same_unit(au, bu):
        return {"error": f"cannot {op} {au} and {bu}: convert one to the other's unit first"}
    if op == "divide" and y == 0:
        return {"error": "division by zero"}
    readings = []
    if op in ("multiply", "divide"):
        px, cu_a, read_a = _as_plain(x, au, op)
        py, cu_b, read_b = _as_plain(y, bu, op)
        readings = [r for r in (read_a, read_b) if r]
    if op == "add":
        value, sign, unit = x + y, "+", au
    elif op == "subtract":
        value, sign, unit = x - y, "-", au
    elif op == "multiply":
        value, sign, unit = px * py, "*", _compose(cu_a, cu_b, op)
    else:
        value, sign, unit = px / py, "/", _compose(cu_a, cu_b, op)
    value = round(value, 10)
    method = f"{x:g} {au} {sign} {y:g} {bu} = {value:g} {unit}"
    if readings:
        method += f" (taking {'; '.join(readings)})"
    return {"result": value, "result_unit": unit, "standard": "arithmetic", "method": method}
