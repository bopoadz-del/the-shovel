"""Formula registry: every calculation declares who owns it and what it does.

A formula is registered by declaring it where it is defined::

    @formula(owner="base", display_name="Rectangular volume",
             description="Volume of a rectangular prism.",
             inputs={"length_m": "m", "width_m": "m", "depth_m": "m"},
             outputs={"volume_m3": "m3"})
    def some_volume(length_m, width_m, depth_m): ...

``owner`` is ``"base"`` (general formulas every hat is offered: unit
conversion, geometry, areas, volumes, percentages, simple rates) or one hat
(``design``, ``quantities``, ``commercial``, ``contracts``, ``planning``,
``procurement``, ``qaqc``, ``safety``). A turn wearing a hat is offered the
base formulas plus that hat's own.

The declaration holds descriptions only -- never a project's value, rate or
worked example. ``display_name`` is what a user reads when an answer credits
the formula ("Delay damages per day -- platform calculator"); the function
name is an internal identifier and never shown. Built-in factors a formula falls back to are returned flagged
as indicative defaults by the formula itself.

Each input is declared as a :class:`Param`: its unit, the range a supplied
value must lie in, the words a user reads for it, and -- for a strength --
the material whose grade label states it ("C30" states a concrete strength
of 30 MPa). The input's type is its annotation in the function signature.
A supplied value of the wrong type or outside the range is never run; the
user is asked for it by its label.

The schema every registry entry carries (formulas here, tools in
app.agents.core.tool_registry), enforced by tests/test_registry_schema.py:

* the entry: ``display_name``;
* each parameter: ``label`` (its display label -- the words a user reads),
  type (a formula's from its signature annotation; a tool's declared as
  ``kind``), ``unit`` and an inclusive range ``lo``..``hi`` for a numeric
  input, and required / optional (a formula's from whether its signature
  gives a default; a tool's declared as ``required``).

``schema_gaps()`` lists what an entry is still missing. The model fills a
calculator's arguments from these declarations, never from per-formula
phrase lists.
"""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

BASE = "base"
HATS = ("design", "quantities", "commercial", "contracts", "planning",
        "procurement", "qaqc", "safety")
OWNERS = (BASE,) + HATS


#: Kinds an input's annotation gives it. Only numeric kinds carry a range.
NUMERIC_KINDS = ("number", "integer", "series")


@dataclass(frozen=True)
class Param:
    """One input as declared: unit, inclusive range of a supplied value (in
    the unit the function receives), the words a user reads, and the
    material whose grade label states this input (``grade``)."""
    unit: str = "-"
    lo: Optional[float] = None
    hi: Optional[float] = None
    label: str = ""
    grade: str = ""
    #: A tool parameter's type (``number``, ``integer``, ``series``, ``text``,
    #: ``boolean``, ``list``, ``object``); a formula's comes from its signature.
    kind: str = ""
    #: A tool parameter's required / optional; a formula's comes from its
    #: signature (no default = required).
    required: Optional[bool] = None

    @property
    def display_label(self) -> str:
        return self.label


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    unit: str
    lo: Optional[float]
    hi: Optional[float]
    label: str
    grade: str
    required: bool
    default: Any = None

    @property
    def display_label(self) -> str:
        return self.label


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    fn: Callable[..., Any]
    owner: str
    description: str
    display_name: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Param] = field(default_factory=dict)
    #: The one output an answer leads with (empty: none declared).
    main_output: str = ""


_REGISTRY: Dict[str, FormulaSpec] = {}


def _declared_param(value: Union[str, Param]) -> Param:
    return value if isinstance(value, Param) else Param(unit=str(value or "-"))


def formula(
    *,
    owner: str,
    description: str,
    display_name: str,
    inputs: Dict[str, Union[str, Param]],
    outputs: Dict[str, str],
    name: Optional[str] = None,
    main_output: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a formula. Raises at import on an unknown owner, an empty
    description, an empty display name or a duplicate name -- a formula
    without an owner, a description or a name a user can read cannot be
    registered at all."""
    if owner not in OWNERS:
        raise ValueError(f"unknown formula owner {owner!r}; one of {OWNERS}")
    if not (description or "").strip():
        raise ValueError("a formula needs a description")
    if not (display_name or "").strip():
        raise ValueError("a formula needs a display_name")
    if main_output and main_output not in outputs:
        raise ValueError(f"main_output {main_output!r} is not one of the declared outputs")

    def _register(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = name or fn.__name__
        existing = _REGISTRY.get(key)
        if existing is not None and existing.fn is not fn:
            raise ValueError(f"formula {key!r} is declared twice")
        params = {k: _declared_param(v) for k, v in inputs.items()}
        _REGISTRY[key] = FormulaSpec(key, fn, owner, description.strip(),
                                     display_name.strip(),
                                     {k: p.unit for k, p in params.items()},
                                     dict(outputs), params, main_output)
        fn.__formula__ = _REGISTRY[key]  # type: ignore[attr-defined]
        return fn

    return _register


def all_specs() -> List[FormulaSpec]:
    _load_all()
    return [spec for _key, spec in sorted(_REGISTRY.items())]


def specs_for(hat: Optional[str]) -> List[FormulaSpec]:
    """Formulas offered to a turn wearing ``hat``: base plus that hat's own.
    ``hat=None`` (no hat selected) is base only."""
    return [s for s in all_specs() if s.owner == BASE or (hat and s.owner == hat)]


def calculators() -> Dict[str, Callable[..., Any]]:
    """name -> function for every declared formula (what construction_calc runs)."""
    return {s.name: s.fn for s in all_specs()}


def get(name: str) -> Optional[FormulaSpec]:
    _load_all()
    return _REGISTRY.get(name)


def formula_modules() -> List[str]:
    """Every module of ``app.agents.base.formulas`` and of each
    ``app.agents.hats.<hat>.formulas`` -- discovered, so a hat's formulas are
    files in its own package and adding one is no core edit."""
    import importlib
    import pkgutil

    import app.agents.hats as hats

    packages = ["app.agents.base.formulas"] + [
        f"app.agents.hats.{info.name}.formulas"
        for info in sorted(pkgutil.iter_modules(hats.__path__), key=lambda i: i.name)
        if info.ispkg
    ]
    mods: List[str] = []
    for pkg in packages:
        try:
            package = importlib.import_module(pkg)
        except ModuleNotFoundError:
            continue
        mods.extend(f"{pkg}.{info.name}"
                    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name))
    return mods


_loaded = False


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    import importlib

    for mod in formula_modules():
        importlib.import_module(mod)
    _loaded = True


def annotation_kind(annotation: Any, default: Any = inspect.Parameter.empty) -> str:
    """``number``, ``integer``, ``series`` (a list of numbers), ``flag``,
    ``table`` (rows or a mapping) or ``text`` -- read from the signature."""
    if annotation is inspect.Parameter.empty:
        if isinstance(default, bool):
            return "flag"
        if isinstance(default, int):
            return "integer"
        return "number" if isinstance(default, float) else "text"
    if isinstance(annotation, type):
        raw = annotation.__name__
    else:
        raw = str(annotation)
    low = raw.replace("typing.", "").replace(" ", "").lower()
    if "bool" in low:
        return "flag"
    if "dict" in low:
        return "table"
    if "list[" in low or low in ("list", "optional[list]"):
        return "series"
    if "float" in low:
        return "number"
    if "int" in low:
        return "integer"
    return "text"


def parameters(spec: FormulaSpec) -> Dict[str, ParamSpec]:
    """Every declared input of ``spec`` with its kind from the signature."""
    sig = inspect.signature(spec.fn)
    out: Dict[str, ParamSpec] = {}
    for key, declared in spec.params.items():
        param = sig.parameters.get(key)
        if param is None:
            continue
        no_default = param.default is inspect.Parameter.empty
        # A declared required / optional wins over the signature: an input
        # can carry a default for the function's own sake and still be a
        # figure the user must give.
        required = no_default if declared.required is None else bool(declared.required)
        out[key] = ParamSpec(
            name=key, kind=annotation_kind(param.annotation, param.default),
            unit=declared.unit, lo=declared.lo, hi=declared.hi,
            label=declared.label, grade=declared.grade, required=required,
            default=None if no_default else param.default,
        )
    return out


# A grade label states a strength in MPa: the letter names the material and
# the first number is the strength (EN 206 / BS 8500 concrete C<cylinder>[/<cube>],
# EN 10025 structural steel S<yield>, BS 4449 / EN 10080 reinforcement B<yield>[A-C]).
_GRADE_FAMILIES: Dict[str, Tuple[str, float, float]] = {
    "c": ("concrete", 8.0, 120.0),
    "s": ("steel", 185.0, 1100.0),
    "b": ("rebar", 250.0, 700.0),
}
# Steel and reinforcement grades are written in capitals; a lowercase b or s
# before a number is a section width or a spacing.
GRADE_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])([CSBc])(\d{1,4})(?:/(\d{1,4}))?([A-C])?(?![A-Za-z0-9/])",
)


def _grade_from_match(match: "re.Match[str]") -> Optional[Tuple[str, float, str]]:
    letter, first, second, suffix = match.groups()
    family, lo, hi = _GRADE_FAMILIES[letter.lower()]
    if (second and family != "concrete") or (suffix and family != "rebar"):
        return None
    strength = float(first)
    if not lo <= strength <= hi:
        return None
    return family, strength, f"{letter.upper()}{first}"


def grade_label(value: Any) -> Optional[Tuple[str, float, str]]:
    """``(material, strength_mpa, short_label)`` when ``value`` is a grade label."""
    if not isinstance(value, str):
        return None
    match = GRADE_LABEL_RE.fullmatch(value.strip())
    return _grade_from_match(match) if match else None


def grade_labels_in(text: str) -> List[Tuple[Tuple[int, int], str, float, str]]:
    """Every grade label written in ``text``: ``(span, material, strength, label)``."""
    found = []
    for match in GRADE_LABEL_RE.finditer(text or ""):
        grade = _grade_from_match(match)
        if grade:
            found.append((match.span(), *grade))
    return found


def _fmt_strength(strength: float) -> str:
    return str(int(strength)) if float(strength).is_integer() else str(strength)


def grade_value(spec: ParamSpec, strength: float, label: str) -> Any:
    """What a grade label supplies to an input of its material."""
    return strength if spec.kind in NUMERIC_KINDS else label


def _place_grade_labels(params: Dict[str, ParamSpec], values: Dict[str, Any]) -> Dict[str, Any]:
    """A grade label goes to the input of its own material, never to another."""
    out = dict(values)
    for key, value in list(values.items()):
        spec = params.get(key)
        if spec is None:
            continue
        if spec.grade and spec.kind == "text" and _number(value) is not None:
            letter = next(k for k, v in _GRADE_FAMILIES.items() if v[0] == spec.grade)
            value = f"{letter.upper()}{_fmt_strength(_number(value))}"
            out[key] = value
        grade = grade_label(value)
        if grade is None:
            continue
        family, strength, label = grade
        if spec.grade == family:
            out[key] = grade_value(spec, strength, label)
            continue
        owners = [k for k, p in params.items() if p.grade == family and out.get(k) in (None, "")]
        if len(owners) == 1:
            out[owners[0]] = grade_value(params[owners[0]], strength, label)
            del out[key]
    return out


_YES = ("true", "yes", "y", "1", "on")
_NO = ("false", "no", "n", "0", "off")


def _rejection(spec: ParamSpec, value: Any, reason: str) -> Dict[str, Any]:
    return {"parameter": spec.name, "label": spec.label, "unit": spec.unit, "value": value,
            "reason": reason, "lo": spec.lo, "hi": spec.hi, "kind": spec.kind}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _in_range(spec: ParamSpec, number: float) -> bool:
    return (spec.lo is None or number >= spec.lo) and (spec.hi is None or number <= spec.hi)


_SI_PREFIX = {"G": 1e9, "M": 1e6, "k": 1e3, "": 1.0, "c": 1e-2, "m": 1e-3}
_SI_UNIT_RE = re.compile(r"^(G|M|k|c|m|)(m|N|Pa|g)([1-4])?$")


def scale_hint(spec: ParamSpec, number: float) -> Optional[Dict[str, Any]]:
    """The other prefix of the declared SI unit under which an out-of-range
    figure lands in range -- the one nearest the middle of the range when
    several do -- or None."""
    m = _SI_UNIT_RE.match(spec.unit or "")
    if not m or number <= 0 or spec.lo is None or spec.hi is None or spec.hi <= 0:
        return None
    prefix, base, power = m.group(1), m.group(2), int(m.group(3) or 1)
    centre = (math.log10(max(spec.lo, spec.hi * 1e-12)) + math.log10(spec.hi)) / 2
    best: Optional[Dict[str, Any]] = None
    for other, size in _SI_PREFIX.items():
        if other == prefix:
            continue
        factor = (size / _SI_PREFIX[prefix]) ** power
        converted = number * factor
        if not _in_range(spec, converted) or converted <= 0:
            continue
        distance = abs(math.log10(converted) - centre)
        if best is None or distance < best["distance"]:
            best = {"unit": f"{other}{base}{power if power > 1 else ''}",
                    "value": converted, "exponent": round(math.log10(factor)), "distance": distance}
    if best is not None:
        best.pop("distance")
    return best


#: A signed figure as people write it: "-0.2", "1,250", ".5", "3e-4".
SIGNED_NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

#: The sign of a figure inside a sentence. A hyphen after a digit or a word
#: is a range or a compound ("10-12", "B-12"), not a minus; after a space,
#: an "x" or "by" it is the figure's sign ("15 x 8 x -0.2").
FIGURE_SIGN = r"(?:(?<![0-9.,])(?<![A-WYZa-wyz])[-+])?"

#: A figure inside a sentence, with its sign: "-0.2", "1,250", "8".
SIGNED_FIGURE = FIGURE_SIGN + r"\d[\d,]*(?:\.\d+)?"

_VALUE_UNIT_RE = re.compile(
    rf"^\s*({SIGNED_NUMBER})\s*((?:1\s*/|[A-Za-z%°µμ‰'\"]).*?)?\s*[.,;]?\s*$")
_UNIT_VALUE_RE = re.compile(
    rf"^\s*([A-Z]{{3}})\s*({SIGNED_NUMBER})\s*([A-Za-z/][\w /\-]*?)?\s*$")


def _unit_head(text: str) -> str:
    """The unit at the start of ``text``: "m long" -> "m"; the whole text
    when no leading run of words is a unit the converter knows."""
    from app.agents.base.formulas.construction_formulas_planning import parse_unit

    words = text.split()
    for end in range(len(words), 0, -1):
        head = " ".join(words[:end])
        if parse_unit(head) is not None:
            return head
    return text.strip()


def value_and_unit(value: Any) -> Optional[Tuple[float, str]]:
    """``"20 ft"`` -> (20.0, "ft"); ``{"value": 20, "unit": "ft"}`` likewise;
    ``"AED 5,000"`` -> (5000.0, "AED"). None for anything without a written unit."""
    from app.agents.base.formulas.construction_formulas_planning import normalise_rate
    if isinstance(value, dict) and "value" in value and value.get("unit"):
        number = _number(value["value"])
        if number is None and isinstance(value["value"], str):
            parsed = value_and_unit(value["value"])
            number = parsed[0] if parsed else None
        return (number, str(value["unit"]).strip()) if number is not None else None
    if not isinstance(value, str):
        return None
    m = _UNIT_VALUE_RE.match(value)
    # "SAR 1,200 per day" / "Per Day" / "perday" / "daily": the currency,
    # then what it is per, in the one spelling normalise_rate gives. A tail
    # that is not a rate is not this form.
    rate = normalise_rate(m.group(3)) if m and m.group(3) else ""
    if m and (not m.group(3) or rate.startswith("/")):
        return float(m.group(2).replace(",", "")), m.group(1) + rate
    m = _VALUE_UNIT_RE.match(value)
    if not m or not m.group(2):
        return None
    return float(m.group(1).replace(",", "")), _unit_head(m.group(2))


def conversion_context(values: Dict[str, Any]) -> Dict[str, Any]:
    """Crew size, working hours per day and working days per week, as numbers, when given."""
    from app.agents.base.formulas.construction_formulas_planning import CONVERSION_CONTEXT

    out: Dict[str, Any] = {}
    for key in CONVERSION_CONTEXT:
        raw = values.get(key)
        number = _number(raw)
        if number is None:
            parsed = value_and_unit(raw)
            number = parsed[0] if parsed else None
            if number is None and isinstance(raw, str):
                try:
                    number = float(raw.replace(",", ""))
                except ValueError:
                    number = None
        if number is not None:
            out[key] = number
    return out


def written_number(number: float) -> str:
    """``number`` as a person writes it: "3,200,000", "0.00001", "6.3"."""
    if float(number).is_integer() and abs(number) < 1e15:
        return f"{int(number):,}"
    text = f"{number:.6g}"
    if "e" in text:
        text = f"{number:.12f}".rstrip("0").rstrip(".")
    return text


def _unit_rejection(spec: ParamSpec, value: Any, reason: str, **extra: Any) -> Dict[str, Any]:
    row = _rejection(spec, value, reason)
    row.update(extra)
    return row


# Declared units that carry no dimension: a figure in them is a plain number.
_DIMENSIONLESS = frozenset({"count", "ratio"})


def convert_figure(spec: ParamSpec, number: float, unit: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One figure written in ``unit``, put into the unit ``spec`` declares by
    the base unit-conversion tool. Returns ``value`` (and ``conversion`` when
    the unit changed) or ``rejection``."""
    from app.agents.base.formulas.construction_formulas_planning import parse_unit, pe_unit_convert

    shown = f"{written_number(number)} {unit}"
    declared, written = parse_unit(spec.unit), parse_unit(unit)
    dimensionless = declared is not None and declared.dimension in _DIMENSIONLESS
    if declared is None or (dimensionless and (written is None or written.dimension
                                                in (declared.dimension, "fraction"))):
        # No unit of its own: a count, a ratio, a factor, or a figure in
        # whatever unit its companions use. The written number stands (an
        # unrecognised word after a count names what is counted; a known
        # unit of another kind, "5 m" for a count, is refused below); a
        # percentage only becomes a fraction for an input whose range is a
        # fraction.
        if (written is not None and written.dimension == "fraction"
                and spec.hi is not None and spec.hi <= 1):
            value = number * written.scale
            return {"value": value,
                    "conversion": {"parameter": spec.name, "value_in": number, "from_unit": unit,
                                   "to_unit": "fraction", "value_out": value, "note": ""}}
        return {"value": number}
    if written is None:
        if declared.dimension == "currency" or unit.strip().lower() == spec.unit.strip().lower():
            return {"value": number}
        # "5,000 per day" for a currency-per-time input: the currency is
        # implied, as it is for a bare figure on a currency input. Only a
        # unit that then reads as the declared kind is taken; it begins with
        # currency, so only a currency-per input can match it.
        implied = parse_unit(f"currency {unit}")
        if implied is not None and implied.dimension == declared.dimension:
            return convert_figure(spec, number, implied.written, context)
        return {"rejection": _unit_rejection(spec, shown, "unknown_unit", given_unit=unit)}
    if written.dimension == "currency" and declared.dimension == "currency":
        return {"value": number}
    ctx = dict(context or {})
    out = pe_unit_convert(number, unit, spec.unit, **ctx)
    if "needs_input" in out:
        return {"rejection": _unit_rejection(spec, shown, "needs_conversion_input", given_unit=unit,
                                             needs=list(out["needs_input"]))}
    if "error" in out:
        return {"rejection": _unit_rejection(spec, shown, "wrong_unit", given_unit=unit,
                                             given_dimension=written.dimension,
                                             expected_dimension=declared.dimension)}
    value = out["value_out"]
    if written.scale == declared.scale and written.dimension == declared.dimension \
            and written.offset == declared.offset and not written.needs:
        return {"value": value}
    return {"value": value,
            "conversion": {"parameter": spec.name, "value_in": number, "from_unit": unit,
                           "to_unit": spec.unit, "value_out": value, "note": out.get("note", "")}}


def convert_written_units(name: str, values: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None,
                          ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every value supplied with a unit, converted to its input's declared
    unit. Returns the values, the conversions made, and a rejection for each
    unit that is not the same kind of quantity or needs a value not given.
    A rejected value is left as supplied, so it is not also reported missing."""
    spec = get(name)
    if spec is None:
        return dict(values), [], []
    params = parameters(spec)
    stated = conversion_context(values)
    out = dict(values)
    conversions: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for key, value in values.items():
        p = params.get(key)
        if p is None or p.kind not in NUMERIC_KINDS:
            continue
        # A context value being converted is not its own context.
        ctx = {k: v for k, v in {**stated, **(context or {})}.items() if k != key}
        items = value if p.kind == "series" and isinstance(value, (list, tuple)) else [value]
        converted: List[Any] = []
        failed = False
        for item in items:
            parsed = value_and_unit(item)
            if parsed is None:
                converted.append(item)
                continue
            got = convert_figure(p, parsed[0], parsed[1], ctx)
            if "rejection" in got:
                rejected.append(got["rejection"])
                failed = True
                break
            converted.append(got["value"])
            if "conversion" in got:
                conversions.append(got["conversion"])
        if failed:
            continue
        out[key] = converted if p.kind == "series" and isinstance(value, (list, tuple)) else converted[0]
    return out, conversions, rejected


def check_inputs(name: str, values: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Supplied ``values`` for formula ``name`` held to each input's declared
    type and range. Returns the values as the formula receives them and one
    rejection per value that cannot be run."""
    spec = get(name)
    if spec is None:
        return dict(values), []
    params = parameters(spec)
    out = _place_grade_labels(params, values)
    rejected: List[Dict[str, Any]] = []
    for key, value in list(out.items()):
        p = params.get(key)
        if p is None or value is None:
            continue
        if p.kind == "flag":
            if isinstance(value, bool):
                continue
            token = str(value).strip().lower()
            if token in _YES or token in _NO:
                out[key] = token in _YES
            else:
                rejected.append(_rejection(p, value, "not_yes_no"))
            continue
        if p.kind == "series":
            items = value if isinstance(value, (list, tuple)) else [value]
            numbers = [_number(v) for v in items]
            if not items:
                rejected.append(_rejection(p, value, "empty"))
            elif any(n is None for n in numbers):
                rejected.append(_rejection(p, value, "not_a_number"))
            elif not all(_in_range(p, n) for n in numbers):
                rejected.append(_rejection(p, value, "out_of_range"))
            continue
        if p.kind not in ("number", "integer"):
            continue
        number = _number(value)
        if number is None:
            rejected.append(_rejection(p, value, "not_a_number"))
            continue
        if p.kind == "integer":
            if not number.is_integer():
                rejected.append(_rejection(p, value, "not_whole"))
                continue
            out[key] = int(number)
        if not _in_range(p, number):
            row = _rejection(p, value, "out_of_range")
            hint = scale_hint(p, number)
            if hint:
                row["scale"] = hint
            rejected.append(row)
    return out, rejected


#: Kinds a tool parameter may declare.
TOOL_KINDS = NUMERIC_KINDS + ("text", "boolean", "list", "object")


def _param_gaps(where: str, kind: str, p: Any, required: Optional[bool]) -> List[str]:
    gaps = []
    if not (p.label or "").strip():
        gaps.append(f"{where}:label")
    if not kind:
        gaps.append(f"{where}:type")
    if required is None:
        gaps.append(f"{where}:required")
    if kind in NUMERIC_KINDS:
        if not p.unit or p.unit == "-":
            gaps.append(f"{where}:unit")
        if p.lo is None or p.hi is None:
            gaps.append(f"{where}:range")
    return gaps


def schema_gaps() -> List[str]:
    """Every field a registry entry is still missing, as
    ``formula:<name>.<param>:<field>`` / ``tool:<name>[.<param>]:<field>``.
    A dimensionless numeric input declares its unit as ``ratio`` or ``count``,
    never ``-``."""
    from app.agents.core import tool_registry

    gaps: List[str] = []
    for spec in all_specs():
        if not spec.display_name.strip():
            gaps.append(f"formula:{spec.name}:display_name")
        for key, p in parameters(spec).items():
            gaps += _param_gaps(f"formula:{spec.name}.{key}", p.kind, p, p.required)
    tool_registry.load()
    for name, t in sorted(tool_registry._TOOLS.items()):
        if not t.display_name.strip():
            gaps.append(f"tool:{name}:display_name")
        if t.params is None:
            gaps.append(f"tool:{name}:params")
            continue
        for key, p in t.params.items():
            kind = p.kind if p.kind in TOOL_KINDS else ""
            gaps += _param_gaps(f"tool:{name}.{key}", kind, p, p.required)
    return sorted(gaps)


def _key_form(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


#: Magnitude words a figure may carry. The bare letter "m" is not one: it is
#: a metre.
_MAGNITUDE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "mio": 1e6,
              "billion": 1e9, "bn": 1e9}
_MAGNITUDE_RE = re.compile(
    r"^(?P<pre>\s*(?:[A-Za-z]{3}|[$€£¥])?\s*)(?P<num>[+-]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<mag>" + "|".join(sorted(_MAGNITUDE, key=len, reverse=True)) + r")\b(?P<post>.*)$",
    re.IGNORECASE)


def _with_magnitude(value: Any) -> Any:
    """"2 million" -> 2000000; "SAR 1.5 bn" -> "SAR 1500000000". Anything else unchanged."""
    if not isinstance(value, str):
        return value
    m = _MAGNITUDE_RE.match(value)
    if not m:
        return value
    number = float(m.group("num").replace(",", "")) * _MAGNITUDE[m.group("mag").lower()]
    shown = int(number) if number.is_integer() else number
    pre, post = m.group("pre").strip(), m.group("post").strip()
    return shown if not pre and not post else " ".join(x for x in (pre, str(shown), post) if x)


def bind_declared(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Arguments bound by the formula's own declarations, the same way for
    every formula: a key that names one input by its name, its declared label,
    the label's initials ("PV" for "planned value") or its name in words
    (case, spaces and separators ignored) is moved onto
    that input, and a figure written with a magnitude word is expanded. An
    input already given, or a key that could name two inputs, is left alone."""
    spec = get(name or "")
    if spec is None or not isinstance(params, dict):
        return params
    from app.lib.source_labels import parameter_words

    forms: Dict[str, List[str]] = {}
    for key, p in spec.params.items():
        words = re.findall(r"[A-Za-z]+", p.label or "")
        initials = "".join(w[0] for w in words) if len(words) >= 2 else ""
        for form in {_key_form(key), _key_form(p.label), _key_form(parameter_words(key, p.unit)),
                     _key_form(initials)}:
            if form:
                forms.setdefault(form, []).append(key)
    out = dict(params)
    for given in list(params):
        if given in spec.params:
            continue
        # A one-character key is a symbol (E, I, w): the symbol binder owns
        # it, unit conversion included.
        if len(_key_form(given)) < 2:
            continue
        targets = sorted(set(forms.get(_key_form(given), [])))
        if len(targets) == 1 and targets[0] not in out:
            out[targets[0]] = out.pop(given)
    # A magnitude word scales an amount with no physical unit (money, a
    # count, a ratio); after an engineering figure "MN" is a meganewton.
    for key, p in spec.params.items():
        if key in out and _norm_unit(p.unit) in _AMOUNT_UNITS:
            out[key] = _with_magnitude(out[key])
    return out


_AMOUNT_UNITS = frozenset({"", "-", "currency", "count", "ratio", "no", "nr"})


def _norm_unit(unit: str) -> str:
    return (unit or "").strip().lower()
