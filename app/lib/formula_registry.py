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

import contextvars
import inspect
import json as _json
import math
import os as _os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path
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
    #: The range begins above ``lo``: ``lo`` itself is not accepted ("more
    #: than 0"). For a quantity that must be positive but has no smallest
    #: value worth naming.
    above_lo: bool = False
    #: A tool parameter's type (``number``, ``integer``, ``series``, ``text``,
    #: ``boolean``, ``list``, ``object``); a formula's comes from its signature.
    kind: str = ""
    #: A tool parameter's required / optional; a formula's comes from its
    #: signature (no default = required).
    required: Optional[bool] = None
    #: For an amount that can be stated either way, which one this input
    #: takes (``BASES``): this period's amount or the total to date. Empty
    #: when the amount has only one reading.
    basis: str = ""
    #: The knowledge-read constant this input falls back to when it is not
    #: given (a :class:`KnowledgeConstant`; see its ``as_input``). A
    #: description of the input names that entry and its value.
    default_from: Any = None
    #: The input this one is "per": its unit's denominator is what that
    #: input states (a consumption "per m2 of work" is per the quantity; a
    #: price "per m3" is per what the consumption yields). The registry then
    #: converts a denominator of the same kind into the other input's unit
    #: and refuses one of another kind. Empty: no other input names it.
    per: str = ""

    def __post_init__(self) -> None:
        if self.basis and self.basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, not {self.basis!r}")

    @property
    def display_label(self) -> str:
        return self.label


#: The bases an amount input can declare: the amount for this period alone,
#: or the total to date (which includes every earlier period).
BASES = ("period", "cumulative")


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
    basis: str = ""
    above_lo: bool = False
    per: str = ""

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
    #: Alternative-required input sets: at least one name in each inner group
    #: must be supplied (e.g. EVM accepts ``pv`` OR ``bcws``). Declared on the
    #: formula so the requirement lives with the kernel, not a central table.
    required_groups: Tuple[Tuple[str, ...], ...] = ()
    #: The published standard / code the formula implements, declared with
    #: the kernel (empty: none declared). Half of what "certified" needs.
    standard: str = ""
    #: Catalogue declarations (all optional; a missing one is listed in the
    #: catalogue entry's ``gaps``): when the formula is the right one, when it
    #: is not and what to use instead -- type-level conditions only -- and
    #: worked examples with invented figures (``declared_examples``).
    when_to_use: str = ""
    when_not: str = ""
    examples: Tuple[Dict[str, Any], ...] = ()


_REGISTRY: Dict[str, FormulaSpec] = {}


#: Registry-level name -> unit DEFAULTS: a last-resort fallback consulted only
#: when a parameter carries no declared unit and its name suffix infers none.
#: Every entry whose parameter the registry now declares is redundant (see
#: tests/test_kt1b_unit_fallback.py), so this table shrinks toward deletion as
#: declarations cover the names; it must never override a declared unit.
NAME_UNIT_DEFAULTS: Dict[str, str] = {
    "pv": "currency", "ev": "currency", "ac": "currency",
    "bcws": "currency", "bcwp": "currency", "acwp": "currency", "bac": "currency",
    "rate_percent": "%", "contract_amount": "currency",
    "water_depth": "m", "raft_thickness": "m", "floor_count": "floors",
    "floor_thickness": "m", "time_days": "d", "time_constant_days": "d",
    "ultimate_shrinkage_microstrain": "µε", "udl_w_kn_m": "kN/m",
    "central_point_load_kn": "kN", "quantity_m3": "m3", "w_c_ratio": "ratio",
    "floor_area_m2": "m2", "panel_length": "m", "wall_thickness": "m",
    "excavation_depth": "m", "total_cost": "currency", "area": "m2",
    "gross_valuation": "currency", "temperature_history_c": "°C",
    "time_intervals_hours": "h", "axial_load_kn": "kN", "column_diameter_mm": "mm",
}


def name_unit_default(name: str) -> str:
    """The last-resort name -> unit default for ``name`` (``""`` if none)."""
    return NAME_UNIT_DEFAULTS.get(str(name), "")


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
    required_groups: Tuple[Tuple[str, ...], ...] = (),
    standard: str = "",
    when_to_use: str = "",
    when_not: str = "",
    examples: Tuple[Dict[str, Any], ...] = (),
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
        shown = declared_examples("formula", key, examples)
        existing = _REGISTRY.get(key)
        if existing is not None and existing.fn is not fn:
            raise ValueError(f"formula {key!r} is declared twice")
        params = {k: _declared_param(v) for k, v in inputs.items()}
        _REGISTRY[key] = FormulaSpec(key, fn, owner, description.strip(),
                                     display_name.strip(),
                                     {k: p.unit for k, p in params.items()},
                                     dict(outputs), params, main_output,
                                     tuple(tuple(g) for g in required_groups),
                                     (standard or "").strip(),
                                     (when_to_use or "").strip(), (when_not or "").strip(),
                                     shown)
        fn.__formula__ = _REGISTRY[key]  # type: ignore[attr-defined]
        return fn

    return _register


#: The keys a declared example carries. ``invented`` must be True: an
#: example's figures are made up for the example, never a document's figure
#: and never a test question's.
EXAMPLE_KEYS = ("invented", "given", "chain", "result")


def declared_examples(kind: str, name: str, examples: Any) -> Tuple[Dict[str, Any], ...]:
    """Check a registry entry's declared ``examples``: a sequence of dicts,
    each with ``invented: True``, a non-empty ``given`` and ``result`` text
    and an optional ``chain`` list of strings, and no other key. Raises
    ``ValueError`` at declaration on anything else."""
    if isinstance(examples, (str, bytes, dict)) or (examples is not None and not hasattr(examples, "__iter__")):
        raise ValueError(f"{kind} {name!r}: examples must be a sequence of dicts")
    out = []
    for example in examples or ():
        if not isinstance(example, dict):
            raise ValueError(f"{kind} {name!r}: an example must be a dict")
        extra = set(example) - set(EXAMPLE_KEYS)
        if extra:
            raise ValueError(f"{kind} {name!r}: unknown example keys {sorted(extra)}")
        if example.get("invented") is not True:
            raise ValueError(f"{kind} {name!r}: an example must say invented: True")
        for key in ("given", "result"):
            if not isinstance(example.get(key), str) or not example[key].strip():
                raise ValueError(f"{kind} {name!r}: an example needs a {key} text")
        chain = example.get("chain", [])
        if not isinstance(chain, (list, tuple)) or not all(isinstance(c, str) for c in chain):
            raise ValueError(f"{kind} {name!r}: an example's chain is a list of strings")
        out.append({"invented": True, "given": example["given"].strip(),
                    "chain": list(chain), "result": example["result"].strip()})
    return tuple(out)


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


@dataclass(frozen=True)
class Constant:
    """A fixed number a calculator's body applies (a unit weight, a code
    factor, a waste allowance): its value, unit, what it is, and the written
    source that states it. ``source`` None means no source is recorded yet;
    the result then says the constant still needs one (D2), never hides it."""

    value: float
    unit: str
    meaning: str
    source: Optional[str] = None

    @classmethod
    def from_knowledge(cls, entry_id: str, field_path: str, meaning: str, *, within: Param) -> "KnowledgeConstant":
        """A constant whose value the shipped knowledge base states.

        ``field_path`` is a dotted path inside the entry (``thresholds.x``).
        ``within`` is the calculator input the figure stands in for: its
        unit is the constant's unit and its declared range is the range the
        figure must fall in, exactly as a value the caller gives would.
        Nothing is read here: the value is read from the entry every time a
        calculator applies it (see :meth:`KnowledgeConstant.resolve`), so
        the run's constants record names the figure and the entry it came
        from, and an entry that is missing or broken fails the one
        calculator that applies it, never the module that declares it."""
        if not isinstance(within, Param):
            raise TypeError("a knowledge-read constant must name the input it stands in for")
        return KnowledgeConstant(entry_id, field_path, meaning, within)

    def resolve(self) -> "Constant":
        """The constant as applied now: a literal constant is itself."""
        return self


def _range_words(lo: Optional[float], hi: Optional[float], unit: str, above_lo: bool = False) -> str:
    """The declared range as a reader sees it (``1 to 5 ratio``; with a
    lower bound that is not itself accepted, ``more than 0, up to 5 ratio``)."""
    if lo is not None and hi is not None:
        span = f"more than {lo:g}, up to {hi:g}" if above_lo else f"{lo:g} to {hi:g}"
    elif lo is not None:
        span = f"more than {lo:g}" if above_lo else f"at least {lo:g}"
    else:
        span = f"at most {hi:g}"
    return f"{span} {unit}" if unit and unit != "-" else span


@dataclass(frozen=True)
class KnowledgeConstant:
    """A declared constant whose number lives in a shipped knowledge entry,
    read when it is applied, never copied into the calculator at import."""

    entry_id: str
    field_path: str
    meaning: str
    #: The calculator input this figure stands in for (unit and range).
    within: Param

    @property
    def unit(self) -> str:
        return self.within.unit

    def as_input(self) -> Param:
        """The input declaration whose unstated value is this constant."""
        return replace(self.within, default_from=self)

    def _fault(self, what: str) -> LookupError:
        return LookupError(f"knowledge entry {self.entry_id!r} {what}")

    def resolve(self) -> Constant:
        """Read the entry now. A missing entry or field, a value that is not
        a number, a number that is not finite (NaN, infinity, too large for
        a float) or a number outside the input's declared range raises,
        naming the entry: a calculator never falls back to a figure nobody
        wrote down, never applies one no calculation can use, and a fault in
        the knowledge base is never reported as a fault in the user's input."""
        from app.blocks import _knowledge

        entry = _knowledge.shipped_entry(self.entry_id)
        if entry is None:
            raise self._fault("is not in the shipped knowledge base")
        node: Any = entry
        for key in self.field_path.split("."):
            if not isinstance(node, dict) or key not in node:
                raise self._fault(f"has no field {self.field_path!r}")
            node = node[key]
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            raise self._fault(f"field {self.field_path!r} is not a number")
        try:
            value = float(node)
        except OverflowError:
            value = math.inf
        if not math.isfinite(value):
            raise self._fault(f"field {self.field_path!r} is not a finite number")
        lo, hi, above = self.within.lo, self.within.hi, self.within.above_lo
        if (lo is not None and (value <= lo if above else value < lo)) or (hi is not None and value > hi):
            raise self._fault(
                f"field {self.field_path!r} states {value:g}, outside what the calculator "
                f"accepts for {self.within.label or 'this input'} "
                f"({_range_words(lo, hi, self.unit, above)})"
            )
        prov = entry.get("provenance") or {}
        written = prov.get("course_reference") or prov.get("source")
        source = f"knowledge entry {self.entry_id}" + (f" ({written})" if written else "")
        return Constant(value, self.unit, self.meaning, source=source)

    def describe(self) -> str:
        """Where an unstated input's value comes from, for a description
        offered before the run: ``from knowledge entry X (1.2)``. When the
        entry cannot be applied the reason is given instead of a value."""
        try:
            return f"from knowledge entry {self.entry_id} ({self.resolve().value:g})"
        except LookupError as exc:
            return f"from knowledge entry {self.entry_id}, which cannot be applied: {exc}"


_CONSTANT_READS: "contextvars.ContextVar[Optional[List[Tuple[str, Constant]]]]" = (
    contextvars.ContextVar("formula_constant_reads", default=None)
)


class Constants:
    """A calculator module's declared constants, read as attributes::

        _K = Constants(soil_unit_weight=Constant(18.0, "kN/m3", "soil unit weight"))
        net_q = q - depth_m * _K.soil_unit_weight

    A read returns the plain value. While the door records (see
    :func:`recording_constants`), each read is noted, so the result states
    exactly the constants this run applied -- not every one declared.
    """

    def __init__(self, **declared: Constant) -> None:
        for key, c in declared.items():
            if not isinstance(c, (Constant, KnowledgeConstant)):
                raise TypeError(f"constant {key} must be declared as a Constant")
        object.__setattr__(self, "_declared", dict(declared))

    def __getattr__(self, name: str) -> float:
        declared = object.__getattribute__(self, "_declared")
        if name not in declared:
            raise AttributeError(name)
        c = declared[name].resolve()
        reads = _CONSTANT_READS.get()
        if reads is not None:
            reads.append((name, c))
        return c.value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("declared constants are read-only")

    def declared(self) -> Dict[str, Union[Constant, KnowledgeConstant]]:
        return dict(object.__getattribute__(self, "_declared"))


class recording_constants:
    """Context manager: the rows of every declared constant read inside it,
    once each, in first-read order (``rows`` after exit)."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self._token: Any = None
        self._reads: List[Tuple[str, Constant]] = []

    def __enter__(self) -> "recording_constants":
        self._token = _CONSTANT_READS.set(self._reads)
        return self

    def __exit__(self, *exc: Any) -> None:
        _CONSTANT_READS.reset(self._token)
        seen: set = set()
        for name, c in self._reads:
            if name in seen:
                continue
            seen.add(name)
            row: Dict[str, Any] = {
                "name": name, "value": c.value, "unit": c.unit, "meaning": c.meaning,
            }
            if c.source:
                row["source"] = c.source
            else:
                row["needs_source"] = True
            self.rows.append(row)


def written_method(fn: Callable[..., Any]) -> str:
    """The method a calculator states in its own words: its docstring's first
    paragraph plus every paragraph that writes an equation (holds ``=``).

    A coefficient or factor the body applies (the 384 of 5wL^4/384EI, a
    partial factor) is stated where the calculator writes its formula; the
    door carries that text so the result says how it was computed. Paragraphs
    with no equation after the first are left out. Empty when the calculator
    has no docstring.
    """
    doc = inspect.getdoc(inspect.unwrap(fn)) or ""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", doc)]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return ""
    kept = [paragraphs[0]] + [p for p in paragraphs[1:] if "=" in p]
    return "\n".join(kept)


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
            basis=declared.basis, above_lo=declared.above_lo, per=declared.per,
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
            "reason": reason, "lo": spec.lo, "hi": spec.hi, "kind": spec.kind,
            "above_lo": spec.above_lo}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _in_range(spec: ParamSpec, number: float) -> bool:
    # Written as "is it inside", never "is it outside": NaN compares False to
    # every bound, so it is inside no range that has one.
    if spec.lo is not None and not (number > spec.lo if spec.above_lo else number >= spec.lo):
        return False
    return spec.hi is None or number <= spec.hi


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
    rf"^\s*({SIGNED_NUMBER})\s*((?:1\s*/|/|[A-Za-z%°µμ‰'\"]).*?)?\s*[.,;]?\s*$")
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

#: Dimensions that measure an amount of work/material. A "count" input here
#: means "units of work, as per the rate" (m2 of blockwork, m3 of concrete),
#: so a stated measurement in one of these is accepted and carried through.
#: "count" itself is the declared dimension and is handled above without a
#: conversion; any other written dimension (a force, a time, a fraction, ...)
#: is refused.
_COUNTABLE_MEASURES = frozenset({"length", "area", "volume", "mass"})


def _measure_fills_count(declared: Any, written: Any, chained: bool = False) -> bool:
    """True when the written unit is the declared one with a countable
    measure (length, area, volume, mass) standing in each place the declared
    unit says ``count`` -- and in no other place. Both units must have the
    same number of parts; a unit equal to the declared one is not a fill.
    A measure stands in a denominator ("per m3") only for an input that
    declares what it is ``per`` (``chained``): only then is that measure
    checked against another input's unit (:func:`_chain_units`)."""
    if declared is None or written is None:
        return False
    want, have = declared.dimension.split("/"), written.dimension.split("/")
    if len(want) != len(have) or want == have:
        return False
    return all(w == h or (w == "count" and h in _COUNTABLE_MEASURES and (i == 0 or chained))
               for i, (w, h) in enumerate(zip(want, have)))


def _noun_fills_count(declared: Any, unit: str, chained: bool = False) -> bool:
    """True when ``unit``, read part by part ("blocks per m2" -> blocks, m2),
    fills the declared unit with a counted noun -- a word the converter does
    not know -- in at least one place the declared unit says ``count``, and
    every other part is the declared kind or a countable measure as
    :func:`_measure_fills_count` allows. The noun names what is counted, as
    an unrecognised word after a plain count does. A noun or a measure stands
    in a denominator only for an input that declares what it is ``per``
    (``chained``). A part that is not a plain word ("m33") is not a noun."""
    from app.agents.base.formulas.construction_formulas_planning import normalise_rate, parse_unit

    if declared is None:
        return False
    want = declared.dimension.split("/")
    parts = [part.strip() for part in normalise_rate(unit).split("/")]
    if len(parts) != len(want) or any(not part for part in parts):
        return False
    nouns = 0
    for i, (w, part) in enumerate(zip(want, parts)):
        atom = parse_unit(part)
        if atom is not None and atom.dimension == w:
            continue
        if w != "count" or not (i == 0 or chained):
            return False
        if atom is None and part.isalpha():
            nouns += 1
        elif atom is None or atom.dimension not in _COUNTABLE_MEASURES:
            return False
    return nouns > 0


def _count_mark(unit: str) -> Optional[str]:
    """The standard mark for a written count unit ("each", "ea", "pcs" ->
    "nr", by the bill-of-quantities unit table), or None when ``unit`` is
    not one."""
    from app.agents.base.formulas.construction_formulas_planning import parse_unit
    from app.lib.boq_units import canon_unit

    mark = canon_unit(unit)
    atom = parse_unit(mark)
    return mark if atom is not None and atom.dimension == "count" else None


def convert_figure(spec: ParamSpec, number: float, unit: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One figure written in ``unit``, put into the unit ``spec`` declares by
    the base unit-conversion tool. Returns ``value`` (and ``conversion`` when
    the unit changed) or ``rejection``."""
    from app.agents.base.formulas.construction_formulas_planning import parse_unit, pe_unit_convert

    shown = f"{written_number(number)} {unit}"
    declared, written = parse_unit(spec.unit), parse_unit(unit)
    dimensionless = declared is not None and declared.dimension in _DIMENSIONLESS
    if _measure_fills_count(declared, written, bool(spec.per)):
        # A declared "count" stands for whatever is counted, so each count
        # part of the declared unit may be a measure the user states ("1,200
        # m3" for a count, "12 m2/day" for a count per day, "0.022 m3/m2" for
        # a count per count, "310 AED/m3" for currency per count). The
        # number is the figure and the stated unit is carried through.
        return {"value": number,
                "conversion": {"parameter": spec.name, "value_in": number,
                               "from_unit": unit, "to_unit": unit, "value_out": number,
                               "note": "count carries the stated unit"}}
    if declared is None or (dimensionless and (written is None or written.dimension
                                                in (declared.dimension, "fraction"))):
        # No unit of its own: a count, a ratio, a factor, or a figure in
        # whatever unit its companions use. The written number stands (an
        # unrecognised word after a count names what is counted; a measure
        # was carried above). A unit of a non-measurable kind ("5 kN" for a
        # count) is refused below. A
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
        if any(_noun_fills_count(declared, text, bool(spec.per))
               for text in (unit, f"currency {unit}")):
            # "12.5 blocks per m2" for a count per count, "3.20 per block"
            # for a price per count: the noun names what is counted.
            return {"value": number,
                    "conversion": {"parameter": spec.name, "value_in": number,
                                   "from_unit": unit, "to_unit": unit, "value_out": number,
                                   "note": "count carries the stated unit"}}
        # "3.20 each" for a price per count: a price written against a count
        # mark is a price per that mark.
        mark = _count_mark(unit)
        implied = parse_unit(f"currency/{mark}" if mark else f"currency {unit}")
        if implied is not None and (implied.dimension == declared.dimension
                                    or _measure_fills_count(declared, implied, bool(spec.per))):
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


def _stated_parts(value: Any) -> Optional[Tuple[str, str]]:
    """The unit ``value`` is stated in, split into what it measures and what
    it is per: "5 L/m2" -> ("L", "m2"), "12 m2" -> ("m2", ""), "9 per
    m3" -> ("currency", "m3"). None when no unit the converter knows is stated."""
    from app.agents.base.formulas.construction_formulas_planning import parse_unit

    parsed = value_and_unit(value)
    if parsed is None:
        return None
    unit = parse_unit(parsed[1]) or parse_unit(f"currency {parsed[1]}")
    if unit is None:
        return None
    head, _, tail = unit.written.partition("/")
    return head.strip(), tail.strip()


def _counted_parts(value: Any) -> Optional[Tuple[str, str]]:
    """As :func:`_stated_parts`, for a unit with a counted noun in a place
    ("12 blocks per m2" -> ("blocks", "m2"), "3 per block" -> ("currency",
    "block")): each part is a unit the converter knows or a plain word.
    None when a part is neither."""
    from app.agents.base.formulas.construction_formulas_planning import normalise_rate

    parsed = value_and_unit(value)
    if parsed is None:
        return None
    for text in (parsed[1], f"currency {parsed[1]}"):
        parts = [part.strip() for part in normalise_rate(text).split("/")]
        if 1 <= len(parts) <= 2 and all(_part_unit(part) is not None for part in parts):
            return parts[0], (parts[1] if len(parts) == 2 else "")
    return None


def _part_unit(part: str) -> Any:
    """One part of a stated unit: the unit the converter reads, or for a
    plain word it does not know (a counted noun, "blocks") a count."""
    from app.agents.base.formulas.construction_formulas_planning import parse_unit

    unit = parse_unit(part)
    if unit is None and part.isalpha():
        return parse_unit("count")
    return unit


def _chain_units(params: Dict[str, ParamSpec], values: Dict[str, Any], out: Dict[str, Any],
                 conversions: List[Dict[str, Any]], rejected: List[Dict[str, Any]]) -> None:
    """Hold every input that declares what it is ``per`` to that input's
    stated unit. Where both state a unit, the denominator of this one and
    what the other measures must be the same kind: the same kind converts
    (a price per m3 against a consumption in litres becomes a price per
    litre), another kind is refused. Where either is a bare figure there is
    nothing stated to check."""
    from app.agents.base.formulas.construction_formulas_planning import pe_unit_convert

    refused = {row["parameter"] for row in rejected}
    for key, p in params.items():
        other = params.get(p.per) if p.per else None
        if other is None or key in refused or p.per in refused:
            continue
        mine, theirs = [_stated_parts(values.get(k)) or _counted_parts(values.get(k))
                        for k in (key, p.per)]
        if not mine or not theirs or not mine[1]:
            continue
        per_unit, measured = _part_unit(mine[1]), _part_unit(theirs[0])
        if per_unit is None or measured is None:
            continue
        if per_unit.dimension != measured.dimension:
            row = _rejection(p, values.get(key), "unit_chain")
            row.update(per=p.per, per_value=values.get(p.per), unit_per=mine[1],
                       per_measures=theirs[0], given_dimension=per_unit.dimension,
                       per_dimension=measured.dimension)
            rejected.append(row)
            out[key] = values.get(key)
            continue
        if per_unit.scale == measured.scale:
            continue
        # One unit of what the other input measures, in this input's
        # denominator: a price per m3 is 0.001 of itself per litre.
        factor = pe_unit_convert(1.0, theirs[0], mine[1]).get("value_out")
        if factor is None:
            continue
        value_in = out[key]
        out[key] = value_in * factor
        conversions.append({"parameter": key, "value_in": value_in,
                            "from_unit": f"{mine[0]}/{mine[1]}", "to_unit": f"{mine[0]}/{theirs[0]}",
                            "value_out": out[key],
                            "note": f"per the {other.label or p.per} in {theirs[0]}"})


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
    _chain_units(params, values, out, conversions, rejected)
    # Currency flows from the data: a money (or money-per-time) value carries
    # its currency in its unit ("SAR/day", "AED"). When the formula takes a
    # currency label and none was supplied, adopt the currency of the first
    # such value. Nothing is invented -- no currency anywhere stays absent.
    if "currency" in params and not str(values.get("currency") or "").strip():
        from app.lib.construction_formulas import _CURRENCY_CODES
        codes = {c.upper() for c in _CURRENCY_CODES}
        for ckey, cval in values.items():
            if params.get(ckey) is None:
                continue
            parsed = value_and_unit(cval)
            if parsed is None:
                continue
            head = parsed[1].split("/", 1)[0].strip().upper()
            if head in codes:
                out["currency"] = head
                break
    return out, conversions, rejected


def check_inputs(name: str, values: Dict[str, Any],
                 written: Optional[Dict[str, Any]] = None,
                 ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Supplied ``values`` for formula ``name`` held to each input's declared
    type and range. Returns the values as the formula receives them and one
    rejection per value that cannot be run. ``written`` holds the values as
    the user wrote them, before any unit conversion: a refusal quotes the
    figure and unit the user gave, not the converted one."""
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
            given = (written or {}).get(key)
            stated = value_and_unit(given)
            if stated is not None and stated[0] != number:
                # Converted from the unit the user wrote: quote that, and
                # offer no rescaling of a figure the user never saw.
                row = _rejection(p, given, "out_of_range")
                row["written"] = [stated[0], stated[1]]
                rejected.append(row)
                continue
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
        params = parameters(spec)
        for key, p in params.items():
            gaps += _param_gaps(f"formula:{spec.name}.{key}", p.kind, p, p.required)
            if p.per and (p.per == key or p.per not in params):
                gaps.append(f"formula:{spec.name}.{key}:per")
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


# ---------------------------------------------------------------------------
# Certified means verified (KS-1).
#
# A formula is *certified* only when BOTH hold:
#   * it declares the published standard it implements (``standard=`` on
#     ``@formula``), and
#   * it has at least one hand-value case -- inputs, the expected outputs
#     worked by hand from that standard, and the working -- and the kernel
#     reproduces every one of its cases right now.
# Nothing sets "certified" by hand: it is derived here from the declaration
# and the kernel's own output on the hand cases, so a kernel that drifts from
# its hand value loses the label on the next run.
#
# Hand-value cases are verification data, not product data: they live in
# tests/hand_values/*.json (override with FORMULA_HAND_VALUES_DIR), one list of
# cases per file:
#   [{"formula": "<registered name>", "inputs": {...},
#     "expect": {"<output key>": <value>, ...}, "rel_tol": 0.001,
#     "working": "<the hand working, step by step>"}]
# A missing or unreadable directory certifies nothing (fail closed).
# ---------------------------------------------------------------------------

#: Relative tolerance a hand case is checked to unless it states its own.
HAND_VALUE_REL_TOL = 1e-3


def hand_values_dir() -> _Path:
    """Where the hand-value cases are read from."""
    override = (_os.getenv("FORMULA_HAND_VALUES_DIR") or "").strip()
    if override:
        return _Path(override)
    return _Path(__file__).resolve().parents[2] / "tests" / "hand_values"


def _case_problem(row: Any) -> str:
    """The first field a hand case is missing ("" when it is well formed)."""
    if not isinstance(row, dict):
        return "not_an_object"
    if not str(row.get("formula") or "").strip():
        return "formula"
    if not isinstance(row.get("inputs"), dict):
        return "inputs"
    if not isinstance(row.get("expect"), dict) or not row["expect"]:
        return "expect"
    if not str(row.get("working") or "").strip():
        return "working"
    return ""


@_lru_cache(maxsize=8)
def _load_hand_values(directory: str) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    """(cases, problems) from every ``*.json`` file in ``directory``. Each case
    carries the file and position it came from."""
    cases: List[Dict[str, Any]] = []
    problems: List[str] = []
    root = _Path(directory)
    if not root.is_dir():
        return (), ("hand_value_dir:missing",)
    for path in sorted(root.glob("*.json")):
        try:
            rows = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"hand_value:{path.name}:unreadable:{type(exc).__name__}")
            continue
        if not isinstance(rows, list):
            problems.append(f"hand_value:{path.name}:not_a_list")
            continue
        for i, row in enumerate(rows):
            problem = _case_problem(row)
            if problem:
                problems.append(f"hand_value:{path.name}[{i}]:{problem}")
                continue
            cases.append({**row, "_file": path.name, "_index": i})
    return tuple(cases), tuple(problems)


def clear_hand_value_cache() -> None:
    """Forget loaded cases and verdicts (a test that writes a directory)."""
    _load_hand_values.cache_clear()
    _verdict.cache_clear()


def hand_value_cases(name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every well-formed hand case, or only those for formula ``name``."""
    cases, _problems = _load_hand_values(str(hand_values_dir()))
    return [c for c in cases if name is None or c["formula"] == name]


def _output(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        if key in result:
            return result[key]
        inner = result.get("result")
        if isinstance(inner, dict) and key in inner:
            return inner[key]
    return None


def _matches(actual: Any, expected: Any, rel_tol: float) -> bool:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return actual == expected
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=1e-9)


def _case_tolerance(case: Dict[str, Any]) -> float:
    tol = case.get("rel_tol")
    if isinstance(tol, (int, float)) and not isinstance(tol, bool) and tol > 0:
        return float(tol)
    return HAND_VALUE_REL_TOL


def hand_value_failures(name: str) -> List[str]:
    """Why the kernel of ``name`` does not reproduce its hand cases (empty when
    every case is reproduced, or when there is no case -- see
    :func:`verified_by_hand`)."""
    spec = get(name)
    failures: List[str] = []
    for case in hand_value_cases(name):
        where = f"{case['_file']}[{case['_index']}]"
        if spec is None:
            failures.append(f"{where}: no registered formula {name!r}")
            continue
        try:
            result = spec.fn(**case["inputs"])
        except Exception as exc:  # noqa: BLE001 -- a raising kernel is a failed case, said so
            failures.append(f"{where}: kernel raised {type(exc).__name__}: {exc}")
            continue
        tol = _case_tolerance(case)
        for key, expected in case["expect"].items():
            actual = _output(result, key)
            if not _matches(actual, expected, tol):
                failures.append(f"{where}: {key} = {actual!r}, hand value {expected!r}")
    return failures


@_lru_cache(maxsize=512)
def _verdict(directory: str, name: str) -> bool:
    return bool(hand_value_cases(name)) and not hand_value_failures(name)


def verified_by_hand(name: str) -> bool:
    """True when ``name`` has at least one hand case and its kernel reproduces
    every one of them."""
    return _verdict(str(hand_values_dir()), str(name or ""))


def certified(name: str) -> bool:
    """Derived, never declared: a registered formula that states its standard
    AND reproduces its hand values."""
    spec = get(str(name or ""))
    return bool(spec is not None and spec.standard and verified_by_hand(spec.name))


def certification_gaps() -> List[str]:
    """The registry check for "certified means verified":

    * ``formula:<name>:standard_or_hand_value`` -- declares neither a standard
      nor a hand case;
    * ``formula:<name>:hand_value_mismatch`` -- a hand case the kernel does not
      reproduce (never acceptable, whatever else the formula declares);
    * ``hand_value:<file>...`` -- a malformed case, or one naming no
      registered formula.
    """
    cases, problems = _load_hand_values(str(hand_values_dir()))
    gaps: List[str] = list(problems)
    names = {spec.name for spec in all_specs()}
    for case in cases:
        if case["formula"] not in names:
            gaps.append(f"hand_value:{case['_file']}[{case['_index']}]:unknown_formula")
    for spec in all_specs():
        has_cases = bool(hand_value_cases(spec.name))
        if not spec.standard and not has_cases:
            gaps.append(f"formula:{spec.name}:standard_or_hand_value")
        if has_cases and hand_value_failures(spec.name):
            gaps.append(f"formula:{spec.name}:hand_value_mismatch")
    return sorted(gaps)
