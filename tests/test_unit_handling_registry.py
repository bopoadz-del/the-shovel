"""A figure written in a unit is converted by the base unit-conversion tool.

One rule for the whole registry, so a formula added later is covered without
editing this file:

* every numeric input declares a unit the converter knows, or none ('-');
* a value written in any other unit of the same kind of quantity is
  converted to the declared unit and never refused, and the calculator gives
  the same result it gives for the declared unit;
* a value of a different kind of quantity is refused and asked for again by
  the input's display label;
* a conversion that needs a value no unit carries (crew size, working hours
  per day) asks for it by its display label, and runs once it is given;
* a minus sign written before a figure is kept, so a value below the
  input's range is refused rather than run as its positive;
* a missing input is asked for by its display label, never its id;
* a default the calculator used is stated by its display label and a value
  as a reader writes it.
"""
from __future__ import annotations

import inspect
import math

import pytest

import app.lib.construction_formulas as cf
from app.agents.base.formulas import construction_formulas_planning as units
from app.lib import formula_registry, source_labels
from app.lib.construction_formulas_quantities import parse_lwt_metres
from app.lib.formula_registry import all_specs

_parse_unit = getattr(units, "parse_unit", None)
_convert_units = getattr(units, "convert_units", None)
_convert_written = getattr(formula_registry, "convert_written_units", None)
_CONTEXT_LABELS = getattr(units, "CONVERSION_CONTEXT", {})
_unit_key = getattr(units, "unit_key", lambda token: str(token).strip().lower())
_NO_CONVERTER = ("registry", "has-no-unit-converter")
_CONTEXT = {"crew_size": 4, "hours_per_day": 8, "days_per_week": 6}

_NUMERIC = formula_registry.NUMERIC_KINDS
_SCALAR = ("number", "integer")


def _ids(pairs):
    return [f"{f}.{p}" for f, p in pairs]


def _param(formula: str, name: str):
    return formula_registry.parameters(formula_registry.get(formula))[name]


def _label(formula: str, name: str) -> str:
    return source_labels.parameter_label(formula, name)


def _numeric_inputs(kinds=_NUMERIC):
    return [(spec.name, name) for spec in all_specs()
            for name, p in formula_registry.parameters(spec).items() if p.kind in kinds]


def _dimensional_inputs(kinds=_SCALAR):
    return [(f, n) for f, n in _numeric_inputs(kinds) if _param(f, n).unit != "-"]


def _inside(p) -> float:
    """A round figure inside the declared range of ``p``."""
    lo, hi = p.lo, p.hi
    floor = lo if lo > 0 else hi * 1e-3
    middle = math.sqrt(floor * hi) if floor > 0 else (lo + hi) / 2
    middle = float(f"{middle:.2g}")
    if p.kind == "integer":
        return float(max(math.ceil(lo), round(middle)))
    return middle if lo <= middle <= hi else (lo + hi) / 2


def _unit_tables():
    tables = {family: dict(table) for family, table in getattr(units, "_FAMILIES", {}).items()}
    tables["temperature"] = {raw: scale for raw, scale in getattr(units, "_TEMPERATURE", {}).items()}
    return tables


def _scale_of(value) -> float:
    return float(value[0] if isinstance(value, tuple) else value)


def _one_per_scale(table, written: str):
    """One unit per distinct factor of ``table``, other than ``written``'s."""
    own = _unit_key(written)
    own_scale = next((_scale_of(v) for raw, v in table.items() if _unit_key(raw) == own), None)
    seen, out = {own_scale}, []
    for raw, value in table.items():
        scale = _scale_of(value)
        if scale in seen:
            continue
        seen.add(scale)
        out.append(raw)
    return out


def _family_of(written: str):
    key = _unit_key(written)
    for family, table in _unit_tables().items():
        if any(_unit_key(raw) == key for raw in table):
            return table
    return None


def _alternates(unit: str):
    """Every other unit of the same kind as ``unit``, one per factor; a
    ratio ("currency/t") varies one part at a time."""
    parts = unit.split("/")
    out = []
    for index, part in enumerate(parts):
        table = _family_of(part)
        if table is None:
            continue
        for other in _one_per_scale(table, part):
            out.append("/".join(parts[:index] + [other] + parts[index + 1:]))
    return out


def _written(number: float) -> str:
    return f"{number:.12g}"


def _same(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


# ── declarations ───────────────────────────────────────────────────────────

NUMERIC = _numeric_inputs()


@pytest.mark.parametrize("formula,name", NUMERIC, ids=_ids(NUMERIC))
def test_every_numeric_input_declares_a_unit_the_converter_knows(formula, name):
    assert _parse_unit is not None, "the base unit-conversion tool cannot read a unit"
    unit = _param(formula, name).unit
    assert unit == "-" or _parse_unit(unit) is not None, f"{formula}.{name} declares {unit!r}"


FAMILY_UNITS = ([(family, raw) for family, table in _unit_tables().items() for raw in table]
                or [_NO_CONVERTER])


@pytest.mark.parametrize("family,unit", FAMILY_UNITS, ids=[f"{f}:{u}" for f, u in FAMILY_UNITS])
def test_every_unit_round_trips_through_its_base(family, unit):
    assert _convert_units is not None, "the base tool has no unit families"
    base = next(iter(_unit_tables()[family]))
    there = _convert_units(12.5, unit, base, **_CONTEXT)
    assert "value_out" in there, there
    back = _convert_units(there["value_out"], base, unit, **_CONTEXT)
    assert math.isclose(back["value_out"], 12.5, rel_tol=1e-9), (there, back)


# ── another unit of the same kind: converted, never refused ────────────────

DIMENSIONAL = _dimensional_inputs()
CONVERTIBLE = [(f, n) for f, n in DIMENSIONAL if _alternates(_param(f, n).unit)] or [_NO_CONVERTER]


@pytest.mark.parametrize("formula,name", CONVERTIBLE, ids=_ids(CONVERTIBLE))
def test_a_value_in_another_unit_of_its_kind_is_converted(formula, name):
    assert _convert_written is not None, "no calculator input is converted by the base tool"
    p = _param(formula, name)
    value = _inside(p)
    for unit in _alternates(p.unit):
        there = _convert_units(value, p.unit, unit, **_CONTEXT)
        assert "value_out" in there, (unit, there)
        typed = f"{there['value_out']!r} {unit}"
        values, conversions, rejected = _convert_written(formula, {name: typed}, dict(_CONTEXT))
        assert rejected == [], (typed, rejected)
        assert math.isclose(values[name], value, rel_tol=1e-9, abs_tol=1e-12), (typed, values)
        assert [c["parameter"] for c in conversions] == [name], (typed, conversions)


def _baseline(spec):
    """Every input the formula requires, inside its declared range."""
    values = {}
    declared = formula_registry.parameters(spec)
    for name, param in inspect.signature(spec.fn).parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue
        p = declared.get(name)
        if p is None or p.kind not in _NUMERIC:
            return None
        values[name] = [_inside(p)] * 2 if p.kind == "series" else _inside(p)
    return values


def _metamorphic_cases():
    cases = []
    for spec in all_specs():
        base = _baseline(spec)
        if not base:
            continue
        out = cf.run_calculation(spec.name, dict(base))
        if out.get("status") != "success":
            continue
        for name in base:
            p = formula_registry.parameters(spec)[name]
            if p.kind in _SCALAR and p.unit != "-" and _alternates(p.unit):
                cases.append((spec.name, name))
    return cases


METAMORPHIC = _metamorphic_cases() or [_NO_CONVERTER]


def test_the_metamorphic_cases_span_the_formula_owners():
    owners = {formula_registry.get(f).owner for f, _n in METAMORPHIC}
    assert len(owners) >= 5, owners


@pytest.mark.parametrize("formula,name", METAMORPHIC, ids=_ids(METAMORPHIC))
def test_the_same_figure_in_another_unit_gives_the_same_result(formula, name):
    assert _convert_units is not None, "no calculator input is converted by the base tool"
    spec = formula_registry.get(formula)
    base = _baseline(spec)
    p = _param(formula, name)
    for unit in _alternates(p.unit):
        there = _convert_units(base[name], p.unit, unit, **_CONTEXT)
        figure = float(f"{there['value_out']:.3g}")
        same = _convert_units(figure, unit, p.unit, **_CONTEXT)["value_out"]
        if not p.lo <= same <= p.hi:
            continue
        expected = cf.run_calculation(formula, {**base, name: same})
        typed = f"{figure:g} {unit}"
        out = cf.run_calculation(formula, {**base, name: typed, **_context_for(formula)})
        assert out.get("status") == expected.get("status") == "success", (typed, out, expected)
        assert _same(out["result"], expected["result"]), (typed, out["result"], expected["result"])
        assert [c["parameter"] for c in out.get("conversions", [])] == [name], (typed, out)


def _context_for(formula: str):
    """The conversion context, for a calculator that does not take it as its own input."""
    taken = inspect.signature(formula_registry.get(formula).fn).parameters
    return {k: v for k, v in _CONTEXT.items() if k not in taken}


# ── another kind of quantity: refused, asked by label ──────────────────────

def _other_kind(unit: str) -> str:
    dimension = _parse_unit(unit).dimension
    return "kN" if dimension != "force" else "m2"


@pytest.mark.parametrize("formula,name", DIMENSIONAL, ids=_ids(DIMENSIONAL))
def test_a_value_of_another_kind_is_refused_and_asked_by_label(formula, name):
    assert _parse_unit is not None
    p = _param(formula, name)
    typed = f"{_written(_inside(p))} {_other_kind(p.unit)}"
    out = cf.run_calculation(formula, {name: typed})
    assert out["status"] == "error" and out.get("needs_input") is True, out
    rows = out["rejected"]
    assert name in [r["parameter"] for r in rows], out
    assert all(r["reason"] == "wrong_unit" for r in rows), out
    question = out["question"]
    assert _label(formula, name) in question, question
    if "_" in name:
        assert name not in question, question
    assert question.rstrip().endswith("?")
    assert "result" not in out


# ── a conversion that needs a value no unit carries ───────────────────────

_EFFORT = ("man-hours", "man-days", "working days")


def _needs_context():
    cases = []
    if _convert_units is None:
        return [("registry", "converter-missing", "man-hours")]
    for formula, name in DIMENSIONAL:
        p = _param(formula, name)
        for unit in _EFFORT:
            out = _convert_units(10, unit, p.unit)
            if out.get("needs_input") and name not in out["needs_input"]:
                cases.append((formula, name, unit))
    return cases


NEEDS_CONTEXT = _needs_context()


@pytest.mark.parametrize("formula,name,unit", NEEDS_CONTEXT,
                         ids=[f"{f}.{n}:{u}" for f, n, u in NEEDS_CONTEXT])
def test_a_conversion_that_needs_crew_or_hours_asks_by_label(formula, name, unit):
    assert _convert_written is not None, "no calculator input is converted by the base tool"
    p = _param(formula, name)
    needs = _convert_units(400, unit, p.unit)["needs_input"]
    values, _conv, rejected = _convert_written(formula, {name: f"400 {unit}"})
    assert [(r["parameter"], r["reason"]) for r in rejected] == [(name, "needs_conversion_input")]
    question = source_labels.input_question(formula, rejected, [])
    for need in needs:
        assert _CONTEXT_LABELS[need] in question, question
        assert need not in question, question
    assert _label(formula, name) in question and question.rstrip().endswith("?"), question

    out = cf.run_calculation(formula, {name: f"400 {unit}"})
    assert out.get("needs_input") is True, out
    assert out["question"] == source_labels.input_question(formula, rejected, out["missing"]), out

    values, conversions, rejected = _convert_written(formula, {name: f"400 {unit}"}, dict(_CONTEXT))
    assert rejected == [], rejected
    expected = _convert_units(400, unit, p.unit, **_CONTEXT)["value_out"]
    assert math.isclose(values[name], expected, rel_tol=1e-9), values
    assert conversions and conversions[0]["parameter"] == name


def test_labour_effort_to_working_days_asks_for_the_crew_size_and_working_hours():
    out = cf.run_calculation("pe_unit_convert", {"value": 400, "from_unit": "man-hours", "to_unit": "days"})
    question = out.get("question", "")
    assert "crew size" in question and "working hours per day" in question, out
    assert "crew_size" not in question and "hours_per_day" not in question
    done = cf.run_calculation("pe_unit_convert", {"value": 400, "from_unit": "man-hours",
                                                  "to_unit": "days", "crew_size": 5, "hours_per_day": 8})
    assert done["status"] == "success" and done["result"]["value_out"] == 10.0, done


# ── a figure written in a sentence ────────────────────────────────────────

def _written_in_prose(unit: str) -> bool:
    prose = {_unit_key(t) for t in getattr(units, "text_unit_tokens", list)()} | {"currency"}
    return all(_unit_key(part) in prose for part in unit.split("/"))


def _text_cases():
    cases = []
    for formula, name in CONVERTIBLE:
        p = _param(formula, name)
        for unit in filter(_written_in_prose, _alternates(p.unit)):
            there = _convert_units(_inside(p), p.unit, unit, **_CONTEXT)
            number = there.get("value_out")
            if number is None or not 1e-3 <= abs(number) <= 1e7 or "e" in _written(number):
                continue
            cases.append((formula, name, unit))
            break
    return cases


TEXT_CASES = _text_cases() if _convert_units else [(*_NO_CONVERTER, "-")]


@pytest.mark.parametrize("formula,name,unit", TEXT_CASES,
                         ids=[f"{f}.{n}:{u}" for f, n, u in TEXT_CASES])
def test_a_figure_written_in_another_unit_in_a_sentence_is_converted(formula, name, unit):
    assert _convert_units is not None, "no calculator input is converted by the base tool"
    spec = formula_registry.get(formula)
    p = _param(formula, name)
    value = _inside(p)
    number = _convert_units(value, p.unit, unit, **_CONTEXT)["value_out"]
    ask = f"Calculate {spec.display_name}: {name} = {_written(number)} {unit}."
    found = cf.extract_calculation_params_from_text(spec.fn, ask)
    got = found.get(name)
    if isinstance(got, str):
        values, _conv, rejected = _convert_written(formula, {name: got}, dict(_CONTEXT))
        assert rejected == [], (ask, got, rejected)
        got = values[name]
    assert isinstance(got, (int, float)) and math.isclose(got, value, rel_tol=1e-6), (ask, found)


# ── a minus sign is kept ──────────────────────────────────────────────────

NON_NEGATIVE = [(f, n) for f, n in _numeric_inputs(_SCALAR) if _param(f, n).lo >= 0]


def _assert_refused_below(formula: str, name: str, out: dict):
    assert out["status"] == "error" and out.get("needs_input") is True, out
    rows = [r for r in out["rejected"] if r["parameter"] == name]
    assert rows and rows[0]["reason"] == "out_of_range", out
    question = out["question"]
    assert _label(formula, name) in question, question
    if "_" in name:
        assert name not in question, question
    assert "result" not in out


@pytest.mark.parametrize("formula,name", NON_NEGATIVE, ids=_ids(NON_NEGATIVE))
def test_a_negative_figure_with_its_unit_is_refused_not_run_as_positive(formula, name):
    p = _param(formula, name)
    value = _inside(p) or (p.hi / 2)
    unit = "" if p.unit == "-" else f" {p.unit}"
    out = cf.run_calculation(formula, {name: f"-{_written(value)}{unit}"})
    _assert_refused_below(formula, name, out)


@pytest.mark.parametrize("formula,name", NON_NEGATIVE, ids=_ids(NON_NEGATIVE))
def test_a_negative_figure_in_a_sentence_keeps_its_sign(formula, name):
    spec = formula_registry.get(formula)
    p = _param(formula, name)
    value = _inside(p) or (p.hi / 2)
    unit = "" if p.unit in ("-", "currency") else f" {source_labels.unit_words(p.unit)}"
    ask = f"Calculate {spec.display_name}: {name} = -{_written(value)}{unit}."
    found = cf.extract_calculation_params_from_text(spec.fn, ask)
    if name not in found:
        pytest.skip("the figure did not bind to this input by its id")
    got = found[name]
    if isinstance(got, str):
        got = formula_registry.value_and_unit(got)[0]
    assert got < 0, (ask, found)


_SEPARATORS = ["x", "×", " by ", "*"]
_PLACEMENTS = ["each", "trailing", "none"]


def _chain(figures, separator, placement):
    if placement == "each":
        parts = [f"{n:g} m" for n in figures]
    elif placement == "trailing":
        parts = [f"{n:g}" for n in figures[:-1]] + [f"{figures[-1]:g} m"]
    else:
        parts = [f"{n:g}" for n in figures]
    sep = separator if separator.strip() in ("by",) else f" {separator} "
    return sep.join(parts)


CHAIN_CASES = [(s, pl, i) for s in _SEPARATORS for pl in _PLACEMENTS for i in range(3)]


@pytest.mark.parametrize("separator,placement,index", CHAIN_CASES,
                         ids=[f"{s.strip()}-{pl}-{i}" for s, pl, i in CHAIN_CASES])
def test_a_dimension_chain_keeps_the_sign_of_each_figure(separator, placement, index):
    figures = [15.0, 8.0, 0.2]
    figures[index] = -figures[index]
    text = f"slab {_chain(figures, separator, placement)}"
    assert parse_lwt_metres(text) == tuple(figures), text


def test_a_range_written_with_a_hyphen_is_not_a_minus():
    assert parse_lwt_metres("block 10-12 x 8 x 0.2 m") != (-12.0, 8.0, 0.2)


_FIGURES = (15.0, 8.0, 0.2)


def _chain_run(formula: str, figures) -> dict:
    spec = formula_registry.get(formula)
    ask = f"Calculate {spec.display_name} for {_chain(list(figures), ' by ', 'each')}."
    return cf.run_calculation(formula, {"text": ask})


def _with(index: int, number: float):
    figures = list(_FIGURES)
    figures[index] = number
    return figures


def _chain_cases():
    """``(formula, index)`` where the result depends on the chain's figure at ``index``."""
    out = []
    for spec in all_specs():
        done = _chain_run(spec.name, _FIGURES)
        if done.get("status") != "success":
            continue
        for index in range(3):
            moved = _chain_run(spec.name, _with(index, _FIGURES[index] * 1.5))
            if moved.get("status") == "success" and not _same(moved["result"], done["result"]):
                out.append((spec.name, index))
    return out


CHAIN_CASES_BY_FORMULA = _chain_cases()


@pytest.mark.parametrize("formula,index", CHAIN_CASES_BY_FORMULA,
                         ids=[f"{f}-{i}" for f, i in CHAIN_CASES_BY_FORMULA])
def test_a_negated_dimension_never_runs_as_its_positive(formula, index):
    positive = _chain_run(formula, _FIGURES)
    out = _chain_run(formula, _with(index, -_FIGURES[index]))
    if out.get("status") == "success":
        assert not _same(out["result"], positive["result"]), out["result"]
    else:
        assert "?" in out.get("question", ""), out


# ── a missing input is asked for by its label ─────────────────────────────

WITH_REQUIRED = [spec.name for spec in all_specs()
                 if cf.run_calculation(spec.name, {}).get("missing")]


def test_most_calculators_report_a_missing_input():
    assert len(WITH_REQUIRED) >= len(all_specs()) // 2, WITH_REQUIRED


@pytest.mark.parametrize("formula", WITH_REQUIRED)
def test_a_missing_input_is_asked_for_by_its_label(formula):
    from app.agents import runtime

    out = cf.run_calculation(formula, {})
    assert out["status"] == "error", out
    question = out.get("question")
    assert question, out
    for name in out["missing"]:
        assert _label(formula, name) in question, (name, question)
        if "_" in name:
            assert name not in question, question
    assert question.rstrip().endswith("?"), question
    agent = runtime.Agent(name="project-assistant", description="t",
                          system_prompt="t", allowed_blocks=["construction"])
    nudge = runtime._nudge_for_failed_tool({"name": "construction_calc", "result": out}, agent)
    assert question in nudge, nudge


# ── a default is stated by its label and a readable value ─────────────────

def _defaults():
    out = []
    for spec in all_specs():
        for name, param in inspect.signature(spec.fn).parameters.items():
            default = param.default
            if default is inspect.Parameter.empty or isinstance(default, bool):
                continue
            if isinstance(default, (int, float)) and float(default) != 0.0:
                out.append((spec.name, name))
            elif isinstance(default, str) and default.strip():
                out.append((spec.name, name))
    return out


DEFAULTS = _defaults()


@pytest.mark.parametrize("formula,name", DEFAULTS, ids=_ids(DEFAULTS))
def test_a_default_is_stated_by_its_label_and_a_readable_value(formula, name):
    default = inspect.signature(formula_registry.get(formula).fn).parameters[name].default
    lines = source_labels.calculator_default_lines(formula, {name: default}, "", passed={})
    lines += [line for _code, line in source_labels.calculator_currency_defaults(formula, {}, "", {})]
    if not lines:
        pytest.skip("this default is not restated (a currency code or a zero)")
    for line in lines:
        assert line.startswith("The "), line
        assert "_" not in line, line
    assert any(_label(formula, name) in line for line in lines), lines
