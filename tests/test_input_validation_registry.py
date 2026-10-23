"""Every calculator input is typed, has a unit and a range, and is held to them.

One rule for the whole registry, so a formula added later is covered without
editing this file:

* every input of every registered formula is declared with a kind, a unit
  and -- for numeric kinds -- an inclusive range whose bounds hold its default;
* a supplied value outside that range, of the wrong type, or not a whole
  number where a count is asked for is refused, the calculator does not run,
  and the user is asked again by the input's display label, never its id;
* a grade label (C30, S355, B500B) binds to the input of its own material
  and to no other input; a code designation (ACI 318, BS 8110) binds nothing.
"""
from __future__ import annotations

import asyncio
import inspect
import math

import pytest

import app.lib.construction_formulas as cf
from app.lib import formula_registry
from app.lib.formula_registry import all_specs

_PARAMETERS = getattr(formula_registry, "parameters", None)


_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _signature_pairs():
    return [(spec.name, name) for spec in all_specs()
            for name, p in inspect.signature(spec.fn).parameters.items()
            if p.kind not in _VARIADIC]


def _declared(kinds=None, grade=None):
    """``(formula, input)`` pairs read from the registry declarations."""
    if _PARAMETERS is None:
        return [("registry", "declares-no-parameters")]
    out = []
    for spec in all_specs():
        for name, p in _PARAMETERS(spec).items():
            if kinds is not None and p.kind not in kinds:
                continue
            if grade is not None and p.grade != grade:
                continue
            out.append((spec.name, name))
    return out


def _param(formula: str, name: str):
    assert _PARAMETERS is not None, "the registry declares no typed parameters"
    spec = formula_registry.get(formula)
    assert spec is not None, formula
    return _PARAMETERS(spec)[name]


def _ids(pairs):
    return [f"{f}.{p}" for f, p in pairs]


def _run_alone(formula: str, values: dict) -> dict:
    return cf.run_calculation(formula, values)


def _label(formula: str, name: str) -> str:
    from app.lib.source_labels import parameter_label

    return parameter_label(formula, name)


# ── declarations ───────────────────────────────────────────────────────────

SIGNATURE_PAIRS = _signature_pairs()


@pytest.mark.parametrize("formula,name", SIGNATURE_PAIRS, ids=_ids(SIGNATURE_PAIRS))
def test_every_input_is_declared_with_a_kind_unit_and_range(formula, name):
    spec = formula_registry.get(formula)
    declared = getattr(spec, "params", {}).get(name)
    assert isinstance(declared, getattr(formula_registry, "Param", ())), (
        f"{formula}.{name} has no Param declaration")
    p = _param(formula, name)
    assert p.kind in ("number", "integer", "series", "flag", "text", "table")
    assert p.unit, f"{formula}.{name} declares no unit"
    if p.kind in formula_registry.NUMERIC_KINDS:
        assert p.lo is not None and p.hi is not None, f"{formula}.{name} has no range"
        assert p.lo < p.hi, f"{formula}.{name} range {p.lo}..{p.hi} is empty"
        default = p.default
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            assert p.lo <= default <= p.hi, (
                f"{formula}.{name} default {default} is outside {p.lo}..{p.hi}")


@pytest.mark.parametrize("formula,name", SIGNATURE_PAIRS, ids=_ids(SIGNATURE_PAIRS))
def test_the_declared_unit_feeds_the_formula_unit_map(formula, name):
    spec = formula_registry.get(formula)
    assert spec.inputs.get(name) == _param(formula, name).unit


# ── out of range: refused, asked by label ──────────────────────────────────

SCALAR_NUMERIC = _declared(kinds=("number", "integer"))


def _outside(p, side: str):
    span = max(abs(p.hi - p.lo), 1.0)
    if side == "above":
        value = p.hi + max(span * 10, abs(p.hi) * 10, 1000.0)
        return math.ceil(value) if p.kind == "integer" else value
    value = p.lo - max(span, abs(p.lo), 1.0)
    return math.floor(value) if p.kind == "integer" else value


@pytest.mark.parametrize("side", ["above", "below"])
@pytest.mark.parametrize("formula,name", SCALAR_NUMERIC, ids=_ids(SCALAR_NUMERIC))
def test_an_out_of_range_value_is_refused_and_asked_by_label(formula, name, side):
    p = _param(formula, name)
    bad = _outside(p, side)
    out = _run_alone(formula, {name: bad})
    assert out["status"] == "error", out
    assert out.get("needs_input") is True, out
    rejected = out["rejected"]
    assert rejected and all(r["reason"] == "out_of_range" and r["value"] == bad
                            for r in rejected), out
    question = out["question"]
    for row in rejected:
        assert _label(formula, row["parameter"]) in question, (row, question)
        if "_" in row["parameter"]:
            assert row["parameter"] not in question, question
    assert question.rstrip().endswith("?")
    assert "result" not in out


SERIES = _declared(kinds=("series",))


@pytest.mark.parametrize("formula,name", SERIES, ids=_ids(SERIES))
def test_an_out_of_range_series_element_is_refused(formula, name):
    p = _param(formula, name)
    out = _run_alone(formula, {name: [p.lo, _outside(p, "above")]})
    assert out.get("needs_input") is True, out
    assert _label(formula, name) in out["question"]


@pytest.mark.parametrize("formula,name", SERIES, ids=_ids(SERIES))
def test_an_empty_series_is_asked_for(formula, name):
    out = _run_alone(formula, {name: []})
    assert out.get("needs_input") is True, out
    assert "at least one number" in out["question"]


# ── wrong type ─────────────────────────────────────────────────────────────

INTEGERS = _declared(kinds=("integer",))
FLAGS = _declared(kinds=("flag",))
NUMBERS = _declared(kinds=("number",))


@pytest.mark.parametrize("formula,name", INTEGERS, ids=_ids(INTEGERS))
def test_a_count_needs_a_whole_number(formula, name):
    p = _param(formula, name)
    inside = p.lo + min(1.5, (p.hi - p.lo) / 2)
    if float(inside).is_integer():
        inside += 0.5
    _values, rejected = formula_registry.check_inputs(formula, {name: inside})
    assert [(r["parameter"], r["reason"]) for r in rejected] == [(name, "not_whole")]


@pytest.mark.parametrize("formula,name", INTEGERS, ids=_ids(INTEGERS))
def test_a_whole_float_count_is_passed_as_an_integer(formula, name):
    p = _param(formula, name)
    whole = float(math.ceil(p.lo))
    values, rejected = formula_registry.check_inputs(formula, {name: whole})
    assert rejected == []
    assert isinstance(values[name], int)


@pytest.mark.parametrize("formula,name", FLAGS, ids=_ids(FLAGS))
def test_a_flag_needs_a_yes_or_a_no(formula, name):
    values, rejected = formula_registry.check_inputs(formula, {name: "yes"})
    assert rejected == [] and values[name] is True
    _values, rejected = formula_registry.check_inputs(formula, {name: "perhaps"})
    assert [(r["parameter"], r["reason"]) for r in rejected] == [(name, "not_yes_no")]


@pytest.mark.parametrize("formula,name", NUMBERS, ids=_ids(NUMBERS))
def test_a_number_input_refuses_words(formula, name):
    _values, rejected = formula_registry.check_inputs(formula, {name: "several"})
    assert [(r["parameter"], r["reason"]) for r in rejected] == [(name, "not_a_number")]


# ── grade labels go to their own material ──────────────────────────────────

_FAMILY_EXAMPLES = {"concrete": ("C30", 30), "steel": ("S355", 355), "rebar": ("B500B", 500)}
GRADED = [pair for material in _FAMILY_EXAMPLES for pair in _declared(grade=material)]


@pytest.mark.parametrize("formula,name", GRADED, ids=_ids(GRADED))
def test_a_grade_label_binds_to_the_input_of_its_material(formula, name):
    p = _param(formula, name)
    label, strength = _FAMILY_EXAMPLES[p.grade]
    spec = formula_registry.get(formula)
    found = cf.extract_calculation_params_from_text(
        spec.fn, f"Calculate {spec.display_name} using grade {label}.")
    expected = label if p.kind == "text" else strength
    assert found.get(name) == expected, found
    others = {k: v for k, v in found.items() if k != name and v in (strength, label)}
    assert not others, f"{label} also bound to {others}"


@pytest.mark.parametrize("formula,name", GRADED, ids=_ids(GRADED))
def test_a_bare_strength_given_to_a_grade_input_is_accepted(formula, name):
    p = _param(formula, name)
    label, strength = _FAMILY_EXAMPLES[p.grade]
    values, rejected = formula_registry.check_inputs(formula, {name: label})
    assert rejected == [], rejected
    assert values[name] == (label if p.kind == "text" else strength)


ALL_FORMULAS = [spec.name for spec in all_specs()]


@pytest.mark.parametrize("label,strength,material", [
    ("C30", 30, "concrete"), ("C35/45", 35, "concrete"),
    ("S355", 355, "steel"), ("B500B", 500, "rebar"),
])
@pytest.mark.parametrize("formula", ALL_FORMULAS)
def test_a_grade_label_never_binds_to_another_input(formula, label, strength, material):
    spec = formula_registry.get(formula)
    found = cf.extract_calculation_params_from_text(
        spec.fn, f"Calculate {spec.display_name} for {label}.")
    for key, value in found.items():
        if value in (strength, label) or (isinstance(value, str) and label in value):
            assert _param(formula, key).grade == material, (
                f"{label} bound to {formula}.{key}={value!r}")


@pytest.mark.parametrize("designation,number", [
    ("ACI 318", 318), ("BS 8110", 8110), ("EN 1992", 1992),
    ("ASTM C39", 39), ("BS EN 206", 206),
])
@pytest.mark.parametrize("formula", ALL_FORMULAS)
def test_a_code_designation_binds_no_figure(formula, designation, number):
    spec = formula_registry.get(formula)
    found = cf.extract_calculation_params_from_text(
        spec.fn, f"Calculate {spec.display_name} to {designation}.")
    figures = {k: v for k, v in found.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool) and v == number}
    assert not figures, f"{designation} bound as {figures}"


# ── the question reaches the user ──────────────────────────────────────────

def test_the_question_offers_the_scale_that_fits_the_range():
    out = _run_alone("beam_deflection_cantilever_point_load",
                     {"p_kn": 10, "span_m": 4, "ec_mpa": 200_000, "i_mm4": 3e-4})
    assert out.get("needs_input") is True, out
    assert "If that figure is in m4" in out["question"]
    assert "1e12" in out["question"]


def test_the_predispatched_calculator_answers_with_its_question():
    from app.agents import runtime

    agent = runtime.Agent(name="project-assistant", description="t",
                          system_prompt="t", allowed_blocks=["construction"])
    ask = "Calculate the risk score: probability 10, impact 3."
    pre = asyncio.run(runtime._predispatch_formula_calc(
        agent, [{"role": "user", "content": ask}], "proj", operator_text=ask))
    assert pre and pre["name"] == "construction_calc", pre
    answer = runtime._calc_input_question(pre)
    assert "probability" in answer and "1 to 5" in answer and answer.endswith("?"), answer


def test_the_model_is_told_to_ask_not_to_retry_with_its_own_value():
    from app.agents import runtime

    agent = runtime.Agent(name="project-assistant", description="t",
                          system_prompt="t", allowed_blocks=["construction"])
    env = _run_alone("score_risk", {"probability": 10, "impact": 3})
    nudge = runtime._nudge_for_failed_tool({"name": "construction_calc", "result": env}, agent)
    assert env["question"] in nudge
    assert "Do not call it again" in nudge


def test_a_value_inside_the_range_still_runs():
    out = _run_alone("score_risk", {"probability": 3, "impact": 4})
    assert out["status"] == "success" and out["result"]["score"] == 12
