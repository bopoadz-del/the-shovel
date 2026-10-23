"""S2b: every figure in a dimension chain keeps its own unit, and a gate never
leaves a calculator turn empty.

Owner UI re-check on 061aa54: "slab 50 ft by 20 ft by 8 in" bound length 50,
width 20 and thickness 8 with one unit, ft, for all three. The "in" was lost
and the calculator reported 8,000 m3, reading the ft figures as metres. The
first ask came back as an empty turn: the figure gate removed the model's own
(correct) figures because they did not match the calculator's (wrong) result,
that sentence was the whole answer, and the coverage footer then hid the empty
answer from the blank-answer guard.

Every case below is generated from the converter's unit table and the formula
registry. None names a project, a document or a question.
"""
from __future__ import annotations

import itertools
import json
import math
import random

import pytest

from app.agents.base.formulas.construction_formulas_planning import (
    _FAMILIES,
    convert_units,
    parse_unit,
)
from app.lib import construction_formulas as cf
from app.lib import formula_registry
from app.lib.construction_formulas_quantities import dimension_chains, parse_lwt_metres

#: Units a building dimension is written in (a mile-long slab is rejected by range, rightly).
_LENGTH_UNITS = sorted(u for u in _FAMILIES["length"] if convert_units(1.0, u, "m")["value_out"] <= 10)
_FIGURES = (50.0, 20.0, 8.0)
_SEPARATORS = (" by ", " x ", " × ", "*")


def _metres(value: float, unit: str) -> float:
    return convert_units(value, unit, "m")["value_out"]


def _chain(figures, units, sep):
    return sep.join(f"{f:g} {u}" if u else f"{f:g}" for f, u in zip(figures, units))


_rng = random.Random(20261009)
_UNIT_TRIPLES = sorted({tuple(_rng.sample(_LENGTH_UNITS, 3)) for _ in range(60)}
                       | {(u, u, v) for u, v in itertools.permutations(("ft", "in", "m", "mm"), 2)})


# ── the chain: one unit per figure ──────────────────────────────────────────

@pytest.mark.parametrize("units", _UNIT_TRIPLES, ids="-".join)
@pytest.mark.parametrize("sep", _SEPARATORS, ids=lambda s: s.strip() or "star")
def test_each_figure_in_a_chain_converts_by_its_own_unit(units, sep):
    text = f"slab {_chain(_FIGURES, units, sep)}"
    got = parse_lwt_metres(text)
    want = tuple(_metres(f, u) for f, u in zip(_FIGURES, units))
    assert got is not None and all(math.isclose(g, w, rel_tol=1e-9) for g, w in zip(got, want)), (text, got)


@pytest.mark.parametrize("unit", _LENGTH_UNITS)
@pytest.mark.parametrize("bare", [(0, 1), (1, 2), (0, 2), (0,), (1,), (2,)])
def test_a_bare_figure_takes_the_nearest_unit_written_in_its_chain(unit, bare):
    units = [None if i in bare else unit for i in range(3)]
    [chain] = dimension_chains(f"slab {_chain(_FIGURES, units, ' x ')}")
    assert [u for _f, u in chain] == [unit] * 3


def test_a_chain_with_no_unit_stays_as_written():
    assert parse_lwt_metres("raft 30 x 20 x 1.5") == (30.0, 20.0, 1.5)


# ── the call: a bare figure is in the unit the ask wrote for it ─────────────

@pytest.mark.parametrize("units", _UNIT_TRIPLES, ids="-".join)
def test_a_call_that_loses_per_figure_units_still_computes_in_them(units):
    """The model passes bare figures and one unit for all three; the ask's own
    units win and the result is in the unit the calculator reports."""
    text = f"What is the concrete volume of a slab {_chain(_FIGURES, units, ' by ')}?"
    call = {"length": _FIGURES[0], "width": _FIGURES[1], "thickness": _FIGURES[2],
            "unit": units[0], "text": text, "waste_factor": 0}
    out = cf.run_calculation("concrete_volume", call)
    assert out["status"] == "success", out
    want = math.prod(_metres(f, u) for f, u in zip(_FIGURES, units))
    assert math.isclose(out["result"]["net_volume_m3"], round(want, 3), rel_tol=1e-3, abs_tol=1e-3), (call, out)


def _unit_inputs():
    """(formula, input, declared unit, another unit of that kind, a unit of another kind)."""
    kinds: dict[str, list[str]] = {}
    for table in _FAMILIES.values():
        for unit in table:
            parsed = parse_unit(unit)
            written = cf._units_written_for(7.0, f"is 7 {unit} here") == [unit]
            if parsed is not None and not parsed.needs and not parsed.offset and written:
                kinds.setdefault(parsed.dimension, []).append(unit)
    other_kind = {dim: next(u for d, us in kinds.items() if d != dim for u in us) for dim in kinds}
    cases = []
    for spec in formula_registry.all_specs():
        for p in formula_registry.parameters(spec).values():
            if p.kind not in ("number", "integer") or not p.unit:
                continue
            target = parse_unit(p.unit)
            if target is None or target.needs or target.offset or target.dimension not in kinds:
                continue
            alts, factors = [], set()
            for u in kinds[target.dimension]:
                f = convert_units(1.0, u, p.unit).get("value_out")
                if f not in (None, 1.0) and round(f, 9) not in factors:
                    factors.add(round(f, 9))
                    alts.append(u)
            if alts:
                cases.append((spec.name, p.name, p.unit, alts, other_kind[target.dimension]))
    return cases


_UNIT_INPUTS = _unit_inputs()


def _attach(case, text, call_unit=None, figure=7.0):
    name, param = case[0], case[1]
    fn = cf.CALCULATORS[name]
    return cf._attach_written_units(name, fn, {param: figure}, {param: figure}, text, call_unit)[param]


@pytest.mark.parametrize("case", _UNIT_INPUTS, ids=lambda c: f"{c[0]}.{c[1]}")
def test_a_bare_figure_takes_the_unit_the_ask_wrote_for_it(case):
    alt = case[3][0]
    assert _attach(case, f"the value is 7 {alt} here") == f"7.0 {alt}"


@pytest.mark.parametrize("case", _UNIT_INPUTS, ids=lambda c: f"{c[0]}.{c[1]}")
def test_a_unit_of_another_kind_is_not_the_figure_unit(case):
    other = case[4]
    assert _attach(case, f"the value is 7 {other} here") == 7.0


@pytest.mark.parametrize("case", _UNIT_INPUTS, ids=lambda c: f"{c[0]}.{c[1]}")
def test_the_call_unit_applies_when_the_ask_writes_none(case):
    alt, other = case[3][0], case[4]
    assert _attach(case, "the value is 7", call_unit=alt) == f"7.0 {alt}"
    assert _attach(case, "the value is 7", call_unit=other) == 7.0


@pytest.mark.parametrize("case", [c for c in _UNIT_INPUTS if len(c[3]) > 1][::3], ids=lambda c: f"{c[0]}.{c[1]}")
def test_two_units_written_for_one_figure_leave_it_as_passed(case):
    first, second = case[3][:2]
    assert _attach(case, f"either 7 {first} or 7 {second}") == 7.0
    assert _attach(case, f"either 7 {first} or 7 {second}", call_unit=second) == f"7.0 {second}"


@pytest.mark.parametrize("case", _UNIT_INPUTS[::5], ids=lambda c: f"{c[0]}.{c[1]}")
def test_a_figure_already_converted_by_the_caller_is_left_alone(case):
    alt = case[3][0]
    assert _attach(case, f"the value is 7 {alt} here", figure=7.5) == 7.5


# ── a gate never leaves a calculator turn empty ─────────────────────────────

def _calc_turn(units):
    from app.agents import runtime as rt

    ask = f"What is the concrete volume of a slab {_chain(_FIGURES, units, ' by ')}?"
    envelope = cf.run_calculation("concrete_volume", {"text": ask})
    messages = [
        {"role": "user", "content": ask},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {
            "name": "construction_calc", "arguments": json.dumps({"calculation": "concrete_volume"})}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "construction_calc",
         "content": rt._tool_result_content(envelope)},
    ]
    return messages, envelope["result"]


_GATES = ("_cost_grounding_gate", "_calc_figure_grounding_gate")


@pytest.mark.parametrize("gate", _GATES + ("citation_provenance",))
@pytest.mark.parametrize("units", _UNIT_TRIPLES[::6], ids="-".join)
def test_an_answer_a_gate_empties_becomes_the_calculator_result(units, gate, monkeypatch):
    from app.agents import citation_provenance
    from app.agents import runtime as rt

    messages, result = _calc_turn(units)
    if gate in _GATES:
        monkeypatch.setattr(rt, gate, lambda text, *a, **k: "")
    else:
        monkeypatch.setattr(citation_provenance, "gate", lambda text, *a, **k: "")
    out = rt._postprocess_answer("Every clause of this answer is removed.", None, messages)
    body = rt._strip_coverage_footer(out)
    assert body and body != rt._SYNTH_CUTOFF_NOTICE, out
    assert f"{result['net_volume_m3']:g}" in body.replace(",", ""), out


def test_the_live_empty_turn_is_answered_by_the_calculator():
    """The turn as it ran on 061aa54: the calculator read ft as m (8,000 m³),
    the model wrote its own converted figure, and the figure gate removed the
    only sentence. The user saw the coverage footer alone."""
    from app.agents import runtime as rt

    ask = "What is the concrete volume of a slab 50 ft by 20 ft by 8 in?"
    envelope = cf.run_calculation("concrete_volume", {"length_m": 50, "width_m": 20, "thickness_m": 8})
    messages = [
        {"role": "user", "content": ask},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {
            "name": "construction_calc", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "construction_calc",
         "content": rt._tool_result_content(envelope)},
    ]
    answer = ("The calculator returned 8,000 m³ (8,400 m³ with 5% waste), but it read the feet as metres; "
              "the correct volume is 18.88 m³ (19.82 m³ with waste).")
    assert rt._calc_figure_grounding_gate(answer, messages).strip() == ""
    out = rt._postprocess_answer(answer, None, messages, project_id="s2b-unindexed-project")
    assert rt._COVERAGE_FOOTER_RE.search(out), out
    body = rt._strip_coverage_footer(out)
    assert body and body != rt._SYNTH_CUTOFF_NOTICE, out


@pytest.mark.parametrize("footer", ["0 of 0 project documents indexed", "3 of 12 project documents indexed"])
def test_a_coverage_footer_alone_is_not_an_answer(footer):
    from app.agents import runtime as rt

    messages, result = _calc_turn(("ft", "ft", "in"))
    out = rt._nonblank_after_empty_synthesis(footer, messages)
    assert out.endswith(footer) and rt._strip_coverage_footer(out), out
    assert f"{result['net_volume_m3']:g}" in out.replace(",", "")
    plain = rt._nonblank_after_empty_synthesis(footer, [{"role": "user", "content": "hello"}])
    assert rt._strip_coverage_footer(plain) == rt._EMPTY_RESPONSE_FALLBACK


def test_an_answer_with_a_body_and_a_footer_is_unchanged():
    from app.agents import runtime as rt

    text = "The slab is 18.88 m³.\n0 of 0 project documents indexed"
    assert rt._nonblank_after_empty_synthesis(text, []) == text
