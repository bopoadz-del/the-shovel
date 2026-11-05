"""Beam solver (P1 of the structural-formulas order) + selection binding.

The owner's live defect: a cantilever ask ran beam_moment_fixed_udl and
printed 83.333 kN.m labelled "Maximum" for a member whose governing moment is
500 kN.m -- a 6x under-estimate of a design moment. These tests pin the
mechanism, not the transcript:

* a recognised support condition BINDS the calculator (P1.1);
* the solver reproduces every closed form -- they are the oracle, and CI
  fails when the two disagree (P1.2);
* untabulated cases (triangular cantilever, propped cantilever under UDL)
  are SOLVED from the boundary conditions, not looked up (P1.2);
* the printed support condition equals the classified condition (P1.3);
* a total load over a length resolves to an intensity by dimensional form
  (force / length), never by phrasing (P1.4).
"""

from __future__ import annotations

import json

import pytest

from app.agents.hats.design.formulas import beam_solver as solver
from app.agents.hats.design.formulas.beam_solver import (
    _finite_in_range,
    beam_governing_moment,
    beam_reactions,
    governing_moment,
    reactions,
    resolve_beam_moment_calc,
    selected_condition,
)
from app.agents.hats.design.formulas.construction_formulas_beam_analysis import (
    beam_moment_fixed_udl,
    beam_moment_point_load,
    beam_moment_simple,
)
from app.lib import construction_formulas as cf


def answer(q: str, name: str = "beam_moment_fixed_udl") -> str:
    """The calc envelope for ``q``, through the live run path with the wrong
    pick ``name`` -- exactly the defective route from the transcript."""
    env = cf.run_calculation(name, {"text": q})
    return json.dumps(env, default=str, sort_keys=True)


# --- P1.1 selection binding --------------------------------------------------


@pytest.mark.parametrize(
    "q,expect",
    [
        ("moment of 100 kN evenly distributed on a 10 m cantilever", "cantilever"),
        ("moment of 100 kN evenly distributed on a 10 m beam fixed both ends",
         "fixed_fixed"),
        ("moment on a propped cantilever of 6 m", "propped_cantilever"),
        ("moment on a simply supported beam of 6 m", "simply_supported"),
        ("moment of 100 kN on a 10 m beam", None),
    ],
)
def test_support_condition_binds_the_calculator(q, expect):
    assert selected_condition(q) == expect


@pytest.mark.parametrize(
    "q,expected_calc",
    [
        ("moment of 100 kN evenly distributed on a 10 m cantilever",
         "beam_governing_moment"),
        ("moment of 100 kN evenly distributed on a 10 m beam fixed both ends",
         "beam_moment_fixed_udl"),
    ],
)
def test_the_resolver_binds_by_the_classified_condition(q, expected_calc):
    name, params = resolve_beam_moment_calc("beam_moment_fixed_udl", {"text": q})
    assert name == expected_calc
    assert params.get("udl_w_kn_m") == 10.0 or params.get("loads")


@pytest.mark.parametrize(
    "q",
    [
        "moment of 100 kN evenly distributed on a 10 m beam",
        "the moment from 200 kN spread across an 8 m member",
    ],
)
def test_an_ask_with_no_support_condition_keeps_the_callers_pick(q):
    name, params = resolve_beam_moment_calc("beam_moment_fixed_udl", {"text": q})
    assert name == "beam_moment_fixed_udl"
    assert "loads" not in params


@pytest.mark.parametrize(
    "params",
    [
        {"udl_w_kn_m": 10, "span_m": 10, "text": "moment on a simply supported beam"},
        {"udl_w_kn_m": 6, "span_m": 4, "text": "moment on a simply supported beam"},
    ],
)
def test_the_resolver_reads_numbers_from_params(params):
    name, out = resolve_beam_moment_calc("beam_moment_fixed_udl", params)
    assert name == "beam_moment_simple"
    assert out.get("udl_w_kn_m") == params["udl_w_kn_m"]


def test_a_cantilever_never_gets_the_fixed_fixed_coefficient():
    out = answer("moment of 100 kN evenly distributed on a 10 m cantilever")
    assert "83.333" not in out and "w*L^2/12" not in out
    assert '"moment_kn_m": -500.0' in out


@pytest.mark.parametrize(
    "q,condition",
    [
        ("moment on a 5 m cantilever", "cantilever"),
        ("the moment for a propped cantilever of 6 m", "propped_cantilever"),
    ],
)
def test_a_classified_condition_without_a_load_errors_naming_the_reason(q, condition):
    env = cf.run_calculation("beam_moment_fixed_udl", {"text": q})
    assert env.get("status") == "error"
    assert condition in str(env.get("error") or "")


# --- P1.2 the solver and its oracles ----------------------------------------


def closed_form(support: str, w: float, L: float) -> float:
    """The textbook closed form, as the oracle for that support condition."""
    if support == "simply_supported":
        return w * L * L / 8.0
    if support == "fixed_fixed":
        return w * L * L / 12.0
    if support == "cantilever":
        return w * L * L / 2.0
    raise AssertionError(f"no closed form for {support}")


@pytest.mark.parametrize(
    "support,w,L,expected",
    [
        ("simply_supported", 10, 10, 125.0),
        ("simply_supported", 6, 4, 12.0),
        ("fixed_fixed", 10, 10, 83.333),
        ("fixed_fixed", 6, 4, 8.0),
        ("cantilever", 10, 10, 500.0),
        ("cantilever", 6, 4, 48.0),
    ],
)
def test_solver_reproduces_every_closed_form(support, w, L, expected):
    got = governing_moment(L, [{"kind": "udl", "w": w}], support)["moment_kn_m"]
    assert abs(abs(got) - expected) < 1e-3


@pytest.mark.parametrize(
    "support,w,L",
    [("cantilever", 10, 10), ("simply_supported", 6, 4)],
)
def test_closed_form_and_solver_never_disagree(support, w, L):
    got = governing_moment(L, [{"kind": "udl", "w": w}], support)["moment_kn_m"]
    assert abs(abs(got) - closed_form(support, w, L)) < 1e-6


@pytest.mark.parametrize(
    "calc,params,expected,condition",
    [
        (beam_moment_simple, {"udl_w_kn_m": 10, "span_m": 10}, 125.0, "simply_supported"),
        (beam_moment_simple, {"udl_w_kn_m": 6, "span_m": 4}, 12.0, "simply_supported"),
        (beam_moment_fixed_udl, {"udl_w_kn_m": 10, "span_m": 10}, 83.333, "fixed_fixed"),
        (beam_moment_fixed_udl, {"udl_w_kn_m": 6, "span_m": 4}, 8.0, "fixed_fixed"),
        (beam_moment_point_load, {"point_load_kn": 100, "span_m": 10}, 250.0, "simply_supported"),
        (beam_moment_point_load, {"point_load_kn": 100, "span_m": 10, "distance_from_left_m": 2.5}, 187.5, "simply_supported"),
    ],
)
def test_the_closed_forms_report_their_support_condition(
    calc, params, expected, condition,
):
    out = calc(**params)
    assert out.get("support_condition") == condition
    assert abs(out.get("max_moment_kn_m") - expected) < 1e-3


@pytest.mark.parametrize(
    "calc,bad_params",
    [
        (beam_moment_simple, {"udl_w_kn_m": 10, "span_m": 0}),
        (beam_moment_fixed_udl, {"udl_w_kn_m": 10, "span_m": 0}),
        (beam_moment_point_load, {"point_load_kn": 100, "span_m": 0}),
        (beam_moment_point_load, {"point_load_kn": 100, "span_m": 10, "distance_from_left_m": 20}),
    ],
)
def test_the_closed_forms_keep_their_error_guards(calc, bad_params):
    out = calc(**bad_params)
    assert "error" in out


@pytest.mark.parametrize(
    "support,loads,expected",
    [
        ("cantilever", [{"kind": "triangular", "w_max": 10}], 166.667),  # w*L^2/6
        ("propped_cantilever", [{"kind": "udl", "w": 10}], 125.0),      # w*L^2/8
    ],
)
def test_untabulated_cases_are_solved_not_looked_up(support, loads, expected):
    got = governing_moment(10, loads, support)["moment_kn_m"]
    assert abs(abs(got) - expected) < 1e-2


@pytest.mark.parametrize("bad_kind", ["nonsense", "helicopter"])
def test_an_unknown_load_kind_errors_naming_it(bad_kind):
    with pytest.raises(ValueError, match="unknown load kind"):
        governing_moment(10, [{"kind": bad_kind, "w": 5}], "cantilever")


def test_overhang_errors_naming_the_missing_support_positions():
    with pytest.raises(ValueError, match="overhang requires"):
        governing_moment(10, [{"kind": "udl", "w": 10}], "overhang")


@pytest.mark.parametrize("support", ["cantilever", "simply_supported"])
def test_a_non_positive_span_errors_naming_it(support):
    with pytest.raises(ValueError, match="span must be > 0"):
        governing_moment(0, [{"kind": "udl", "w": 10}], support)


@pytest.mark.parametrize("support", ["cantilever", "simply_supported"])
def test_reactions_reject_a_non_positive_span(support):
    with pytest.raises(ValueError, match="span must be > 0"):
        reactions(0, [{"kind": "udl", "w": 10}], support)


@pytest.mark.parametrize("value,lo,hi,expected", [
    (2.5, 0.0, 4.0, 2.5),
    (8.0, 0.0, 4.0, None),
])
def test_finite_in_range_filters_by_the_range(value, lo, hi, expected):
    assert _finite_in_range(value, lo, hi) == expected


@pytest.mark.parametrize("support", ["propped_cantilever", "fixed_fixed"])
def test_the_solver_names_the_support_condition_it_computed(support):
    out = governing_moment(10, [{"kind": "udl", "w": 10}], support)
    assert out["support"] == support
    assert "x" in out["expression"] or "C" not in out["expression"]


@pytest.mark.parametrize(
    "w,L,expected",
    [(10, 10, 50.0), (6, 4, 12.0)],
)
def test_reactions_balance_a_udl(w, L, expected):
    out = beam_reactions(L, "simply_supported", [{"kind": "udl", "w": w}])
    assert out["reaction_left_kn"] == expected
    assert out["reaction_right_kn"] == expected


@pytest.mark.parametrize(
    "w,L,expected_shear,expected_moment",
    [(10, 10, 100.0, 500.0), (6, 4, 24.0, 48.0)],
)
def test_cantilever_reactions_and_fixed_moment(w, L, expected_shear, expected_moment):
    fixed = beam_reactions(L, "cantilever", [{"kind": "udl", "w": w}])
    assert fixed["reaction_left_kn"] == expected_shear
    assert abs(fixed["moment_left_kn_m"]) == expected_moment


@pytest.mark.parametrize("loads", [
    {"kind": "udl", "w": 10},
    {"kind": "udl", "w": 6},
])
def test_the_solver_accepts_a_single_load_dict(loads):
    out = beam_governing_moment(10, "cantilever", loads)
    assert out.get("moment_kn_m") == -(loads["w"] * 100 / 2)


# --- P1.3 the printed condition ---------------------------------------------


@pytest.mark.parametrize(
    "q,condition,key",
    [
        ("moment of 100 kN evenly distributed on a 10 m cantilever", "cantilever", "support"),
        ("moment of 100 kN evenly distributed on a 10 m beam fixed both ends", "fixed_fixed", "support_condition"),
    ],
)
def test_the_printed_condition_equals_the_classified_condition(q, condition, key):
    env = cf.run_calculation("beam_moment_fixed_udl", {"text": q})
    assert env.get("status") == "success", env
    assert env["result"][key] == condition
    assert env["result"][key] == selected_condition(q)


# --- P1.4 dimensional-form operand resolution --------------------------------


@pytest.mark.parametrize(
    "phrasing",
    [
        "moment of 100 kN evenly distributed on a 10 m cantilever",
        "on the cantilever, 100 kN spread evenly over the 10 m span: the moment",
    ],
)
def test_total_load_over_length_resolves_by_dimension(phrasing):
    env = cf.run_calculation("beam_moment_fixed_udl", {"text": phrasing})
    assert env.get("status") == "success", env
    assert abs(env["result"]["moment_kn_m"]) == 500.0


@pytest.mark.parametrize(
    "span_m,w,expected",
    [(10, 10, 500.0), (4, 6, 48.0)],
)
def test_an_intensity_ask_is_left_as_an_intensity(span_m, w, expected):
    env = cf.run_calculation(
        "beam_governing_moment",
        {"text": f"moment on a {span_m} m cantilever under {w} kN/m", "span_m": span_m,
         "support": "cantilever", "loads": [{"kind": "udl", "w": w}]},
    )
    assert env.get("status") == "success", env
    assert abs(env["result"]["moment_kn_m"]) == expected
