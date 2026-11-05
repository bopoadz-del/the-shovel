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
    governing_moment,
    reactions,
    resolve_beam_moment_calc,
    selected_condition,
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


def test_a_cantilever_never_gets_the_fixed_fixed_coefficient():
    out = answer("moment of 100 kN evenly distributed on a 10 m cantilever")
    assert "83.333" not in out and "w*L^2/12" not in out
    assert '"moment_kn_m": -500.0' in out


def test_a_classified_condition_without_a_load_errors_naming_the_reason():
    env = cf.run_calculation("beam_moment_fixed_udl", {"text": "moment on a 5 m cantilever"})
    assert env.get("status") == "error"
    assert "cantilever" in str(env.get("error") or "")


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
    "support,loads,expected",
    [
        ("cantilever", [{"kind": "triangular", "w_max": 10}], 166.667),  # w*L^2/6
        ("propped_cantilever", [{"kind": "udl", "w": 10}], 125.0),      # w*L^2/8
    ],
)
def test_untabulated_cases_are_solved_not_looked_up(support, loads, expected):
    got = governing_moment(10, loads, support)["moment_kn_m"]
    assert abs(abs(got) - expected) < 1e-2


def test_the_solver_names_the_support_condition_it_computed():
    out = governing_moment(10, [{"kind": "udl", "w": 10}], "propped_cantilever")
    assert out["support"] == "propped_cantilever"
    assert "x" in out["expression"] or "C" not in out["expression"]


def test_reactions_balance_a_udl():
    out = reactions(10, [{"kind": "udl", "w": 10}], "simply_supported")
    assert out["reaction_left_kn"] == 50.0
    assert out["reaction_right_kn"] == 50.0
    fixed = reactions(10, [{"kind": "udl", "w": 10}], "cantilever")
    assert fixed["reaction_left_kn"] == 100.0
    assert abs(fixed["moment_left_kn_m"]) == 500.0


# --- P1.3 the printed condition ---------------------------------------------


def test_the_printed_condition_equals_the_classified_condition():
    q = "moment of 100 kN evenly distributed on a 10 m cantilever"
    env = cf.run_calculation("beam_moment_fixed_udl", {"text": q})
    assert env.get("status") == "success", env
    assert env["result"]["support"] == "cantilever"
    assert env["result"]["support"] == selected_condition(q)


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


def test_an_intensity_ask_is_left_as_an_intensity():
    env = cf.run_calculation(
        "beam_governing_moment",
        {"text": "moment on a 10 m cantilever under 10 kN/m", "span_m": 10,
         "support": "cantilever", "loads": [{"kind": "udl", "w": 10}]},
    )
    assert env.get("status") == "success", env
    assert abs(env["result"]["moment_kn_m"]) == 500.0
