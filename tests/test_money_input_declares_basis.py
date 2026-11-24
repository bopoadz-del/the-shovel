"""KS-2: an amount input says whether it takes this period's amount or the
total to date, and the calculator offer the model reads prints it.

Live misread this guards against: a "valuation to date" figure was bound to
an input that takes one period's gross, so earlier payments were never taken
off. The offer's labels said "gross valuation (currency, ...)" with no basis.
Every test builds its own input; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.agents.driver import tools as driver_tools
from app.lib import formula_registry as fr


@pytest.mark.parametrize("basis", ["period", "cumulative"])
def test_a_known_basis_is_accepted(basis):
    assert fr.Param("currency", 0, 1e9, label="amount", basis=basis).basis == basis


@pytest.mark.parametrize("basis", ["to date", "monthly"])
def test_an_unknown_basis_is_refused_at_declaration(basis):
    """``Param.__post_init__`` refuses it, so a typo in a formula's
    declaration fails at import instead of printing no basis."""
    with pytest.raises(ValueError, match="basis must be one of"):
        fr.Param("currency", 0, 1e9, label="amount", basis=basis)


@pytest.mark.parametrize("unit", ["currency", "%"])
def test_no_basis_is_the_default(unit):
    assert fr.Param(unit).basis == ""


def _spec(name: str, basis: str, required: bool = True) -> fr.ParamSpec:
    return fr.ParamSpec(name=name, kind="number", unit="currency", lo=0, hi=1e9,
                        label=name.replace("_", " "), grade="", required=required,
                        basis=basis)


@pytest.mark.parametrize("basis", ["period", "cumulative"])
def test_the_offer_line_states_the_declared_basis(basis):
    line = driver_tools._input_line("amount_x", _spec("amount_x", basis))
    assert driver_tools.BASIS_WORDS[basis] in line
    other = next(b for b in fr.BASES if b != basis)
    assert driver_tools.BASIS_WORDS[other] not in line


@pytest.mark.parametrize("required", [True, False])
def test_an_input_without_a_basis_states_none(required):
    line = driver_tools._input_line("amount_y", _spec("amount_y", "", required))
    assert not any(words in line for words in driver_tools.BASIS_WORDS.values())
    assert ("optional" in line) is (not required)


def _two_amounts(this_period: float, to_date: float = 0.0) -> dict:
    return {"this_period": this_period, "to_date": to_date}


@pytest.mark.parametrize("bases", [("period", "cumulative"), ("cumulative", "period")])
def test_parameters_carry_each_declared_basis(bases):
    spec = fr.FormulaSpec(
        name="two_amounts_probe", fn=_two_amounts, owner=fr.BASE,
        description="d", display_name="Two amounts",
        params={
            "this_period": fr.Param("currency", 0, 1e9, label="a", basis=bases[0]),
            "to_date": fr.Param("currency", 0, 1e9, label="b", basis=bases[1]),
        },
    )
    got = fr.parameters(spec)
    assert (got["this_period"].basis, got["to_date"].basis) == bases


@pytest.mark.parametrize("hat", [None, "commercial"])
def test_the_registered_period_payment_input_says_so_in_the_offer(hat):
    """The calculator that nets one period's gross declares that basis, and
    the catalogue both chat paths read prints it on that input's line."""
    _specs, lines = driver_tools.calculator_catalogue(hat)
    line = next(ln for ln in lines if ln.startswith("- calculate_interim_payment:"))
    gross = next(part for part in line.split("; ") if "gross_valuation = " in part)
    assert driver_tools.BASIS_WORDS["period"] in gross


def test_the_default_chat_path_offer_carries_the_basis():
    from app.agents.runtime import _construction_calc_tool_schema

    desc = _construction_calc_tool_schema()["function"]["description"]
    assert driver_tools.BASIS_WORDS["period"] in desc


def test_every_registered_basis_is_known():
    for spec in fr.all_specs():
        for key, p in fr.parameters(spec).items():
            assert p.basis in ("",) + fr.BASES, (spec.name, key)
