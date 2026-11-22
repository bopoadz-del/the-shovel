"""An output whose NAME carries a unit must DECLARE that unit (exam 6 M22 class).

The statement builder (``calculation_statement``) states only the outputs
whose declared unit is not ``''`` / ``'-'``, and labels each figure with the
declared unit. So an output named ``..._psi`` declared ``'-'`` is never stated,
and one named ``..._l_m`` declared ``'m'`` is stated under the wrong unit. The
model then labels the bare figure itself (live M22: a kgf/cm2 figure printed
as "bar").

This guard reads the declaration of EVERY registered formula. It holds no list
of formula names. The unit tokens come from the registry's own bind-suffix
table (``_BIND_UNIT_SUFFIXES``), plus a small table of compound tokens that the
bind table reads only by their tail (``_l_m`` is litres per metre, not metres).

Every checker decision is pinned below with at least two constructed inputs.
"""
from __future__ import annotations

from typing import Optional

import pytest

from app.lib import formula_registry
from app.lib.construction_formulas import _BIND_UNIT_SUFFIXES

#: Compound unit tokens -> the declared units they accept. Longest token wins,
#: so a compound is never read by its tail ("_l_m" before "_m").
COMPOUND_TOKENS: dict[str, frozenset[str]] = {
    "_kg_cm2": frozenset({"kgf/cm2", "kg/cm2"}),
    "_l_m": frozenset({"L/m"}),
    "_l_m3": frozenset({"L/m3"}),
    "_n_mm2": frozenset({"N/mm2", "MPa"}),
    "_psi": frozenset({"psi"}),
    "_min": frozenset({"min"}),
}

_UNITLESS = frozenset({"", "-"})


def _tokens() -> list[str]:
    toks = {suf for suf, _unit in _BIND_UNIT_SUFFIXES} | set(COMPOUND_TOKENS)
    return sorted(toks, key=len, reverse=True)


def unit_token(key: str) -> Optional[str]:
    """The longest unit token ``key`` ends with, or None.

    A token right after ``_per`` names the denominator of a rate
    (``shifts_per_day``): it says nothing about the unit of the figure itself,
    so it is not read as one.
    """
    k = str(key).lower()
    for tok in _tokens():
        if k.endswith(tok) and len(k) > len(tok):
            if k[: -len(tok)].endswith("_per"):
                return None
            return tok
    return None


def declaration_problem(key: str, declared: str) -> Optional[str]:
    """Why ``declared`` cannot be the unit of an output named ``key`` (or None)."""
    tok = unit_token(key)
    if tok is None:
        return None
    if str(declared).strip() in _UNITLESS:
        return f"{key!r} names unit token {tok!r} but declares no unit ({declared!r})"
    accepted = COMPOUND_TOKENS.get(tok)
    if accepted is not None and str(declared).strip() not in accepted:
        return f"{key!r} names {tok!r} but declares {declared!r} (expected one of {sorted(accepted)})"
    return None


# ── the checker, decision by decision ─────────────────────────────────────


@pytest.mark.parametrize("key", ["anchor_force_psi", "cure_time_min"])
def test_a_unit_named_output_declared_unitless_is_a_problem(key):
    assert declaration_problem(key, "-")
    assert declaration_problem(key, "")


@pytest.mark.parametrize("key, declared", [("anchor_force_psi", "psi"), ("cure_time_min", "min")])
def test_a_unit_named_output_declaring_its_unit_is_fine(key, declared):
    assert declaration_problem(key, declared) is None


@pytest.mark.parametrize("key", ["slurry_take_l_m", "bond_stress_kg_cm2"])
def test_a_compound_token_is_not_read_by_its_tail(key):
    # "_l_m" ends in "_m" and "_kg_cm2" in "_m2"; the longer compound token wins.
    assert unit_token(key) in COMPOUND_TOKENS
    assert declaration_problem(key, "m")


@pytest.mark.parametrize("key, declared", [("slurry_take_l_m", "L/m"), ("bond_stress_kg_cm2", "kgf/cm2")])
def test_a_compound_token_accepts_its_own_unit(key, declared):
    assert declaration_problem(key, declared) is None


@pytest.mark.parametrize("key", ["pours_per_day", "lifts_per_m"])
def test_a_rate_denominator_is_not_the_figures_unit(key):
    assert unit_token(key) is None
    assert declaration_problem(key, "-") is None


@pytest.mark.parametrize("key", ["pours_count", "verdict"])
def test_an_output_naming_no_unit_may_be_unitless(key):
    assert unit_token(key) is None
    assert declaration_problem(key, "-") is None


@pytest.mark.parametrize("key, declared", [("pour_volume_m3", "m3"), ("cover_mm", "mm")])
def test_a_simple_token_accepts_a_declared_unit(key, declared):
    # Simple tokens only require SOME declared unit; their spelling is the
    # bind table's business (declared 'days' vs inferred 'd' both state).
    assert unit_token(key) is not None
    assert declaration_problem(key, declared) is None


# ── the registry sweep ────────────────────────────────────────────────────


def test_every_registered_output_named_with_a_unit_declares_it():
    specs = formula_registry.all_specs()
    assert len(specs) > 96  # the sweep reads the whole registry, not a sample
    problems = [
        f"{spec.name}: {why}"
        for spec in specs
        for key, declared in spec.outputs.items()
        if (why := declaration_problem(key, declared))
    ]
    assert not problems, "\n".join(problems)
