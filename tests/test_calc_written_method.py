"""D1 rule C at the door: a calculation states the calculator's own written
method, so a coefficient or factor its body applies is not silent.

Every calculator here is invented in this file; nothing reads a real
calculator's text.
"""

from __future__ import annotations

import pytest

from app.lib import formula_registry


def _with_equation_later(a: float, b: float) -> float:
    """Invented panel load.

    Notes for the maintainer with no equation.

    load = 7.25 * a * b, the 7.25 factor from the invented code table.
    """
    return 7.25 * a * b


def _with_equation_first(a: float) -> float:
    """Invented strip: s = a / 3.5 (invented spacing rule).

    History paragraph that writes no equation.
    """
    return a / 3.5


def _without_doc(a: float) -> float:
    return a * 2.75


def _blank_doc(a: float) -> float:
    """   """
    return a * 4.5


@pytest.mark.parametrize(
    ("fn", "kept", "dropped"),
    [
        (_with_equation_later, ["Invented panel load.", "load = 7.25 * a * b"],
         "Notes for the maintainer"),
        (_with_equation_first, ["s = a / 3.5"], "History paragraph"),
    ],
)
def test_written_method_keeps_first_and_equation_paragraphs(fn, kept, dropped):
    method = formula_registry.written_method(fn)
    for text in kept:
        assert text in method
    assert dropped not in method


@pytest.mark.parametrize("fn", [_without_doc, _blank_doc])
def test_written_method_is_empty_without_a_written_method(fn):
    assert formula_registry.written_method(fn) == ""


def _register(name: str, fn):
    formula_registry.formula(
        owner=formula_registry.BASE,
        description="Invented calculator for the written-method door test.",
        display_name=name.replace("_", " "),
        inputs={"a": formula_registry.Param("m", 0, 1000, label="length a"),
                **({"b": formula_registry.Param("m", 0, 1000, label="length b")}
                   if "b" in fn.__code__.co_varnames[: fn.__code__.co_argcount] else {})},
        outputs={"value": "-"},
        name=name,
    )(fn)


@pytest.fixture
def invented_calculators(monkeypatch):
    from app.lib import construction_formulas as cf

    names = {"zz_invented_panel_load": _with_equation_later,
             "zz_invented_strip": _with_equation_first,
             "zz_invented_no_doc": _without_doc,
             "zz_invented_blank_doc": _blank_doc}
    for name, fn in names.items():
        _register(name, fn)
        monkeypatch.setitem(cf.CALCULATORS, name, fn)
    yield names
    for name in names:
        formula_registry._REGISTRY.pop(name, None)


@pytest.mark.parametrize(
    ("name", "params", "value", "factor", "dropped"),
    [
        ("zz_invented_panel_load", {"a": 2, "b": 4}, 58.0, "7.25",
         "Notes for the maintainer"),
        ("zz_invented_strip", {"a": 7}, 2.0, "3.5", "History paragraph"),
    ],
)
def test_door_states_the_written_method_with_its_factor(
        invented_calculators, name, params, value, factor, dropped):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(name, params)
    assert env["status"] == "success", env
    assert env["result"]["value"] == pytest.approx(value)
    assert factor in env["method"]
    assert dropped not in env["method"]


@pytest.mark.parametrize(
    ("name", "a", "value"),
    [
        ("zz_invented_no_doc", 2, 5.5),
        ("zz_invented_blank_doc", 3, 13.5),
    ],
)
def test_door_adds_no_method_when_the_calculator_writes_none(
        invented_calculators, name, a, value):
    from app.lib.construction_formulas import run_calculation

    env = run_calculation(name, {"a": a})
    assert env["status"] == "success", env
    assert env["result"]["value"] == pytest.approx(value)
    assert "method" not in env
