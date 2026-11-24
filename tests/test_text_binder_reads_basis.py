"""The ask-text binder reads the basis an amount input declares.

An input declared as this period's amount never takes a figure the ask wrote
as a total to date, nor the reverse; the input stays open so the calculator
asks for it. Before this, the binder bound by label alone, and the
supplied-input graft could then replace the model's answer (which saw the
basis in the labelled offer) with one worked from the wrong figure.

Every formula here is invented in the test; no product formula is named.
"""
from __future__ import annotations

import pytest

from app.lib import formula_registry


def _period_fn(work_value):
    return {"value": work_value}


def _cumulative_fn(paid_value):
    return {"value": paid_value}


def _plain_fn(stock_value):
    return {"value": stock_value}


@pytest.fixture
def invented():
    rows = {
        "zz_basis_period": (_period_fn, "work_value", "work value", "period"),
        "zz_basis_cumulative": (_cumulative_fn, "paid_value", "paid value", "cumulative"),
        "zz_basis_plain": (_plain_fn, "stock_value", "stock value", ""),
    }
    for name, (fn, key, label, basis) in rows.items():
        formula_registry.formula(
            owner=formula_registry.BASE,
            description="Invented calculator for the basis binding test.",
            display_name=name.replace("_", " "),
            inputs={key: formula_registry.Param("currency", 0, 1e13, label=label, basis=basis)},
            outputs={"value": "currency"}, name=name,
        )(fn)
    yield {name: row[0] for name, row in rows.items()}
    for name in rows:
        formula_registry._REGISTRY.pop(name, None)


def _bind(fn, text):
    from app.lib.construction_formulas import extract_calculation_params_from_text

    return extract_calculation_params_from_text(fn, text)


@pytest.mark.parametrize(
    ("name", "key", "text", "expected"),
    [
        # Period input: a to-date figure is refused, a this-period one binds.
        ("zz_basis_period", "work_value", "work value 1,250,000 to date", None),
        ("zz_basis_period", "work_value", "work value 1,250,000 this period", 1250000.0),
        # Cumulative input: the reverse.
        ("zz_basis_cumulative", "paid_value", "paid value 840,000 this month", None),
        ("zz_basis_cumulative", "paid_value", "paid value 840,000 so far", 840000.0),
    ],
)
def test_declared_basis_refuses_a_figure_written_on_the_other(invented, name, key, text, expected):
    assert _bind(invented[name], text).get(key) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The cue sits in another figure's clause: this figure is unmarked.
        ("work value 1,250,000, and 9,000,000 to date", 1250000.0),
        # The cue sits in this figure's clause.
        ("work value 9,000,000 to date, and 1,250,000 retained", None),
    ],
)
def test_cue_marks_only_the_figure_in_its_own_clause(invented, text, expected):
    assert _bind(invented["zz_basis_period"], text).get("work_value") == expected


@pytest.mark.parametrize(
    ("name", "key", "expected"),
    [
        # An input that declares no basis keeps binding by label as before.
        ("zz_basis_plain", "stock_value", 1250000.0),
        ("zz_basis_period", "work_value", None),
    ],
)
def test_only_a_declared_basis_refuses(invented, name, key, expected):
    label = key.replace("_", " ")
    assert _bind(invented[name], f"{label} 1,250,000 to date").get(key) == expected


@pytest.mark.parametrize(
    ("cue", "selected"),
    [("to date", False), ("this period", True)],
)
def test_selector_does_not_pick_a_formula_from_the_wrong_basis(invented, cue, selected):
    from app.lib.construction_formulas import formula_completed_by_user_text

    got = formula_completed_by_user_text(f"calculate the zz basis period: work value 1,250,000 {cue}")
    assert (got is not None and got[0] == "zz_basis_period") is selected


@pytest.mark.parametrize("basis", formula_registry.BASES)
def test_every_basis_has_cue_words(basis):
    assert formula_registry.BASIS_CUES[basis]
    assert set(formula_registry.BASIS_CUES) == set(formula_registry.BASES)
