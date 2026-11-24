"""S-K2 item 7: resource-based cost line = qty x consumption x waste x cited price.

Every expected value is worked out here from the method on inputs this file
builds itself; nothing is read back from the engine's arithmetic. Each
decision is checked on two inputs (mutation gate).
"""
from __future__ import annotations

import pytest

from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    resource_cost_line,
)

# Two seeded lines: (quantity, consumption per unit, price per resource unit).
LINES = [
    (240.0, 0.35, 410.0),
    (1_850.0, 12.5, 3.2),
]


def _base(q, c, p):
    return {"quantity": q, "consumption_per_unit": c, "price_per_resource_unit": p}


@pytest.mark.parametrize("q,c,p", LINES)
def test_line_without_waste_or_markup_is_qty_times_consumption_times_price(q, c, p):
    out = resource_cost_line(**_base(q, c, p))
    assert out["resource_quantity"] == pytest.approx(q * c)
    assert out["resource_cost"] == pytest.approx(round(q * c * p, 2))
    assert out["line_total"] == pytest.approx(round(q * c * p, 2))
    assert out["waste_factor"] is None
    assert out["markup_amount"] is None


@pytest.mark.parametrize("q,c,p", LINES)
@pytest.mark.parametrize("waste", [5.0, 12.5])
def test_waste_multiplies_the_resource_quantity(q, c, p, waste):
    out = resource_cost_line(**_base(q, c, p), waste_percent=waste)
    factor = 1 + waste / 100
    assert out["waste_factor"] == pytest.approx(factor)
    assert out["resource_quantity_net"] == pytest.approx(q * c)
    assert out["resource_quantity"] == pytest.approx(q * c * factor)
    assert out["resource_cost"] == pytest.approx(round(q * c * factor * p, 2))


@pytest.mark.parametrize("q,c,p", LINES)
def test_absent_waste_is_said_not_assumed(q, c, p):
    out = resource_cost_line(**_base(q, c, p))
    assert any("No waste allowance supplied" in w for w in out["working"])
    with_waste = resource_cost_line(**_base(q, c, p), waste_percent=5.0)
    assert not any("No waste allowance supplied" in w for w in with_waste["working"])


@pytest.mark.parametrize("q,c,p", LINES)
def test_no_default_markup(q, c, p):
    out = resource_cost_line(**_base(q, c, p), waste_percent=5.0)
    assert out["markup_amount"] is None
    assert out["line_total"] == out["resource_cost"]
    assert any("no default markup" in w for w in out["working"])


@pytest.mark.parametrize("q,c,p", LINES)
@pytest.mark.parametrize("markup", [0.0, 15.0])
def test_supplied_markup_is_added_on_the_resource_cost(q, c, p, markup):
    out = resource_cost_line(**_base(q, c, p), waste_percent=5.0,
                             markup_percent=markup)
    cost = q * c * 1.05 * p
    assert out["markup_amount"] == pytest.approx(round(cost * markup / 100, 2))
    assert out["line_total"] == pytest.approx(round(cost * (1 + markup / 100), 2))
    assert not any("no default markup" in w for w in out["working"])


@pytest.mark.parametrize("absent", ["quantity", "consumption_per_unit",
                                    "price_per_resource_unit"])
@pytest.mark.parametrize("q,c,p", LINES)
def test_a_missing_project_fact_is_named_not_defaulted(absent, q, c, p):
    args = _base(q, c, p)
    args[absent] = None
    out = resource_cost_line(**args)
    assert out["missing"] == [absent]
    assert "line_total" not in out


@pytest.mark.parametrize("negative", ["quantity", "consumption_per_unit",
                                      "price_per_resource_unit",
                                      "waste_percent", "markup_percent"])
@pytest.mark.parametrize("value", [-1.0, -0.01])
def test_negative_inputs_are_refused(negative, value):
    args = {**_base(*LINES[0]), "waste_percent": 5.0, "markup_percent": 10.0}
    args[negative] = value
    out = resource_cost_line(**args)
    assert "error" in out
    assert "line_total" not in out


@pytest.mark.parametrize("q,c,p", LINES)
def test_zero_inputs_are_allowed(q, c, p):
    out = resource_cost_line(quantity=0.0, consumption_per_unit=c,
                             price_per_resource_unit=p, waste_percent=0.0)
    assert out["line_total"] == 0.0
    assert out["waste_factor"] == 1.0


@pytest.mark.parametrize("q,c,p", LINES)
def test_working_prints_each_step(q, c, p):
    out = resource_cost_line(**_base(q, c, p), waste_percent=5.0,
                             markup_percent=10.0)
    text = " | ".join(out["working"])
    for step in ("Net resource", "With waste", "Resource cost", "Markup"):
        assert step in text
    assert f"{round(q * c * 1.05 * p * 1.1, 2):,.2f}" in text


@pytest.mark.parametrize("q,c,p", LINES)
def test_registered_and_run_through_the_registry(q, c, p):
    from app.lib.construction_formulas import CALCULATORS

    fn = CALCULATORS["resource_cost_line"]
    assert fn(**_base(q, c, p))["line_total"] == pytest.approx(round(q * c * p, 2))


def test_registry_module_reexports_the_hat_formula():
    from app.lib import construction_formulas_quantities as lib

    assert lib.resource_cost_line is resource_cost_line


_FIGURE = r"(-?[\d,]*\.?\d+)"


def _figures(line: str) -> list[float]:
    import re
    return [float(t.replace(",", "")) for t in re.findall(_FIGURE, line)]


@pytest.mark.parametrize("q,c,p", LINES + [(2_500_000.0, 1.0, 0.125),
                                           (1_234_567.0, 0.004, 17.375)])
def test_printed_figures_multiply_to_the_printed_result(q, c, p):
    # The working is read as a person reads it: the figures as printed, not
    # the engine's floats. Each step must reconcile on its own figures.
    out = resource_cost_line(**_base(q, c, p), waste_percent=7.5)
    net, waste, cost = (next(w for w in out["working"] if w.startswith(head))
                        for head in ("Net resource", "With waste", "Resource cost"))
    a, b, r = _figures(net)
    assert (a, b) == pytest.approx((q, c)) and r == pytest.approx(a * b)
    _, n, f, g = _figures(waste)
    assert n == pytest.approx(r) and g == pytest.approx(n * f)
    g2, price, total = _figures(cost)
    assert g2 == pytest.approx(g) and price == pytest.approx(p)
    assert total == pytest.approx(g2 * price, abs=0.005)
    assert "e+" not in " ".join(out["working"]).lower()
