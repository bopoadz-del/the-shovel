"""K-T4-a: crane rates are the caller's inputs. Each guard G6 listed has a
two-input test (quantities :55/:58/:63/:65, procurement :79/:81/:90)."""
from __future__ import annotations

import pytest

from app.agents.hats.procurement.formulas.construction_formulas import crane_planning
from app.agents.hats.quantities.formulas.construction_formulas import crane_cost_estimate

BASE = dict(num_cranes=2, crane_capacity_tons=10, duration_months=3)


# :55 a negative dry-hire rate is refused; zero is a valid rate.
@pytest.mark.parametrize("rate,raises", [(-1.0, True), (0.0, False)])
def test_negative_dry_hire_rate_is_refused(rate, raises):
    kw = dict(BASE, dry_hire_rate_per_month=rate, include_operator=False, include_riggers=False)
    if raises:
        with pytest.raises(ValueError, match="dry_hire_rate_per_month"):
            crane_cost_estimate(**kw)
    else:
        assert crane_cost_estimate(**kw)["grand_total"] == 0


# :58/:63 an empty or absent operator rate, operator included, is a needs_input return.
@pytest.mark.parametrize("op_rate", ["", None])
def test_missing_operator_rate_returns_needs_input(op_rate):
    out = crane_cost_estimate(**BASE, dry_hire_rate_per_month=1000, include_riggers=False,
                              operator_rate_per_month=op_rate)
    assert out == {"error": "crew included but its rate was not given",
                   "needs_input": ["operator_rate_per_month"]}


# :65 an excluded operator costs 0 even when a rate is supplied.
@pytest.mark.parametrize("include,expected", [(False, 0.0), (True, 400.0)])
def test_excluded_operator_rate_is_not_charged(include, expected):
    out = crane_cost_estimate(**BASE, dry_hire_rate_per_month=1000, include_riggers=False,
                              include_operator=include, operator_rate_per_month=400)
    assert out["operator_per_month"] == expected
    assert out["per_crane_monthly"] == 1000 + expected


PLAN = dict(total_lift_demand_tons=5000, crane_capacity_tons=10)


# procurement :79 an empty-string rate is the same as no rate.
@pytest.mark.parametrize("rate", ["", None])
def test_empty_rate_gives_count_only(rate):
    out = crane_planning(**PLAN, dry_hire_rate_per_month=rate)
    assert out["monthly_rate"] is None and out["total_monthly_cost"] is None
    assert any("needs the project's dry hire rate" in n for n in out["notes"])


# :81/:90 the currency suffix and the total-cost note.
@pytest.mark.parametrize("currency,suffix", [("EUR", " EUR"), ("", "")])
def test_total_note_carries_the_given_currency(currency, suffix):
    out = crane_planning(**PLAN, dry_hire_rate_per_month=1000, currency=currency)
    n = out["cranes_required"]
    assert out["total_monthly_cost"] == n * 1000.0
    assert out["notes"][-1] == f"Total crane cost: {n * 1000:,.0f}{suffix}/month"
    assert out["notes"][-2] == f"Require {n} crane(s) @ 1,000{suffix}/month each"


# :60 an empty or absent riggers rate, riggers included, is a needs_input return.
@pytest.mark.parametrize("rg_rate", ["", None])
def test_missing_riggers_rate_returns_needs_input(rg_rate):
    out = crane_cost_estimate(**dict(BASE, dry_hire_rate_per_month=100.0, include_operator=False,
                                     include_riggers=True, riggers_rate_per_month=rg_rate))
    assert out["needs_input"] == ["riggers_rate_per_month"]


@pytest.mark.parametrize("rg_rate,want", [(10.0, 2 * 3 * 110.0), (30.0, 2 * 3 * 130.0)])
def test_a_given_riggers_rate_is_charged(rg_rate, want):
    out = crane_cost_estimate(**dict(BASE, dry_hire_rate_per_month=100.0, include_operator=False,
                                     include_riggers=True, riggers_rate_per_month=rg_rate))
    assert out["total_project_cost"] == pytest.approx(want)


# quantities :49-50 negative counts / durations are refused.
@pytest.mark.parametrize("field", ["num_cranes", "duration_months"])
def test_negative_count_or_duration_is_refused(field):
    kw = dict(BASE, dry_hire_rate_per_month=100.0, include_operator=False, include_riggers=False)
    kw[field] = -1
    with pytest.raises(ValueError, match="num_cranes and duration_months"):
        crane_cost_estimate(**kw)


# procurement :63-64 a negative lift demand is refused; zero gives no cranes.
@pytest.mark.parametrize("demand,raises", [(-5.0, True), (0.0, False)])
def test_negative_lift_demand_is_refused(demand, raises):
    if raises:
        with pytest.raises(ValueError, match="total_lift_demand_tons"):
            crane_planning(total_lift_demand_tons=demand, crane_capacity_tons=50)
    else:
        assert crane_planning(total_lift_demand_tons=demand, crane_capacity_tons=50)["cranes_required"] == 0
