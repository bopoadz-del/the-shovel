"""Rates, ratios and currency are the project's inputs, never built in."""
import pytest

from app.lib.construction_formulas import run_calculation

from app.agents.hats.quantities.formulas.construction_formulas import (
    cost_buildup_concrete, cost_buildup_formwork, cost_buildup_rebar,
    mobilization_cost_estimate,
)
from app.agents.hats.planning.formulas.construction_formulas import (
    electrical_installation_sequence, supervision_ratio,
)
from app.agents.hats.contracts.formulas.construction_formulas_commercial import (
    delay_damages_daily,
)


@pytest.mark.parametrize("kg", [10000, 250])
def test_rebar_without_rates_asks_for_them(kg):
    out = run_calculation("cost_buildup_rebar", {"quantity_kg": kg})
    assert out["status"] == "needs_input"
    assert set(out["missing"]) == {"material_price_per_t", "labour_rate_per_hr", "crane_rate_per_hr"}


@pytest.mark.parametrize("price,cur", [(2000, "SAR"), (900, "USD")])
def test_rebar_follows_the_given_rates(price, cur):
    out = cost_buildup_rebar(quantity_kg=1000, material_price_per_t=price, labour_rate_per_hr=10,
                             crane_rate_per_hr=100, waste_pct=0, indirect_pct=0, markup_pct=0,
                             labour_mhr_t=0, crane_hr_t=0, currency=cur)
    assert out["selling_price_per_t"] == pytest.approx(price, rel=1e-3)
    assert out["currency"] == cur


@pytest.mark.parametrize("rate", [40, 80])
def test_formwork_follows_the_given_supply_rate(rate):
    out = cost_buildup_formwork(area_m2=10, shuttering_supply_per_m2=rate, scaffolding_per_m2_day=0,
                                labour_rate_per_hr=0, crane_rate_per_hr=0)
    assert "error" not in out
    assert "shuttering_supply_per_m2" in run_calculation("cost_buildup_formwork", {"area_m2": 10})["missing"]


def test_concrete_without_prices_asks():
    assert "cement_price_per_t" in run_calculation("cost_buildup_concrete", {"quantity_m3": 100})["missing"]


@pytest.mark.parametrize("office", [100, 300])
def test_mobilization_uses_given_rates(office):
    kw = dict(num_personnel=10, duration_months=2, include_camp=False, include_transport=False,
              include_safety_medical=False, office_rate_per_person_month=office, office_setup_cost=0,
              it_rate_per_person_month=0, warehouse_equipment_cost=0, demobilization_cost=0)
    assert mobilization_cost_estimate(**kw)["grand_total"] == office * 10 * 2


def test_mobilization_asks_only_for_included_items():
    out = mobilization_cost_estimate(num_personnel=10, duration_months=2, include_camp=False)
    assert "camp_rate_per_person_month" not in out["needs_input"]
    assert "office_rate_per_person_month" in out["needs_input"]


@pytest.mark.parametrize("ratio,expected", [(1000, 5), (2500, 2)])
def test_supervision_uses_given_ratio(ratio, expected):
    out = supervision_ratio(concrete_m3=5000, concrete_m3_per_supervisor=ratio,
                            supervisors_per_hse_officer=8, staff_per_document_controller=15)
    assert out["civil_supervisors"] == expected


def test_supervision_asks_only_for_quantities_given():
    out = supervision_ratio(structural_steel_t=100)
    assert "steel_t_per_supervisor" in out["needs_input"]
    assert "concrete_m3_per_supervisor" not in out["needs_input"]


@pytest.mark.parametrize("rates,days", [({"a": 100}, 10), ({"a": 50, "b": 250}, 24)])
def test_electrical_stages_follow_given_rates(rates, days):
    assert electrical_installation_sequence(1000, 1, stage_output_rates=rates)["total_days"] == days
    assert run_calculation("electrical_installation_sequence", {"floor_area_m2": 1000})["missing"] == ["stage_output_rates"]


@pytest.mark.parametrize("cur", ["SAR", "AED"])
def test_delay_damages_currency_is_the_contracts(cur):
    assert delay_damages_daily(0.1, 1_000_000, cur)["currency"] == cur
    # No currency given -> none is invented (K-T5 ruling ii): the field is None
    # and the note carries no symbol and no leading double space.
    unlabelled = delay_damages_daily(0.1, 1_000_000)
    assert unlabelled["currency"] is None
    assert "SAR" not in unlabelled["note"] and "  " not in unlabelled["note"]


@pytest.mark.parametrize("cur,unit", [("SAR", "SAR/m3"), ("USD", "USD/m3")])
def test_the_headline_names_the_given_currency(cur, unit):
    from app.agents.driver import render
    from tests.example_project_rates import CONCRETE_RATES

    env = run_calculation("cost_buildup_concrete", {"quantity_m3": 100, **{**CONCRETE_RATES, "currency": cur}})
    assert render.headline(env) == f"Headline: Selling price is 308.83 {unit}."
