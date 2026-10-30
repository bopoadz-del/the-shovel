"""K-T4-b: every new branch in the slice-2 rate changes has a test that fails
when that branch is mutated. Each test runs on two inputs."""
import re

import pytest

from app.agents.hats.planning.formulas.construction_formulas import (
    electrical_installation_sequence, supervision_ratio,
)
from app.agents.hats.quantities.formulas.construction_formulas import (
    cost_buildup_concrete, cost_buildup_formwork, cost_buildup_rebar, mobilization_cost_estimate,
)
from app.agents.hats.contracts.formulas.construction_formulas_commercial import (
    delay_damages_daily,
)
from app.agents.driver.render import _label_without_unit, _outputs, currency_unit, headline
from app.agents.runtime import _format_deterministic_cost_calc
from app.lib.construction_formulas import describe_calculation_params

RATIOS = dict(supervisors_per_hse_officer=4, staff_per_document_controller=10)


# --- supervision_ratio --------------------------------------------------
@pytest.mark.parametrize("blank", [None, ""])
def test_supervision_blank_quantity_ratio_is_asked_for(blank):
    out = supervision_ratio(concrete_m3=500, concrete_m3_per_supervisor=blank, **RATIOS)
    assert out["needs_input"] == ["concrete_m3_per_supervisor"]


@pytest.mark.parametrize("missing", ["supervisors_per_hse_officer", "staff_per_document_controller"])
@pytest.mark.parametrize("blank", [None, ""])
def test_supervision_staff_ratios_are_asked_for_once_supervisors_result(missing, blank):
    kw = dict(RATIOS, **{missing: blank})
    out = supervision_ratio(area_m2=1000, area_m2_per_supervisor=250, **kw)
    assert out["needs_input"] == [missing]


@pytest.mark.parametrize("q,ratio,sup,hse", [(1000, 300, 4, 1), (2000, 250, 8, 2)])
def test_supervision_counts_round_up_from_the_given_ratios(q, ratio, sup, hse):
    out = supervision_ratio(area_m2=q, area_m2_per_supervisor=ratio, **RATIOS)
    assert out["general_supervisors"] == sup
    assert out["hse_officers"] == hse


# --- electrical_installation_sequence -----------------------------------
@pytest.mark.parametrize("rates", [{}, None])
def test_electrical_without_stage_rates_asks(rates):
    out = electrical_installation_sequence(1000, 1, stage_output_rates=rates)
    assert out["needs_input"] == ["stage_output_rates"]


@pytest.mark.parametrize("rates", [[("rough_in", 100)], "rough_in=100"])
def test_electrical_non_mapping_rates_ask_instead_of_crashing(rates):
    out = electrical_installation_sequence(1000, 1, stage_output_rates=rates)
    assert out["needs_input"] == ["stage_output_rates"]


@pytest.mark.parametrize("area,days", [(1000, 10), (2050, 21)])
def test_electrical_zero_rate_stage_is_skipped(area, days):
    out = electrical_installation_sequence(area, 1, stage_output_rates={"rough_in": 100, "testing": 0})
    assert out["stages"] == {"rough_in": days}
    assert out["total_days"] == days


# --- render ---------------------------------------------------------------
@pytest.mark.parametrize("key", ["cost_per", "rate_per"])
def test_a_key_whose_own_words_end_in_per_keeps_it(key):
    assert _label_without_unit(key, "") == key.replace("_", " ")


@pytest.mark.parametrize("cur", ["EUR", "USD"])
def test_a_key_naming_two_currencies_ignores_the_given_one(cur):
    assert currency_unit("sar_to_aed_rate", "currency/m3", cur) == "currency/m3"


@pytest.mark.parametrize("cur,want", [("eur", "EUR/m3"), (" usd ", "USD/m3")])
def test_a_neutral_key_takes_the_given_currency(cur, want):
    assert currency_unit("selling_price_per_m3", "currency/m3", cur) == want


# --- delay_damages_daily --------------------------------------------------
@pytest.mark.parametrize("base,text", [(1000, "0.1% of 1,000.00 = 1.00 per"), (5000, "0.1% of 5,000.00 = 5.00 per")])
def test_delay_damages_note_without_currency_names_none(base, text):
    out = delay_damages_daily(rate_percent=0.1, contract_amount=base)
    assert out["note"].startswith(text)
    assert out["currency"] == ""


@pytest.mark.parametrize("cur", [" usd ", "eur"])
def test_delay_damages_currency_is_normalised(cur):
    out = delay_damages_daily(rate_percent=0.1, contract_amount=1000, currency=cur)
    code = cur.strip().upper()
    assert out["currency"] == code
    assert out["note"].startswith(f"0.1% of {code} 1,000.00 = {code} 1.00")


# --- mobilization_cost_estimate -------------------------------------------
ALL = dict(office_rate_per_person_month=100, camp_rate_per_person_month=200,
           transport_rate_per_person_month=50, safety_medical_rate_per_person_month=30,
           it_rate_per_person_month=10, office_setup_cost=1000, camp_setup_cost=2000,
           safety_setup_cost=300, warehouse_equipment_cost=0, demobilization_cost=0)


@pytest.mark.parametrize("flag,line", [
    ("include_offices", "offices"), ("include_camp", "camp"),
    ("include_transport", "transport"), ("include_safety_medical", "safety_medical"),
])
@pytest.mark.parametrize("n", [5, 10])
def test_an_excluded_item_costs_nothing_even_with_its_rate_given(flag, line, n):
    out = mobilization_cost_estimate(num_personnel=n, duration_months=2, **{flag: False}, **ALL)
    assert out["breakdown"][line] == 0
    other = mobilization_cost_estimate(num_personnel=n, duration_months=2, **ALL)
    assert other["breakdown"][line] > 0
    assert out["recurring_monthly"] < other["recurring_monthly"]


# --- cost_buildup_concrete / rebar -----------------------------------------
CONC = dict(cement_price_per_t=300, aggregate_price_per_t=50, water_price_per_m3=5,
            microsilica_price_per_kg=2, plasticizer_price_per_lit=3, plant_cost_per_m3=20,
            labour_cost_per_m3=30, erection_per_m3=10, power_per_m3=5, indirect_per_m3=5)


@pytest.mark.parametrize("cur,pat", [("", r"total \d+ for "), ("USD", r"total \d+ USD for ")])
def test_concrete_note_names_currency_only_when_given(cur, pat):
    out = cost_buildup_concrete(quantity_m3=100, currency=cur, **CONC)
    assert re.search(pat, out["note"]), out["note"]


@pytest.mark.parametrize("kg", [500, 2000])
def test_rebar_total_is_rate_times_quantity(kg):
    out = cost_buildup_rebar(quantity_kg=kg, material_price_per_t=1000, labour_rate_per_hr=0,
                             crane_rate_per_hr=0, waste_pct=0, indirect_pct=0, markup_pct=0)
    assert out["material_per_t"] == 1000
    assert out["total_project_value"] == 1000 * kg / 1000


# --- runtime cost answer / param units --------------------------------------
@pytest.mark.parametrize("cur,want", [
    ("", "Selling price 100.00 per m3. Total 1,000."),
    ("USD", "Selling price 100.00 USD/m3. Total 1,000 USD."),
])
def test_cost_answer_uses_the_result_currency_only(cur, want):
    payload = {"calculation": "cost_buildup_concrete",
               "result": {"selling_price_per_m3": 100, "total_project_value": 1000, "currency": cur}}
    assert _format_deterministic_cost_calc(payload) == want


@pytest.mark.parametrize("name,fn,key,unit", [
    ("cost_buildup_rebar", cost_buildup_rebar, "material_price_per_t", "currency/t"),
    ("cost_buildup_concrete", cost_buildup_concrete, "cement_price_per_t", "currency/t"),
])
def test_declared_money_unit_wins_over_name_suffix(name, fn, key, unit):
    rows = {r["name"]: r["unit"] for r in describe_calculation_params(fn, name)}
    assert rows[key] == unit


# --- each changed function's own first guard ---------------------------------
REBAR = dict(material_price_per_t=1000, labour_rate_per_hr=10, crane_rate_per_hr=100)
FORM = dict(shuttering_supply_per_m2=40, scaffolding_per_m2_day=1, labour_rate_per_hr=10, crane_rate_per_hr=100)


@pytest.mark.parametrize("call", [
    lambda q: cost_buildup_concrete(quantity_m3=q, **CONC),
    lambda q: cost_buildup_rebar(quantity_kg=q, **REBAR),
    lambda q: cost_buildup_formwork(area_m2=q, **FORM),
    lambda q: electrical_installation_sequence(q, 1, stage_output_rates={"rough_in": 100}),
])
@pytest.mark.parametrize("q", [-5, -0.5])
def test_a_negative_quantity_is_refused(call, q):
    with pytest.raises(ValueError):
        call(q)


@pytest.mark.parametrize("floors", [0, -1])
def test_electrical_needs_at_least_one_floor(floors):
    with pytest.raises(ValueError):
        electrical_installation_sequence(100, floors, stage_output_rates={"rough_in": 100})


@pytest.mark.parametrize("pct,base", [(-0.1, 1000), (0.1, -1000)])
def test_delay_damages_refuses_negative_operands(pct, base):
    out = delay_damages_daily(rate_percent=pct, contract_amount=base)
    assert out == {"error": "rate_percent and contract_amount must be >= 0."}


@pytest.mark.parametrize("key,unit,want", [("selling_price_sar", "SAR/m3", "selling price"),
                                           ("bid_value_usd", "USD", "bid value")])
def test_a_trailing_currency_word_is_dropped_from_the_label(key, unit, want):
    assert _label_without_unit(key, unit) == want


@pytest.mark.parametrize("unit", ["m3", "days"])
def test_a_non_money_unit_is_left_alone(unit):
    assert currency_unit("selling_sar", unit, "USD") == unit


def _rebar_env(cur):
    res = cost_buildup_rebar(quantity_kg=2000, currency=cur, **REBAR)
    return {"calculation": "cost_buildup_rebar", "result": res}, res


@pytest.mark.parametrize("cur", ["USD", "EUR"])
def test_headline_leads_with_the_declared_main_output_in_its_currency(cur):
    env, res = _rebar_env(cur)
    line = headline(env)
    assert "selling price" in line.lower()
    assert f"{cur}/t" in line
    assert f"{res['selling_price_per_t']:,.0f}".split(",")[0] in line


@pytest.mark.parametrize("cur", ["USD", "EUR"])
def test_outputs_carry_the_result_currency_and_drop_primary_mirrors(cur):
    env, _ = _rebar_env(cur)
    env["result"] = dict(env["result"], primary_result={"value": 1, "unit": "x", "label": "y"},
                         value=1, unit="x", label="y")
    outputs, labels, units = _outputs(env, {"quantity_kg": 2000})
    assert units["selling_price_per_t"] == f"{cur}/t"
    assert "quantity_kg" not in outputs
    assert not {"value", "unit", "label"} & set(outputs)
