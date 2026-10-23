"""Standing-exit B last-5 bind after LIVE tip 53b8b29 (37/42).

LIVE FAILs:
  beam_shear_simple          — cause (list-wrapped w/span → float(list) TypeError)
  concrete_maturity_strength — tool_error (list-literal assignment / ask CSV)
  concrete_mix_slip_form     — tool_error (extra kwargs / list params)
  cost_buildup_rebar         — wrong_number (waste_pct=10 treated as 1000% waste)
  evaluate_tender            — tool_error (bidders alias / raw array / ask text)

Formulas stay the gated oracles. Bind / alias / extract-from-ask / coerce
is what this file pins. Never invent a figure that is not in the ask.
"""
from __future__ import annotations

import json

import pytest

from app.lib.construction_formulas import (
    CALCULATORS,
    bind_calculation_params,
    coerce_calc_params,
    run_calculation,
)
from tests.test_calc_standing_exit_bind import (
    _assert_named_missing,
    _err,
    _ok,
    _tool,
)
from tests.test_construction_calc_tool import _agent, _call, _run


# ── beam_shear_simple (LIVE: cause error) ──────────────────────────────────

# V = wL/2. Hand: 20×6/2 = 60 kN.
_SHEAR_REQUIRED = (
    ("udl_w_kn_m", "kN/m"),
    ("span_m", "m"),
)


def test_beam_shear_list_wrapped_scalars_are_60_not_cause():
    """Live sent w=[20], span=[6] — float(list) set envelope['cause']."""
    inner = _ok("beam_shear_simple", {"w": [20], "span": [6]})
    assert inner["max_shear_kn"] == pytest.approx(60.0, abs=0.01)
    env = run_calculation("beam_shear_simple", {"w": [20], "L": [6]})
    assert env.get("status") == "success", env
    assert "cause" not in env
    assert env["result"]["max_shear_kn"] == pytest.approx(60.0, abs=0.01)


def test_beam_shear_json_object_string_with_list_values():
    inner = _ok("beam_shear_simple", '{"w": [20], "L": [6]}')
    assert inner["max_shear_kn"] == pytest.approx(60.0, abs=0.01)


def test_beam_shear_positional_pair_binds_w_and_span():
    """params [20, 6] / '[20, 6]' are the live-probe positional shape."""
    inner = _ok("beam_shear_simple", [20, 6])
    assert inner["max_shear_kn"] == pytest.approx(60.0, abs=0.01)
    via_str = _ok("beam_shear_simple", "[20, 6]")
    assert via_str["max_shear_kn"] == pytest.approx(60.0, abs=0.01)
    env = _tool("beam_shear_simple", {"params": [20, 6]})
    assert env.get("ok") is True, env
    assert env["result"]["result"]["max_shear_kn"] == pytest.approx(60.0, abs=0.01)


def test_beam_shear_empty_still_missing_not_zero():
    env = _err("beam_shear_simple", {})
    _assert_named_missing(env, _SHEAR_REQUIRED)
    assert env.get("cause") in (None, "")


# ── concrete_maturity_strength (LIVE: tool_error) ──────────────────────────

_MATURITY_ASK = (
    "Nurse-Saul maturity: temperatures 20, 22, 25 C at 6, 6, 6 hour intervals"
)


def test_concrete_maturity_list_literal_assignment_string():
    """temperature_history_c=[20, 22, 25], time_intervals_hours=[6, 6, 6]."""
    inner = _ok(
        "concrete_maturity_strength",
        "temperature_history_c=[20, 22, 25], time_intervals_hours=[6, 6, 6]",
    )
    assert inner["maturity_index_c_hrs"] == pytest.approx(582.0, abs=0.5)
    assert inner["predicted_strength_n_mm2"] == pytest.approx(6.9, abs=0.1)


def test_concrete_maturity_positional_pair_of_lists():
    inner = _ok("concrete_maturity_strength", [[20, 22, 25], [6, 6, 6]])
    assert inner["maturity_index_c_hrs"] == pytest.approx(582.0, abs=0.5)
    via_json = _ok("concrete_maturity_strength", "[[20, 22, 25], [6, 6, 6]]")
    assert via_json["percent_of_28d"] == pytest.approx(17.2, abs=0.2)


def test_concrete_maturity_extract_csv_from_ask():
    inner = _ok("concrete_maturity_strength", {"text": _MATURITY_ASK})
    assert inner["maturity_index_c_hrs"] == pytest.approx(582.0, abs=0.5)
    assert inner["predicted_strength_n_mm2"] == pytest.approx(6.9, abs=0.1)
    env = _tool("concrete_maturity_strength", {"text": _MATURITY_ASK})
    assert env.get("ok") is True, env
    assert env["result"]["result"]["percent_of_28d"] == pytest.approx(17.2, abs=0.2)


def test_concrete_maturity_partial_ask_does_not_invent_hours():
    env = _err("concrete_maturity_strength", {
        "text": "concrete maturity temperatures 20, 22, 25 C",
    })
    assert "time_intervals_hours" in env["missing"]
    assert "missing required" in env["error"]


# ── concrete_mix_slip_form (LIVE: tool_error) ──────────────────────────────

def test_concrete_mix_slip_form_extra_kwargs_do_not_typeerror():
    """Documented slip-form mix is 1:2:2.6 @ w/c=0.42 → 398 kg cement.

    Live sent slump/w_c next to calculation. Extra keys must not TypeError
    (envelope cause) — they are ignored, constants stay documented.
    """
    inner = _ok("concrete_mix_slip_form", {
        "slump_mm": 150, "w_c_ratio": 0.42, "text": "slip form mix",
    })
    assert inner["cement_kg_m3"] == pytest.approx(398.0, abs=0.5)
    assert inner["w_c_ratio"] == pytest.approx(0.42, abs=1e-9)
    direct = CALCULATORS["concrete_mix_slip_form"](slump=150, wc=0.42)
    assert direct["cement_kg_m3"] == pytest.approx(398.0, abs=0.5)


def test_concrete_mix_slip_form_list_params_still_succeed():
    inner = _ok("concrete_mix_slip_form", [])
    assert inner["cement_kg_m3"] == pytest.approx(398.0, abs=0.5)
    via_str = _ok("concrete_mix_slip_form", "[]")
    assert via_str["retarder_20c_lit_m3"] == pytest.approx(3.8, abs=1e-9)
    env = _tool("concrete_mix_slip_form", {"params": {"slump": 150}})
    assert env.get("ok") is True, env
    assert env["result"]["result"]["cement_kg_m3"] == pytest.approx(398.0, abs=0.5)


# ── cost_buildup_rebar (LIVE: wrong_number) ────────────────────────────────

# Documented: 1 t @ 2600, waste 10% → material 2860; sell 4856 SAR/t.
_REBAR_SELL = 4856


def test_cost_buildup_rebar_waste_percent_10_is_fraction_not_1000pct():
    """Live sent waste_pct=10 (percent) and scored selling 40590 as wrong_number."""
    inner = _ok("cost_buildup_rebar", {"quantity_kg": 1000, "waste_pct": 10})
    assert inner["material_sar_t"] == pytest.approx(2860, abs=1)
    assert inner["selling_price_sar_t"] == pytest.approx(_REBAR_SELL, abs=2)
    via_alias = _ok("cost_buildup_rebar", {"qty": 1000, "waste": 10})
    assert via_alias["selling_price_sar_t"] == pytest.approx(_REBAR_SELL, abs=2)


def test_cost_buildup_rebar_price_alias_and_tonne_quantity():
    inner = _ok("cost_buildup_rebar", {"qty": 1000, "price": 2600})
    assert inner["selling_price_sar_t"] == pytest.approx(_REBAR_SELL, abs=2)
    tonne = _ok("cost_buildup_rebar", {"quantity_t": 1})
    assert tonne["selling_price_sar_t"] == pytest.approx(_REBAR_SELL, abs=2)
    assert tonne["total_project_value_sar"] == pytest.approx(_REBAR_SELL, abs=2)


def test_cost_buildup_rebar_extract_from_ask():
    inner = _ok("cost_buildup_rebar", {
        "text": "cost buildup rebar 1000 kg at 2600 SAR/t",
    })
    assert inner["material_sar_t"] == pytest.approx(2860, abs=1)
    assert inner["selling_price_sar_t"] == pytest.approx(_REBAR_SELL, abs=2)
    env = _tool("cost_buildup_rebar", {
        "text": "rebar cost build-up for 1 tonne @ 2600 SAR/t",
    })
    assert env.get("ok") is True, env
    assert env["result"]["result"]["selling_price_sar_t"] == pytest.approx(
        _REBAR_SELL, abs=2,
    )


def test_cost_buildup_rebar_empty_is_missing_quantity():
    env = _err("cost_buildup_rebar", {})
    assert "quantity_kg" in env["missing"]
    assert "missing required" in env["error"]


# ── evaluate_tender (LIVE: tool_error) ─────────────────────────────────────

_BIDDERS = [
    {"name": "Bidder A", "technical_score": 85, "commercial_score": 75,
     "hse_score": 90, "local_content_score": 60},
    {"name": "Bidder B", "technical_score": 78, "commercial_score": 82,
     "hse_score": 85, "local_content_score": 70},
    {"name": "Bidder C", "technical_score": 70, "commercial_score": 88,
     "hse_score": 80, "local_content_score": 75},
]
_TENDER_ASK = (
    "Per XYZ-603, evaluate the following three balanced bidders: "
    "Bidder A — technical 85, commercial 75, HSE 90, local content 60; "
    "Bidder B — technical 78, commercial 82, HSE 85, local content 70; "
    "Bidder C — technical 70, commercial 88, HSE 80, local content 75. "
    "Which bidder is recommended and what is the ranking?"
)


def _assert_bidder_a(inner):
    assert inner["recommended"]["name"] == "Bidder A"
    assert inner["recommended"]["weighted_total"] == pytest.approx(80.10, abs=0.01)
    assert inner["ranked_tenderers"][0]["name"] == "Bidder A"


def test_evaluate_tender_bidders_alias_binds():
    inner = _ok("evaluate_tender", {"bidders": _BIDDERS})
    _assert_bidder_a(inner)
    via_bids = _ok("evaluate_tender", {"bids": _BIDDERS})
    _assert_bidder_a(via_bids)


def test_evaluate_tender_raw_array_params_bind():
    inner = _ok("evaluate_tender", _BIDDERS)
    _assert_bidder_a(inner)
    via_json = _ok("evaluate_tender", json.dumps(_BIDDERS))
    _assert_bidder_a(via_json)
    env = _tool("evaluate_tender", {"params": _BIDDERS})
    assert env.get("ok") is True, env
    _assert_bidder_a(env["result"]["result"])


def test_evaluate_tender_extract_from_ask():
    inner = _ok("evaluate_tender", {"text": _TENDER_ASK})
    _assert_bidder_a(inner)
    env = _tool("evaluate_tender", {"text": _TENDER_ASK})
    assert env.get("ok") is True, env
    _assert_bidder_a(env["result"]["result"])


def test_evaluate_tender_empty_is_missing_tenderers():
    env = _err("evaluate_tender", {})
    assert "tenderers" in env["missing"]
    assert "missing required" in env["error"]


def test_evaluate_tender_partial_ask_does_not_invent_scores():
    env = _err("evaluate_tender", {
        "text": "evaluate tender for Bidder A and Bidder B",
    })
    assert "tenderers" in env["missing"]


# ── coerce helpers stay unregistered ───────────────────────────────────────

def test_last5_helpers_are_not_registered_calculators():
    assert "coerce_calc_params" not in CALCULATORS
    assert "bind_calculation_params" not in CALCULATORS
    parsed = coerce_calc_params("temperature_history_c=[20, 22, 25], dt=[6, 6, 6]")
    assert parsed["temperature_history_c"] == [20, 22, 25]
    assert parsed["dt"] == [6, 6, 6]
    assert coerce_calc_params([20, 6])["_positional"] == [20, 6]


def test_last5_bind_does_not_leak_envelope_on_rebar():
    leaked = {
        "qty": 1000, "price": 2600,
        "text": "compute rebar",
        "formula": "not-a-kwarg",
        "project_id": "ws-b5",
        "input": {"noise": 1},
    }
    bound = bind_calculation_params(CALCULATORS["cost_buildup_rebar"], leaked)
    for key in ("text", "formula", "input", "project_id", "noise"):
        assert key not in bound, bound
    assert bound["quantity_kg"] == 1000
    assert bound["material_price_sar_t"] == 2600
