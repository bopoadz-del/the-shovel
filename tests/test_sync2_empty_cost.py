"""SYNC2 Agent A — empty turns after calc (A2) and user-supplied rates (A3).

Live master_corpus FAILs on 72fcfd2:

A2-1  plastering 3,400 m2 @ 42 m2/gang-day, gang SAR 1,950
      → construction_calc ×6 then raw JSON
      ``{"status":"error","error":"Unknown calculation 'None'."}``
A2-2  weight of 12 tonnes of Y16 bars in metres run
      → construction_calc then "I was unable to generate a response…"

A3-1  14 pad footings 2.4×2.4×0.6 m @ SAR 410/m3 + 5% waste + 10% contingency
      → refused "I don't have a rate on file" despite the user giving 410
A3-2  500 m of 300 mm uPVC @ SAR 95 per metre
      → refused the user-supplied 95

Live tip a8498b38: A2 empty is 0/2. construction_calc is called with
empty / incomplete kwargs (or succeeds with a formula-only note / 1 m
demo). Same class as empty-args payment_certificate. See
uploads/SYNC2_A2_empty_repro.
"""
from __future__ import annotations

import json
import re

import pytest

from app.agents.runtime import (
    _CG_REFUSAL,
    _EMPTY_RESPONSE_FALLBACK,
    _cost_grounding_gate,
    _format_any_calc_result,
    _format_tool_error_plain,
    _graft_complete_calc_answer,
    _looks_like_formula_template,
    _looks_like_tool_error_json,
    _plain_sentence_for_tool_error_json,
    _postprocess_answer,
    _recover_answer_from_tool_messages,
    _sanitize_final_text,
    _text_needs_tool_recovery,
)
from app.lib.construction_formulas import run_calculation


# ── Live asks (synthetic figures only) ──────────────────────────────────────

A3_PADS = (
    "Take off concrete volume for 14 pad footings each 2.4×2.4×0.6 m, "
    "price at SAR 410/m3, include 5% waste and 10% contingency"
)
A3_PIPE = "Price 500 m of 300 mm uPVC drainage pipe at SAR 95 per metre"

# 14 × 2.4 × 2.4 × 0.6 = 48.384; ×1.05 = 50.8032; ×410 = 20,829.31; ×1.10 = 22,912.24
A3_PADS_ANSWER = (
    "Net volume 14 × 2.4 × 2.4 × 0.6 = 48.384 m3. "
    "With 5% waste: 50.803 m3. At the instructed SAR 410/m3 = SAR 20,829.31. "
    "Plus 10% contingency: SAR 22,912.24."
)
A3_PIPE_ANSWER = (
    "500 m × SAR 95 per metre = SAR 47,500 (operator-instructed rate)."
)

UNKNOWN_NONE = json.dumps({
    "status": "error",
    "error": "Unknown calculation 'None'.",
    "available": ["concrete_volume", "rebar_weight", "excavation_volume"],
    "request_id": "req-sync2-a2",
})

REBAR_OK = {
    "status": "success",
    "calculation": "rebar_weight",
    "result": {
        "unit_mass_kg_m": 1.578,
        "total_mass_kg": 12000.0,
        "total_mass_t": 12.0,
        "note": (
            "Unit mass = 1.578 kg/m for Y16; 12 tonnes is 12,000 kg → "
            "7,605 metres run."
        ),
    },
}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _tool(payload, *, name="construction_calc") -> dict:
    return {"role": "tool", "name": name, "content": json.dumps(payload)}


# ── A3: user-supplied rates ground derived arithmetic ───────────────────────


def test_a3_pad_footings_user_rate_with_waste_and_contingency_passes():
    msgs = [_user(A3_PADS)]
    assert _cost_grounding_gate(A3_PADS_ANSWER, None, msgs) == A3_PADS_ANSWER
    assert _cost_grounding_gate(A3_PADS_ANSWER, None, msgs) != _CG_REFUSAL


def test_a3_pipe_user_rate_times_qty_passes():
    msgs = [_user(A3_PIPE)]
    assert _cost_grounding_gate(A3_PIPE_ANSWER, None, msgs) == A3_PIPE_ANSWER
    assert "47,500" in _cost_grounding_gate(A3_PIPE_ANSWER, None, msgs)


def test_a3_plastering_user_gang_rate_times_days_passes():
    ask = (
        "How much will the remaining 3,400 m2 of plastering cost if "
        "productivity stays at 42 m2 per gang-day and a gang costs "
        "SAR 1,950 per day?"
    )
    answer = "3,400 / 42 gang-days × SAR 1,950 = SAR 157,857.14."
    msgs = [_user(ask)]
    assert _cost_grounding_gate(answer, None, msgs) == answer


def test_a3_invented_rate_still_refused():
    msgs = [_user("What is the unit rate for ready-mix C40?")]
    answer = "Use SAR 650/m3 for C40 ready-mix."
    assert _cost_grounding_gate(answer, None, msgs) == _CG_REFUSAL


def test_a3_user_gave_410_but_answer_invents_650_refused():
    msgs = [_user(A3_PADS)]
    answer = "Price the pads at SAR 650/m3 — that is the market rate."
    assert _cost_grounding_gate(answer, None, msgs) == _CG_REFUSAL


# ── A2: tool-error JSON never reaches the user ──────────────────────────────


def test_a2_unknown_calculation_json_is_detected():
    assert _looks_like_tool_error_json(UNKNOWN_NONE) is True
    assert _looks_like_tool_error_json('{"answer": "ok", "result": 1}') is False
    assert _looks_like_tool_error_json("Plain sentence about a calculation.") is False


def test_a2_sanitize_replaces_unknown_calc_json_with_plain_sentence():
    out = _sanitize_final_text(UNKNOWN_NONE)
    assert "{" not in out
    assert "Unknown calculation" not in out
    assert "available" not in out
    assert "calculator" in out.lower() or "formula" in out.lower()
    assert "req-sync2-a2" in out


def test_a2_markdown_fenced_error_json_is_also_plain():
    fenced = "```json\n" + UNKNOWN_NONE + "\n```"
    out = _sanitize_final_text(fenced)
    assert "{" not in out
    assert "req-sync2-a2" in out


def test_a2_format_keeps_request_id_from_payload():
    sentence = _format_tool_error_plain(
        {"status": "error", "error": "Unknown calculation 'None'.",
         "request_id": "abc-123"},
    )
    assert "Request id: abc-123" in sentence
    assert "{" not in sentence


def test_a2_empty_after_failed_calc_is_plain_sentence_not_json():
    msgs = [
        _user("How much will the remaining 3,400 m2 of plastering cost "
              "if productivity stays at 42 m2 per gang-day and a gang "
              "costs SAR 1,950 per day?"),
        _tool(json.loads(UNKNOWN_NONE)),
    ]
    recovered = _recover_answer_from_tool_messages(_EMPTY_RESPONSE_FALLBACK, msgs)
    assert recovered != _EMPTY_RESPONSE_FALLBACK
    assert "{" not in recovered
    assert "unable to generate" not in recovered.lower()
    assert "calculator" in recovered.lower() or "formula" in recovered.lower()


def test_a2_empty_after_successful_calc_uses_tool_note():
    msgs = [
        _user("What is the weight of 12 tonnes of Y16 bars in metres run?"),
        _tool(REBAR_OK),
    ]
    recovered = _recover_answer_from_tool_messages(_EMPTY_RESPONSE_FALLBACK, msgs)
    assert recovered != _EMPTY_RESPONSE_FALLBACK
    assert "unable to generate" not in recovered.lower()
    assert "7,605" in recovered or "1.578" in recovered
    assert "{" not in recovered


def test_a2_postprocess_empty_turn_after_calc_is_not_silence():
    msgs = [
        _user("What is the weight of 12 tonnes of Y16 bars in metres run?"),
        _tool(REBAR_OK),
    ]
    out = _postprocess_answer("", None, msgs)
    assert out.strip()
    assert out != _EMPTY_RESPONSE_FALLBACK
    assert "unable to generate" not in out.lower()
    assert "7,605" in out or "1.578" in out


def test_a2_postprocess_error_json_is_not_raw_json():
    msgs = [
        _user("How much will the remaining 3,400 m2 of plastering cost "
              "if a gang costs SAR 1,950 per day?"),
        _tool(json.loads(UNKNOWN_NONE)),
    ]
    raw = _sanitize_final_text(UNKNOWN_NONE)
    out = _postprocess_answer(raw, None, msgs)
    assert "{" not in out
    assert out != _CG_REFUSAL
    assert "unable to generate" not in out.lower()


def test_a2_text_needs_recovery_on_error_json_and_empty():
    assert _text_needs_tool_recovery("")
    assert _text_needs_tool_recovery(_EMPTY_RESPONSE_FALLBACK)
    assert _text_needs_tool_recovery(UNKNOWN_NONE)
    assert not _text_needs_tool_recovery("Y16 unit mass is 1.578 kg/m.")


def test_a2_plain_helper_matches_sanitize():
    assert _plain_sentence_for_tool_error_json(UNKNOWN_NONE) == _sanitize_final_text(
        UNKNOWN_NONE
    )


# ── Live tip a8498b38: formula-only / 1 m demo after a green calc ───────────

A2_PLASTER = (
    "How much will the remaining 3,400 m2 of plastering cost if "
    "productivity stays at 42 m2 per gang-day and a gang costs "
    "SAR 1,950 per day?"
)
A2_REBAR = "What is the weight of 12 tonnes of Y16 bars in metres run?"

# Live third construction_calc on A2-1 — duration only, formula-name note.
A2_1_LIVE_OK = {
    "status": "success",
    "calculation": "productivity_manpower_duration",
    "result": {
        "formulas_used": ["Duration = Quantity / Daily Production"],
        "standard": "PE formula sheet (Productivity / Manpower / Duration)",
        "duration": 80.9524,
        "duration_units": "days (same period as daily_production)",
        "note": "Duration = Quantity / Daily Production",
    },
    "note": (
        "Deterministic engineering calculation. Any cost figure uses the unit "
        "rates provided (or indicative GCC defaults if none were given)."
    ),
}

# Live second construction_calc on A2-2 — 1 m unit-mass demo.
A2_2_LIVE_OK = {
    "status": "success",
    "calculation": "rebar_weight",
    "result": {
        "unit_mass_kg_m": 1.5783,
        "total_mass_kg": 1.58,
        "total_mass_t": 0.0016,
        "note": (
            "Unit mass = (pi/4)*(16.0/1000)^2*7850 = 1.5783 kg/m; "
            "x 1 m x 1 = 1.58 kg."
        ),
    },
}

A2_1_FORMULA_SHIPPED = (
    "Duration = Quantity / Daily Production\n"
    "3354 of 3354 project documents indexed"
)
A2_2_DEMO_SHIPPED = (
    "Unit mass = (pi/4)*(16.0/1000)^2*7850 = 1.5783 kg/m; "
    "x 1 m x 1 = 1.58 kg.\n"
    "3354 of 3354 project documents indexed"
)


def _has_sar_plaster_cost(text: str) -> bool:
    compact = (text or "").replace(",", "").replace(" ", "")
    return bool(re.search(r"SAR", text or "", re.I)) and "157" in compact


def _has_metres_run(text: str) -> bool:
    compact = (text or "").replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*m(?:etres?)?\b", compact, re.I)
    return bool(match) and float(match.group(1)) > 100


def test_a2_1_live_params_compute_duration_and_cost():
    env = run_calculation("productivity_manpower_duration", {
        "quantity": 3400,
        "unit": "m2",
        "productivity_rate": 42,
        "rate_unit": "m2 per gang-day",
        "crew_cost_per_day": 1950,
    })
    assert env["status"] == "success", env
    inner = env["result"]
    assert inner["duration"] == pytest.approx(3400 / 42, abs=0.01)
    assert inner["total_cost"] == pytest.approx(3400 / 42 * 1950, abs=0.5)
    assert "157" in str(inner["total_cost"])


def test_a2_2_weight_to_length_first_call():
    env = run_calculation("rebar_weight", {
        "bar_diameter_mm": 16,
        "total_weight_kg": 12000,
        "mode": "weight_to_length",
    })
    assert env["status"] == "success", env
    inner = env["result"]
    metres = inner.get("metres_run") or inner["total_length_m"]
    assert metres == pytest.approx(12000 / inner["unit_mass_kg_m"], abs=1.0)
    assert metres > 1000


def test_a2_formula_template_needs_recovery():
    assert _looks_like_formula_template("Duration = Quantity / Daily Production")
    assert _looks_like_formula_template(A2_1_FORMULA_SHIPPED)
    assert _text_needs_tool_recovery("Duration = Quantity / Daily Production")
    assert _text_needs_tool_recovery(A2_1_FORMULA_SHIPPED)
    assert not _text_needs_tool_recovery("Y16 unit mass is 1.578 kg/m.")


def test_a2_1_format_skips_formula_only_note():
    formatted = _format_any_calc_result(A2_1_LIVE_OK)
    assert formatted
    assert "Duration = Quantity / Daily Production" != formatted
    assert "80.95" in formatted or "80.952" in formatted


def test_a2_1_live_formula_note_recovers_sar_cost():
    msgs = [_user(A2_PLASTER), _tool(A2_1_LIVE_OK)]
    recovered = _recover_answer_from_tool_messages(A2_1_FORMULA_SHIPPED, msgs)
    grafted = _graft_complete_calc_answer(recovered, msgs)
    out = _postprocess_answer(A2_1_FORMULA_SHIPPED, None, msgs)
    for text in (grafted, out):
        assert _has_sar_plaster_cost(text), text
        assert "unable to generate" not in text.lower()
        assert "{" not in text


def test_a2_2_live_1m_demo_recovers_metres_run():
    msgs = [_user(A2_REBAR), _tool(A2_2_LIVE_OK)]
    grafted = _graft_complete_calc_answer(A2_2_DEMO_SHIPPED, msgs)
    out = _postprocess_answer(A2_2_DEMO_SHIPPED, None, msgs)
    for text in (grafted, out):
        assert _has_metres_run(text), text
        assert "unable to generate" not in text.lower()


def test_a2_1_empty_turn_after_duration_only_calc_has_cost():
    msgs = [_user(A2_PLASTER), _tool(A2_1_LIVE_OK)]
    out = _postprocess_answer("", None, msgs)
    assert _has_sar_plaster_cost(out), out


def test_a2_2_empty_turn_after_1m_demo_has_metres():
    msgs = [_user(A2_REBAR), _tool(A2_2_LIVE_OK)]
    out = _postprocess_answer(_EMPTY_RESPONSE_FALLBACK, None, msgs)
    assert _has_metres_run(out), out


def test_a2_empty_kwargs_name_required_params_with_units():
    """Empty / incomplete construction_calc must name every required input."""
    prod = run_calculation("productivity_manpower_duration", {})
    assert prod["status"] == "error"
    err = prod["error"]
    assert "quantity (qty)" in err
    assert "daily_production (qty/day)" in err
    assert "crew_cost_per_day (currency/day)" in err
    assert "productivity_rate (qty/gang-day)" in err

    rebar = run_calculation("rebar_weight", {})
    assert rebar["status"] == "error"
    err = rebar["error"]
    assert "bar_diameter_mm (mm)" in err
    assert "total_length_m (m)" in err
    assert "total_weight_kg (kg)" in err
    assert "weight_to_length" in err
    assert "missing 1 required positional" not in err.lower()

    incomplete = run_calculation("rebar_weight", {
        "bar_diameter_mm": 16, "total_weight_kg": 12000, "mode": "weight_to_length",
    })
    assert incomplete["status"] == "success", incomplete


def test_a2_empty_kwargs_plain_error_keeps_units():
    env = run_calculation("rebar_weight", {})
    sentence = _format_tool_error_plain(env)
    assert "bar_diameter_mm (mm)" in sentence
    assert "total_weight_kg (kg)" in sentence
    assert "{" not in sentence
