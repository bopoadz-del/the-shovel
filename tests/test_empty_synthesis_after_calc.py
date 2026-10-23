"""Cost-gate A3-1: construction_calc succeeds, force_synthesis yields 0 tokens.

Live SO probe on a8498b3 (SO_A_probe_a8498b3.md):

    "Take off the concrete volume for 14 pad footings, each 2.4 x 2.4 x 0.6 m,
     and price it at SAR 410 per m3. Also show 5% waste and 10% contingency
     on the concrete cost."

    construction_calc succeeded, then 0 synthesis tokens — silent blank bubble.

Class: a successful concrete take-off plus a user-supplied unit rate must
never ship empty / fallback / volume-only. Compose volume × rate × waste ×
contingency from the ask (do not invent a rate). Parked D1 / C3 and
#643/#645/#649 stay out of scope.
"""
from __future__ import annotations

import json

from app.agents.runtime import (
    _EMPTY_RESPONSE_FALLBACK,
    _SYNTH_CUTOFF_NOTICE,
    _postprocess_answer,
    _recover_answer_from_tool_messages,
)
from app.lib.construction_formulas_commercial import (
    compose_user_priced_takeoff_from_ask,
    format_user_priced_takeoff_line,
    query_asks_user_priced_takeoff,
)


# Exact live probe ask (spacing / "x" not "×").
A3_1 = (
    "Take off the concrete volume for 14 pad footings, each 2.4 x 2.4 x 0.6 m, "
    "and price it at SAR 410 per m3. Also show 5% waste and 10% contingency "
    "on the concrete cost."
)

# 14 × 2.4 × 2.4 × 0.6 = 48.384; ×1.05 = 50.8032;
# ×410 = 20,829.31; ×1.10 = 22,912.24
NET = 48.384
WASTE_VOL = 50.8032
BASE_COST = 20_829.31
TOTAL = 22_912.24

E4_RAFT = (
    "Concrete volume for a raft 30x20x1.5 m including your documented waste "
    "factor."
)
A7 = "What is the Advance Payment amount under the contract?"
NO_RATE = (
    "Take off the concrete volume for 14 pad footings, each 2.4 x 2.4 x 0.6 m."
)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _calc_tool(payload: dict) -> dict:
    return {
        "role": "tool",
        "name": "construction_calc",
        "content": json.dumps(payload),
    }


def _volume_ok(*, volume: float, net: float | None = None) -> dict:
    inner = {"volume_m3": volume, "waste_factor": 0.05}
    if net is not None:
        inner["net_volume_m3"] = net
    return {
        "status": "success",
        "calculation": "concrete_volume",
        "result": inner,
    }


def _msgs(ask: str = A3_1, payload: dict | None = None) -> list[dict]:
    extra = [] if payload is None else [_calc_tool(payload)]
    return [_user(ask), *extra]


def _assert_priced(out: str) -> None:
    compact = (out or "").replace(",", "").replace(" ", "")
    assert out.strip(), "blank bubble after successful calc"
    assert out.strip() != _EMPTY_RESPONSE_FALLBACK
    assert "48.384" in compact
    assert "410" in compact
    assert "20829.31" in compact
    assert "22912.24" in compact


# ── Ask class ──────────────────────────────────────────────────────────────


def test_live_a3_1_is_a_user_priced_takeoff_ask():
    assert query_asks_user_priced_takeoff(A3_1)
    assert not query_asks_user_priced_takeoff(E4_RAFT)
    assert not query_asks_user_priced_takeoff(A7)
    assert not query_asks_user_priced_takeoff(NO_RATE)


def test_compose_from_the_live_ask_is_22912():
    composed = compose_user_priced_takeoff_from_ask(A3_1)
    assert composed is not None
    assert abs(composed["net_volume_m3"] - NET) < 1e-6
    assert abs(composed["volume_with_waste_m3"] - WASTE_VOL) < 1e-6
    assert abs(composed["base_cost"] - BASE_COST) < 0.02
    assert abs(composed["total_cost"] - TOTAL) < 0.02
    assert composed["unit_rate"] == 410.0
    assert composed["currency"] == "SAR"
    line = format_user_priced_takeoff_line(composed)
    _assert_priced(line)


def test_compose_does_not_invent_a_rate():
    assert compose_user_priced_takeoff_from_ask(NO_RATE) is None
    assert compose_user_priced_takeoff_from_ask(E4_RAFT) is None


# ── Empty / 0-token synthesis after a successful calc ──────────────────────


def test_empty_force_synthesis_after_full_volume_calc_composes_price():
    """Live shape: calc returned the take-off, synthesis emitted 0 tokens."""
    msgs = _msgs(payload=_volume_ok(volume=WASTE_VOL, net=NET))
    out = _postprocess_answer("", None, msgs)
    _assert_priced(out)


def test_empty_fallback_after_calc_is_not_a_blank_or_volume_only_bubble():
    msgs = _msgs(payload=_volume_ok(volume=WASTE_VOL, net=NET))
    out = _postprocess_answer(_EMPTY_RESPONSE_FALLBACK, None, msgs)
    _assert_priced(out)
    assert "unable to generate" not in out.lower()


def test_one_footing_tool_volume_does_not_drop_the_count_of_14():
    """fw-calc / L×W×D steal often returns one pad (3.456 m3)."""
    msgs = _msgs(payload=_volume_ok(volume=3.6288, net=3.456))
    recovered = _recover_answer_from_tool_messages("", msgs)
    # Volume-only recovery is not enough for A3-1.
    out = _postprocess_answer(recovered or "", None, msgs)
    _assert_priced(out)
    assert "3.456" not in out.replace(",", "")


def test_unwrap_hostile_envelope_still_composes_from_the_ask():
    """If the tool envelope cannot be formatted, still price the ask."""
    payload = {
        "ok": True,
        "name": "construction_calc",
        "result": {
            "status": "success",
            "calculation": "concrete_volume",
            "data": {"note": "ok"},
        },
    }
    out = _postprocess_answer("", None, _msgs(payload=payload))
    _assert_priced(out)


def test_successful_calc_plus_truly_empty_never_ships_blank(monkeypatch):
    """Last-chance: if compose is off, emit a cut-off sentence, not ''."""
    monkeypatch.setenv("COMPOSE_USER_PRICED_TAKEOFF", "0")
    monkeypatch.setenv("COMPOSE_CONCRETE_VOLUME", "0")
    msgs = _msgs(payload=_volume_ok(volume=WASTE_VOL, net=NET))
    # Recovery may still write a volume line; strip that path by using
    # an envelope with no formatable volume and empty synthesis.
    hostile = {
        "status": "success",
        "calculation": "concrete_volume",
        "result": {"ok": True},
    }
    msgs = _msgs(payload=hostile)
    out = _postprocess_answer("", None, msgs)
    assert out.strip()
    assert "22912" not in out.replace(",", "")
    assert out.strip() in (
        _SYNTH_CUTOFF_NOTICE,
        _EMPTY_RESPONSE_FALLBACK,
    ) or _SYNTH_CUTOFF_NOTICE in out


def test_e4_raft_without_a_user_rate_is_not_priced():
    out = _postprocess_answer("", None, _msgs(E4_RAFT))
    compact = (out or "").replace(",", "")
    assert "20,829.31" not in out
    assert "20829.31" not in compact
    assert "22,912.24" not in out
