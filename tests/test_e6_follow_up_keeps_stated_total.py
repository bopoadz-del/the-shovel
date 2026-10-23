"""E6 follow-up must price the stated total, not one pile cap.

Live SET5 close-out, tip 209bc83, one chat in six:

    Seed: "How much concrete for 24 pile caps, each 2.5 m by 2.5 m by 1.2 m deep?"
    Follow-up: "Add 7% waste to that total and price it at SAR 420 per cubic metre."

The right continuation is the stated total, 24 x 7.5 = 180 m3 net, then the
waste the operator just named: 180 x 1.07 = 192.60 m3, x SAR 420 = SAR 80,892.

The miss answered 8.025 m3 (one cap, 7.5 x 1.07) and about SAR 3,370.50.
The documented 5% default overriding a stated 7% is a different bug and is
already gone — these tests do not ask for that.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _inject_user_ask_into_construction_calc_args,
    _postprocess_answer,
)
from app.lib.construction_formulas import run_calculation

SEED = (
    "How much concrete for 24 pile caps, each 2.5 m by 2.5 m by 1.2 m deep?"
)
FOLLOW = "Add 7% waste to that total and price it at SAR 420 per cubic metre."

NET = 24 * 2.5 * 2.5 * 1.2  # 180
WITH_WASTE = NET * 1.07  # 192.60
PRICE = WITH_WASTE * 420  # 80892
ONE_UNIT = 2.5 * 2.5 * 1.2 * 1.07  # 8.025
ONE_UNIT_PRICE = ONE_UNIT * 420  # 3370.50


def _compact(text: str) -> str:
    return (text or "").replace(",", "").replace(" ", "")


def _msgs(answer_tool: dict | None = None) -> list[dict]:
    rows: list[dict] = [
        {"role": "user", "content": SEED},
        {
            "role": "assistant",
            "content": "Net concrete for 24 pile caps is 180 m³ (24 × 2.5 × 2.5 × 1.2).",
        },
        {"role": "user", "content": FOLLOW},
    ]
    if answer_tool is not None:
        rows.append(answer_tool)
    return rows


def _assert_stated_total(text: str) -> None:
    compact = _compact(text)
    assert "192.60" in compact or "192.6" in compact
    assert "80892" in compact
    # The one-cap miss, with or without its price, must not be the answer.
    assert "8.025" not in compact
    assert "3370" not in compact


def test_the_sheets_arithmetic_is_192_60_and_80892():
    assert NET == pytest.approx(180)
    assert WITH_WASTE == pytest.approx(192.60)
    assert PRICE == pytest.approx(80892)
    assert ONE_UNIT == pytest.approx(8.025)
    assert ONE_UNIT_PRICE == pytest.approx(3370.50)


def test_concrete_volume_does_not_drop_a_passed_quantity_of_24():
    """Bind used to discard quantity — concrete_volume has no such parameter —
    so 24 unit caps came back as one cap."""
    result = run_calculation(
        "concrete_volume",
        {
            "length_m": 2.5,
            "width_m": 2.5,
            "thickness_m": 1.2,
            "waste_factor": 0.07,
            "quantity": 24,
            "text": FOLLOW,
        },
    )
    assert result.get("status") == "success"
    inner = result["result"]
    assert inner["net_volume_m3"] == pytest.approx(NET)
    assert inner["volume_m3"] == pytest.approx(WITH_WASTE)
    assert inner["volume_m3"] != pytest.approx(ONE_UNIT)


def test_follow_up_calc_keeps_the_count_from_the_prior_turn():
    """Unit dims on the tool call, count only in the previous ask.

    That is the handoff that priced one cap: length/width/thickness are the
    'each' size, and nothing multiplies by 24.
    """
    result = run_calculation(
        "concrete_volume",
        {
            "length_m": 2.5,
            "width_m": 2.5,
            "thickness_m": 1.2,
            "waste_factor": 0.07,
            "text": FOLLOW,
            "prior_text": SEED,
        },
    )
    assert result.get("status") == "success"
    inner = result["result"]
    assert inner["net_volume_m3"] == pytest.approx(NET)
    assert inner["volume_m3"] == pytest.approx(WITH_WASTE)
    assert inner["waste_factor"] == pytest.approx(0.07)
    assert inner["volume_m3"] != pytest.approx(ONE_UNIT)
    # 180 x 1.05 = 189 is the documented default. The follow-up said 7%.
    assert inner["volume_m3"] != pytest.approx(NET * 1.05)


def test_one_unit_follow_up_answer_is_replaced_with_the_stated_total():
    bad = (
        "With 7% waste that is 8.025 m³ "
        "(2.5 × 2.5 × 1.2 × 1.07). At SAR 420 per cubic metre: SAR 3,370.50."
    )
    out = _postprocess_answer(bad, None, _msgs())
    _assert_stated_total(out)


def test_a_follow_up_that_already_states_the_total_is_kept():
    good = (
        "Continuing from 180 m³ net. With 7% waste: 192.60 m³. "
        "At SAR 420/m³ that is SAR 80,892.00."
    )
    out = _postprocess_answer(good, None, _msgs())
    assert out == good
    _assert_stated_total(out)


def test_handoff_injects_the_prior_count_before_the_calc_runs():
    args = _inject_user_ask_into_construction_calc_args(
        {
            "calculation": "concrete_volume",
            "length_m": 2.5,
            "width_m": 2.5,
            "thickness_m": 1.2,
            "waste_factor": 0.07,
        },
        FOLLOW,
        history=[{"role": "user", "content": SEED}],
    )
    params = {
        key: value
        for key, value in args.items()
        if key not in ("calculation",)
    }
    result = run_calculation(args.get("calculation") or "concrete_volume", params)
    inner = result["result"]
    assert inner["volume_m3"] == pytest.approx(WITH_WASTE)
    assert inner["net_volume_m3"] == pytest.approx(NET)
    assert inner["volume_m3"] != pytest.approx(ONE_UNIT)


def test_a_single_pile_cap_is_not_multiplied_up_to_24():
    """No stated count means one element. Do not invent 24."""
    result = run_calculation(
        "concrete_volume",
        {
            "length_m": 2.5,
            "width_m": 2.5,
            "thickness_m": 1.2,
            "waste_factor": 0.07,
            "text": FOLLOW,
            "prior_text": (
                "How much concrete for a pile cap, 2.5 m by 2.5 m by 1.2 m deep?"
            ),
        },
    )
    inner = result["result"]
    assert inner["volume_m3"] == pytest.approx(ONE_UNIT)
    assert inner["net_volume_m3"] == pytest.approx(7.5)
