"""An out-of-range percentage for a fraction input is quoted as the user
wrote it: 150% reads 150%, not 1.5%. Synthetic calculator."""
from __future__ import annotations

from tests.lane5_core_support import Param, registered


def _share(amount: float, share: float) -> dict:
    return {"value": amount * share}


def test_an_out_of_range_percentage_is_quoted_as_written():
    from app.lib.construction_formulas import run_calculation

    with registered("lane5_share_gate", _share, "Qx share", {
        "amount": Param("currency", lo=0, hi=1e12, label="amount"),
        "share": Param("fraction", lo=0, hi=1, label="share"),
    }, {"value": "currency"}):
        env = run_calculation("lane5_share_gate", {"text": "qx share of 500,000 at 150%",
                                                   "amount": 500000, "share": 150})
    assert env.get("needs_input") is True, env
    assert "150%" in env["question"] and "1.5%" not in env["question"]
    assert "0% to 100%" in env["question"]


def _grow(volume: float, swell: float = 0.2) -> dict:
    return {"value": volume * (1 + swell)}


def test_a_converted_percentage_is_not_divided_again():
    """150% converts to 1.5 for a 0..1 input and is rejected, never re-read
    as 1.5% and computed (lane 5: bulking 150% computed as 1.5%)."""
    from app.lib.construction_formulas import run_calculation

    with registered("lane5_swell_gate", _grow, "Qx swell", {
        "volume": Param("m3", lo=0, hi=1e9, label="volume"),
        "swell": Param("fraction", lo=0, hi=1, label="swell", required=False),
    }, {"value": "m3"}):
        bad = run_calculation("lane5_swell_gate", {"volume": 240, "swell": "150%"})
        raw = run_calculation("lane5_swell_gate", {"volume": 240, "swell": 150})
        ok = run_calculation("lane5_swell_gate", {"volume": 240, "swell": "20%"})
    assert bad.get("needs_input") is True and "150%" in bad["question"], bad
    assert raw.get("needs_input") is True and "150%" in raw["question"] and "15,000%" not in raw["question"], raw
    assert ok["result"]["value"] == 288
