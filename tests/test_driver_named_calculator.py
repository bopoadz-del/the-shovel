"""An ask that names a platform calculator tells the model so before it
chooses: what that calculator computes comes from running it. Synthetic
calculator, scripted model."""
from __future__ import annotations

from tests.lane5_core_support import Param, call, drive, registered


def _gate(blend: float) -> dict:
    return {"value": blend * 2}


_INPUTS = {"blend": Param("ratio", lo=0, hi=1, label="blend")}


def test_the_model_is_told_the_ask_names_a_calculator(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)
    with registered("lane5_named_gate", _gate, "Qxnet blend gate", _INPUTS, {"value": "ratio"}):
        _events, seen = drive(monkeypatch, "What is the qxnet blend gate for a blend of 0.4?", [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": "lane5_named_gate", "params": {"blend": 0.4}}, "c1")]},
            {"content": "The blend gate is 0.8."},
        ])
    first = seen[0]["messages"]
    notes = [m["content"] for m in first if m.get("role") == "system" and "Qxnet blend gate" in m["content"]]
    assert notes and "lane5_named_gate" in notes[0]
    assert first[-1]["role"] == "user"  # the ask stays the last message


def test_an_ask_that_names_no_calculator_gets_no_note(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)
    _events, seen = drive(monkeypatch, "Summarise the meeting notes.", [{"content": "Done."}])
    assert not [m for m in seen[0]["messages"] if m.get("role") == "system"
                and "names the platform calculator" in m["content"]]
