"""#392 grader point 6: the REAL turn loop, the model stubbed at the last hop.

The driver loop runs as it does for a user (tool dispatch, the calculator,
post-processing, the token stream). Only the provider call is scripted: the
model asks for a calculation, the calculator succeeds, and the model's
synthesis comes back empty -- or as markdown with no words. The user must
see the calculator's figure, never the generic "unable to generate" line.

Invented dimensions; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK
from app.lib import construction_formulas as cf
from tests.lane5_core_support import answer_text, call, drive, install_driver_stream

PITS = [(9, 4, 2), (13, 5, 1.5)]
EMPTY_SYNTHESES = ["", "**\n\n---\n* "]


def _bank(length, width, depth) -> float:
    out = cf.run_calculation("excavation_volume",
                             {"length_m": length, "width_m": width, "depth_m": depth})
    assert out["status"] == "success", out
    return float(out["result"]["bank_volume_m3"])


@pytest.mark.parametrize("synthesis", EMPTY_SYNTHESES)
@pytest.mark.parametrize("pit", PITS)
def test_a_successful_calculation_with_an_empty_synthesis_shows_the_figure(
        monkeypatch, pit, synthesis):
    length, width, depth = pit
    install_driver_stream(monkeypatch)
    events, seen = drive(monkeypatch, f"Bank volume of a pit {length} x {width} x {depth} m?", [
        {"content": "", "tool_calls": [call("construction_calc", {
            "calculation": "excavation_volume",
            "params": {"length_m": length, "width_m": width, "depth_m": depth}}, "c1")]},
        {"content": synthesis},
    ])
    text = answer_text(events)
    bank = _bank(length, width, depth)
    assert f"{bank:g}" in text.replace(",", ""), text
    assert _EMPTY_RESPONSE_FALLBACK not in text
    assert "unable to generate" not in text.lower()
    assert "---" not in text
    # The calculator ran inside the loop: the model saw its result.
    assert any(m.get("role") == "tool" for m in seen[-1]["messages"])


@pytest.mark.parametrize("synthesis", EMPTY_SYNTHESES)
def test_a_failed_calculation_with_an_empty_synthesis_says_why(monkeypatch, synthesis):
    install_driver_stream(monkeypatch)
    events, _ = drive(monkeypatch, "Bank volume of the pit?", [
        {"content": "", "tool_calls": [call("construction_calc", {
            "calculation": "zz_no_such_calculation", "params": {}}, "c1")]},
        {"content": synthesis},
    ])
    text = answer_text(events)
    assert text.strip(), events
    assert _EMPTY_RESPONSE_FALLBACK not in text
    assert "unable to generate" not in text.lower()
    # The plain sentence that the tool failed -- never the calculator's own
    # error text; the wordless reply is not shown. The calculation name the
    # MODEL sent is not in the user's question, so it is not echoed.
    from app.agents.core.computed_answer import LABELS

    assert "zz_no_such_calculation" not in text, text
    assert f"{LABELS['en']['asked']}:" not in text, text
    assert LABELS["en"]["tool_error"] in text, text
    assert "Unknown calculation" not in text, text
    assert "---" not in text
