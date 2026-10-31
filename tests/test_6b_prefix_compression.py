"""6b part (a) coverage for the driver generation-dispatch call site.

The llm_first_token stamp lives in app/agents/driver/loop.py::stream (at t0).
CI-M mutates stream() because part (a) touches it; its first `if` is `if note:`
(the named-calculator-note insert). drop_rule replaces that `if` with `pass`,
so the note never reaches the model. A turn that NAMES a calculator must carry
that note into the messages the model sees -> this reddens the mutant. Driving
the turn also exercises the stream() call site that carries the dispatch stamp.

(Part (b) prefix compression was dropped: the ~5k "Other calculators" input
labels are load-bearing per tested contracts -- test_delay_damages_ask_figure_
bound_once needs delay_damages_daily's full `amount_per_day = ...` label at
no-hat, and test_an_optional_input needs the `(optional)` markers -- so a real
reduction needs those tests updated, which is an owner/orchestrator call. See
REPORT_MAIN_0015Z / 0030Z. tools.py is unchanged vs main.)
"""
from __future__ import annotations

import pytest

# Dotted import so CI-M selects this test for app.agents.driver.loop::stream.
import app.agents.driver.loop as driver_loop  # noqa: F401


@pytest.mark.parametrize("phrasing", [
    "Please use the Lane5 Widget Sizer.",
    "Run the Lane5 Widget Sizer calculator for me.",
])
def test_stream_injects_the_named_calculator_note_so_it_reaches_the_model(monkeypatch, phrasing):
    from tests.lane5_core_support import Param, drive, install_driver_stream, registered

    def fn(width_m: float):
        return {"value": width_m}

    install_driver_stream(monkeypatch)  # route driver llm.stream through the scripted llm.call
    with registered("lane5_widget_sizer", fn, "Lane5 Widget Sizer",
                    {"width_m": Param("m", lo=0, hi=1e6, label="width")}, {"value": "m"}):
        _events, seen = drive(monkeypatch, phrasing, [{"content": "ok"}])
    assert seen, "the driver made no model call"
    note_reached = any(
        m.get("role") == "system" and "lane5_widget_sizer" in (m.get("content") or "")
        for call in seen for m in call["messages"]
    )
    assert note_reached, "named-calculator note did not reach the model (stream `if note:` dropped?)"
