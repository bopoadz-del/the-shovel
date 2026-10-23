"""A tool that answered "I can't" has not delivered anything.

Live on d8d9573, master_corpus, one run in three:

    "If Milestone 1 is 30 days late, what are the milestone delay damages?"

The user was shown this, verbatim, as the answer:

    <｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="construction_calc">...
    The calculator returned a null result -- it did not accept the parameter
    names I passed ... I won't substitute a hand-computed number.

Two defects, one behind the other.

1. The model called ``construction_calc`` with a calculation name that does
   not exist. The tool said so, clearly, and listed the 84 that do. But the
   envelope around that reply was ``ok: True`` -- the block RAN -- and
   ``_should_force_synthesis`` only looked at the envelope. It declared the
   deliverable delivered and disarmed every tool. The model, holding an error
   and the list of valid names, could not try again.
2. So it tried anyway, in the only channel left: it wrote the tool call as
   text. The streamed-synthesis leak guard knew about XML tool markup and raw
   tool JSON. It did not know DSML, and sent it to the browser.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    _looks_like_tool_markup_leak,
    _should_force_synthesis,
    _strip_dsml,
)
from app.lib.construction_formulas import run_calculation

# Exact live text. Do not tidy it.
LIVE_LEAK = (
    "<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name=\"construction_calc\">"
    "<｜｜DSML｜｜ parameter name=\"calculation\" string=\"true\">"
    "calculate_delay_damages</｜｜DSML｜｜ parameter>"
    "<｜｜DSML｜｜ parameter name=\"params\" string=\"false\">"
    "{\"contract_price\": 1000000000.0, \"rate_percent_per_day\": 0.015, "
    "\"days\": 30}</｜｜DSML｜｜ parameter></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>"
    "The calculator returned a null result."
)
LIVE_LEAK_AFTER_PROSE = (
    "The calculator returned a zeroed result. Let me re-run it with the "
    "correct inputs." + LIVE_LEAK
)


def _envelope(payload, *, name="construction_calc", ok=True):
    """What the tool loop holds: the block ran (ok), and said <payload>."""
    return {"name": name, "ok": ok, "result": payload}


# ── 1. a failed calculation is not the deliverable ────────────────────────

def test_the_live_call_really_is_an_error_with_the_valid_names_attached():
    """Pins the premise: the tool's reply was good. The model was given what
    it needed to recover, and then had the tool taken away."""
    reply = run_calculation("calculate_delay_damages", {"days": 30})
    assert reply["status"] == "error"
    assert "Unknown calculation" in reply["error"]
    assert "delay_damages_daily" in reply["available"]


def test_unknown_name_still_recovers_when_ask_text_names_the_calculator():
    """Ask prose may name the real calculator; the unknown name string must not."""
    recovered = run_calculation("calculate_delay_damages", {
        "text": "delay_damages_daily",
        "rate_percent": 0.015,
        "contract_amount": 1_000_000_000,
    })
    assert recovered.get("status") == "success", recovered
    assert recovered.get("calculation") == "delay_damages_daily"
    assert recovered["result"]["daily_amount"] == pytest.approx(150_000.0)


def test_an_unknown_calculation_does_not_disarm_the_tools():
    reply = run_calculation("calculate_delay_damages", {"days": 30})
    assert not _should_force_synthesis(_envelope(reply))


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "error", "error": "Bad parameters for concrete_volume: ..."},
        {"status": "failed", "error": "boq_processor: no rows parsed"},
        {"error": "No file_path provided"},
        # Blocks nest their own reply one level down.
        {"status": "success", "result": {"status": "error", "error": "Unknown calculation"}},
    ],
)
def test_any_tool_that_reports_failure_in_its_payload_is_not_a_deliverable(payload):
    """Shape-invariance: not only construction_calc, and not only one spelling
    of failure."""
    assert not _should_force_synthesis(_envelope(payload, name="some_block"))


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "success", "calculation": "concrete_volume", "result": {"value": 945.0}},
        {"status": "success", "result": {"items": [1, 2, 3], "total": 40560.0}},
        # A successful result may legitimately CONTAIN the word, or an empty
        # error slot. Neither is a failure.
        {"status": "success", "result": {"value": 1.0, "error": None}},
        {"status": "success", "result": {"errors": [], "value": 2.0}},
        {"status": "success", "result": {"note": "margin of error 2%", "value": 3.0}},
    ],
)
def test_a_tool_that_delivered_still_forces_synthesis(payload):
    """The control. Force-synthesis exists for a reason; a fix that switched
    it off for good results would bring back the endless tool loop."""
    assert _should_force_synthesis(_envelope(payload))


def test_an_envelope_level_failure_is_still_honoured():
    ok_payload = {"status": "success", "result": {"value": 1.0}}
    assert not _should_force_synthesis(_envelope(ok_payload, ok=False))


# ── 2. DSML is tool markup, wherever it is streamed from ──────────────────

def test_the_live_leak_is_recognised_as_tool_markup():
    assert _looks_like_tool_markup_leak(LIVE_LEAK)
    assert _looks_like_tool_markup_leak(LIVE_LEAK_AFTER_PROSE)


def test_it_is_recognised_as_soon_as_the_marker_has_arrived():
    """The guard runs on every streamed delta. It has to fire on the first
    fragment, before a single line of markup is flushed to the browser."""
    assert _looks_like_tool_markup_leak("Some prose.\n<｜｜DSML｜｜ cal")
    assert _looks_like_tool_markup_leak("<|DSML|invoke name=")


@pytest.mark.parametrize(
    "text",
    [
        "The milestone delay damages are SAR 7,895,270.05 for 30 days.",
        "| Item | Value |\n|---|---|\n| Rate | 0.015% |",
        "Use the formula <rate> x <days> x <Contract Price>.",
        "The DSM-L classification in the drawing register is not a tool call.",
    ],
)
def test_an_ordinary_answer_is_not_mistaken_for_markup(text):
    assert not _looks_like_tool_markup_leak(text)


def test_what_is_kept_after_a_leak_never_contains_the_markup():
    for leak in (LIVE_LEAK, LIVE_LEAK_AFTER_PROSE):
        kept = _strip_dsml(leak)
        assert "DSML" not in kept and "invoke" not in kept
        assert "contract_price" not in kept
