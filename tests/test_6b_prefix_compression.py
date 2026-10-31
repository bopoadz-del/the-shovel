"""6b part (b): the no-hat construction_calc prefix is compressed -- every
other hat's calculator stays CALLABLE (in the enum) but its prefix line is just
name + display, not its full input list (~5k tokens saved on the no-hat turn).
ORDERS_31OCT_1355Z path B + compression: reachability kept, lines compressed.

Also covers driver/loop.py::stream (where the 6b llm_dispatch stamp lives): a
no-note turn must not inject an empty system message -- the stream() `if note:`
drop_rule mutant reddens that test.
"""
from __future__ import annotations

import json

import pytest

# Dotted imports so CI-M selects these for the changed modules.
import app.agents.driver.tools as dtools
import app.agents.driver.loop as driver_loop  # noqa: F401  (selection anchor for stream())
from app.agents import catalog
from app.lib import formula_registry as fr

_HATS = sorted({h.discipline.value for h in catalog.list_hats()})
_A_HAT = _HATS[0] if _HATS else None


@pytest.mark.parametrize("hat", [None, _A_HAT])
def test_every_calculator_stays_reachable_in_the_enum(hat):
    # Compression must NOT drop reachability: the enum lists every registered
    # formula regardless of hat (path B). This is the invariant the cut would
    # have broken; it stays intact.
    enum = dtools.calculator_schema(hat)["function"]["parameters"]["properties"]["calculation"]["enum"]
    assert set(enum) == {s.name for s in fr.all_specs()}


@pytest.mark.parametrize("hat", [None, _A_HAT])
def test_other_hat_calculators_are_named_but_compressed(hat):
    # Each unselected-hat formula appears as a terse "- name: display" line (so
    # the model knows it exists and can call it), WITHOUT its per-input labels.
    # Dropping the `if others:` block (drop_rule) would strip these names ->
    # this reddens it.
    desc = dtools.calculator_schema(hat)["function"]["description"]
    base = {s.name for s in fr.specs_for(hat)}
    others = [s for s in fr.all_specs() if s.name not in base]
    for spec in others:
        line_at = desc.find(f"- {spec.name}:")
        assert line_at >= 0, f"other calculator not named: {spec.name}"
        line = desc[line_at:desc.find(chr(10), line_at)]
        # compressed: the verbose "Inputs: ... = ..." label list is gone
        assert "Inputs:" not in line, (spec.name, line)


@pytest.mark.parametrize("hat", [None, _A_HAT])
def test_selected_hat_formulas_keep_their_full_input_lines(hat):
    # The base / selected-hat formulas are NOT compressed -- they keep "Inputs:"
    # so the model fills params directly (only unselected hats are terse).
    desc = dtools.calculator_schema(hat)["function"]["description"]
    for spec in fr.specs_for(hat):
        if not fr.parameters(spec):
            continue
        at = desc.find(f"- {spec.name}:")
        assert at >= 0
        line = desc[at:desc.find(chr(10), at) if desc.find(chr(10), at) > 0 else len(desc)]
        assert "Inputs:" in line, (spec.name, line)


@pytest.mark.parametrize("phrasing", [
    "Please use the Lane5 Widget Sizer.",
    "Run the Lane5 Widget Sizer calculator for me.",
])
def test_stream_injects_the_named_calculator_note_so_it_reaches_the_model(monkeypatch, phrasing):
    # stream()'s first `if` is `if note:` guarding the named-calculator-note
    # insert. When the ask NAMES a calculator, that note must reach the model.
    # drop_rule replaces `if note:` with `pass`, so the insert never runs and
    # the note never reaches the model -> this assertion reddens the mutant.
    # Driving the turn also covers the stream() call site that carries the 6b
    # llm_dispatch stamp.
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
