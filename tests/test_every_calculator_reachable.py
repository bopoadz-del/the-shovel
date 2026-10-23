"""A calculator the turn needs is never out of reach before its hat is
chosen: the calculator tool lists every registered calculator, the current
hat's in full and the rest briefly, and a call to any of them runs."""
from __future__ import annotations

from tests.lane5_core_support import Param, call, drive, install_driver_stream, registered, tool_envelope


def test_every_registered_calculator_is_offered_with_no_hat():
    from app.agents.driver import tools
    from app.lib import formula_registry

    schema = tools.calculator_schema(None)
    enum = schema["function"]["parameters"]["properties"]["calculation"]["enum"]
    assert set(enum) == {s.name for s in formula_registry.all_specs()}


def test_a_hat_calculator_runs_before_any_hat_is_chosen(monkeypatch):
    install_driver_stream(monkeypatch)

    def fn(span_m: float) -> dict:
        return {"value": span_m * 3}

    from tests import lane5_core_support as support

    original = support.install_formula

    def as_hat(name, fn, display, inputs, outputs):  # registered by a hat, not base
        wrapped = original(name, fn, display, inputs, outputs)
        from app.lib import formula_registry
        spec = formula_registry.get(name)
        object.__setattr__(spec, "owner", "design") if hasattr(spec, "__dataclass_fields__") else None
        return wrapped

    monkeypatch.setattr(support, "install_formula", as_hat)
    with registered("lane5_hat_calc", fn, "Qx hat span", {"span_m": Param("m", lo=0, hi=1e6, label="span")},
                    {"value": "kN"}):
        from app.agents.driver import tools
        enum = tools.calculator_schema(None)["function"]["parameters"]["properties"]["calculation"]["enum"]
        assert "lane5_hat_calc" in enum
        _events, seen = drive(monkeypatch, "qx hat span for a span of 4 m", [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": "lane5_hat_calc", "params": {"span_m": 4}}, "c1")]},
            {"content": "It is 12 kN."},
        ])
    env = tool_envelope(seen)
    assert env.get("status") == "success" and env["result"]["value"] == 12
