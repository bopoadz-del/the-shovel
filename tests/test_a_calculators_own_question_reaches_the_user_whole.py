"""A calculator's own question reaches the user whole.

Lane 2 + lane 5 on 838e85a (6/6): an out-of-range percent got "The bulking
given (150%) is outside what the Excavation volume calculator accepts." --
no range, no question. The calculator's question says "(0% to 100%). What
bulking should I use?"; the driver released it, and the exit figure check
removed the range it could not source, and the question with it. The
platform's own words come from the tool's result and keep their figures; the
model's words are checked as before. Invented pit.
"""
from tests.lane5_core_support import answer_text, call, drive, install_driver_stream


def test_an_out_of_range_percent_states_the_range_and_asks(monkeypatch):
    install_driver_stream(monkeypatch)
    events, _ = drive(monkeypatch, "Pit 10 x 8 x 3 m, bulking 150%. Loose volume?", [
        {"content": "", "tool_calls": [call("construction_calc", {"calculation": "excavation_volume", "params": {
            "length_m": 10, "width_m": 8, "depth_m": 3, "bulking_factor": 1.5}}, "c1")]},
        {"content": "It is outside the range."},
    ])
    text = answer_text(events)
    assert "0% to 100%" in text
    assert "?" in text and "What bulking should I use" in text


def test_the_models_own_unbacked_figure_is_still_removed(monkeypatch):
    install_driver_stream(monkeypatch)
    events, _ = drive(monkeypatch, "What is the loose volume of a 10 x 8 x 3 m pit?", [
        {"content": "The loose volume is 987.65 m3."},
    ])
    assert "987.65" not in answer_text(events)
