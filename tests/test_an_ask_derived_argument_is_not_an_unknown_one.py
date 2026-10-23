"""An argument the server read out of the ask's words is never an "unknown
argument" that fails the run.

Lane 5 on 2371fe8 (3/3): a complete diaphragm-wall call (panel 6 m, 0.8 m,
25 m, 10 panels) failed with "Unknown argument(s) ...: wall, width_m": the
ask extractors read "6 x 0.8 x 25 m" and "wall" out of the user's words as
arguments this calculator does not take. Only an argument the caller sent
can be unknown; one the server guessed from the words is dropped. Invented
panel.
"""
from app.lib import construction_formulas as cf

PANEL = {"panel_length": 6, "wall_thickness": 0.8, "excavation_depth": 25, "panel_count": 10}


def test_a_complete_call_runs_whatever_the_words_say():
    for ask in ("Diaphragm wall: 10 panels of 6 x 0.8 x 25 m. Concrete volume plus 10%?",
                "dwall panels: 10 no., 6m long, 800 mm thick, 25 m deep, +10% overbreak"):
        out = cf.run_calculation("diaphragm_wall_panel_volume", {"text": ask, **PANEL})
        assert out["status"] == "success", out.get("error")
        assert out["result"]["volume_with_waste_m3"] == 1320.0


def test_an_argument_the_caller_sent_is_still_unknown():
    out = cf.run_calculation("diaphragm_wall_panel_volume", {"text": "panels", **PANEL, "bogus_key": 3})
    assert out["status"] == "error" and "bogus_key" in out["error"]
