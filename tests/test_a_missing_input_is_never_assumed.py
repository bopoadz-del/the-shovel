"""A calculator never invents an input the caller did not give.

LANE5 #976 FAIL: a missing crew size silently became 1, and the answer
quoted a per-crew-hour figure for a crew nobody named. Order 2130Z item 2,
one rule for every calculator: a missing input is reported -- the result
carries what CAN be computed, with its unit, names the input it needed
(``needs_input``) and says so in the input's own words (``not_computed``).
"""
from app.lib.construction_formulas import run_calculation


def _run(**params):
    out = run_calculation("productivity_rate", params)
    return out.get("result", out)


def test_without_a_crew_the_per_man_hour_rate_is_given_and_the_crew_figures_are_not():
    r = _run(output_quantity=120, labor_hours=40)
    assert r["rate_per_worker_hour"] == 3.0
    assert "rate_per_crew_hour" not in r and "crew_hours" not in r
    assert r["needs_input"] == ["crew_size"]
    assert "crew size" in r["not_computed"] and "no value is assumed" in r["not_computed"]


def test_with_the_crew_the_crew_hour_rate_follows():
    r = _run(output_quantity=120, labor_hours=40, crew_size=4)
    assert r["rate_per_crew_hour"] == 12.0 and r["crew_hours"] == 10.0
    assert "needs_input" not in r and "not_computed" not in r


def test_a_fractional_average_crew_is_used_as_given():
    r = _run(output_quantity=120, labor_hours=40, crew_size=2.5)
    assert r["crew_hours"] == 16.0 and r["rate_per_crew_hour"] == 7.5


def test_a_zero_or_negative_crew_is_refused_not_replaced():
    for crew in (0, -3):
        out = run_calculation("productivity_rate", {"output_quantity": 120, "labor_hours": 40, "crew_size": crew})
        assert out.get("status") in ("error", "rejected") or "error" in out, out
        assert "rate_per_crew_hour" not in str(out)
