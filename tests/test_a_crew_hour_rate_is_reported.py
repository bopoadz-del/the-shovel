"""The productivity rate reports the crew-hour rate beside the worker-hour one.

Lane 5 on 408d440 (order 2055Z item 2): "120 m2 by a crew of 4 in 40
man-hours, m2 per crew-hour?" answered 3 (the per-worker-hour rate) where
12 is right: the calculator offered no crew-hour figure. It now reports
rate_per_crew_hour = output / (man-hours / crew) and the crew-hours; the
per-worker-hour figures (and their no-double-count rule) are unchanged.
"""
from app.lib import construction_formulas as cf


def test_the_crew_hour_rate_and_the_worker_hour_rate_are_both_reported():
    r = cf.run_calculation("productivity_rate", {"output_quantity": 120, "labor_hours": 40, "crew_size": 4})["result"]
    assert r["rate_per_crew_hour"] == 12.0 and r["crew_hours"] == 10.0
    assert r["rate_per_worker_hour"] == 3.0 and r["rate_per_hour"] == 3.0


def test_a_crew_of_one_has_equal_rates():
    r = cf.run_calculation("productivity_rate", {"output_quantity": 120, "labor_hours": 40, "crew_size": 1})["result"]
    assert r["rate_per_crew_hour"] == r["rate_per_worker_hour"] == 3.0
