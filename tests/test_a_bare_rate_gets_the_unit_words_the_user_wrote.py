"""A bare figure passed to a text input gets the unit words the user wrote.

Lane 2 on 9c1d155: "1,500 m2 per gang-day" came out as "manhours required
40 h" again. The rate input is text: the formula reads the unit words written
on its value (#950). The model passed a bare 1500, so the formula saw no unit.
The server now attaches the words the ask writes right after that figure, up
to the end of the clause. Invented quantities.
"""
from app.lib import construction_formulas as cf


def _run(ask, **params):
    return cf.run_calculation("productivity_manpower_duration", {"text": ask, **params})["result"]


def test_a_bare_day_rate_gives_a_duration_not_man_hours():
    out = _run("Blockwork 60,000 m2 at a productivity of 1,500 m2 per gang-day. How long?",
               quantity=60000, productivity=1500)
    assert out.get("duration") == 40.0
    assert "manhours_required" not in out


def test_a_bare_hour_rate_still_gives_man_hours():
    out = _run("Plaster 6,000 m2 at 12 m2 per man-hour. Man-hours?", quantity=6000, productivity=12)
    assert out.get("manhours_required") == 500.0


def test_the_words_after_a_figure_stop_at_the_clause():
    assert cf._words_written_after(1500, "at 1,500 m2 per gang-day. How long?") == "m2 per gang-day"
    assert cf._words_written_after(12, "12 m2 per crew-hour, 4 crews") == "m2 per crew-hour"
    assert cf._words_written_after(7, "no such figure here") == ""
