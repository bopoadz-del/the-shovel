"""Activities typed in the message are a schedule the tool can use.

#703 taught the container to accept an ``activities`` list, and D3 stayed 0/5
on live: the turn never reaches the tool-call path. ``message_wants_look_ahead``
matches first, the pre-dispatch runs ``look_ahead`` itself, and the model is
never asked for arguments -- so the list was never passed and the answer stayed:

    "Error: No schedule provided - pass schedule_file pointing to a Primavera
     P6 .xer, or activities with start and finish dates"

Reading the message inside the container covers every dispatch path at once
(tool call, container action, pre-dispatch). A half-written row is dropped, not
completed by guesswork. Synthetic activities throughout.
"""
import asyncio
import inspect

import pytest

import app.containers.construction.schedule as schedule_mod

CONTAINER = next(
    obj for obj in vars(schedule_mod).values()
    if inspect.isclass(obj) and hasattr(obj, "look_ahead")
)

TYPED = (
    "Today is 21 September 2026. Using ONLY the activities listed below give me a "
    "two-week look-ahead: Blinding 15 Sep to 25 Sep 2026; Rebar fixing 28 Sep to "
    "2 Oct 2026; Formwork 4 Oct to 20 Oct 2026; Backfill 6 Oct to 10 Oct 2026; "
    "Survey 1 Sep to 10 Sep 2026 (complete)."
)


def _names_from(text):
    return [a["name"] for a in schedule_mod._activities_from_text(text)]


def _look_ahead(params):
    return asyncio.run(CONTAINER().look_ahead({}, params))


# ── reading the programme out of the message ───────────────────────────────

def test_the_five_activities_are_read_with_their_dates():
    rows = schedule_mod._activities_from_text(TYPED)
    assert [r["name"] for r in rows] == [
        "Blinding", "Rebar fixing", "Formwork", "Backfill", "Survey"]
    assert rows[0] == {"name": "Blinding", "start": "2026-09-15", "finish": "2026-09-25"}


def test_a_year_stated_once_fixes_the_pair():
    # "15 Sep to 25 Sep 2026": the first date carries no year.
    rows = schedule_mod._activities_from_text("Blinding 15 Sep to 25 Sep 2026")
    assert rows == [{"name": "Blinding", "start": "2026-09-15", "finish": "2026-09-25"}]


@pytest.mark.parametrize("line,start,finish", [
    ("Rebar fixing 2026-09-28 to 2026-10-02", "2026-09-28", "2026-10-02"),
    ("Formwork 04/10/2026 - 20/10/2026", "2026-10-04", "2026-10-20"),
    ("Blinding 15 September 2026 until 25 September 2026", "2026-09-15", "2026-09-25"),
])
def test_the_spellings_an_operator_actually_types(line, start, finish):
    rows = schedule_mod._activities_from_text(line)
    assert rows and rows[0]["start"] == start and rows[0]["finish"] == finish


# ── it never completes a row by guessing ───────────────────────────────────

def test_a_row_with_one_date_is_dropped():
    assert _names_from("Painting 15 Sep 2026") == []


def test_a_row_with_no_dates_is_dropped():
    assert _names_from("Painting and tiling to follow") == []


def test_a_bare_year_is_not_a_date_pair():
    assert _names_from("Mobilisation 2026 to 2027") == []


def test_prose_around_the_programme_is_not_an_activity():
    rows = schedule_mod._activities_from_text(TYPED)
    assert "Today is 21 September 2026" not in [r["name"] for r in rows]
    assert len(rows) == 5


# ── end to end through look_ahead, every dispatch path ─────────────────────

def test_the_message_alone_produces_the_look_ahead():
    out = _look_ahead({"user_message": TYPED, "as_of": "2026-09-21", "days": 14})
    assert out["status"] == "success"
    assert out["source"] == "activities_supplied_in_request"
    assert {a["name"] for a in out["activities"]} == {"Blinding", "Rebar fixing", "Formwork"}


@pytest.mark.parametrize("key", ["user_message", "message", "brief"])
def test_every_carrier_of_the_message_is_read(key):
    out = _look_ahead({key: TYPED, "as_of": "2026-09-21", "days": 14})
    assert out["status"] == "success"
    assert len(out["activities"]) == 3


def test_an_explicit_list_still_wins_over_the_text():
    explicit = [{"name": "Only this", "start": "2026-09-22", "finish": "2026-09-24"}]
    out = _look_ahead({"activities": explicit, "user_message": TYPED,
                       "as_of": "2026-09-21", "days": 14})
    assert [a["name"] for a in out["activities"]] == ["Only this"]


def test_a_message_with_no_programme_still_errors():
    out = _look_ahead({"user_message": "Give me a two-week look-ahead please",
                       "as_of": "2026-09-21", "days": 14})
    assert out["status"] == "error"
    assert "schedule_file" in out["error"] and "activities" in out["error"]


def test_an_unreadable_month_is_dropped_not_defaulted():
    # The line looks like an activity and the shape matches, but "Smarch" is
    # not a month. Dropping it is the only honest option: defaulting to
    # January would put a real-looking activity in the window.
    assert _names_from("Blinding 15 Smarch 2026 to 25 Sep 2026") == []
    assert _names_from("Blinding 15 Sep 2026 to 25 Smarch 2026") == []
