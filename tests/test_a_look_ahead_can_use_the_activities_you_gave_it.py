"""A look-ahead can be built from the activities in the message.

Live SET4.1 D3 on 3403cb5, 0/5. The operator listed five activities and said
not to ask for a programme file; every run answered:

    "Error: No schedule file provided - pass schedule_file pointing to a
     Primavera P6 .xer ... Because the tool requires a Primavera P6 .xer
     schedule file to run, no activities, dates, or statuses could be
     processed."

The data needed was in the question. The tool's own docstring claimed an
honest error "when no .xer / no activities", but it never looked for
activities. A named .xer still wins; an empty request still errors; nothing
is ever invented. Synthetic activities throughout.
"""
import asyncio
import inspect

import pytest

import app.containers.construction.schedule as schedule_mod

CONTAINER = next(
    obj for obj in vars(schedule_mod).values()
    if inspect.isclass(obj) and hasattr(obj, "look_ahead")
)

ACTIVITIES = [
    {"name": "Blinding", "start": "2026-09-15", "finish": "2026-09-25"},
    {"name": "Rebar fixing", "start": "2026-09-28", "finish": "2026-10-02"},
    {"name": "Formwork", "start": "2026-10-04", "finish": "2026-10-20"},
    {"name": "Backfill", "start": "2026-10-06", "finish": "2026-10-10"},
    {"name": "Survey", "start": "2026-09-01", "finish": "2026-09-10"},
]


def _run(params, data=None):
    return asyncio.run(CONTAINER().look_ahead(data or {}, params))


def _names(out):
    return [a.get("name") for a in out.get("activities", [])]


# ── the live case ──────────────────────────────────────────────────────────

def test_the_window_is_built_from_the_activities_in_the_request():
    out = _run({"activities": ACTIVITIES, "as_of": "2026-09-21", "days": 14})
    assert out["status"] == "success"
    assert out["source"] == "activities_supplied_in_request"
    assert set(_names(out)) == {"Blinding", "Rebar fixing", "Formwork"}


def test_an_activity_starting_after_the_window_is_out():
    out = _run({"activities": ACTIVITIES, "as_of": "2026-09-21", "days": 14})
    assert "Backfill" not in _names(out), "Backfill starts 6 Oct, past the window"


def test_a_finished_activity_is_out():
    out = _run({"activities": ACTIVITIES, "as_of": "2026-09-21", "days": 14})
    assert "Survey" not in _names(out), "Survey finished 10 Sep"


# ── it never invents, and never silently drops the ask ─────────────────────

def test_an_empty_request_still_errors_and_names_both_ways_in():
    out = _run({"days": 14})
    assert out["status"] == "error"
    assert "schedule_file" in out["error"] and "activities" in out["error"]


def test_a_row_without_dates_is_dropped_not_guessed():
    rows = ACTIVITIES + [{"name": "Painting"}, {"name": "Tiling", "start": "2026-09-22"}]
    out = _run({"activities": rows, "as_of": "2026-09-21", "days": 14})
    assert "Painting" not in _names(out)
    assert "Tiling" not in _names(out)
    assert out["total_activities_in_schedule"] == len(ACTIVITIES)


def test_nothing_appears_that_was_not_supplied():
    out = _run({"activities": ACTIVITIES, "as_of": "2026-09-21", "days": 14})
    assert set(_names(out)) <= {a["name"] for a in ACTIVITIES}


def test_an_empty_overlap_is_a_valid_answer_not_an_error():
    out = _run({"activities": ACTIVITIES, "as_of": "2027-01-01", "days": 14})
    assert out["status"] == "success"
    assert _names(out) == []


# ── field spellings and precedence ─────────────────────────────────────────

def test_other_field_spellings_are_accepted():
    rows = [{"activity": "Blinding", "from": "2026-09-15", "to": "2026-09-25"},
            {"title": "Formwork", "early_start": "2026-10-04", "end": "2026-10-20"}]
    out = _run({"activities": rows, "as_of": "2026-09-21", "days": 14})
    assert set(_names(out)) == {"Blinding", "Formwork"}


def test_a_named_schedule_file_still_takes_precedence():
    # Both supplied: the file is the source of record, so a missing file is an
    # error rather than a silent fall-back to the message's activities.
    out = _run({"activities": ACTIVITIES, "schedule_file": "no_such_programme.xer",
                "as_of": "2026-09-21", "days": 14})
    assert out["status"] == "error"
    assert "no_such_programme.xer" in out["error"]


def test_activities_in_input_data_work_as_well_as_params():
    out = _run({"as_of": "2026-09-21", "days": 14}, data={"activities": ACTIVITIES})
    assert out["status"] == "success"
    assert set(_names(out)) == {"Blinding", "Rebar fixing", "Formwork"}


def test_the_tool_advertises_the_activities_parameter():
    # The model cannot pass what the schema does not offer.
    from tests import agent_source
    source = agent_source.text()  # runtime.py and app/agents/core/
    assert '"activities": {' in source and "listed in the message" in source


@pytest.mark.parametrize("bad", [None, "not a list", 42])
def test_a_malformed_activities_value_is_ignored_not_crashed(bad):
    out = _run({"activities": bad, "days": 14})
    assert out["status"] == "error"
    assert "schedule_file" in out["error"]
