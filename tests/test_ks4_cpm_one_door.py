"""KS-4: every schedule path runs its CPM through the calculation envelope.

``run_cpm`` is the one door: it runs the registered ``cpm_network`` kernel by
``run_calculation`` and hands back the same ``CPMOutput`` the bare pass
gives. Product code outside the kernel never calls ``compute_cpm`` itself.
"""
from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import math

import pytest

from app.lib import construction_formulas as cf
from app.lib.pm_computations import CircularDependencyError, compute_cpm, run_cpm
from app.schemas.cpm import CPMInput, WorkCalendar

ROOT = Path(__file__).resolve().parents[1]

# The kernel itself and the module that defines the pass are the only places
# allowed to call the bare pass.
_PASS_OWNERS = {
    "app/lib/pm_computations.py",
    "app/agents/hats/planning/formulas/construction_formulas_planning.py",
}


def _typed_network():
    return [
        {"id": "P1", "name": "set out", "duration": 4},
        {"id": "P2", "duration": 6,
         "predecessors": [{"predecessor_id": "P1", "type": "SS", "lag": 2}]},
        {"id": "P3", "duration": 3, "predecessors": [{"predecessor_id": "P1"}]},
        {"id": "P4", "duration": 2,
         "predecessors": [{"predecessor_id": "P2"},
                          {"predecessor_id": "P3", "type": "FS", "lag": 1}]},
    ]


def _chain_network():
    return [
        {"id": "K", "duration": 7},
        {"id": "L", "duration": 2, "predecessors": [{"predecessor_id": "K"}]},
        {"id": "M", "duration": 9},
    ]


@pytest.mark.parametrize("acts, settings", [
    (_typed_network(), {}),
    (_chain_network(), {"project_start": date(2027, 3, 4),
                        "calendar": WorkCalendar(work_weekdays=[0, 1, 2, 3, 4, 5],
                                                 holidays=[date(2027, 3, 9)]),
                        "near_critical_band_days": 3}),
])
def test_envelope_output_equals_the_bare_pass(acts, settings):
    data = CPMInput(activities=acts, **settings)
    assert run_cpm(data) == compute_cpm(data)


@pytest.mark.parametrize("band, expect_named", [(None, False), (4, True)])
def test_near_critical_only_inside_a_stated_band(band, expect_named):
    data = CPMInput(activities=_chain_network(), near_critical_band_days=band)
    out = run_cpm(data)
    assert (out.near_critical is not None) is expect_named


@pytest.mark.parametrize("start, dated", [(None, False), (date(2027, 1, 11), True)])
def test_dates_only_from_a_stated_start(start, dated):
    out = run_cpm(CPMInput(activities=_typed_network(), project_start=start))
    assert (out.project_finish is not None) is dated
    assert all((r.early_start is not None) is dated for r in out.results)


@pytest.mark.parametrize("acts, calls", [
    (_chain_network(), ["cpm_network"]),
    ([], []),
])
def test_run_cpm_goes_through_run_calculation(monkeypatch, acts, calls):
    """A network is run by the envelope; an empty one has nothing to pass
    over and never reaches it."""
    seen = []
    real = cf.run_calculation

    def spy(name, params=None):
        seen.append(name)
        return real(name, params)

    monkeypatch.setattr(cf, "run_calculation", spy)
    data = CPMInput(activities=acts)
    assert run_cpm(data) == compute_cpm(data)
    assert seen == calls


@pytest.mark.parametrize("acts, exc_type", [
    ([{"id": "R", "duration": 1, "predecessors": [{"predecessor_id": "S"}]},
      {"id": "S", "duration": 2, "predecessors": [{"predecessor_id": "R"}]}],
     CircularDependencyError),
    ([{"id": "T", "duration": 1}, {"id": "T", "duration": 3}], ValueError),
])
def test_refusal_comes_back_as_the_kind_the_kernel_names(acts, exc_type):
    with pytest.raises(ValueError) as info:
        run_cpm(CPMInput(activities=acts))
    assert type(info.value) is exc_type


@pytest.mark.parametrize("status, refused", [
    ("success", False),
    ("error", True),
    ("needs_input", True),
])
def test_only_a_success_envelope_is_read_as_a_schedule(monkeypatch, status, refused):
    """The envelope's status decides, not whether a result happens to ride
    along: a non-success envelope is a refusal even when it carries a
    result-shaped body."""
    data = CPMInput(activities=_chain_network())
    good = cf.run_calculation("cpm_network", {
        "activities": [{"id": a.id, "duration": a.duration,
                        "predecessors": [p.predecessor_id for p in a.predecessors]}
                       for a in data.activities]})
    assert good["status"] == "success"
    env = dict(good, status=status, error="refused here")
    monkeypatch.setattr(cf, "run_calculation", lambda name, params=None: env)
    if refused:
        with pytest.raises(ValueError, match="refused here"):
            run_cpm(data)
    else:
        assert run_cpm(data) == compute_cpm(data)


@pytest.mark.parametrize("acts, kind", [
    ([{"id": "U", "duration": 1, "predecessors": "V"},
      {"id": "V", "duration": 1, "predecessors": "U"}], "CircularDependencyError"),
    ([{"id": "W", "duration": 1}, {"id": "W", "duration": 2}], "ValueError"),
])
def test_envelope_carries_the_refusal_kind(acts, kind):
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env["status"] == "error"
    assert env.get("error_type") == kind


def _bare_pass_calls(source: str):
    """Line numbers of every call to ``compute_cpm`` in ``source`` (by name
    or as an attribute); text inside strings is not a call."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "compute_cpm":
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("source, count", [
    ("from m import compute_cpm; out = compute_cpm(x)", 1),
    ("PROMPT = 'call compute_cpm(data)'; out = run_cpm(x)", 0),
])
def test_the_guard_counts_calls_not_text(source, count):
    assert len(_bare_pass_calls(source)) == count


def _app_parts():
    """Every top-level part of ``app``: each package, and ``.`` for the
    modules that sit directly in it. Together they cover the whole tree."""
    return ["."] + sorted(p.name for p in (ROOT / "app").iterdir()
                          if p.is_dir() and p.name != "__pycache__")


@pytest.mark.parametrize("part", _app_parts())
def test_no_product_module_calls_the_bare_pass(part):
    base = ROOT / "app" / part
    paths = base.glob("*.py") if part == "." else base.rglob("*.py")
    offenders = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel in _PASS_OWNERS:
            continue
        for n in _bare_pass_calls(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{n}")
    assert offenders == []


# -- the callers that switched to the one door keep their own decisions ------
# (KS-4 moved these onto ``run_cpm``; each decision is held by two inputs.)

def _scheduler():
    from app.containers.construction.schedule import ConstructionScheduleMixin
    return ConstructionScheduleMixin()


def _wbs_rows():
    return [
        {"id": "A1", "name": "excavate", "duration_days": 5, "predecessors": [],
         "resources": ["civil"], "wbs_phase": "1"},
        {"id": "A2", "name": "blind", "duration_days": 2, "predecessors": ["A1"],
         "resources": ["civil"], "wbs_phase": "1"},
    ]


# 2027-05-03 is a Monday; 7 Mon-Fri working days end on Tue 05-11. The
# finish ``compute_cpm`` dates is working-day offset 7 (``nth_working_day``),
# the next working day, Wed 05-12; that convention predates KS-4.
@pytest.mark.parametrize("start, finish", [
    (None, None),
    ("2027-05-03", "2027-05-12"),
])
def test_attach_cpm_dates_activities_only_from_a_stated_start(start, finish):
    acts, summary, err = _scheduler()._attach_cpm_to_activities(_wbs_rows(), start)
    assert err is None
    assert [a["early_finish_day"] for a in acts] == [5, 7]
    assert summary["total_duration_days"] == 7
    assert summary.get("project_finish") == finish
    assert ("early_start_date" in acts[0]) is (start is not None)
    if start is not None:
        assert acts[0]["early_start_date"] == start


@pytest.mark.parametrize("rows, err_part", [
    (_wbs_rows(), None),
    ([dict(_wbs_rows()[0], predecessors=["A2"]), _wbs_rows()[1]], "ircular"),
])
def test_attach_cpm_refusal_leaves_the_activities_unchanged(rows, err_part):
    acts, summary, err = _scheduler()._attach_cpm_to_activities(rows, None)
    if err_part is None:
        assert err is None and summary["critical_count"] == 2
    else:
        assert err_part in err and acts == rows and summary["critical_count"] == 0


@pytest.mark.parametrize("acts, status, totals", [
    ([{"id": "H1", "name": "frame", "duration": 6},
      {"id": "H2", "name": "roof", "duration": 3}], "success", [2, 1]),
    ([], "error", None),
])
def test_histogram_from_activity_counts_buckets_cpm_spans(acts, status, totals):
    out = _scheduler()._histogram_from_activity_counts(acts, "week", reason="no crew")
    assert out["status"] == status
    if totals is not None:
        assert [p["total"] for p in out["periods"]] == totals


@pytest.mark.parametrize("rows, status, error_part", [
    ([{"id": "Q1", "name": "pour", "duration_days": 4},
      {"id": "Q2", "name": "cure", "duration_days": 7, "predecessors": ["Q1"]}],
     "success", None),
    ([{"id": "Q3", "name": "fit out", "duration_days": "UNKNOWN"}],
     "error", "no dated activities"),
])
def test_resource_histogram_phases_an_inline_schedule(rows, status, error_part):
    import asyncio
    out = asyncio.run(_scheduler().resource_histogram({"activities": rows}, {}))
    assert out["status"] == status
    if error_part is not None:
        assert error_part in out["error"]
    else:
        assert out["histogram_kind"] == "activity_count"


def _session(acts):
    from app.schemas.project_session import ProjectSession
    s = ProjectSession.new("ks4-one-door", user_id="test")
    if acts is not None:
        s.data["activities"] = acts
    return s


def _step(kind, key=""):
    from app.schemas.execution_plan import PlanStep
    return PlanStep(type=kind, output_key=key)


@pytest.mark.parametrize("acts", [_typed_network(), _chain_network()])
def test_plan_executor_compute_cpm_and_gantt_use_the_one_door(acts):
    import asyncio
    from app.core.plan_executor import PlanExecutor
    ex, session = PlanExecutor(), _session(acts)
    want = compute_cpm(CPMInput(activities=acts))
    value = asyncio.run(ex._step_compute_cpm(_step("compute_cpm"), session))
    assert value["project_duration"] == want.project_duration
    assert session.data["cpm_results"] == value
    bars = asyncio.run(ex._step_gantt(_step("gantt", "bars"), session))
    assert len(bars) == len(want.results)
    assert session.data["bars"] == bars


@pytest.mark.parametrize("acts, ok", [(_chain_network(), True), (None, False)])
def test_plan_executor_histogram_needs_activities(acts, ok):
    import asyncio
    from app.core.plan_executor import PlanExecutionError, PlanExecutor
    session = _session(acts)
    step = _step("resource_histogram")
    if ok:
        value = asyncio.run(PlanExecutor()._step_resource_histogram(step, session))
        assert session.data["manpower"] == value
    else:
        with pytest.raises(PlanExecutionError):
            asyncio.run(PlanExecutor()._step_resource_histogram(step, session))


# ── Ids are kept as given; a turn ask never edits the caller's own band ─────

def _spaced(first, second):
    return [{"id": first, "duration": 2},
            {"id": second, "duration": 3, "predecessors": [{"predecessor_id": first}]}]


@pytest.mark.parametrize("first, second", [("A1 ", "A2 "), (" K", "L  ")])
def test_run_cpm_keeps_each_id_exactly_as_given(first, second):
    data = CPMInput(activities=_spaced(first, second))
    out = run_cpm(data)
    assert out == compute_cpm(data)
    assert [r.id for r in out.results] == [first, second]
    assert out.critical_path == [first, second]


@pytest.mark.parametrize("plain, spaced", [("A", "A "), ("B", " B")])
def test_ids_that_differ_by_a_space_are_two_activities(plain, spaced):
    data = CPMInput(activities=[{"id": plain, "duration": 4},
                                {"id": spaced, "duration": 1}])
    out = run_cpm(data)
    assert out == compute_cpm(data)
    assert {r.id for r in out.results} == {plain, spaced}


@pytest.mark.parametrize("first, second, duration", [
    ("A1 ", "A2 ", 5),   # hand: 0-2, 2-5
    (" P", "Q ", 5),
])
def test_container_network_pass_joins_spaced_ids(first, second, duration):
    from app.containers.construction import ConstructionContainer
    acts = [{"id": first, "duration": 2},
            {"id": second, "duration": 3, "predecessors": [{"id": first}]}]
    out = ConstructionContainer()._calculate_cpm({"activities": acts})
    assert out["critical_path"] == [first, second]
    assert out["project_duration_days"] == duration


@pytest.mark.parametrize("blank", ["", "   "])
def test_kernel_refuses_a_blank_id(blank):
    env = cf.run_calculation("cpm_network", {"activities": [{"id": blank, "duration": 2}]})
    assert env.get("status") != "success"
    assert "has no id" in str(env.get("error"))


@pytest.mark.parametrize("raw, ids", [
    ("A, B", ["A", "B"]),                       # a written list: pieces trimmed
    ([{"id": " C"}, "D ,E"], [" C", "D", "E"]),  # a mapping's id kept as written
])
def test_links_trim_only_ids_read_out_of_a_written_list(raw, ids):
    from app.agents.hats.planning.formulas.construction_formulas_planning import (
        _network_links,
    )
    assert [link["predecessor_id"] for link in _network_links(raw)] == ids


def _banded(lag):
    return [{"id": "A", "duration": 5},
            {"id": "B", "duration": 3, "predecessors": [{"predecessor_id": "A"}]},
            {"id": "C", "duration": 1, "predecessors": [{"predecessor_id": "A"}]},
            {"id": "D", "duration": 4,
             "predecessors": [{"predecessor_id": "B"},
                              {"predecessor_id": "C", "lag": lag}]}]


@pytest.mark.parametrize("band, lag, ask", [
    (2, -2, "Run the critical path with a near-critical band of two days. "
            "D has a lead of -2 days."),
    (5, -5, "Flag anything within a working week of critical; "
            "the link from C has a lag of -5."),
])
def test_a_turn_ask_never_flips_the_callers_band(band, lag, ask):
    data = CPMInput(activities=_banded(lag), near_critical_band_days=band)
    token = cf.set_current_ask(ask)
    try:
        out = run_cpm(data)
        assert cf._CURRENT_ASK.get() == ask   # the turn's ask is back afterwards
    finally:
        cf._CURRENT_ASK.reset(token)
    assert out == compute_cpm(data)
    assert out.near_critical_band_days == band


# ── A link joins its activity when the two differ only by spaces ────────────

@pytest.mark.parametrize("acts, path, duration", [
    ([{"id": "A ", "duration": 2},
      {"id": "B", "duration": 3, "predecessors": ["A "]}], ["A ", "B"], 5),
    ([{"id": "A ", "duration": 2},
      {"id": "B", "duration": 3, "predecessors": "A "}], ["A ", "B"], 5),
    ([{"id": " K", "duration": 4},
      {"id": "L", "duration": 1, "predecessors": [" K"]}], [" K", "L"], 5),
    ([{"id": "A", "duration": 6},
      {"id": "B", "duration": 3, "predecessors": [{"id": "A "}]}], ["A", "B"], 9),
])
def test_chat_calculator_joins_a_link_written_with_other_spaces(acts, path, duration):
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env.get("status") == "success", env
    assert env["result"]["project_duration"] == duration
    assert env["result"]["critical_path"] == path


@pytest.mark.parametrize("form", [
    lambda link: [{"id": link}],   # a mapping
    lambda link: [link],           # an id as one item of a list
    lambda link: link,             # an id given on its own as text
])
@pytest.mark.parametrize("link, duration", [("A ", 1 + 3), ("A", 7 + 3)])
def test_an_exact_id_wins_over_a_spaced_twin(form, link, duration):
    # "A" (7 days) and "A " (1 day) are two activities; a link joins the one
    # it names exactly, never its twin, however the link is written.
    acts = [{"id": "A", "duration": 7}, {"id": "A ", "duration": 1},
            {"id": "B", "duration": 3, "predecessors": form(link)}]
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env.get("status") == "success", env
    row = {r["id"]: r for r in env["result"]["activities"]}["B"]
    assert row["early_start"] == duration - 3


@pytest.mark.parametrize("link", ["A  ", "Z"])
def test_a_link_with_no_single_match_is_refused(link):
    # "A  " trims to two activities ("A" and "A "); "Z" to none: no guess.
    acts = [{"id": "A", "duration": 7}, {"id": "A ", "duration": 1},
            {"id": "B", "duration": 3, "predecessors": [{"id": link}]}]
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env.get("status") != "success"
    assert "unknown predecessor" in str(env.get("error"))
    assert repr(link) in str(env.get("error"))


@pytest.mark.parametrize("band, accepted", [(100000, True), (100001, False)])
def test_band_upper_bound_is_the_declared_range(band, accepted):
    acts = [{"id": "A", "duration": 2},
            {"id": "B", "duration": 3, "predecessors": ["A"]}]
    env = cf.run_calculation("cpm_network", {"activities": acts,
                                             "near_critical_band_days": band})
    assert (env.get("status") == "success") is accepted, env


@pytest.mark.parametrize("preds, start", [("A , C", 7), ("C, A ", 7),
                                          ("C ,A", 7), ("A ", 1)])
def test_comma_list_ids_lose_only_the_separator_spaces(preds, start):
    # Inside "x, y" the spaces belong to the separator, so a piece reads as
    # its trimmed text ("A", the 7-day twin); an id on its own keeps its text.
    acts = [{"id": "A", "duration": 7}, {"id": "A ", "duration": 1},
            {"id": "C", "duration": 2},
            {"id": "B", "duration": 3, "predecessors": preds}]
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env.get("status") == "success", env
    row = {r["id"]: r for r in env["result"]["activities"]}["B"]
    assert row["early_start"] == start


@pytest.mark.parametrize("preds, start", [(" ", 0), (["\t", "C"], 2),
                                          ("A, ", 7)])
def test_a_blank_predecessor_is_no_link(preds, start):
    # Text with nothing but spaces names no activity: it adds no link, it is
    # not refused as an unknown predecessor.
    acts = [{"id": "A", "duration": 7}, {"id": "C", "duration": 2},
            {"id": "B", "duration": 3, "predecessors": preds}]
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env.get("status") == "success", env
    row = {r["id"]: r for r in env["result"]["activities"]}["B"]
    assert row["early_start"] == start


# A value the caller gave that the schedule refuses is asked for again by the
# registry's own question for that input (its label, and its declared range
# where it has one). No text of the schema's refusal reaches the answer.
_SCHEMA_TEXT = ("should", "[type=", "pydantic", "CPMInput", "WorkCalendar",
                "predecessor_id", "validation error", "input_value")


def _link_net(link):
    return [{"id": "A", "duration": 2},
            {"id": "B", "duration": 3, "predecessors": [{"id": "A", **link}]}]


@pytest.mark.parametrize("given, field", [
    ({"activities": _link_net({"type": "QQ"})}, "activities"),
    ({"activities": _link_net({"type": "start-to-start"})}, "activities"),
    ({"activities": _link_net({"lag": "x"})}, "activities"),
    ({"activities": _link_net({"lag": 1.5})}, "activities"),
    ({"activities": _chain_network(), "project_start": "next monday"}, "project_start"),
    ({"activities": _chain_network(), "project_start": "2026-13-40"}, "project_start"),
    ({"activities": _chain_network(), "calendar": "Sun-Thu"}, "calendar"),
    ({"activities": _chain_network(), "calendar": {"holidays": "Eid"}}, "calendar"),
])
def test_a_refused_input_is_asked_for_by_its_label_only(given, field):
    env = cf.run_calculation("cpm_network", given)
    assert env["status"] == "error" and env["needs_input"] is True
    assert env["missing"] == [field]
    from app.lib.source_labels import parameter_label
    label = parameter_label("cpm_network", field)
    assert f"What {label} should I use?" in env["question"]
    # the one "should" left is the registry's own question for the input
    own = env["error"].replace(f"What {label} should I use?", "")
    assert not any(t in own for t in _SCHEMA_TEXT), env["error"]
    # nothing the caller sent is echoed: neither the rows nor the bad value
    assert not any(t in env["error"] for t in ("{", "[", "'", "\"", "Sun-Thu", "monday"))


@pytest.mark.parametrize("link, key", [({"type": "QQ"}, "type"), ({"lag": "x"}, "lag")])
def test_a_refused_link_names_the_activity_and_the_key_typed(link, key):
    env = cf.run_calculation("cpm_network", {"activities": _link_net(link)})
    assert f"activity B ({key})" in env["error"]


@pytest.mark.parametrize("links, said", [
    ([{"id": "A", "type": "QQ"}, {"id": "C", "type": "ZZ"}], "activity B (type)."),
    ([{"id": "A", "lag": "x"}, {"id": "C", "type": "QQ"}],
     "activity B (lag) and 1 more."),
])
def test_one_refused_part_is_said_once(links, said):
    acts = [{"id": "A", "duration": 2}, {"id": "C", "duration": 1},
            {"id": "B", "duration": 3, "predecessors": links}]
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert said in env["error"], env["error"]


@pytest.mark.parametrize("n", [1500, 40])
def test_a_large_refused_network_is_answered_short(n):
    acts = [{"id": f"T{i}", "duration": 1,
             "predecessors": [{"id": f"T{i - 1}", "type": "QQ"}] if i else []}
            for i in range(n)]
    env = cf.run_calculation("cpm_network", {"activities": acts})
    assert env["missing"] == ["activities"]
    assert len(env["error"]) < 300, len(env["error"])
    assert f"activity T1 (type) and {n - 2:,} more" in env["error"]


@pytest.mark.parametrize("band", [0, 0.0, -1])
def test_band_zero_is_refused_by_the_declared_range(band):
    acts = _chain_network()
    env = cf.run_calculation("cpm_network", {"activities": acts,
                                             "near_critical_band_days": band})
    assert env["status"] == "error"
    assert "near-critical float band" in env["question"]
    assert "outside what" in env["error"]
    # and the schedule itself refuses the same figure
    with pytest.raises(Exception):
        CPMInput(activities=acts, near_critical_band_days=band)


@pytest.mark.parametrize("band", [math.ulp(0.0), 0.5])
def test_the_smallest_positive_band_is_accepted_by_both(band):
    env = cf.run_calculation("cpm_network", {"activities": _chain_network(),
                                             "near_critical_band_days": band})
    assert env["status"] == "success", env
    CPMInput(activities=_chain_network(), near_critical_band_days=band)


@pytest.mark.parametrize("given", [
    {"project_start": "2026-01-05"},
    {"calendar": {"work_weekdays": [6, 0, 1, 2, 3]}},
    {"near_critical_band_days": 0.5},
    {"project_start": "2026-01-05", "calendar": {"holidays": ["2026-01-06"]},
     "near_critical_band_days": 3},
])
def test_a_readable_setting_is_not_refused(given):
    env = cf.run_calculation("cpm_network", {"activities": _chain_network(), **given})
    assert env["status"] == "success", env
    assert "needs_input" not in env


@pytest.mark.parametrize("start, cal", [("next monday", "Sun-Thu"),
                                        ("someday", {"work_weekdays": "x"})])
def test_every_refused_input_is_named(start, cal):
    env = cf.run_calculation("cpm_network", {"activities": _chain_network(),
                                             "project_start": start, "calendar": cal})
    assert env["missing"] == ["project_start", "calendar"]
    assert "project start date" in env["question"] and "working calendar" in env["question"]


@pytest.mark.parametrize("calendar", [
    {"work_weekdays": "Sun-Thu", "holidays": "Eid"},
    {"work_weekdays": "weekdays", "holidays": 5},
])
def test_an_input_refused_twice_is_asked_for_once(calendar):
    env = cf.run_calculation("cpm_network", {"activities": _chain_network(),
                                             "calendar": calendar})
    assert env["missing"] == ["calendar"]
    assert env["error"].count("working calendar (working weekdays and holidays) given") == 1


# The band's range is "more than 0": the bound is said as it is meant, to the
# user and to the model, never as the smallest number a float can hold.

@pytest.mark.parametrize("band, shown", [(0, "0"), ("0", "0"), (-1, "-1"), (-0.5, "-0.5"), (100001, "100,001")])
def test_a_refused_band_names_the_range_as_a_person_would(band, shown):
    env = cf.run_calculation("cpm_network", {"activities": _chain_network(),
                                             "near_critical_band_days": band})
    assert env["status"] == "error"
    assert env["error"] == (
        f"The near-critical float band given ({shown} working days) is outside what the Critical path "
        "of an activity network calculator accepts (more than 0 working days, up to 100,000 working days). "
        "What near-critical float band should I use?")
    for raw in ("e-324", "e-3", "4.94", "5e-"):
        assert raw not in env["error"] and raw not in str(env.get("question") or "")


@pytest.mark.parametrize("name, part", [
    ("near_critical_band_days", "working days, more than 0, up to 100000, optional"),
    ("activities", "activities with durations and predecessors"),
    ("project_start", "project start date"),
])
def test_the_models_input_lines_for_the_schedule_say_what_is_declared(name, part):
    from app.agents.driver import tools as driver_tools
    from app.lib import formula_registry as fr

    params = fr.parameters(fr.get("cpm_network"))
    line = driver_tools._input_line(name, params[name])
    assert part in line
    assert ("more than" in line) is (name == "near_critical_band_days")
    assert "e-324" not in line and "0 to 100000" not in line


@pytest.mark.parametrize("link_type", ["QQ", "ZZ"])
def test_a_schema_refusal_from_the_kernel_names_its_kind(link_type):
    from app.agents.hats.planning.formulas.construction_formulas_planning import cpm_network

    out = cpm_network(activities=[{"id": "A", "duration": 2},
                                  {"id": "B", "duration": 3,
                                   "predecessors": [{"predecessor_id": "A", "type": link_type}]}])
    assert out["error_type"] == "ValueError"

