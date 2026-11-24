"""KS-4: every schedule path runs its CPM through the calculation envelope.

``run_cpm`` is the one door: it runs the registered ``cpm_network`` kernel by
``run_calculation`` and hands back the same ``CPMOutput`` the bare pass
gives. Product code outside the kernel never calls ``compute_cpm`` itself.
"""
from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

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
