"""Tests for pm_computations — Reasoning Engine Plan 1 (CPM core)."""

from datetime import date, date as _date

import pytest

from app.schemas.cpm import Activity, CPMInput, Dependency, DependencyType, WorkCalendar
from app.lib.pm_computations import (
    CircularDependencyError, topological_order, cpm_forward_pass, compute_cpm,
)


def test_activity_defaults():
    a = Activity(id="A", duration=5)
    assert a.predecessors == [] and a.name == ""


def test_dependency_defaults_to_fs():
    d = Dependency(predecessor_id="A")
    assert d.type == DependencyType.FS and d.lag == 0


def test_negative_duration_rejected():
    with pytest.raises(Exception):
        Activity(id="A", duration=-1)


def test_empty_activity_id_rejected():
    with pytest.raises(Exception):
        Activity(id="", duration=1)


def test_nth_working_day_zero_is_first_working_day():
    # 2026-05-18 is a Monday
    assert WorkCalendar().nth_working_day(date(2026, 5, 18), 0) == date(2026, 5, 18)


def test_nth_working_day_skips_weekend():
    # Mon 18 + 5 working days -> Mon 25 (skips Sat 23 / Sun 24)
    assert WorkCalendar().nth_working_day(date(2026, 5, 18), 5) == date(2026, 5, 25)


def test_nth_working_day_skips_holiday():
    cal = WorkCalendar(holidays=[date(2026, 5, 19)])
    assert cal.nth_working_day(date(2026, 5, 18), 1) == date(2026, 5, 20)


def test_nth_working_day_advances_off_weekend_start():
    # Sat 23, offset 0 -> Mon 25
    assert WorkCalendar().nth_working_day(date(2026, 5, 23), 0) == date(2026, 5, 25)


def _act(id, dur, preds=None):
    return Activity(
        id=id, duration=dur,
        predecessors=[Dependency(predecessor_id=p) for p in (preds or [])],
    )


def _index(acts):
    return {a.id: a for a in acts}


def test_topological_order_linear_chain():
    acts = [_act("C", 1, ["B"]), _act("A", 1), _act("B", 1, ["A"])]
    assert topological_order(acts) == ["A", "B", "C"]


def test_topological_order_detects_cycle():
    acts = [_act("A", 1, ["B"]), _act("B", 1, ["A"])]
    with pytest.raises(CircularDependencyError):
        topological_order(acts)


def test_topological_order_rejects_unknown_predecessor():
    with pytest.raises(ValueError):
        topological_order([_act("A", 1, ["GHOST"])])


def test_forward_pass_linear_chain():
    acts = [_act("A", 3), _act("B", 2, ["A"]), _act("C", 4, ["B"])]
    fwd = cpm_forward_pass(_index(acts), topological_order(acts))
    assert fwd["A"] == (0, 3)
    assert fwd["B"] == (3, 5)
    assert fwd["C"] == (5, 9)


def test_forward_pass_parallel_takes_max():
    acts = [_act("A", 3), _act("B", 2, ["A"]), _act("C", 5, ["A"]),
            _act("D", 1, ["B", "C"])]
    fwd = cpm_forward_pass(_index(acts), topological_order(acts))
    assert fwd["D"] == (8, 9)  # max(B.EF 5, C.EF 8)


def test_forward_pass_fs_lag():
    acts = [_act("A", 3),
            Activity(id="B", duration=2, predecessors=[
                Dependency(predecessor_id="A", type=DependencyType.FS, lag=2)])]
    fwd = cpm_forward_pass(_index(acts), topological_order(acts))
    assert fwd["B"] == (5, 7)


def test_forward_pass_negative_lag_overlaps():
    # FS lag -2: B starts 2 working days before A finishes
    acts = [_act("A", 6),
            Activity(id="B", duration=4, predecessors=[
                Dependency(predecessor_id="A", type=DependencyType.FS, lag=-2)])]
    fwd = cpm_forward_pass(_index(acts), topological_order(acts))
    assert fwd["B"] == (4, 8)  # A.EF 6 + lag -2 = 4


def test_forward_pass_start_to_start():
    acts = [_act("A", 10),
            Activity(id="B", duration=4, predecessors=[
                Dependency(predecessor_id="A", type=DependencyType.SS, lag=2)])]
    fwd = cpm_forward_pass(_index(acts), topological_order(acts))
    assert fwd["B"] == (2, 6)


def test_forward_pass_finish_to_finish():
    # A duration=5 -> EF=5. B has FF dep on A, lag=0, duration=3.
    # cand = p_ef + lag - a.duration = 5 + 0 - 3 = 2
    # B.ES=2, B.EF=5
    acts = [_act("A", 5),
            Activity(id="B", duration=3, predecessors=[
                Dependency(predecessor_id="A", type=DependencyType.FF, lag=0)])]
    fwd = cpm_forward_pass(_index(acts), topological_order(acts))
    assert fwd["B"] == (2, 5)


def test_forward_pass_start_to_finish():
    # A duration=5 -> ES=0. B has SF dep on A, lag=6, duration=4.
    # cand = p_es + lag - a.duration = 0 + 6 - 4 = 2
    # B.ES=2, B.EF=6
    acts = [_act("A", 5),
            Activity(id="B", duration=4, predecessors=[
                Dependency(predecessor_id="A", type=DependencyType.SF, lag=6)])]
    fwd = cpm_forward_pass(_index(acts), topological_order(acts))
    assert fwd["B"] == (2, 6)


def test_compute_cpm_identifies_critical_path():
    # long path A(3)->B(5)->D(2)=10; A->C(2)->D has float
    acts = [_act("A", 3), _act("B", 5, ["A"]), _act("C", 2, ["A"]),
            _act("D", 2, ["B", "C"])]
    out = compute_cpm(CPMInput(activities=acts))
    assert out.project_duration == 10
    assert out.critical_path == ["A", "B", "D"]
    by_id = {r.id: r for r in out.results}
    assert by_id["C"].total_float == 3
    assert by_id["C"].is_critical is False
    assert by_id["A"].is_critical is True


def test_compute_cpm_free_float():
    acts = [_act("A", 3), _act("B", 5, ["A"]), _act("C", 2, ["A"]),
            _act("D", 2, ["B", "C"])]
    by_id = {r.id: r for r in compute_cpm(CPMInput(activities=acts)).results}
    assert by_id["C"].free_float == 3


def test_compute_cpm_near_critical():
    acts = [_act("A", 3), _act("B", 5, ["A"]), _act("C", 4, ["A"]),
            _act("D", 2, ["B", "C"])]
    assert "C" in compute_cpm(CPMInput(activities=acts)).near_critical


def test_compute_cpm_projects_dates():
    out = compute_cpm(CPMInput(activities=[_act("A", 5)],
                               project_start=_date(2026, 5, 18)))
    assert out.results[0].early_start == _date(2026, 5, 18)
    assert out.results[0].early_finish == _date(2026, 5, 25)


def test_compute_cpm_empty_input():
    out = compute_cpm(CPMInput(activities=[]))
    assert out.project_duration == 0 and out.results == []


def test_compute_cpm_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        compute_cpm(CPMInput(activities=[_act("A", 1), _act("A", 2)]))


def test_compute_cpm_with_ff_dependency():
    # A(5) -> B(3) via FF lag=0.
    # Forward: A.ES=0, A.EF=5. B: cand = 5+0-3 = 2, B.ES=2, B.EF=5.
    # Project duration = max(5, 5) = 5.
    # Backward: B.LF=5, B.LS=2. A (succ=B via FF): cand = B.LF - lag = 5-0=5.
    #   A.LF=5, A.LS=0. Both TF=0 -> both critical.
    acts = [_act("A", 5),
            Activity(id="B", duration=3, predecessors=[
                Dependency(predecessor_id="A", type=DependencyType.FF, lag=0)])]
    out = compute_cpm(CPMInput(activities=acts))
    assert out.project_duration == 5
    by_id = {r.id: r for r in out.results}
    assert by_id["A"].early_start_day == 0
    assert by_id["A"].early_finish_day == 5
    assert by_id["B"].early_start_day == 2
    assert by_id["B"].early_finish_day == 5
    assert by_id["A"].total_float == 0
    assert by_id["B"].total_float == 0
    assert set(out.critical_path) == {"A", "B"}


def test_compute_cpm_realistic_network():
    """A 7-activity network with parallel branches, a lag, and an SS link."""
    acts = [
        _act("MOB", 5),
        _act("EXC", 10, ["MOB"]),
        _act("FND", 15, ["EXC"]),
        Activity(id="STL", duration=20, predecessors=[
            Dependency(predecessor_id="FND", type=DependencyType.FS, lag=2)]),
        Activity(id="MEP", duration=18, predecessors=[
            Dependency(predecessor_id="FND", type=DependencyType.SS, lag=5)]),
        _act("ENV", 12, ["STL"]),
        _act("FIT", 10, ["ENV", "MEP"]),
    ]
    out = compute_cpm(CPMInput(activities=acts, project_start=_date(2026, 6, 1)))
    # longest path: MOB5 + EXC10 + FND15 + lag2 + STL20 + ENV12 + FIT10 = 74
    assert out.project_duration == 74
    assert out.critical_path == ["MOB", "EXC", "FND", "STL", "ENV", "FIT"]
    by_id = {r.id: r for r in out.results}
    assert by_id["MEP"].is_critical is False      # MEP branch is shorter
    assert by_id["MEP"].total_float > 0
    assert out.project_finish is not None
    assert 0 <= out.critical_percentage <= 100
