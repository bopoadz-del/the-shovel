"""Tests for pm_computations extended — Reasoning Engine Plan 1b."""

import pytest

from app.schemas.cpm import (
    Activity, CPMInput, Dependency, GanttBar, HistogramPeriod,
    ResourceAssignment, ResourceHistogram,
)
from app.lib.pm_computations import (
    compress_schedule, compute_cpm, gantt_data, resource_histogram,
)

_PERIOD_DAYS = {"week": 5, "month": 21}


def _act(act_id, dur, preds=None, resources=None):
    return Activity(
        id=act_id, duration=dur,
        predecessors=[Dependency(predecessor_id=p) for p in (preds or [])],
        resources=resources or [],
    )


def _index(results, act_id):
    return next(r for r in results if r.id == act_id)


def test_resource_assignment_defaults():
    r = ResourceAssignment(trade="electrician")
    assert r.count == 1.0


def test_activity_carries_resources():
    a = Activity(id="A", duration=5,
                 resources=[ResourceAssignment(trade="steelfixer", count=8)])
    assert a.resources[0].trade == "steelfixer"
    assert a.resources[0].count == 8


def test_histogram_and_gantt_models_construct():
    hp = HistogramPeriod(index=0, label="W1", total=12, by_trade={"x": 12})
    rh = ResourceHistogram(period_unit="week", periods=[hp], peak_total=12,
                           peak_period="W1", by_trade_totals={"x": 12},
                           total_manhours=480)
    assert rh.peak_total == 12
    bar = GanttBar(id="A", name="Mob", start_day=0, end_day=5, is_critical=True)
    assert bar.end_day == 5


def test_resource_histogram_buckets_by_week():
    # A: 10 working days, crew of 4 -> spans weeks 0 and 1
    acts = [_act("A", 10, resources=[ResourceAssignment(trade="civil", count=4)])]
    out = compute_cpm(CPMInput(activities=acts))
    hist = resource_histogram(out.results, acts, period_unit="week")
    assert hist.period_unit == "week"
    assert len(hist.periods) == 2
    assert hist.periods[0].by_trade["civil"] == 4
    assert hist.peak_total == 4


def test_resource_histogram_sums_concurrent_trades():
    # A and B both run in week 0, different trades
    acts = [
        _act("A", 5, resources=[ResourceAssignment(trade="civil", count=6)]),
        _act("B", 5, resources=[ResourceAssignment(trade="mep", count=3)]),
    ]
    out = compute_cpm(CPMInput(activities=acts))
    hist = resource_histogram(out.results, acts, period_unit="week")
    assert hist.periods[0].total == 9
    assert hist.peak_total == 9


def test_resource_histogram_total_manhours():
    # crew 4 x 10 days x 8 h = 320
    acts = [_act("A", 10, resources=[ResourceAssignment(trade="civil", count=4)])]
    out = compute_cpm(CPMInput(activities=acts))
    hist = resource_histogram(out.results, acts, period_unit="week")
    assert hist.total_manhours == 320


def test_gantt_data_one_bar_per_activity():
    acts = [_act("A", 3), _act("B", 5, ["A"])]
    out = compute_cpm(CPMInput(activities=acts))
    bars = gantt_data(out.results)
    assert len(bars) == 2
    a = next(b for b in bars if b.id == "A")
    assert (a.start_day, a.end_day) == (0, 3)
    assert a.is_critical is True


def test_gantt_data_sorted_by_start():
    acts = [_act("A", 3), _act("B", 5, ["A"]), _act("C", 2)]
    bars = gantt_data(compute_cpm(CPMInput(activities=acts)).results)
    starts = [b.start_day for b in bars]
    assert starts == sorted(starts)


def test_compress_schedule_shortens_project():
    # critical chain A(3)->B(5)->D(2)=10; cut B by 3 -> 7
    acts = [_act("A", 3), _act("B", 5, ["A"]), _act("C", 2, ["A"]),
            _act("D", 2, ["B", "C"])]
    baseline = compute_cpm(CPMInput(activities=acts))
    revised, delta = compress_schedule(CPMInput(activities=acts), {"B": 3})
    assert baseline.project_duration == 10
    assert revised.project_duration == 7
    assert delta == 3


def test_compress_schedule_rejects_unknown_activity():
    acts = [_act("A", 3)]
    with pytest.raises(ValueError):
        compress_schedule(CPMInput(activities=acts), {"GHOST": 1})


def test_compress_schedule_clamps_at_zero_duration():
    # cutting more than the duration floors at 0, never negative
    acts = [_act("A", 3), _act("B", 4, ["A"])]
    revised, _delta = compress_schedule(CPMInput(activities=acts), {"B": 99})
    assert revised.project_duration == 3  # B floored to duration 0


def test_compress_schedule_noop_when_cut_has_no_float_impact():
    acts = [_act("A", 3), _act("B", 5, ["A"]), _act("C", 2, ["A"]),
            _act("D", 2, ["B", "C"])]
    revised, delta = compress_schedule(CPMInput(activities=acts), {"C": 1})
    assert delta == 0
    assert revised.project_duration == 10


# ── W2 (schedule engine) — hand-solved CPM oracle ───────────────────────────
# A classic textbook AON network with a fully hand-solved critical path. This is
# the committed hand-oracle for Part-B W2: the schedule engine's forward/backward
# pass, float, and critical-path selection must reproduce it EXACTLY.
#
#   A(3);  B(4)<-A;  C(2)<-A;  D(5)<-B;  E(1)<-C;  F(2)<-D,E
#   Forward: A ES0/EF3, B 3/7, C 3/5, D 7/12, E 5/6, F 12/14  -> duration 14
#   Backward from 14: F 12/14, D 7/12, E 11/12, B 3/7, C 9/11, A 0/3
#   Total float: A0 B0 C6 D0 E6 F0  ->  critical path A-B-D-F
def test_cpm_matches_hand_solved_textbook_network():
    acts = [_act("A", 3), _act("B", 4, ["A"]), _act("C", 2, ["A"]),
            _act("D", 5, ["B"]), _act("E", 1, ["C"]), _act("F", 2, ["D", "E"])]
    out = compute_cpm(CPMInput(activities=acts))

    assert out.project_duration == 14
    assert list(out.critical_path) == ["A", "B", "D", "F"]

    expected = {
        # id:  (ES, EF, LS, LF, total_float, is_critical)
        "A": (0, 3, 0, 3, 0, True),
        "B": (3, 7, 3, 7, 0, True),
        "C": (3, 5, 9, 11, 6, False),
        "D": (7, 12, 7, 12, 0, True),
        "E": (5, 6, 11, 12, 6, False),
        "F": (12, 14, 12, 14, 0, True),
    }
    for aid, (es, ef, ls, lf, tf, crit) in expected.items():
        r = _index(out.results, aid)
        assert (r.early_start_day, r.early_finish_day) == (es, ef), aid
        assert (r.late_start_day, r.late_finish_day) == (ls, lf), aid
        assert r.total_float == tf, aid
        assert r.is_critical is crit, aid
