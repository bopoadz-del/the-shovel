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


@pytest.mark.parametrize("spans, counts, period_total, peak, peak_period", [
    # A 5 persons days 0-3, then B 4 persons days 3-5: both touch week 1, so the
    # bucket sums 9, but no day ever holds more than A's 5.
    ([(3, []), (2, ["A"])], {"A": 5, "B": 4}, 9, 5, "W1"),
    # A 6 persons days 0-4, B 6 persons days 4-8: week 1 sums 12 (B starts
    # inside it), yet the busiest day holds one crew of 6, which is in week 1.
    ([(4, []), (4, ["A"])], {"A": 6, "B": 6}, 12, 6, "W1"),
])
def test_resource_histogram_peak_is_one_day_not_one_bucket(spans, counts, period_total, peak, peak_period):
    acts = [_act("A" if i == 0 else "B", dur, preds,
                 resources=[ResourceAssignment(trade="crew", count=counts["A" if i == 0 else "B"])])
            for i, (dur, preds) in enumerate(spans)]
    out = compute_cpm(CPMInput(activities=acts))
    hist = resource_histogram(out.results, acts, period_unit="week")
    assert hist.periods[0].total == period_total
    assert hist.peak_total == peak
    assert hist.peak_period == peak_period


@pytest.mark.parametrize("dur, synthetic_count, qty, weeks", [
    # 10 days, 80 man-hours exported by P6: 8 per day -> 40 in each 5-day week.
    (10, 99, 80, [40.0, 40.0]),
    # 6 days, 120 man-hours: 20 per day -> 100 in week 1, 20 in the 1-day tail.
    (6, 7, 120, [100.0, 20.0]),
])
def test_resource_histogram_prefers_the_exported_task_rows(dur, synthetic_count, qty, weeks):
    """A real TASKRSRC export outranks the synthetic activity.resources guess.

    The synthetic crew count is deliberately absurd (99 / 7) so that a code path
    that silently falls back to it is visible in the totals.
    """
    acts = [_act("A", dur, resources=[ResourceAssignment(trade="entered-by-hand", count=synthetic_count)])]
    out = compute_cpm(CPMInput(activities=acts))
    hist = resource_histogram(
        out.results, acts, period_unit="week",
        task_resources=[{"task_code": "A", "target_qty": qty, "rsrc_id": "from-p6"}],
    )
    assert hist.by_trade_totals == {"from-p6": float(qty)}
    assert "entered-by-hand" not in hist.by_trade_totals
    assert hist.total_manhours == float(qty)
    assert [p.total for p in hist.periods] == weeks


@pytest.mark.parametrize("bad_unit", ["quarter", "day"])
def test_resource_histogram_rejects_an_unknown_period_unit(bad_unit):
    acts = [_act("A", 5, resources=[ResourceAssignment(trade="civil", count=2)])]
    out = compute_cpm(CPMInput(activities=acts))
    with pytest.raises(ValueError, match="period_unit"):
        resource_histogram(out.results, acts, period_unit=bad_unit,
                           task_resources=[{"task_code": "A", "target_qty": 40, "rsrc_id": "lab"}])


@pytest.mark.parametrize("parallel, period_totals, peak, peak_period", [
    # A alone days 0-9: 160 man-hours over 10 days = 16 mh/day = 2 persons;
    # B alone days 10-19: 240 man-hours = 24 mh/day = 3 persons. Week 3 sums
    # 120 man-hours and week 4 sums 120 too, but no day holds more than B's 3
    # persons - a bucket total is man-hours, not a headcount.
    (False, [80.0, 80.0, 120.0, 120.0], 3.0, "W3"),
    # A and B both run days 0-9: 2 + 3 persons every day, so the busiest day
    # holds 5 while week 1 sums 200 man-hours.
    (True, [200.0, 200.0], 5.0, "W1"),
])
def test_taskrsrc_peak_is_the_busiest_day_in_persons(parallel, period_totals, peak, peak_period):
    """The stored P6 form takes the same day-level peak as a synthetic network.

    TASKRSRC carries man-hours, so the peak needs the module's 8-hour day to
    become persons; the period totals stay man-hours.
    """
    acts = [_act("A", 10), _act("B", 10)]
    if not parallel:
        acts[1] = _act("B", 10, preds=["A"])
    out = compute_cpm(CPMInput(activities=acts))
    hist = resource_histogram(
        out.results, acts, period_unit="week",
        task_resources=[
            {"task_code": "A", "target_qty": 160, "rsrc_id": "lab"},
            {"task_code": "B", "target_qty": 240, "rsrc_id": "lab"},
        ],
    )
    assert [p.total for p in hist.periods] == period_totals
    assert hist.peak_total == peak
    assert hist.peak_period == peak_period


@pytest.mark.parametrize("with_export", [False, True])
def test_histogram_with_no_placeable_work_has_no_peak(with_export):
    rows = ([{"task_code": "A", "target_qty": 40, "rsrc_id": "lab"}]
            if with_export else None)
    hist = resource_histogram([], [], period_unit="week", task_resources=rows)
    assert hist.peak_total == 0.0
    assert hist.peak_period == ""
    assert hist.total_manhours == 0.0


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
