# Reasoning Engine — Plan 1b: pm_computations extended (resource / gantt / compress)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.
> Part of the Reasoning Engine — see `2026-05-20-reasoning-engine-INDEX.md`. Depends on Plan 1.

**Goal:** Extend `app/lib/pm_computations.py` with three more pure functions the reasoner/code-gen will lean on: `resource_histogram` (time-phased manpower), `gantt_data` (bars for a chart), `compress_schedule` (apply duration cuts and re-run CPM).

**Architecture:** All pure functions, no AI, no I/O. New Pydantic models go in `app/schemas/cpm.py`; functions append to `app/lib/pm_computations.py`. Excel writing and `.xer` import are deliberately **not** here — they are I/O and live in Plan 6.

**Tech Stack:** Python 3.11, Pydantic v2. Reuses `compute_cpm` from Plan 1.

**Conventions:** same as Plan 1 — working-day offsets; a "period" is a fixed run of working days (week = 5, month = 21). `resource_histogram` reports concurrent headcount per period (an activity contributes its crew to every period it overlaps).

**Run tests:** `& .venv\Scripts\python.exe -m pytest <path> -q` from `C:\Users\example\The_Fork`. **Plan 1 must be complete first.**

---

### Task 1: Resource, histogram & gantt schemas

**Files:**
- Modify: `app/schemas/cpm.py` (add models + a field on `Activity`)
- Test: `tests/test_pm_extended.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pm_extended.py`:

```python
"""Tests for pm_computations extended — Reasoning Engine Plan 1b."""

import pytest

from app.schemas.cpm import (
    Activity, GanttBar, HistogramPeriod, ResourceAssignment, ResourceHistogram,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: FAIL — `ImportError: cannot import name 'ResourceAssignment'`

- [ ] **Step 3: Add the schemas**

In `app/schemas/cpm.py`, add this model **above** the `Activity` class:

```python
class ResourceAssignment(BaseModel):
    trade: str
    count: float = 1.0  # crew size / headcount on the activity
```

Add a `resources` field to `Activity` (after `predecessors`):

```python
    resources: List[ResourceAssignment] = Field(default_factory=list)
```

Add these models at the **end** of the file:

```python
class HistogramPeriod(BaseModel):
    index: int
    label: str
    total: float
    by_trade: Dict[str, float]


class ResourceHistogram(BaseModel):
    period_unit: str            # 'week' | 'month'
    periods: List[HistogramPeriod]
    peak_total: float
    peak_period: str
    by_trade_totals: Dict[str, float]
    total_manhours: float


class GanttBar(BaseModel):
    id: str
    name: str
    start_day: int
    end_day: int
    is_critical: bool
```

Add `Dict` to the typing import at the top of the file — change `from typing import List, Optional` to:

```python
from typing import Dict, List, Optional
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/cpm.py tests/test_pm_extended.py
git commit -m "feat(pm): resource, histogram & gantt schemas (plan 1b)"
```

---

### Task 2: resource_histogram

**Files:**
- Modify: `app/lib/pm_computations.py` (add `resource_histogram`)
- Test: `tests/test_pm_extended.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_extended.py`:

```python
from app.schemas.cpm import CPMInput, Dependency
from app.lib.pm_computations import compute_cpm, resource_histogram

_PERIOD_DAYS = {"week": 5, "month": 21}


def _act(id, dur, preds=None, resources=None):
    return Activity(
        id=id, duration=dur,
        predecessors=[Dependency(predecessor_id=p) for p in (preds or [])],
        resources=resources or [],
    )


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: FAIL — `ImportError: cannot import name 'resource_histogram'`

- [ ] **Step 3: Implement resource_histogram**

Append to `app/lib/pm_computations.py`. First extend the schema import at the top of the file to:

```python
from app.schemas.cpm import (
    Activity, CPMInput, CPMOutput, CPMResult, DependencyType,
    GanttBar, HistogramPeriod, ResourceHistogram,
)
```

Then append:

```python
_PERIOD_LENGTH = {"week": 5, "month": 21}
_HOURS_PER_DAY = 8


def resource_histogram(
    results: List[CPMResult],
    activities: List[Activity],
    period_unit: str = "week",
) -> ResourceHistogram:
    """Time-phased manpower. An activity contributes its crew to every period
    its early-date span overlaps (concurrent headcount, not man-days)."""
    length = _PERIOD_LENGTH.get(period_unit, 5)
    res_by_id = {a.id: a.resources for a in activities}
    es_ef = {r.id: (r.early_start_day, r.early_finish_day) for r in results}

    if results:
        last_day = max(ef for (_es, ef) in es_ef.values())
        n_periods = max(1, -(-last_day // length))  # ceil division
    else:
        n_periods = 0

    periods: List[HistogramPeriod] = []
    by_trade_totals: Dict[str, float] = {}
    total_manhours = 0.0

    for p in range(n_periods):
        p_start, p_end = p * length, (p + 1) * length
        by_trade: Dict[str, float] = {}
        for rid, (es, ef) in es_ef.items():
            if ef <= p_start or es >= p_end:
                continue  # activity does not overlap this period
            for res in res_by_id.get(rid, []):
                by_trade[res.trade] = by_trade.get(res.trade, 0.0) + res.count
        periods.append(HistogramPeriod(
            index=p, label=f"{period_unit[0].upper()}{p + 1}",
            total=round(sum(by_trade.values()), 2), by_trade=by_trade,
        ))

    for a in activities:
        es, ef = es_ef.get(a.id, (0, 0))
        span = ef - es
        for res in a.resources:
            by_trade_totals[res.trade] = (
                by_trade_totals.get(res.trade, 0.0) + res.count
            )
            total_manhours += res.count * span * _HOURS_PER_DAY

    peak = max(periods, key=lambda hp: hp.total, default=None)
    return ResourceHistogram(
        period_unit=period_unit,
        periods=periods,
        peak_total=peak.total if peak else 0.0,
        peak_period=peak.label if peak else "",
        by_trade_totals=by_trade_totals,
        total_manhours=round(total_manhours, 2),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/pm_computations.py tests/test_pm_extended.py
git commit -m "feat(pm): resource_histogram — time-phased manpower"
```

---

### Task 3: gantt_data

**Files:**
- Modify: `app/lib/pm_computations.py` (add `gantt_data`)
- Test: `tests/test_pm_extended.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_extended.py`:

```python
from app.lib.pm_computations import gantt_data


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: FAIL — `ImportError: cannot import name 'gantt_data'`

- [ ] **Step 3: Implement gantt_data**

Append to `app/lib/pm_computations.py`:

```python
def gantt_data(results: List[CPMResult]) -> List[GanttBar]:
    """One Gantt bar per activity, sorted by early start (then early finish)."""
    bars = [
        GanttBar(
            id=r.id, name=r.name,
            start_day=r.early_start_day, end_day=r.early_finish_day,
            is_critical=r.is_critical,
        )
        for r in results
    ]
    bars.sort(key=lambda b: (b.start_day, b.end_day))
    return bars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/pm_computations.py tests/test_pm_extended.py
git commit -m "feat(pm): gantt_data — bars for the chart"
```

---

### Task 4: compress_schedule

**Files:**
- Modify: `app/lib/pm_computations.py` (add `compress_schedule`)
- Test: `tests/test_pm_extended.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_extended.py`:

```python
from app.lib.pm_computations import compress_schedule


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: FAIL — `ImportError: cannot import name 'compress_schedule'`

- [ ] **Step 3: Implement compress_schedule**

Append to `app/lib/pm_computations.py`:

```python
def compress_schedule(
    data: CPMInput, reductions: Dict[str, int]
) -> Tuple[CPMOutput, int]:
    """Apply working-day duration cuts to named activities and re-run CPM.

    `reductions` maps activity id -> working days to remove (floored at 0
    duration). Returns (revised CPMOutput, days saved vs the baseline).
    Raises ValueError if an id is not in the network.
    """
    ids = {a.id for a in data.activities}
    unknown = set(reductions) - ids
    if unknown:
        raise ValueError(f"Unknown activity ids: {', '.join(sorted(unknown))}")

    baseline = compute_cpm(data)
    revised_acts = []
    for a in data.activities:
        if a.id in reductions:
            new_dur = max(0, a.duration - reductions[a.id])
            revised_acts.append(a.model_copy(update={"duration": new_dur}))
        else:
            revised_acts.append(a)

    revised = compute_cpm(data.model_copy(update={"activities": revised_acts}))
    delta = baseline.project_duration - revised.project_duration
    return revised, delta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_extended.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/pm_computations.py tests/test_pm_extended.py
git commit -m "feat(pm): compress_schedule — apply cuts and re-run CPM"
```

---

### Task 5: Regression check

**Files:** none — verification only.

- [ ] **Step 1: Run the full suite**

Run: `& .venv\Scripts\python.exe -m pytest --ignore=tests/browser -q`
Expected: PASS — 274 passed (263 after Plan 1 + 11 new), 85 skipped

- [ ] **Step 2: Commit (only if anything needed touching)**

If the suite is green with no further changes, there is nothing to commit — Tasks 1–4 already committed their work. If a regression appeared, fix it, then:

```bash
git add -A
git commit -m "fix(pm): resolve regression from plan 1b"
```

---

## Self-Review

**Spec coverage** (Reasoning Engine §5.4 / §7.1 — library functions):
- `resource_histogram` → Task 2 ✅
- `gantt_data` → Task 3 ✅
- `compress_schedule` → Task 4 ✅ (mechanical compression; the LLM *chooses* what to compress — that is the reasoner's job in Plan 5)
- `write_schedule_excel`, `parse_xer`, `excel_templates.py` → **not here** — they are I/O, moved to Plan 6 (index updated).

**Placeholder scan:** none — complete code or exact command in every step.

**Type consistency:** `resource_histogram(List[CPMResult], List[Activity], str) -> ResourceHistogram`. `gantt_data(List[CPMResult]) -> List[GanttBar]`. `compress_schedule(CPMInput, Dict[str,int]) -> Tuple[CPMOutput, int]`. All reuse `compute_cpm` and the Plan 1 schemas; new models (`ResourceAssignment`, `HistogramPeriod`, `ResourceHistogram`, `GanttBar`) are defined in Task 1 before use. Every task has a failing-test step first.

---

**Plan 1b complete.** Next: Plan 2 (Session State Store).
