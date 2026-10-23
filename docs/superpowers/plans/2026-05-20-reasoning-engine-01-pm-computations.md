# Reasoning Engine — Plan 1: pm_computations (CPM core)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Part of the Reasoning Engine — see `2026-05-20-reasoning-engine-INDEX.md`.

**Goal:** Build the Critical Path Method as **reusable library functions** in `app/lib/pm_computations.py` — not a block. Generated code (Plan 4) and the reasoner (Plan 5) import these tested functions instead of re-deriving the algorithm.

**Architecture:** `app/schemas/cpm.py` holds Pydantic models. `app/lib/pm_computations.py` holds pure functions: `topological_order`, `cpm_forward_pass`, `cpm_backward_pass`, `calculate_float`, and the `compute_cpm` orchestrator. No AI, no I/O, no block registration. Importable as `from app.lib.pm_computations import compute_cpm`.

**Tech Stack:** Python 3.11, Pydantic v2 (installed), stdlib only — topological sort via Kahn's algorithm, no graph library.

**Conventions:**
- ES/EF/LS/LF are **working-day offsets** (integers, day 0 = project start).
- `EF = ES + duration` — a 0-duration milestone has `ES == EF`.
- The `*_day` integer fields on `CPMResult` are authoritative; the `date` fields are a calendar projection and are `None` when no `project_start` is given.
- `is_critical` ⇔ `total_float <= 0` (in this model float is never negative, so this is exact equality with 0).
- **Finish-date projection caveat:** a finish offset projects via `nth_working_day(start, EF)`, i.e. the *start of the day after* the last working day. So `early_finish` reads one working day beyond the activity's actual last day. This is fine for a library feeding generated code (the `*_day` ints are authoritative), but **Plan 6 (user-facing Excel/Gantt) must subtract one working day when displaying finish dates.**

**Run tests:** `& .venv\Scripts\python.exe -m pytest <path> -q` from `C:\Users\example\The_Fork`.

---

### Task 1: CPM Pydantic schemas

**Files:**
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/cpm.py`
- Test: `tests/test_pm_computations.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pm_computations.py`:

```python
"""Tests for pm_computations — Reasoning Engine Plan 1 (CPM core)."""

import pytest

from app.schemas.cpm import Activity, CPMInput, Dependency, DependencyType


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas.cpm'`

- [ ] **Step 3: Write the schemas**

Create `app/schemas/__init__.py` (empty).

Create `app/schemas/cpm.py`:

```python
"""Pydantic schemas for the CPM engine — Reasoning Engine Plan 1."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DependencyType(str, Enum):
    FS = "FS"  # finish-to-start (default)
    SS = "SS"  # start-to-start
    FF = "FF"  # finish-to-finish
    SF = "SF"  # start-to-finish


class Dependency(BaseModel):
    predecessor_id: str
    type: DependencyType = DependencyType.FS
    lag: int = 0  # working days; may be negative


class WorkCalendar(BaseModel):
    """A working-day calendar. Monday=0 .. Sunday=6.

    The nth_working_day() method is added in Task 2.
    """
    work_weekdays: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    holidays: List[date] = Field(default_factory=list)


class Activity(BaseModel):
    id: str = Field(min_length=1)
    name: str = ""
    duration: int = Field(ge=0)  # working days
    predecessors: List[Dependency] = Field(default_factory=list)
    wbs_code: str = ""


class CPMResult(BaseModel):
    id: str
    name: str
    duration: int
    early_start_day: int
    early_finish_day: int
    late_start_day: int
    late_finish_day: int
    total_float: int
    free_float: int
    is_critical: bool
    early_start: Optional[date] = None
    early_finish: Optional[date] = None
    late_start: Optional[date] = None
    late_finish: Optional[date] = None


class CPMInput(BaseModel):
    activities: List[Activity]
    project_start: Optional[date] = None
    calendar: WorkCalendar = Field(default_factory=WorkCalendar)


class CPMOutput(BaseModel):
    results: List[CPMResult]
    project_duration: int
    project_finish: Optional[date] = None
    critical_path: List[str]
    critical_percentage: float
    near_critical: List[str]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/__init__.py app/schemas/cpm.py tests/test_pm_computations.py
git commit -m "feat(cpm): CPM Pydantic schemas (reasoning engine plan 1)"
```

---

### Task 2: WorkCalendar working-day arithmetic

**Files:**
- Modify: `app/schemas/cpm.py` (add `nth_working_day` to `WorkCalendar`)
- Test: `tests/test_pm_computations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_computations.py`:

```python
from datetime import date

from app.schemas.cpm import WorkCalendar


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: FAIL — `AttributeError: 'WorkCalendar' object has no attribute 'nth_working_day'`

- [ ] **Step 3: Add the method**

In `app/schemas/cpm.py`, change the imports line `from datetime import date` to:

```python
from datetime import date, timedelta
```

Then add this method inside the `WorkCalendar` class (after the `holidays` field):

```python
    def nth_working_day(self, start: date, n: int) -> date:
        """Calendar date of the n-th working day (0-indexed); offset 0 is the
        first working day on or after `start`."""
        work = set(self.work_weekdays)
        hol = set(self.holidays)

        def is_working(d: date) -> bool:
            return d.weekday() in work and d not in hol

        d = start
        while not is_working(d):
            d += timedelta(days=1)
        count = 0
        while count < n:
            d += timedelta(days=1)
            if is_working(d):
                count += 1
        return d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/cpm.py tests/test_pm_computations.py
git commit -m "feat(cpm): WorkCalendar working-day arithmetic"
```

---

### Task 3: Library package + topological sort

**Files:**
- Create: `app/lib/__init__.py`
- Create: `app/lib/pm_computations.py`
- Test: `tests/test_pm_computations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_computations.py`:

```python
from app.lib.pm_computations import CircularDependencyError, topological_order


def _act(id, dur, preds=None):
    return Activity(
        id=id, duration=dur,
        predecessors=[Dependency(predecessor_id=p) for p in (preds or [])],
    )


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.lib.pm_computations'`

- [ ] **Step 3: Create the library and topological sort**

Create `app/lib/__init__.py` (empty).

Create `app/lib/pm_computations.py`:

```python
"""Reusable project-management computations — Reasoning Engine Plan 1.

Pure functions, no AI, no I/O. Generated code (Plan 4) and the reasoner
(Plan 5) import these instead of re-deriving the algorithms.

CPM math runs in working-day offsets (integers). See the plan header for the
offset conventions.
"""

from typing import Dict, List, Tuple

from app.schemas.cpm import (
    Activity, CPMInput, CPMOutput, CPMResult, DependencyType,
)


class CircularDependencyError(ValueError):
    """Raised when the activity network contains a cycle."""


def topological_order(activities: List[Activity]) -> List[str]:
    """Activity ids in dependency order (Kahn's algorithm).

    Raises ValueError for an unknown predecessor, CircularDependencyError for
    a cycle. Ties broken by id, so the order is deterministic.
    """
    ids = {a.id for a in activities}
    indegree: Dict[str, int] = {a.id: 0 for a in activities}
    successors: Dict[str, List[str]] = {a.id: [] for a in activities}

    for a in activities:
        for dep in a.predecessors:
            if dep.predecessor_id not in ids:
                raise ValueError(
                    f"Activity '{a.id}' references unknown predecessor "
                    f"'{dep.predecessor_id}'"
                )
            indegree[a.id] += 1
            successors[dep.predecessor_id].append(a.id)

    queue = sorted(i for i, d in indegree.items() if d == 0)
    order: List[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for succ in successors[nid]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)
        queue.sort()

    if len(order) != len(activities):
        cycle = sorted(set(indegree) - set(order))
        raise CircularDependencyError(
            f"Circular dependency among: {', '.join(cycle)}"
        )
    return order
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/__init__.py app/lib/pm_computations.py tests/test_pm_computations.py
git commit -m "feat(cpm): pm_computations library + topological sort"
```

---

### Task 4: Forward pass

**Files:**
- Modify: `app/lib/pm_computations.py` (add `cpm_forward_pass`)
- Test: `tests/test_pm_computations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_computations.py`:

```python
from app.lib.pm_computations import cpm_forward_pass


def _index(acts):
    return {a.id: a for a in acts}


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: FAIL — `ImportError: cannot import name 'cpm_forward_pass'`

- [ ] **Step 3: Implement the forward pass**

Append to `app/lib/pm_computations.py`:

```python
def cpm_forward_pass(
    acts: Dict[str, Activity], order: List[str]
) -> Dict[str, Tuple[int, int]]:
    """Compute (ES, EF) working-day offsets. `order` must be topological."""
    es: Dict[str, int] = {}
    ef: Dict[str, int] = {}
    for nid in order:
        a = acts[nid]
        start = 0
        for dep in a.predecessors:
            p_es, p_ef = es[dep.predecessor_id], ef[dep.predecessor_id]
            if dep.type == DependencyType.FS:
                cand = p_ef + dep.lag
            elif dep.type == DependencyType.SS:
                cand = p_es + dep.lag
            elif dep.type == DependencyType.FF:
                cand = p_ef + dep.lag - a.duration
            else:  # SF
                cand = p_es + dep.lag - a.duration
            start = max(start, cand)
        es[nid] = start
        ef[nid] = start + a.duration
    return {nid: (es[nid], ef[nid]) for nid in order}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: PASS (16 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/pm_computations.py tests/test_pm_computations.py
git commit -m "feat(cpm): forward pass with FS/SS/FF/SF + lag"
```

---

### Task 5: Backward pass, float, and compute_cpm

**Files:**
- Modify: `app/lib/pm_computations.py` (add `cpm_backward_pass`, `calculate_float`, `compute_cpm`)
- Test: `tests/test_pm_computations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_computations.py`:

```python
from datetime import date as _date

from app.lib.pm_computations import compute_cpm


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_cpm'`

- [ ] **Step 3: Implement backward pass, float, and the orchestrator**

Append to `app/lib/pm_computations.py`:

```python
def _successor_map(acts: Dict[str, Activity]) -> Dict[str, list]:
    """Map each activity id to a list of (successor_id, dep_type, lag)."""
    succ: Dict[str, list] = {nid: [] for nid in acts}
    for a in acts.values():
        for dep in a.predecessors:
            succ[dep.predecessor_id].append((a.id, dep.type, dep.lag))
    return succ


def cpm_backward_pass(
    acts: Dict[str, Activity], order: List[str], project_duration: int
) -> Dict[str, Tuple[int, int]]:
    """Compute (LS, LF) working-day offsets. `order` must be topological."""
    succ = _successor_map(acts)
    ls: Dict[str, int] = {}
    lf: Dict[str, int] = {}
    for nid in reversed(order):
        a = acts[nid]
        finish = project_duration
        for (s_id, s_type, lag) in succ[nid]:
            s_ls, s_lf = ls[s_id], lf[s_id]
            if s_type == DependencyType.FS:
                cand = s_ls - lag
            elif s_type == DependencyType.SS:
                cand = s_ls - lag + a.duration
            elif s_type == DependencyType.FF:
                cand = s_lf - lag
            else:  # SF
                cand = s_lf - lag + a.duration
            finish = min(finish, cand)
        lf[nid] = finish
        ls[nid] = finish - a.duration
    return {nid: (ls[nid], lf[nid]) for nid in acts}


def calculate_float(
    acts: Dict[str, Activity], fwd: Dict[str, Tuple[int, int]]
) -> Dict[str, int]:
    """Free float per activity: how far it can slip without delaying any
    successor's early dates. Returns -1 for activities with no successor
    (the caller substitutes total float)."""
    succ = _successor_map(acts)
    ff: Dict[str, int] = {}
    for nid, a in acts.items():
        es_j, ef_j = fwd[nid]
        slacks = []
        for (s_id, s_type, lag) in succ[nid]:
            es_k, ef_k = fwd[s_id]
            if s_type == DependencyType.FS:
                slacks.append(es_k - (ef_j + lag))
            elif s_type == DependencyType.SS:
                slacks.append(es_k - (es_j + lag))
            elif s_type == DependencyType.FF:
                slacks.append(ef_k - (ef_j + lag))
            else:  # SF
                slacks.append(ef_k - (es_j + lag))
        ff[nid] = max(0, min(slacks)) if slacks else -1
    return ff


def compute_cpm(data: CPMInput) -> CPMOutput:
    """Run the full Critical Path Method over an activity network."""
    activities = data.activities
    if not activities:
        return CPMOutput(results=[], project_duration=0, project_finish=None,
                         critical_path=[], critical_percentage=0.0,
                         near_critical=[])

    acts = {a.id: a for a in activities}
    if len(acts) != len(activities):
        raise ValueError("Duplicate activity ids in input")

    order = topological_order(activities)
    fwd = cpm_forward_pass(acts, order)
    project_duration = max(ef for (_es, ef) in fwd.values())
    bwd = cpm_backward_pass(acts, order, project_duration)
    ff = calculate_float(acts, fwd)
    cal, start = data.calendar, data.project_start

    def proj(offset: int):
        return cal.nth_working_day(start, offset) if start else None

    results: List[CPMResult] = []
    for nid in order:
        a = acts[nid]
        es, ef = fwd[nid]
        ls, lf = bwd[nid]
        tf = ls - es
        results.append(CPMResult(
            id=a.id, name=a.name, duration=a.duration,
            early_start_day=es, early_finish_day=ef,
            late_start_day=ls, late_finish_day=lf,
            total_float=tf,
            free_float=tf if ff[nid] < 0 else ff[nid],
            is_critical=(tf <= 0),
            early_start=proj(es), early_finish=proj(ef),
            late_start=proj(ls), late_finish=proj(lf),
        ))

    critical = sorted((r for r in results if r.is_critical),
                      key=lambda r: (r.early_start_day, r.early_finish_day))
    near = [r.id for r in results if 0 < r.total_float <= 5]
    return CPMOutput(
        results=results,
        project_duration=project_duration,
        project_finish=proj(project_duration),
        critical_path=[r.id for r in critical],
        critical_percentage=round(len(critical) / len(results) * 100, 1),
        near_critical=near,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: PASS (22 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/pm_computations.py tests/test_pm_computations.py
git commit -m "feat(cpm): backward pass, float, compute_cpm orchestrator"
```

---

### Task 6: Realistic integration test + regression

**Files:**
- Test: `tests/test_pm_computations.py`

- [ ] **Step 1: Write the integration test**

Append to `tests/test_pm_computations.py`:

```python
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
```

- [ ] **Step 2: Run the test**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_computations.py -q`
Expected: PASS (23 passed)

- [ ] **Step 3: Run the full suite — confirm no regression**

Run: `& .venv\Scripts\python.exe -m pytest --ignore=tests/browser -q`
Expected: PASS — 263 passed (240 prior + 23 new), 85 skipped

- [ ] **Step 4: Commit**

```bash
git add tests/test_pm_computations.py
git commit -m "test(cpm): realistic network integration test"
```

---

## Self-Review

**Spec coverage** (Reasoning Engine §5.4 / §7 — CPM as a *library function*):
- `cpm_forward_pass` → Task 4 ✅
- `cpm_backward_pass` → Task 5 ✅
- `calculate_float` → Task 5 ✅
- Topological sort + cycle detection → Task 3 ✅
- Calendar arithmetic → Task 2 ✅
- FS/SS/FF/SF + positive **and negative** lag → Tasks 4–5 ✅
- Importable by generated code (`from app.lib.pm_computations import ...`) → package created Task 3 ✅
- **Not** a registered block — correct per spec §7 ✅

**Deferred to Plan 1b** (`2026-05-20-reasoning-engine-01b-*.md`): `resource_histogram`,
`compress_schedule`, `gantt_data`, `write_schedule_excel`, `parse_xer`, and
`app/lib/excel_templates.py`. Plan 1b also owns the finish-date display fix
(subtract one working day) noted in this plan's header.

**Placeholder scan:** none — every step has complete code or an exact command.

**Type consistency:** `compute_cpm(CPMInput) -> CPMOutput`. `cpm_forward_pass` /
`cpm_backward_pass` take `Dict[str, Activity]` + topological `order`, return
`Dict[str, Tuple[int, int]]`. `calculate_float` takes `Dict[str, Activity]` +
the forward-pass dict, returns `Dict[str, int]`. `topological_order(List[Activity])
-> List[str]`. Every task has a failing-test step before its implementation.

---

**Plan 1 complete.** Next: Plan 1b, then Plan 2. See the index for the full sequence.
