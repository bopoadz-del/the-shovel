# Reasoning Engine — Plan 6: API, UI & output

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.
> Part of the Reasoning Engine — see `2026-05-20-reasoning-engine-INDEX.md`. **Depends on Plan 5.**

**Goal:** Expose the reasoner to users and finish the I/O layer. Three strands:
1. **API** — `app/routers/project.py` with `POST /v1/project/ask`, mounted in `app/main.py`, backed by a process-wide session store.
2. **UI** — a project-chat panel wired into `app/static/index.html` (the platform's only frontend; the old React app was deleted).
3. **I/O** — `parse_xer` (Primavera `.xer` import) added to `app/lib/pm_computations.py`, plus `write_schedule_excel` and the reusable `app/lib/excel_templates.py`. Per spec §7.1, I/O is the *only* thing that stays code-level here — the algorithms already live in Plans 1/1b.

**Architecture:**
- `app/lib/excel_templates.py` — pure formatting helpers built on `openpyxl` (already in `requirements.txt`, line 36): column styling, a Gantt-bar cell painter, a histogram block. No project logic — just spreadsheet shaping.
- `parse_xer` — appended to `app/lib/pm_computations.py`: parses the tab-delimited Primavera `.xer` text format into `Activity` objects. Pure parsing, no file I/O (caller passes the text).
- `write_schedule_excel` — also appended to `app/lib/pm_computations.py`: takes a `CPMOutput` (+ optional `ResourceHistogram`) and a path, writes a formatted `.xlsx` using `excel_templates`. This is the one genuinely I/O function.
- `app/routers/project.py` — `POST /v1/project/ask`: resolves/creates a `ProjectSession`, runs `ProjectReasonerBlock`, saves the session, returns the answer. Uses the same `require_api_key` dependency as `app/routers/chat.py`.
- `app/main.py` — MODIFIED: import and `include_router(project.router)`; initialise a module-level session store in the `lifespan` startup.
- `app/static/index.html` — MODIFIED: a "Project" mode that posts to `/v1/project/ask` and renders the answer + artifacts.

**Finish-date display fix:** per Plan 1's header, a finish offset projects one
working day beyond the activity's actual last day. `write_schedule_excel` is
user-facing, so it **subtracts one working day** when showing finish dates
(Task 3).

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, `openpyxl` (installed), vanilla JS. No new dependencies.

**Run tests:** `& .venv\Scripts\python.exe -m pytest <path> -q` from `C:\Users\example\The_Fork`. **Plan 5 must be complete first.**

---

### Task 1: parse_xer — Primavera .xer import

**Files:**
- Modify: `app/lib/pm_computations.py` (append `parse_xer`)
- Test: `tests/test_pm_io.py`

The `.xer` format is tab-delimited text. Lines starting with `%T` open a table,
`%F` gives that table's field names, `%R` is a data row, `%E` ends the file. We
need the `TASK` table (activities) and `TASKPRED` table (relationships).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pm_io.py`:

```python
"""Tests for pm_computations I/O — Reasoning Engine Plan 6."""

import pytest

from app.lib.pm_computations import parse_xer


_MINIMAL_XER = "\t".join(["%T", "TASK"]) + "\n" + \
    "\t".join(["%F", "task_id", "task_code", "task_name", "target_drtn_hr_cnt"]) + "\n" + \
    "\t".join(["%R", "1001", "A", "Mobilise", "40"]) + "\n" + \
    "\t".join(["%R", "1002", "B", "Excavate", "80"]) + "\n" + \
    "\t".join(["%T", "TASKPRED"]) + "\n" + \
    "\t".join(["%F", "task_id", "pred_task_id", "pred_type", "lag_hr_cnt"]) + "\n" + \
    "\t".join(["%R", "1002", "1001", "PR_FS", "0"]) + "\n" + \
    "%E\n"


def test_parse_xer_reads_activities():
    acts = parse_xer(_MINIMAL_XER)
    assert {a.id for a in acts} == {"A", "B"}
    mob = next(a for a in acts if a.id == "A")
    assert mob.name == "Mobilise"
    assert mob.duration == 5            # 40 hr / 8 hr-per-day


def test_parse_xer_reads_predecessors():
    acts = parse_xer(_MINIMAL_XER)
    exc = next(a for a in acts if a.id == "B")
    assert len(exc.predecessors) == 1
    assert exc.predecessors[0].predecessor_id == "A"


def test_parse_xer_maps_relationship_types():
    xer = _MINIMAL_XER.replace("PR_FS", "PR_SS")
    exc = next(a for a in parse_xer(xer) if a.id == "B")
    from app.schemas.cpm import DependencyType
    assert exc.predecessors[0].type == DependencyType.SS


def test_parse_xer_empty_input_returns_empty_list():
    assert parse_xer("%E\n") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_io.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_xer'`

- [ ] **Step 3: Implement parse_xer**

Append to `app/lib/pm_computations.py`. First extend the schema import at the
top of the file to include `Dependency`:

```python
from app.schemas.cpm import (
    Activity, CPMInput, CPMOutput, CPMResult, Dependency, DependencyType,
    GanttBar, HistogramPeriod, ResourceHistogram,
)
```

Then append:

```python
_XER_PRED_TYPE = {
    "PR_FS": DependencyType.FS, "PR_SS": DependencyType.SS,
    "PR_FF": DependencyType.FF, "PR_SF": DependencyType.SF,
}


def parse_xer(text: str) -> List[Activity]:
    """Parse Primavera P6 `.xer` text into Activity objects.

    Pure parsing — the caller is responsible for reading the file. Reads the
    TASK table (activities) and TASKPRED table (relationships). Durations come
    from `target_drtn_hr_cnt` converted to working days at 8 h/day; lag from
    `lag_hr_cnt` likewise. Unknown columns are ignored.
    """
    tables: Dict[str, Dict] = {}
    current: str = ""
    fields: List[str] = []

    for line in text.splitlines():
        if not line:
            continue
        cells = line.split("\t")
        tag = cells[0]
        if tag == "%T":
            current = cells[1] if len(cells) > 1 else ""
            tables[current] = {"fields": [], "rows": []}
        elif tag == "%F":
            fields = cells[1:]
            if current in tables:
                tables[current]["fields"] = fields
        elif tag == "%R":
            if current in tables:
                row = dict(zip(tables[current]["fields"], cells[1:]))
                tables[current]["rows"].append(row)
        elif tag == "%E":
            break

    task_rows = tables.get("TASK", {}).get("rows", [])
    pred_rows = tables.get("TASKPRED", {}).get("rows", [])
    if not task_rows:
        return []

    # task_id -> task_code (the human id used as Activity.id)
    code_by_tid = {r.get("task_id"): r.get("task_code") or r.get("task_id")
                   for r in task_rows}

    preds_by_tid: Dict[str, List[Dependency]] = {}
    for r in pred_rows:
        tid = r.get("task_id")
        pred_tid = r.get("pred_task_id")
        pred_code = code_by_tid.get(pred_tid)
        if not tid or not pred_code:
            continue
        ptype = _XER_PRED_TYPE.get(r.get("pred_type", "PR_FS"),
                                   DependencyType.FS)
        lag_days = round(float(r.get("lag_hr_cnt") or 0) / _HOURS_PER_DAY)
        preds_by_tid.setdefault(tid, []).append(Dependency(
            predecessor_id=pred_code, type=ptype, lag=int(lag_days),
        ))

    activities: List[Activity] = []
    for r in task_rows:
        tid = r.get("task_id")
        dur_days = round(
            float(r.get("target_drtn_hr_cnt") or 0) / _HOURS_PER_DAY
        )
        activities.append(Activity(
            id=code_by_tid.get(tid) or tid,
            name=r.get("task_name") or "",
            duration=max(0, int(dur_days)),
            predecessors=preds_by_tid.get(tid, []),
        ))
    return activities
```

> `_HOURS_PER_DAY = 8` was already defined in Plan 1b — reuse it.

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_io.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/pm_computations.py tests/test_pm_io.py
git commit -m "feat(pm-io): parse_xer — Primavera .xer import (reasoning engine plan 6)"
```

---

### Task 2: excel_templates — reusable spreadsheet formatting

**Files:**
- Create: `app/lib/excel_templates.py`
- Test: `tests/test_pm_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_io.py`:

```python
from openpyxl import Workbook

from app.lib.excel_templates import (
    header_row, paint_gantt_row, write_histogram_block,
)


def test_header_row_writes_bold_titles():
    wb = Workbook(); ws = wb.active
    header_row(ws, 1, ["ID", "Name", "Duration"])
    assert ws["A1"].value == "ID"
    assert ws["C1"].value == "Duration"
    assert ws["A1"].font.bold is True


def test_paint_gantt_row_fills_only_the_bar_span():
    wb = Workbook(); ws = wb.active
    # bar from day 2..5 across a 0..9 day grid, starting at column 3
    paint_gantt_row(ws, row=2, first_col=3, start_day=2, end_day=5,
                    total_days=10, is_critical=False)
    # day 0,1 empty; day 2 (col 5) filled; day 5 (col 8) empty (end exclusive)
    assert ws.cell(row=2, column=3).fill.start_color.rgb in (None, "00000000")
    assert ws.cell(row=2, column=5).fill.start_color.rgb is not None
    assert ws.cell(row=2, column=5).fill.start_color.rgb != "00000000"


def test_write_histogram_block_writes_period_totals():
    wb = Workbook(); ws = wb.active
    next_row = write_histogram_block(
        ws, start_row=1,
        periods=[{"label": "W1", "total": 12}, {"label": "W2", "total": 8}],
    )
    assert ws["A2"].value == "W1" and ws["B2"].value == 12
    assert ws["A3"].value == "W2" and ws["B3"].value == 8
    assert next_row == 4            # row after the block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_io.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.lib.excel_templates'`

- [ ] **Step 3: Write excel_templates**

Create `app/lib/excel_templates.py`:

```python
"""Reusable Excel formatting helpers — Reasoning Engine Plan 6.

Built on openpyxl. No project logic — only spreadsheet shaping (headers,
Gantt-bar cells, histogram blocks) so write_schedule_excel stays thin.
"""

from typing import Dict, List

from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.worksheet import Worksheet

_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="8B5CF6")
_CRITICAL_FILL = PatternFill("solid", fgColor="EF4444")  # red
_NORMAL_FILL = PatternFill("solid", fgColor="60A5FA")    # blue


def header_row(ws: Worksheet, row: int, titles: List[str]) -> None:
    """Write a bold, filled header row starting at column A."""
    for col, title in enumerate(titles, start=1):
        cell = ws.cell(row=row, column=col, value=title)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL


def paint_gantt_row(
    ws: Worksheet, row: int, first_col: int,
    start_day: int, end_day: int, total_days: int, is_critical: bool,
) -> None:
    """Fill the cells for one activity's bar across a day grid.

    The grid runs `total_days` columns from `first_col`. Cells for days in
    [start_day, end_day) are filled — red if critical, blue otherwise.
    """
    fill = _CRITICAL_FILL if is_critical else _NORMAL_FILL
    for day in range(total_days):
        if start_day <= day < end_day:
            ws.cell(row=row, column=first_col + day).fill = fill


def write_histogram_block(
    ws: Worksheet, start_row: int, periods: List[Dict],
) -> int:
    """Write a manpower histogram: a header then one row per period
    (label, total). Returns the row index just after the block."""
    header_row(ws, start_row, ["Period", "Manpower"])
    row = start_row + 1
    for p in periods:
        ws.cell(row=row, column=1, value=p.get("label"))
        ws.cell(row=row, column=2, value=p.get("total"))
        row += 1
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_io.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/excel_templates.py tests/test_pm_io.py
git commit -m "feat(pm-io): excel_templates — reusable spreadsheet formatting"
```

---

### Task 3: write_schedule_excel

**Files:**
- Modify: `app/lib/pm_computations.py` (append `write_schedule_excel`)
- Test: `tests/test_pm_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pm_io.py`:

```python
from openpyxl import load_workbook

from app.lib.pm_computations import compute_cpm, write_schedule_excel
from app.schemas.cpm import Activity, CPMInput, Dependency


def _chain():
    return [
        Activity(id="A", name="Mob", duration=3),
        Activity(id="B", name="Exc", duration=5,
                 predecessors=[Dependency(predecessor_id="A")]),
        Activity(id="C", name="Fnd", duration=2,
                 predecessors=[Dependency(predecessor_id="B")]),
    ]


def test_write_schedule_excel_creates_a_file(tmp_path):
    out = compute_cpm(CPMInput(activities=_chain()))
    path = tmp_path / "schedule.xlsx"
    written = write_schedule_excel(out, str(path))
    assert written == str(path)
    assert path.exists()


def test_write_schedule_excel_lists_every_activity(tmp_path):
    out = compute_cpm(CPMInput(activities=_chain()))
    path = tmp_path / "schedule.xlsx"
    write_schedule_excel(out, str(path))
    wb = load_workbook(path)
    ws = wb["Schedule"]
    ids = [ws.cell(row=r, column=1).value for r in range(2, 5)]
    assert ids == ["A", "B", "C"]


def test_write_schedule_excel_finish_date_subtracts_one_working_day(tmp_path):
    from datetime import date
    # A: duration 3, start Mon 2026-05-18 -> last working day is Wed 2026-05-20.
    out = compute_cpm(CPMInput(activities=[Activity(id="A", name="X", duration=3)],
                               project_start=date(2026, 5, 18)))
    path = tmp_path / "s.xlsx"
    write_schedule_excel(out, str(path))
    wb = load_workbook(path)
    ws = wb["Schedule"]
    # finish-date column shows the actual last working day, not EF+1.
    finish_cells = [ws.cell(row=2, column=c).value for c in range(1, 12)]
    assert "2026-05-20" in [str(v) for v in finish_cells]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_io.py -q`
Expected: FAIL — `ImportError: cannot import name 'write_schedule_excel'`

- [ ] **Step 3: Implement write_schedule_excel**

Append to `app/lib/pm_computations.py`. Add the imports it needs near the top
of the file (after the existing typing import):

```python
from datetime import timedelta
from typing import Optional
```

Then append:

```python
def write_schedule_excel(
    output: CPMOutput,
    path: str,
    histogram: "Optional[ResourceHistogram]" = None,
) -> str:
    """Write a CPMOutput to a formatted .xlsx and return the path.

    This is the one genuinely I/O function in the library. Produces a
    'Schedule' sheet (activity table + a Gantt grid) and, when `histogram` is
    given, a 'Manpower' sheet.

    DISPLAY NOTE: CPMResult finish offsets project one working day beyond the
    activity's actual last day (see Plan 1 header). This function is
    user-facing, so finish DATES shown here subtract one working day. The
    *_day integer columns keep the raw offsets.
    """
    from openpyxl import Workbook
    from app.lib.excel_templates import (
        header_row, paint_gantt_row, write_histogram_block,
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Schedule"

    total_days = max((r.early_finish_day for r in output.results), default=0)
    gantt_first_col = 9  # day grid starts after the 8 table columns
    header_row(ws, 1, [
        "ID", "Name", "Duration",
        "Early Start", "Early Finish", "Total Float", "Critical",
        "",  # spacer before the day grid
    ] + [f"D{d}" for d in range(total_days)])

    def _finish_date(r):
        # r.early_finish projects EF+1; show the real last working day.
        if r.early_finish is None:
            return ""
        return str(r.early_finish - timedelta(days=1))

    for i, r in enumerate(output.results, start=2):
        ws.cell(row=i, column=1, value=r.id)
        ws.cell(row=i, column=2, value=r.name)
        ws.cell(row=i, column=3, value=r.duration)
        ws.cell(row=i, column=4,
                value=str(r.early_start) if r.early_start else "")
        ws.cell(row=i, column=5, value=_finish_date(r))
        ws.cell(row=i, column=6, value=r.total_float)
        ws.cell(row=i, column=7, value="YES" if r.is_critical else "")
        paint_gantt_row(
            ws, row=i, first_col=gantt_first_col,
            start_day=r.early_start_day, end_day=r.early_finish_day,
            total_days=total_days, is_critical=r.is_critical,
        )

    if histogram is not None:
        hs = wb.create_sheet("Manpower")
        write_histogram_block(
            hs, start_row=1,
            periods=[{"label": p.label, "total": p.total}
                     for p in histogram.periods],
        )

    wb.save(path)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_pm_io.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add app/lib/pm_computations.py tests/test_pm_io.py
git commit -m "feat(pm-io): write_schedule_excel — formatted .xlsx output"
```

---

### Task 4: project router — POST /v1/project/ask

**Files:**
- Create: `app/routers/project.py`
- Test: `tests/test_project_router.py`

The router resolves a `ProjectSession`, runs `ProjectReasonerBlock`, saves the
session, and returns the answer. It uses the shared `require_api_key` dependency
(`app/dependencies.py`) like every other v1 route. The session store is created
in `app/main.py` (Task 5) and exposed via a `get_project_store()` accessor so
tests can inject an in-memory store.

- [ ] **Step 1: Write the failing test**

Create `tests/test_project_router.py`:

```python
"""Tests for the project router — Reasoning Engine Plan 6."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import project as project_router
from app.core.session_store import InMemorySessionStore


@pytest.fixture
def client(monkeypatch):
    # Inject a fresh in-memory store and a scripted mock reasoner so the
    # route is tested without the configured LLM.
    store = InMemorySessionStore()
    monkeypatch.setattr(project_router, "_store", store)

    from app.blocks.project_reasoner import ProjectReasonerBlock

    class _MockReasoner(ProjectReasonerBlock):
        async def _call_llm(self, prompt):
            if "PLAN" in prompt and "{" not in prompt:
                pass
            # first call = plan JSON, second = answer
            if not getattr(self, "_called", False):
                self._called = True
                return json.dumps({"understanding": "u",
                                    "steps": [{"type": "compute_cpm"}]})
            return "Project duration is 10 days."

    monkeypatch.setattr(project_router, "_reasoner_factory",
                        lambda: _MockReasoner())
    return TestClient(app)


def test_ask_creates_session_and_returns_answer(client):
    resp = client.post("/v1/project/ask", json={
        "session_id": "p1",
        "request": "what is the duration?",
        "activities": [
            {"id": "A", "duration": 3, "predecessors": []},
            {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
            {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
        ],
    }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Project duration is 10 days."
    assert body["session_id"] == "p1"


def test_ask_persists_session_across_calls(client):
    payload = {"session_id": "p2", "request": "go",
               "activities": [{"id": "A", "duration": 3, "predecessors": []}]}
    client.post("/v1/project/ask", json=payload,
                headers={"Authorization": "Bearer dev"})
    # second call without activities — the session must still hold them
    resp = client.post("/v1/project/ask",
                        json={"session_id": "p2", "request": "again"},
                        headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    stored = project_router._store.get("p2")
    assert stored is not None
    assert stored.data["activities"][0]["id"] == "A"


def test_ask_rejects_empty_request(client):
    resp = client.post("/v1/project/ask",
                        json={"session_id": "p3", "request": "  "},
                        headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 422 or resp.json().get("status") == "error"
```

> `Bearer dev` works because the suite runs with `ENV=development` — the same
> dev key the existing chat-router tests rely on. Match whatever the other
> router tests in `tests/` use if this differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_router.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.routers.project'`

- [ ] **Step 3: Write the router**

Create `app/routers/project.py`:

```python
"""Project reasoning API — Reasoning Engine Plan 6.

POST /v1/project/ask — run the Project Reasoner over a persistent session.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import require_api_key
from app.core.session_store import SessionStore, get_session_store

router = APIRouter()

# Process-wide session store. app/main.py overwrites this at startup with the
# shared instance; tests monkeypatch it. Lazily created so importing the module
# never fails.
_store: SessionStore = get_session_store()


def get_project_store() -> SessionStore:
    """Accessor for the active session store (overridable in tests)."""
    return _store


def _reasoner_factory():
    """Build a ProjectReasonerBlock. Indirected so tests can swap in a mock."""
    from app.blocks.project_reasoner import ProjectReasonerBlock
    return ProjectReasonerBlock()


class ProjectAskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    request: str
    # optional: load/replace the session's activity list on this turn
    activities: Optional[List[Dict[str, Any]]] = None


@router.post("/v1/project/ask")
async def project_ask(
    body: ProjectAskRequest, auth: dict = Depends(require_api_key)
):
    """Answer a project question. Creates the session on first use, persists
    it after the turn so follow-up questions build on prior state."""
    if not body.request.strip():
        raise HTTPException(422, "request must not be empty")

    session = _store.get_or_create(body.session_id)
    if body.activities is not None:
        session.data["activities"] = body.activities

    reasoner = _reasoner_factory()
    result = await reasoner.process({"request": body.request,
                                     "session": session})

    _store.save(session)   # persist the turn — history, computed state, cache

    return {
        "session_id": body.session_id,
        "status": result.get("status"),
        "answer": result.get("answer", ""),
        "understanding": result.get("understanding", ""),
        "plan": result.get("plan"),
        "execution": result.get("execution"),
        "artifacts": [a.model_dump() for a in session.artifacts],
        "error": result.get("error"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_router.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/routers/project.py tests/test_project_router.py
git commit -m "feat(api): POST /v1/project/ask project reasoning route"
```

---

### Task 5: Mount the router + init the session store in main.py

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_project_router.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_router.py`:

```python
def test_project_route_is_mounted():
    paths = {r.path for r in app.routes}
    assert "/v1/project/ask" in paths


def test_main_initialises_a_shared_store():
    # app/main.py must put a SessionStore on app.state at startup.
    with TestClient(app):              # triggers the lifespan startup
        from app.core.session_store import SessionStore
        assert isinstance(app.state.project_store, SessionStore)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_router.py -q`
Expected: FAIL — `test_project_route_is_mounted` fails: `/v1/project/ask` not in routes.

- [ ] **Step 3: Wire up main.py**

In `app/main.py`, add `project` to the routers import block (the
`from app.routers import (...)` list, alphabetically near `projects`):

```python
    project,
    projects,
```

In the `lifespan` startup function, after `init_db()`, initialise and share the
session store:

```python
    from app.core.session_store import get_session_store
    from app.routers import project as project_router
    app.state.project_store = get_session_store()
    project_router._store = app.state.project_store
    logger.info("Project session store ready: %s",
                type(app.state.project_store).__name__)
```

In the "Include all routers" block, add the router (next to
`app.include_router(projects.router)`):

```python
app.include_router(project.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_router.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_project_router.py
git commit -m "feat(api): mount project router + init session store at startup"
```

---

### Task 6: Project-chat UI in index.html

**Files:**
- Modify: `app/static/index.html`
- Test: `tests/test_project_ui.py`

`app/static/index.html` is the platform's only frontend (the React app was
deleted). It already has a chat surface (`#messages`, `sendMessage()`, the
`api` client object, `addMessage()`). This task adds a **Project mode**: a
toggle in the header that routes the user's message to `/v1/project/ask`
instead of `/v1/chat`, keyed by a per-tab `projectSessionId`.

Static-HTML changes are verified by asserting the markup/JS is present (the
repo has no JS test runner; mirror how other UI assertions in `tests/` work).

- [ ] **Step 1: Write the failing test**

Create `tests/test_project_ui.py`:

```python
"""Smoke checks for the project-chat UI — Reasoning Engine Plan 6."""

from pathlib import Path

import pytest

_HTML = Path("app/static/index.html").read_text(encoding="utf-8")


def test_ui_has_a_project_mode_toggle():
    assert 'id="projectModeToggle"' in _HTML


def test_ui_has_an_askProject_function():
    assert "function askProject" in _HTML or "askProject =" in _HTML


def test_ui_posts_to_the_project_endpoint():
    assert "/v1/project/ask" in _HTML


def test_ui_generates_a_project_session_id():
    assert "projectSessionId" in _HTML


def test_sendMessage_routes_to_project_mode():
    # sendMessage must branch to askProject when project mode is on.
    assert "askProject" in _HTML and "projectMode" in _HTML
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_ui.py -q`
Expected: FAIL — `test_ui_has_a_project_mode_toggle` fails: `projectModeToggle` not in the HTML.

- [ ] **Step 3: Edit index.html**

**3a. Add the toggle to the header.** In the `<header>` block (around line 127,
after the `agentPicker` `<select>` and before the closing `</header>`), add:

```html
            <label style="display:flex;align-items:center;gap:6px;color:#888;font-size:12px;cursor:pointer;">
                <input type="checkbox" id="projectModeToggle" onchange="onProjectModeChange(this.checked)">
                Project mode
            </label>
```

**3b. Add the project-mode JS.** Inside the `<script>` block, near the other
chat helpers (after `addMessage` / `setOutcomes`, before `sendMessage`), add:

```javascript
// ── Project reasoning mode (Reasoning Engine Plan 6) ──────────────
let projectMode = false;
// Per-tab session id so follow-up questions build on prior state.
const projectSessionId = 'proj-' + Math.random().toString(36).slice(2, 11);

function onProjectModeChange(on) {
    projectMode = on;
    addMessage(
        on
            ? 'Project mode ON — I will reason over your project (CPM, manpower, compression, custom analysis).'
            : 'Project mode OFF — back to standard chat.',
        'system', { persist: false }
    );
}

async function askProject(message) {
    setChatBusy(true);
    showIndicator(true);
    try {
        const resp = await fetch(`${API_BASE}/v1/project/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`,
            },
            body: JSON.stringify({
                session_id: projectSessionId,
                request: message,
            }),
        });
        if (!resp.ok) {
            const err = await readApiError(resp);
            surfaceError(err, 'Project reasoning');
            addMessage(`Project reasoning failed: ${err}`, 'error');
            return;
        }
        const data = await resp.json();
        addMessage(data.answer || '(no answer)', 'assistant');
        if (data.understanding) {
            setOutcomes(
                `<div class="outcome-item">
                   <div class="outcome-label">Understood as</div>
                   <div class="outcome-value" style="font-size:13px;">`
                + (data.understanding) + `</div></div>`);
        }
        if (Array.isArray(data.artifacts)) {
            data.artifacts.forEach(a => addMessage(
                `📎 Artifact: ${a.name} (${a.type})`, 'system',
                { persist: false }));
        }
    } catch (e) {
        surfaceError(String(e), 'Project reasoning');
        addMessage(`Project reasoning failed: ${e}`, 'error');
    } finally {
        setChatBusy(false);
        showIndicator(false);
    }
}
```

**3c. Route `sendMessage` into project mode.** In `sendMessage()`, immediately
after the message text is read and echoed with `addMessage(text, 'user')` and
the input is cleared, add the branch (place it before the existing chat/chain
dispatch):

```javascript
    if (projectMode) {
        await askProject(text);
        return;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_ui.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html tests/test_project_ui.py
git commit -m "feat(ui): project-chat mode wired to /v1/project/ask"
```

---

### Task: Regression check

**Files:** none — verification only.

- [ ] **Step 1: Run the full suite**

Run: `& .venv\Scripts\python.exe -m pytest --ignore=tests/browser -q`
Expected: PASS — all prior tests still pass, plus the new tests from Tasks 1–6
(10 in `test_pm_io.py`, 5 in `test_project_router.py`, 5 in `test_project_ui.py`).

- [ ] **Step 2: Manual UI smoke check (optional but recommended)**

Start the app, open the UI, tick "Project mode", and ask "what is the critical
path?" — confirm the request hits `/v1/project/ask`. With `KIMI_API_KEY`
still unset the reasoner returns a plan-build error; that is expected until the
key is funded. The route, persistence, and UI wiring are fully covered by the
mock-LLM tests above.

- [ ] **Step 3: Commit** — nothing to commit unless a regression was fixed.

---

## Self-Review

**Spec coverage** (Reasoning Engine §6 — API & UI; §7.1 — I/O stays code-level):
- `POST /v1/project/ask` — the user-facing reasoning endpoint → Task 4 ✅
- Router mounted + session store initialised at startup → Task 5 ✅
- Project-chat UI in the (only) frontend, `index.html` → Task 6 ✅
- `parse_xer` — Primavera import (I/O) → Task 1 ✅
- `write_schedule_excel` + `excel_templates.py` — Excel output (I/O) → Tasks 2 & 3 ✅
- Finish-date display fix (subtract one working day) from Plan 1's header → Task 3 ✅

**LLM-key blocker handling:** the router and UI need no key — every router test
injects a scripted mock reasoner via `monkeypatch` on `_reasoner_factory`. The
genuine reasoner round-trip is already covered by Plan 5's
`tests/test_project_reasoner_live.py` (`skipif` on the key); no separate live
test is added here. The manual UI smoke check (regression Step 2) notes the
expected plan-build error while the key is unfunded.

**Dependency check:** uses `ProjectReasonerBlock` (Plan 5), `SessionStore` /
`get_session_store` / `InMemorySessionStore` (Plan 2), `compute_cpm` / CPM
schemas / `_HOURS_PER_DAY` (Plans 1 & 1b), `require_api_key` and the router
pattern (`app/routers/chat.py`), `openpyxl` (`requirements.txt` line 36). All
exist before this plan runs — no new dependency.

**Out of scope (noted):** streaming the reasoner's answer (the chat route
streams; `/v1/project/ask` returns one JSON body — acceptable for v1). Writing
the produced `.xlsx` into `session.artifacts` and serving it for download is a
follow-up — `write_schedule_excel` exists and the response already surfaces
`artifacts`, but no plan step calls it yet; add a `write_excel` step type to
`PlanExecutor` in a later iteration. The 50 MB session-size guard deferred since
Plan 2 should be enforced in `project_ask` around `_store.save(session)` — flagged,
not implemented here to keep Task 4 focused.

**Risk:** the `.xer` format varies by P6 version — `parse_xer` reads the common
`TASK` / `TASKPRED` tables and the `target_drtn_hr_cnt` / `lag_hr_cnt` columns;
calendars, WBS, and resource assignments in the `.xer` are not imported (Plan 1b
resources come from elsewhere). Real-world `.xer` files should be tested once
available — Task 1's fixture is a minimal synthetic file.

**Placeholder scan:** none — every step has complete code or an exact command.
The three `app/main.py` edits and the three `index.html` edits each specify the
exact anchor location.

**Type consistency:** `parse_xer(str) -> List[Activity]`. `write_schedule_excel(CPMOutput, str, Optional[ResourceHistogram]) -> str`. `header_row(Worksheet, int, List[str]) -> None`, `paint_gantt_row(...) -> None`, `write_histogram_block(Worksheet, int, List[Dict]) -> int`. `ProjectAskRequest` is a Pydantic v2 model; `project_ask` returns a dict. Every task has a failing-test step before its implementation.

---

**Plan 6 complete.** This finishes the Reasoning Engine plan set (1, 1b, 2–6).
Update the INDEX status checkboxes after all plans are implemented and green.
