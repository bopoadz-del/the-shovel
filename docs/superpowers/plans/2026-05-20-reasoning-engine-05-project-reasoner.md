# Reasoning Engine — Plan 5: Project Reasoner (the LLM agent)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.
> Part of the Reasoning Engine — see `2026-05-20-reasoning-engine-INDEX.md`. **Depends on Plans 1, 1b, 2, 4.**

**Goal:** The reasoning layer itself — the agent that turns a free-text project question into an answer. It runs the four-phase loop from the spec: **UNDERSTAND** the request, **PLAN** a sequence of steps, **EXECUTE** those steps against the session store and the `app/lib` functions / the Plan-4 code generator, and **DELIVER** a written answer with artifacts.

**Architecture:**
- `app/schemas/execution_plan.py` — `PlanStep` and `ExecutionPlan` Pydantic models. A plan is an ordered list of typed steps the executor can run.
- `app/core/plan_executor.py` — `PlanExecutor`: takes an `ExecutionPlan` + a `ProjectSession`, runs each step, writes results into `session.data`, returns a `PlanRunResult`. No AI — pure dispatch over a step-type registry.
- `app/prompts/reasoner_system.py` — `build_reasoner_prompt(session, request)`: a pure function that assembles the dynamic system prompt, injecting what the session already knows (activities loaded? CPM computed? artifacts?) so follow-up questions build on prior state.
- `app/blocks/project_reasoner.py` — `ProjectReasonerBlock`, a `UniversalBlock`. Orchestrates UNDERSTAND→PLAN→EXECUTE→DELIVER. The two LLM calls (plan, deliver) are isolated behind `_call_llm` so mock-LLM tests subclass it; live tests hit the configured LLM.

**Step types (v1):**
- `compute_cpm` — run `compute_cpm` over `session.data["activities"]`, store `cpm_results`.
- `resource_histogram` — run `resource_histogram`, store `manpower`.
- `gantt` — run `gantt_data`, store `gantt`.
- `compress` — run `compress_schedule` with a `reductions` arg, store `compressed`.
- `generate_code` — delegate to `FormulaExecutorV2Block` (Plan 4) for novel logic; store the result under the step's `output_key`.

**LLM provider:** the configured LLM (same shape as Plan 4 / `ChatBlock`). `KIMI_API_KEY` is **pending refill** — this plan ships **mock-LLM tests only**; the live test is written and `skipif` on the key (Task 6).

**Tech Stack:** Python 3.11, Pydantic v2, `httpx`. No new dependencies.

**Run tests:** `& .venv\Scripts\python.exe -m pytest <path> -q` from `C:\Users\example\The_Fork`. **Plans 1, 1b, 2 and 4 must be complete first.**

---

### Task 1: ExecutionPlan schemas

**Files:**
- Create: `app/schemas/execution_plan.py`
- Test: `tests/test_project_reasoner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_project_reasoner.py`:

```python
"""Tests for the Project Reasoner — Reasoning Engine Plan 5."""

import pytest

from app.schemas.execution_plan import ExecutionPlan, PlanStep


def test_plan_step_defaults():
    s = PlanStep(type="compute_cpm")
    assert s.args == {} and s.output_key == ""


def test_execution_plan_construct():
    plan = ExecutionPlan(
        understanding="user wants the critical path",
        steps=[PlanStep(type="compute_cpm")],
    )
    assert len(plan.steps) == 1
    assert plan.understanding


def test_execution_plan_parses_from_llm_json():
    # The reasoner's PLAN call returns JSON — it must validate straight in.
    raw = {
        "understanding": "compute then compress",
        "steps": [
            {"type": "compute_cpm"},
            {"type": "compress", "args": {"reductions": {"B": 3}}},
        ],
    }
    plan = ExecutionPlan.model_validate(raw)
    assert plan.steps[1].type == "compress"
    assert plan.steps[1].args["reductions"] == {"B": 3}


def test_empty_step_type_rejected():
    with pytest.raises(Exception):
        PlanStep(type="")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas.execution_plan'`

- [ ] **Step 3: Write the schemas**

Create `app/schemas/execution_plan.py`:

```python
"""Execution-plan schemas — Reasoning Engine Plan 5.

An ExecutionPlan is the reasoner's PLAN-phase output: an ordered list of
typed steps the PlanExecutor knows how to run.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One executable step.

    `type` selects a handler in PlanExecutor (compute_cpm, resource_histogram,
    gantt, compress, generate_code). `args` carries step-specific parameters.
    `output_key` names where the result lands in session.data — defaults to a
    handler-specific key when blank.
    """
    type: str = Field(min_length=1)
    args: Dict[str, Any] = Field(default_factory=dict)
    output_key: str = ""
    description: str = ""


class ExecutionPlan(BaseModel):
    """The reasoner's plan for one user request."""
    understanding: str = ""        # the UNDERSTAND-phase restatement
    steps: List[PlanStep] = Field(default_factory=list)


class StepResult(BaseModel):
    """Outcome of running one PlanStep."""
    type: str
    output_key: str
    status: str                    # 'success' | 'error'
    error: str = ""


class PlanRunResult(BaseModel):
    """Outcome of running a whole ExecutionPlan."""
    status: str                    # 'success' | 'partial' | 'error'
    step_results: List[StepResult] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/execution_plan.py tests/test_project_reasoner.py
git commit -m "feat(reasoner): ExecutionPlan schemas (reasoning engine plan 5)"
```

---

### Task 2: PlanExecutor — library step types

**Files:**
- Create: `app/core/plan_executor.py`
- Test: `tests/test_project_reasoner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_reasoner.py`:

```python
from app.core.plan_executor import PlanExecutor
from app.core.session_store import InMemorySessionStore


_ACTIVITIES = [
    {"id": "A", "duration": 3, "predecessors": []},
    {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
    {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
]


def _session_with_activities():
    s = InMemorySessionStore().get_or_create("s1")
    s.data["activities"] = _ACTIVITIES
    return s


@pytest.mark.asyncio
async def test_executor_runs_compute_cpm_step():
    session = _session_with_activities()
    plan = ExecutionPlan(steps=[PlanStep(type="compute_cpm")])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "success"
    assert session.data["cpm_results"]["project_duration"] == 10


@pytest.mark.asyncio
async def test_executor_runs_compress_step():
    session = _session_with_activities()
    plan = ExecutionPlan(steps=[
        PlanStep(type="compute_cpm"),
        PlanStep(type="compress", args={"reductions": {"B": 3}}),
    ])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "success"
    assert session.data["compressed"]["revised"]["project_duration"] == 7
    assert session.data["compressed"]["days_saved"] == 3


@pytest.mark.asyncio
async def test_executor_runs_gantt_and_histogram_steps():
    session = _session_with_activities()
    plan = ExecutionPlan(steps=[
        PlanStep(type="compute_cpm"),
        PlanStep(type="gantt"),
        PlanStep(type="resource_histogram", args={"period_unit": "week"}),
    ])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "success"
    assert len(session.data["gantt"]) == 3
    assert session.data["manpower"]["period_unit"] == "week"


@pytest.mark.asyncio
async def test_executor_reports_unknown_step_type():
    session = _session_with_activities()
    plan = ExecutionPlan(steps=[PlanStep(type="teleport")])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "error"
    assert "teleport" in result.step_results[0].error


@pytest.mark.asyncio
async def test_executor_compute_cpm_without_activities_errors():
    session = InMemorySessionStore().get_or_create("s1")  # no activities
    plan = ExecutionPlan(steps=[PlanStep(type="compute_cpm")])
    result = await PlanExecutor().run(plan, session)
    assert result.status == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.plan_executor'`

- [ ] **Step 3: Write the executor**

Create `app/core/plan_executor.py`:

```python
"""Plan executor — Reasoning Engine Plan 5.

Runs an ExecutionPlan's steps against a ProjectSession. No AI: each step type
maps to a handler that calls a tested app/lib function (Plans 1/1b) or the
Plan-4 code generator, then writes the result into session.data.
"""

from typing import Any, Dict

from app.schemas.cpm import CPMInput
from app.schemas.execution_plan import (
    ExecutionPlan, PlanRunResult, PlanStep, StepResult,
)
from app.schemas.project_session import ProjectSession
from app.lib.pm_computations import (
    compute_cpm, resource_histogram, gantt_data, compress_schedule,
)


class PlanExecutionError(Exception):
    """A step could not run (bad args, missing state)."""


def _require_cpm_input(session: ProjectSession) -> CPMInput:
    activities = session.data.get("activities")
    if not activities:
        raise PlanExecutionError(
            "No activities in the session — load a schedule first."
        )
    return CPMInput.model_validate({"activities": activities})


class PlanExecutor:
    """Dispatches ExecutionPlan steps. Stateless — safe to reuse."""

    async def run(
        self, plan: ExecutionPlan, session: ProjectSession
    ) -> PlanRunResult:
        results = []
        any_error = False
        for step in plan.steps:
            try:
                await self._run_step(step, session)
                results.append(StepResult(
                    type=step.type,
                    output_key=step.output_key,
                    status="success",
                ))
            except Exception as e:                       # noqa: BLE001
                any_error = True
                results.append(StepResult(
                    type=step.type,
                    output_key=step.output_key,
                    status="error",
                    error=str(e),
                ))
        if not any_error:
            status = "success"
        elif any(r.status == "success" for r in results):
            status = "partial"
        else:
            status = "error"
        return PlanRunResult(status=status, step_results=results)

    async def _run_step(self, step: PlanStep, session: ProjectSession) -> None:
        handler = getattr(self, f"_step_{step.type}", None)
        if handler is None:
            raise PlanExecutionError(f"Unknown step type: '{step.type}'")
        await handler(step, session)

    # ── library step handlers ────────────────────────────────────────────
    async def _step_compute_cpm(self, step, session):
        out = compute_cpm(_require_cpm_input(session))
        session.data[step.output_key or "cpm_results"] = out.model_dump(mode="json")

    async def _step_resource_histogram(self, step, session):
        out = compute_cpm(_require_cpm_input(session))
        hist = resource_histogram(
            out.results, _require_cpm_input(session).activities,
            period_unit=step.args.get("period_unit", "week"),
        )
        session.data[step.output_key or "manpower"] = hist.model_dump(mode="json")

    async def _step_gantt(self, step, session):
        out = compute_cpm(_require_cpm_input(session))
        bars = gantt_data(out.results)
        session.data[step.output_key or "gantt"] = [
            b.model_dump(mode="json") for b in bars
        ]

    async def _step_compress(self, step, session):
        reductions = step.args.get("reductions")
        if not isinstance(reductions, dict):
            raise PlanExecutionError(
                "compress step needs an args.reductions dict"
            )
        reductions = {k: int(v) for k, v in reductions.items()}
        revised, days_saved = compress_schedule(
            _require_cpm_input(session), reductions
        )
        session.data[step.output_key or "compressed"] = {
            "revised": revised.model_dump(mode="json"),
            "days_saved": days_saved,
        }
```

> `_step_generate_code` is added in Task 3 — keep the failing-test discipline.

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add app/core/plan_executor.py tests/test_project_reasoner.py
git commit -m "feat(reasoner): PlanExecutor with CPM/histogram/gantt/compress steps"
```

---

### Task 3: PlanExecutor — generate_code step (delegates to Plan 4)

**Files:**
- Modify: `app/core/plan_executor.py` (add `_step_generate_code`)
- Test: `tests/test_project_reasoner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_reasoner.py`:

```python
from app.blocks.formula_executor_v2 import FormulaExecutorV2Block


class _MockCodeGen(FormulaExecutorV2Block):
    """Code-gen double — returns canned code, no the configured LLM call."""
    async def _call_llm(self, prompt):
        return "result = a + b"


@pytest.mark.asyncio
async def test_executor_runs_generate_code_step():
    session = InMemorySessionStore().get_or_create("s1")
    plan = ExecutionPlan(steps=[PlanStep(
        type="generate_code",
        args={"task": "add a and b", "variables": {"a": 4, "b": 6}},
        output_key="sum",
    )])
    executor = PlanExecutor(code_block=_MockCodeGen([]))
    result = await executor.run(plan, session)
    assert result.status == "success"
    assert session.data["sum"]["result"] == 10


@pytest.mark.asyncio
async def test_generate_code_step_requires_a_task():
    session = InMemorySessionStore().get_or_create("s1")
    plan = ExecutionPlan(steps=[PlanStep(type="generate_code", args={})])
    result = await PlanExecutor(code_block=_MockCodeGen([])).run(plan, session)
    assert result.status == "error"
```

> `_MockCodeGen` reuses Plan 4's `_MockLLMBlock` pattern. Its `__init__`
> accepts a (here empty) scripted list — see `tests/test_formula_executor_v2.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: FAIL — `TypeError: PlanExecutor() takes no arguments` (no `code_block` param yet).

- [ ] **Step 3: Add the generate_code step**

In `app/core/plan_executor.py`, give `PlanExecutor` a constructor that holds an
optional code-gen block, and add the handler. Add to the top of the class:

```python
    def __init__(self, code_block=None):
        """`code_block` is a FormulaExecutorV2Block (Plan 4). Constructed
        lazily on first use when not injected, so library-only plans need no
        LLM block."""
        self._code_block = code_block

    def _get_code_block(self):
        if self._code_block is None:
            from app.blocks.formula_executor_v2 import FormulaExecutorV2Block
            self._code_block = FormulaExecutorV2Block()
        return self._code_block
```

Add the handler alongside the other `_step_*` methods:

```python
    async def _step_generate_code(self, step, session):
        task = step.args.get("task")
        if not task:
            raise PlanExecutionError("generate_code step needs an args.task")
        variables = dict(step.args.get("variables") or {})
        # let generated code read prior session state when asked
        for key in step.args.get("from_session", []):
            if key in session.data:
                variables[key] = session.data[key]
        out = await self._get_code_block().process({
            "task": task, "variables": variables, "session": session,
        })
        if out.get("status") != "success":
            raise PlanExecutionError(
                out.get("error", "code generation failed")
            )
        session.data[step.output_key or "generated"] = out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add app/core/plan_executor.py tests/test_project_reasoner.py
git commit -m "feat(reasoner): generate_code step delegating to formula_executor_v2"
```

---

### Task 4: Dynamic reasoner system prompt

**Files:**
- Create: `app/prompts/reasoner_system.py`
- Test: `tests/test_project_reasoner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_reasoner.py`:

```python
from app.prompts.reasoner_system import build_reasoner_prompt


def test_reasoner_prompt_lists_the_phases():
    p = build_reasoner_prompt(InMemorySessionStore().get_or_create("s"), "hi")
    for phase in ("UNDERSTAND", "PLAN", "EXECUTE", "DELIVER"):
        assert phase in p


def test_reasoner_prompt_advertises_step_types():
    p = build_reasoner_prompt(InMemorySessionStore().get_or_create("s"), "hi")
    for t in ("compute_cpm", "resource_histogram", "gantt", "compress",
              "generate_code"):
        assert t in p


def test_reasoner_prompt_reflects_empty_session():
    p = build_reasoner_prompt(InMemorySessionStore().get_or_create("s"), "hi")
    assert "no activities" in p.lower() or "not loaded" in p.lower()


def test_reasoner_prompt_reflects_loaded_state():
    s = _session_with_activities()
    s.data["cpm_results"] = {"project_duration": 10}
    p = build_reasoner_prompt(s, "now compress B")
    # the prompt must tell the LLM CPM is already done, so it skips re-running
    assert "cpm" in p.lower()
    assert "now compress B" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.prompts.reasoner_system'`

- [ ] **Step 3: Write the prompt builder**

Create `app/prompts/reasoner_system.py`:

```python
"""Dynamic reasoner system prompt — Reasoning Engine Plan 5.

Pure string building. The prompt is rebuilt every turn so it reflects what the
session already knows — a follow-up question then builds on prior state.
"""

from app.schemas.project_session import ProjectSession

_PHASES = """\
You are the Project Reasoner for a construction-project intelligence platform.
Work in four phases:
  UNDERSTAND — restate what the user is really asking for.
  PLAN       — produce an ordered list of steps that answers it.
  EXECUTE    — (the platform runs your plan; you do not run it yourself).
  DELIVER    — write a clear answer from the executed results.

When asked to PLAN, reply with ONLY a JSON object:
  {"understanding": "...", "steps": [{"type": "...", "args": {...},
   "output_key": "...", "description": "..."}]}

AVAILABLE STEP TYPES:
  - compute_cpm          run the Critical Path Method over loaded activities.
  - resource_histogram   time-phased manpower; args: {"period_unit": "week"|"month"}.
  - gantt                Gantt bars for the loaded schedule.
  - compress             shorten the schedule; args: {"reductions": {"<id>": <days>}}.
  - generate_code        novel/custom logic; args: {"task": "...",
                         "variables": {...}, "from_session": ["<key>", ...]}.
"""


def _state_summary(session: ProjectSession) -> str:
    d = session.data
    lines = []
    activities = d.get("activities")
    if activities:
        lines.append(f"- activities: {len(activities)} loaded.")
    else:
        lines.append("- activities: NOT loaded (no schedule yet).")
    for key, label in (
        ("cpm_results", "CPM already computed"),
        ("manpower", "resource histogram already computed"),
        ("gantt", "Gantt data already computed"),
        ("compressed", "a compressed schedule already exists"),
    ):
        if key in d:
            lines.append(f"- {label} (session key '{key}') — reuse it, "
                         f"do not recompute unless the user changed inputs.")
    if session.artifacts:
        lines.append(f"- {len(session.artifacts)} artifact(s) already produced.")
    return "\n".join(lines)


def build_reasoner_prompt(session: ProjectSession, request: str) -> str:
    """Assemble the reasoner system prompt for the current turn."""
    return (
        f"{_PHASES}\n\n"
        f"CURRENT SESSION STATE:\n{_state_summary(session)}\n\n"
        f"USER REQUEST:\n{request}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add app/prompts/reasoner_system.py tests/test_project_reasoner.py
git commit -m "feat(reasoner): dynamic system-prompt builder reflecting session state"
```

---

### Task 5: ProjectReasonerBlock — the UNDERSTAND→PLAN→EXECUTE→DELIVER loop

**Files:**
- Create: `app/blocks/project_reasoner.py`
- Modify: `app/blocks/__init__.py` (register `project_reasoner`)
- Test: `tests/test_project_reasoner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_reasoner.py`:

```python
import json

from app.blocks.project_reasoner import ProjectReasonerBlock
from app.blocks import BLOCK_REGISTRY, get_block


class _MockReasoner(ProjectReasonerBlock):
    """Reasoner double — scripts the PLAN call's JSON and the DELIVER call's
    answer. `_call_llm` is invoked twice per turn: first PLAN, then DELIVER."""

    def __init__(self, plan_json, answer, **kw):
        super().__init__(**kw)
        self._plan_json = plan_json
        self._answer = answer
        self.calls = 0

    async def _call_llm(self, prompt: str) -> str:
        self.calls += 1
        return self._plan_json if self.calls == 1 else self._answer


@pytest.mark.asyncio
async def test_reasoner_runs_full_loop():
    session = _session_with_activities()
    plan_json = json.dumps({
        "understanding": "user wants the critical path",
        "steps": [{"type": "compute_cpm"}],
    })
    block = _MockReasoner(plan_json, "The critical path is A-B-C, 10 days.")
    out = await block.process({"request": "what is the critical path?",
                               "session": session})
    assert out["status"] == "success"
    assert out["answer"] == "The critical path is A-B-C, 10 days."
    assert out["understanding"] == "user wants the critical path"
    assert session.data["cpm_results"]["project_duration"] == 10
    assert block.calls == 2          # one PLAN call, one DELIVER call


@pytest.mark.asyncio
async def test_reasoner_records_turn_in_history():
    session = _session_with_activities()
    plan_json = json.dumps({"understanding": "x",
                            "steps": [{"type": "compute_cpm"}]})
    block = _MockReasoner(plan_json, "done")
    await block.process({"request": "go", "session": session})
    roles = [m.role for m in session.history]
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_reasoner_handles_bad_plan_json():
    session = _session_with_activities()
    block = _MockReasoner("not json at all", "unused")
    out = await block.process({"request": "go", "session": session})
    assert out["status"] == "error"
    assert "plan" in out["error"].lower()


@pytest.mark.asyncio
async def test_reasoner_reports_step_failure():
    session = InMemorySessionStore().get_or_create("s1")   # no activities
    plan_json = json.dumps({"understanding": "x",
                            "steps": [{"type": "compute_cpm"}]})
    block = _MockReasoner(plan_json, "unused")
    out = await block.process({"request": "go", "session": session})
    assert out["status"] in ("error", "partial")


def test_reasoner_is_registered():
    assert "project_reasoner" in BLOCK_REGISTRY
    assert get_block("project_reasoner") is ProjectReasonerBlock
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.blocks.project_reasoner'`

- [ ] **Step 3: Write the reasoner block**

Create `app/blocks/project_reasoner.py`:

```python
"""Project Reasoner — Reasoning Engine Plan 5.

The LLM agent. One turn: UNDERSTAND + PLAN (one LLM call returning plan JSON)
-> EXECUTE (PlanExecutor runs the steps) -> DELIVER (one LLM call writing the
answer from the executed results).
"""

import json
import os
from typing import Any, Dict

import httpx

from app.core.universal_base import UniversalBlock
from app.core.plan_executor import PlanExecutor
from app.prompts.reasoner_system import build_reasoner_prompt
from app.schemas.execution_plan import ExecutionPlan
from app.schemas.project_session import ProjectSession


def _extract_json(text: str) -> dict:
    """Pull the first {...} object out of an LLM reply (it may add prose or
    fences). Raises ValueError when there is no parsable object."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in LLM reply")
    return json.loads(text[start:end + 1])


class ProjectReasonerBlock(UniversalBlock):
    name = "project_reasoner"
    version = "1.0.0"
    description = (
        "Reasoning agent: UNDERSTAND -> PLAN -> EXECUTE -> DELIVER over a "
        "project session."
    )
    layer = 3
    tags = ["domain", "construction", "reasoning", "agent", "llm"]
    requires = []

    default_config = {"model": "kimi-k2.6"}

    ui_schema = {
        "input": {
            "type": "text",
            "placeholder": "Ask anything about your project...",
            "multiline": True,
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "answer", "type": "markdown", "label": "Answer"},
                {"name": "understanding", "type": "text", "label": "Understood as"},
            ],
        },
        "quick_actions": [
            {"icon": "🧭", "label": "Critical path", "prompt": "What is the critical path?"},
            {"icon": "⏱️", "label": "Compress", "prompt": "How can I finish 2 weeks sooner?"},
        ],
    }

    async def _call_llm(self, prompt: str) -> str:
        """the configured LLM call. Overridden by test doubles."""
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            raise RuntimeError("KIMI_API_KEY not configured")
        model = self.config.get("model", "kimi-k2.6")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.moonshot.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.2},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"the configured LLM API error (HTTP {resp.status_code})"
                )
            return resp.json()["choices"][0]["message"]["content"]

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}
        request = data.get("request") or data.get("text") \
            or params.get("request") \
            or (str(input_data) if not isinstance(input_data, dict) else "")
        session: ProjectSession = data.get("session") or params.get("session")

        if not request.strip():
            return {"status": "error", "error": "No request provided"}
        if session is None:
            return {"status": "error", "error": "No session provided"}

        session.add_message("user", request)

        # ── UNDERSTAND + PLAN ────────────────────────────────────────────
        try:
            plan_reply = await self._call_llm(
                build_reasoner_prompt(session, request)
            )
            plan = ExecutionPlan.model_validate(_extract_json(plan_reply))
        except Exception as e:                              # noqa: BLE001
            return {"status": "error",
                    "error": f"Could not build a plan: {e}"}

        # ── EXECUTE ──────────────────────────────────────────────────────
        run = await PlanExecutor().run(plan, session)

        # ── DELIVER ──────────────────────────────────────────────────────
        deliver_prompt = (
            f"You planned and executed steps for this request:\n{request}\n\n"
            f"UNDERSTANDING: {plan.understanding}\n"
            f"EXECUTION STATUS: {run.status}\n"
            f"RESULTS (session data):\n"
            f"{json.dumps(session.data, default=str)[:6000]}\n\n"
            f"Write a clear, concise answer for the user from these results. "
            f"If the status is error or partial, explain what is missing."
        )
        try:
            answer = await self._call_llm(deliver_prompt)
        except Exception as e:                              # noqa: BLE001
            answer = f"(Could not generate the written answer: {e})"

        session.add_message("assistant", answer)

        status = "success" if run.status == "success" else run.status
        return {
            "status": status,
            "answer": answer,
            "understanding": plan.understanding,
            "plan": plan.model_dump(mode="json"),
            "execution": run.model_dump(mode="json"),
        }
```

In `app/blocks/__init__.py`, add the import after the Plan-4 v2 import:

```python
from .project_reasoner import ProjectReasonerBlock
```

And add the registry entry inside `BLOCK_REGISTRY`, after the
`"formula_executor_v2": FormulaExecutorV2Block,` line:

```python
    "project_reasoner":     ProjectReasonerBlock,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner.py -q`
Expected: PASS (20 passed)

- [ ] **Step 5: Commit**

```bash
git add app/blocks/project_reasoner.py app/blocks/__init__.py tests/test_project_reasoner.py
git commit -m "feat(reasoner): ProjectReasonerBlock — UNDERSTAND/PLAN/EXECUTE/DELIVER loop"
```

---

### Task 6: Live end-to-end test (skipped until the key is funded)

**Files:**
- Test: `tests/test_project_reasoner_live.py`

- [ ] **Step 1: Write the live test**

Create `tests/test_project_reasoner_live.py`:

```python
"""LIVE the configured LLM end-to-end test — Reasoning Engine Plan 5.

Skipped until KIMI_API_KEY is funded. Acceptance check for the real
reasoning loop. Mock-LLM coverage is in tests/test_project_reasoner.py.
"""

import os

import pytest

from app.blocks.project_reasoner import ProjectReasonerBlock
from app.core.session_store import InMemorySessionStore

pytestmark = pytest.mark.skipif(
    not os.getenv("KIMI_API_KEY"),
    reason="KIMI_API_KEY not configured — pending refill",
)


@pytest.mark.asyncio
async def test_live_reasoner_answers_critical_path_question():
    session = InMemorySessionStore().get_or_create("live1")
    session.data["activities"] = [
        {"id": "A", "duration": 3, "predecessors": []},
        {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
        {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
    ]
    block = ProjectReasonerBlock()
    out = await block.process({
        "request": "What is the project duration and the critical path?",
        "session": session,
    })
    assert out["status"] == "success"
    assert "10" in out["answer"]


@pytest.mark.asyncio
async def test_live_reasoner_follow_up_uses_prior_state():
    session = InMemorySessionStore().get_or_create("live2")
    session.data["activities"] = [
        {"id": "A", "duration": 3, "predecessors": []},
        {"id": "B", "duration": 5, "predecessors": [{"predecessor_id": "A"}]},
        {"id": "C", "duration": 2, "predecessors": [{"predecessor_id": "B"}]},
    ]
    block = ProjectReasonerBlock()
    await block.process({"request": "Compute the critical path.",
                         "session": session})
    out = await block.process({
        "request": "Now shorten B by 3 days — what is the new duration?",
        "session": session,
    })
    assert out["status"] == "success"
    assert "7" in out["answer"]
```

- [ ] **Step 2: Run the test**

Run: `& .venv\Scripts\python.exe -m pytest tests/test_project_reasoner_live.py -q`
Expected (key unset): `2 skipped`. Expected (key funded): `2 passed`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_project_reasoner_live.py
git commit -m "test(reasoner): live the configured LLM e2e test (skipped until key funded)"
```

---

### Task: Regression check

**Files:** none — verification only.

- [ ] **Step 1: Run the full suite**

Run: `& .venv\Scripts\python.exe -m pytest --ignore=tests/browser -q`
Expected: PASS — all prior tests still pass, plus 20 new from this plan and 2
skipped from the live test.

- [ ] **Step 2: Commit** — nothing to commit unless a regression was fixed.

---

## Self-Review

**Spec coverage** (Reasoning Engine §4 — the four-phase loop; §5.1 — the reasoner):
- UNDERSTAND + PLAN as one LLM call returning an `ExecutionPlan` → Tasks 1 & 5 ✅
- EXECUTE via `PlanExecutor` over the session + `app/lib` functions + Plan-4 code-gen → Tasks 2, 3 & 5 ✅
- DELIVER as a second LLM call writing the answer from executed results → Task 5 ✅
- Dynamic system prompt reflecting prior session state (follow-ups build on it) → Task 4 ✅
- Conversation history recorded on the session → Task 5 (`add_message`) ✅
- Reasoner registered as a block → Task 5 ✅

**LLM-key blocker handling:** the two LLM calls per turn go through the single
`_call_llm` seam — `_MockReasoner` scripts both. The live round-trip is
`tests/test_project_reasoner_live.py`, `skipif` on the key; it runs unchanged
once the key is funded.

**Dependency check:** uses `compute_cpm` / `resource_histogram` / `gantt_data` /
`compress_schedule` (Plans 1 + 1b), `CPMInput` / schemas (Plan 1), `ProjectSession` /
`InMemorySessionStore` (Plan 2), `FormulaExecutorV2Block` (Plan 4). All exist
before this plan runs.

**Out of scope (noted):** persisting the session back to the store after a turn
is the API layer's job — Plan 6's `/v1/project/ask` calls `store.save(session)`.
Artifact creation (Excel files) is also Plan 6. The 50 MB session-size guard
deferred from Plan 2 is still deferred — enforce in Plan 6 around `store.save`.
Streaming the answer is not done here; Plan 6 may add it.

**Placeholder scan:** none — every step has complete code or an exact command.

**Type consistency:** `ExecutionPlan` / `PlanStep` / `StepResult` / `PlanRunResult`
are Pydantic v2 models. `PlanExecutor.run(ExecutionPlan, ProjectSession) -> PlanRunResult`.
`build_reasoner_prompt(ProjectSession, str) -> str`. `ProjectReasonerBlock.process(Any, Dict) -> Dict`;
`_call_llm(str) -> str` is the only AI seam. Every task has a failing-test step
before its implementation.

---

**Plan 5 complete.** Next: Plan 6 (API, UI & output).
