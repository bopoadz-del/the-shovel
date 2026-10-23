# Project Intelligence Reasoning Engine — Plan Index

> **Master index.** The Reasoning Engine spec covers multiple independent
> subsystems, so it is decomposed into **7 plans** (1, 1b, 2–6). Each ships
> working, testable software on its own. Execute them in dependency order.

**Goal:** Insert an AI reasoning layer between the user and The Fork's block
catalogue — understand a request, plan a solution, generate/run code, deliver
the result — so the platform answers *any* project question instead of one
hardcoded pipeline.

**Supersedes:** the earlier "6 hardcoded schedule blocks" approach (the L2
Schedule Engine spec). Per §7 of the Reasoning Engine spec, CPM and friends are
**reusable library functions**, not blocks. Only I/O stays as blocks.

---

## Architecture

```
USER ──▶ LAYER 1: REASONING ENGINE
              UNDERSTAND → PLAN → EXECUTE → DELIVER
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
   TOOL BLOCKS  CODE-GEN   STATE STORE
   (pdf, ocr,   (sandboxed  (session:
   excel I/O)   formula_    activities,
                executor)   cpm, history)
                  │
                  ▼
   app/lib/pm_computations.py  ← tested functions the generated code imports
```

**Hardcode / library / generate decision (spec §7.1):**
- **Block** — I/O only: PDF/OCR extraction, Excel writing, API calls.
- **Library function** (`app/lib/`) — well-defined algorithms: CPM, resource
  histogram, Gantt, calendar math. Tested once, imported by generated code.
- **Generated on the fly** — project-specific / novel logic: WBS decomposition,
  duration estimation, compression, custom queries.

## Prerequisites & blockers

| Need | Status | Affects |
|------|--------|---------|
| LLM API key — `KIMI_API_KEY` in `.env` | ⏳ user has a the configured LLM key, **pending refill** | Plans 4 & 5 get full end-to-end tests once the key is funded; until then they ship with mock-LLM tests |
| `RestrictedPython` package | ❌ not installed — Plan 3 adds it to `requirements.txt` | Plan 3 |
| Redis | ❌ not present — **not required**; Plan 2 uses in-memory backend, Redis adapter optional | Plan 2 (prod only) |

## The plans

| # | Plan | Ships | LLM key? | Depends on |
|---|------|-------|----------|-----------|
| 1 | `pm_computations` — CPM core | `app/lib/pm_computations.py` + `app/schemas/cpm.py`: topological sort, forward/backward pass, float, `compute_cpm` | No | — |
| 1b | `pm_computations` extended (pure compute) | `resource_histogram`, `gantt_data`, `compress_schedule` | No | 1 |
| 2 | Session State Store | `app/core/session_store.py` + `app/schemas/project_session.py`: swappable dict/Redis backend, TTL | No | — |
| 3 | Sandbox | `app/core/sandbox.py`: RestrictedPython exec, import whitelist, state injection, output capture | No | — (adds `RestrictedPython` dep) |
| 4 | `formula_executor_v2` — code generation | `app/blocks/formula_executor_v2.py` + `app/prompts/codegen_system.py`: LLM code-gen → sandbox → cache → retry | **Yes** | 3 |
| 5 | Project Reasoner | `app/blocks/project_reasoner.py`, `app/core/plan_executor.py`, `app/schemas/execution_plan.py`, `app/prompts/reasoner_system.py` | **Yes** | 1, 1b, 2, 4 |
| 6 | API, UI & output | `app/routers/project.py` (`POST /v1/project/ask`) + project-chat UI + `write_schedule_excel`, `app/lib/excel_templates.py`, `parse_xer` (I/O) | runtime only | 5 |

**Dependency DAG:** `1, 2, 3` are independent (do in parallel); `1b` needs `1`
→ `4` needs `3` → `5` needs `1 + 1b + 2 + 4` → `6` needs `5`.

## File map (spec §6, adjusted for this repo)

| File | Plan | Note |
|------|------|------|
| `app/schemas/cpm.py` | 1 | CPM Pydantic models |
| `app/lib/pm_computations.py` | 1, 1b, 6 | CPM (1); resource/gantt/compress (1b); excel/xer I/O (6) |
| `app/lib/excel_templates.py` | 6 | reusable Excel formatting (Gantt bars, histograms) |
| `app/schemas/project_session.py` | 2 | session state models |
| `app/core/session_store.py` | 2 | dict + Redis backends |
| `app/core/sandbox.py` | 3 | RestrictedPython jail |
| `app/blocks/formula_executor_v2.py` | 4 | enhanced code generator |
| `app/prompts/codegen_system.py` | 4 | code-gen system prompt |
| `app/schemas/execution_plan.py` | 5 | `ExecutionPlan` model |
| `app/core/plan_executor.py` | 5 | runs plan steps |
| `app/blocks/project_reasoner.py` | 5 | the LLM agent |
| `app/prompts/reasoner_system.py` | 5 | dynamic system-prompt builder |
| `app/routers/project.py` | 6 | `/v1/project/ask` |
| `app/static/index.html` | 6 | project-chat UI (NOT React — that frontend was deleted) |
| `app/blocks/formula_executor.py` | 4 | MODIFY → deprecation wrapper to v2 |
| `app/blocks/__init__.py` | 4, 5 | MODIFY → register new blocks |
| `app/main.py` | 6 | MODIFY → mount router, init session store |
| `requirements.txt` | 3 | MODIFY → add `RestrictedPython` |

## Status

Legend: doc = plan document written · impl = implemented + tests passing.

| Plan | Doc | Impl |
|------|-----|------|
| 1 — pm_computations CPM core | ✅ | ✅ |
| 1b — pm_computations extended | ✅ | ✅ |
| 2 — session state store | ✅ | ✅ |
| 3 — sandbox | ✅ | ✅ |
| 4 — formula_executor v2 | ✅ | ✅ |
| 5 — project reasoner | ✅ | ✅ |
| 6 — API & UI | ✅ | ✅ |

All 7 plans implemented. Reasoning Engine suite: 128 passed, 1 skipped
(Redis + live-LLM tests skip without `REDIS_URL` / `KIMI_API_KEY`).

Execute Plan N only after Plan N-1's tests pass.
