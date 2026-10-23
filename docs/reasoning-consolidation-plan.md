# Reasoning & Orchestration Consolidation Plan

Status: SUPERSEDED (2026-07-18). This was a PROPOSAL (paper only — no code
changes), author-reviewed 2026-07-02. The reasoning/orchestration consolidation
that actually shipped is the predefined-orchestrator path (`ORCHESTRATOR_PREDEFINED`,
live=1 on prod) + the discipline-hats layer (`FORK_HATS_ENABLED`, default-off),
not the design below. Retained for history.

## Decisions locked (operator, 2026-07-02)

- **Everything through the orchestrator.** The orchestrator is the SINGLE front
  door — no `agents.py` bypass. Every request passes its decision, which
  chooses predefined reasoning vs dynamic agent vs fast chat. The agent stays,
  as one arm it dispatches to (long tail). This is the TARGET end-state, reached
  via the phased pilot below — not a big-bang rewrite of the live path.
- **`project_reasoner` becomes PART OF the orchestrator block** (decision 2).
  Not a separate module, not a sibling arm — its PLAN/EXECUTE/DELIVER logic is
  absorbed INTO the orchestrator block itself. The orchestrator block is the
  single brain: it decides, per request, whether to run its own predefined
  reasoning or to create/dispatch a dynamic agent. The standalone
  `project_reasoner` block is dissolved into the orchestrator.
- **API calls** (decision 4): steps and materialization execute as API calls —
  the universal `/v1/execute` and the render endpoints. Plan steps CALL blocks/
  endpoints; rendering is NOT inlined into steps. Keeps the single-endpoint
  design and lets the same endpoints serve direct API use.
- DELIVER form signal (decision 3): default = intent classification, with an
  explicit user override ("just give me the file"). Assumed unless changed.

## 1. The problem: two reasoning engines, one of them mis-layered

The platform grew two different "reasoning" implementations, and the live chat
path picked one and stranded the other.

- **Heavy-reasoning tool-agent** — a tool-calling loop in `app/agents/runtime.py`.
  A generative chat turn reaches it: `/v1/chat/stream` -> `_classify_intent`
  -> (generative + confidence) -> `get_agent("heavy-reasoning")` ->
  `agent.chat()`. `generate_wbs` runs here as a synthetic tool. **This is live.**
- **`project_reasoner`** — a structured UNDERSTAND -> PLAN -> EXECUTE
  (`PlanExecutor`) -> DELIVER engine (commit `2d3fecc`, "Plan 5"). It is:
  - packaged as a peer block (`ProjectReasonerBlock(UniversalBlock)`, `name`,
    `ui_schema`, `process()`), yet
  - **not registered** in `app/blocks/__init__.py` / `block_registry.py`, and
  - reachable only from the `/v1/project` route (`app/routers/project.py`),
    which the chat UI does not call. **This is dormant for chat.**

Two smells:
1. **An orchestrator wearing a block costume.** A leaf block does one thing
   (parse a BOQ, OCR a page). `project_reasoner` *plans and dispatches other
   blocks* via `PlanExecutor`. That is orchestration, sitting in the block
   layer as an unregistered peer of the blocks it should command.
2. **The decision lives in the wrong place.** Whether a turn produces a
   document or answers inline is currently decided by "did `generate_wbs`
   happen to run" (the export-offer heuristic in `_build_exports_from_audit`),
   not by the user's intent.

## 2. Target: one orchestration layer, two arms

```
                          user message
                               |  (single front door — no bypass)
   +===================================================================+
   |                     ORCHESTRATOR BLOCK                            |
   |                                                                   |
   |   classify intent + confidence                                   |
   |            |                                                      |
   |            v         decide                                      |
   |   +-------------------------------+                              |
   |   | PREDEFINED REASONING          |  <- project_reasoner's       |
   |   | UNDERSTAND->PLAN->EXECUTE->   |     PLAN/EXECUTE/DELIVER,     |
   |   | DELIVER  (known workflows)    |     absorbed INTO this block  |
   |   +-------------------------------+                              |
   |            OR                                                     |
   |   +-------------------------------+                              |
   |   | CREATE / DISPATCH AGENT       |  <- tool loop, for the       |
   |   | (dynamic, open-ended)         |     long tail                |
   |   +-------------------------------+                              |
   |            OR  fast single-LLM answer (small talk / Q&A)          |
   +===================================================================+
                               |  issues API calls (/v1/execute + endpoints)
                               v
        +-----------------------------------------+
        |  STEPS  (compute, no orchestration)          |
        |  extract | build-WBS | cost-load | render    |
        +-----------------------------------------+
                               |
        +-----------------------------------------+
        |  LEAF BLOCKS (single responsibility)         |
        |  document_engine | boq_processor | ocr | pdf |
        +-----------------------------------------+
                               |
                     stage in ProjectSession
                               |
                     DELIVER decides FORM:
                  inline answer | document | link
```

The reasoner is **absorbed into the orchestrator block** — it is not a peer
block and not a separate module. The orchestrator block IS the brain: it holds
the predefined reasoning and decides, per request, whether to run it or to
create an agent. Everything flows through this one block.

## 3. The orchestrator block's decision (all inside one block)

The orchestrator block does classify + decide + reason/dispatch itself. It
absorbs today's `smart_orchestrator` classifier AND `project_reasoner`'s
PLAN/EXECUTE/DELIVER, so the decision and the predefined reasoning live in the
SAME block — not a router handing off to sibling components.

Inside the block: classify intent + confidence -> one of three:
1. **Known workflow** (action maps to a registered Plan template, high
   confidence) -> run the block's OWN **predefined reasoning**
   (PLAN/EXECUTE/DELIVER). Deterministic, auditable, cheap (no LLM planning
   round-trip), low hallucination.
2. **Generative but no matching workflow** -> the block **creates/dispatches a
   dynamic agent** (today's tool loop) for the long tail.
3. **Q&A / small talk** -> fast single-LLM answer.

`GENERATIVE_INTENTS` already contains the workflow intents (e.g. "produce
schedule"); the block adds a **workflow registry**: `intent action -> Plan
template`. `project_reasoner` is dissolved — its logic is now the block's
predefined-reasoning path; the standalone block and its `/v1/project`-only
wiring go away.

## 4. A predefined workflow is a Plan (data, not hardcoded orchestration)

A Plan template is an ordered list of typed steps + a DELIVER spec. Steps run
via `PlanExecutor` against a `ProjectSession` (the "staging area" — this is the
"sandbox" concept, NOT `sandbox.py`, which is POSIX code isolation).

For a KNOWN workflow the step list is a fixed template — the LLM is used only
to (a) pull parameters at UNDERSTAND and (b) write the answer at DELIVER. No
LLM planning call, so it is deterministic and cannot hallucinate the steps.
(The agent path remains for tasks whose steps genuinely cannot be pre-scripted;
there the LLM plans dynamically.)

New step-type vocabulary `PlanExecutor` needs (it has none of these today):
- `extract_document` -> `document_engine` (generic structured extraction)
- `build_wbs` -> `ConstructionContainer.generate_wbs` (+ lead times from a
  prior step)
- `cost_load` -> `schedule_bridge` + `pm_excel` (build the workbook object)
- `render_artifact` -> materialize xlsx/pdf **only if** the plan/DELIVER wants
  a document

## 5. Materialization: stage, then deliver per intent

The plan's outputs (WBS data, workbook object, extracted facts) are staged in
the session. DELIVER chooses the FORM from the classified intent:

| User intent | DELIVER form |
|-------------|--------------|
| "how long is procurement?" (question) | inline text from staged data, no file |
| "produce / export the schedule" (deliverable) | materialize workbook -> document |
| "give me a link" | materialize -> link |

The form is gated on **intent**, not on "a tool ran." This is the fix to the
current eager-offer behavior: a question about the schedule gets an answer, not
a download button.

## 6. Layer moves (what goes where)

| Component | Today | Target |
|-----------|-------|--------|
| `project_reasoner` | unregistered peer block, `/v1/project` only | DISSOLVED into the orchestrator block — its PLAN/EXECUTE/DELIVER becomes the block's own predefined-reasoning path; standalone block + `/v1/project` wiring removed |
| the orchestrator block | `smart_orchestrator` only classifies (returns an action) | becomes the single brain: classify + decide + run predefined reasoning + create/dispatch agent, all in one block |
| heavy-reasoning tool-agent | live for all generative turns | the orchestrator block CREATES it, for the LONG TAIL only (open-ended) |
| `document_engine` | generic extract (already) | unchanged — stays a leaf, called by `extract_document` step |
| `generate_wbs`, `schedule_bridge`, `pm_excel` | tool + libs + endpoint bodies | re-homed as `build_wbs` / `cost_load` / `render_artifact` step types |
| `export/schedule-from-*`, `cost-boq`, `evm` endpoints | do extract->build->render inline | thin materialize-a-staged-artifact endpoints, callable by a plan step AND directly by API |
| `_build_exports_from_audit` offer heuristic | fires when `generate_wbs` ran | retired — replaced by DELIVER's intent-gated materialization |

Nothing built so far is thrown away — the schedule compute (bridge, pm_excel,
extraction) moves *layer* from "endpoint + tool" to "plan step."

## 7. Pilot: the schedule pipeline (one workflow, end to end)

Prove the whole spine on a single workflow before touching the rest.

Intent "produce schedule" -> predefined plan:
1. `extract_document` (if an RFP/BOD is in context) -> lead times + milestones
2. `build_wbs` -> WBS + injected long-lead procurement
3. `cost_load` -> workbook object (man-days S-curve, milestones)
4. DELIVER:
   - question intent -> inline summary (duration, critical path, procurement)
   - deliverable intent -> `render_artifact` -> document

Acceptance: a schedule *question* returns an inline answer with no file; a
"produce/export a schedule" returns the workbook; the RFP's real lead times
drive it. This replaces the current endpoint+offer approach for schedule only.

## 8. Migration phases (non-breaking, pilot-first)

- **Phase 0** — this document.
- **Phase 1** — build the consolidated router branch + the 4 step types +
  DELIVER form logic, wired for the schedule workflow ONLY. Tool-agent stays
  live for everything else. Ship behind a flag; verify acceptance above.
- **Phase 2** — migrate BOQ pricing, cost estimate, EVM to predefined plans.
- **Phase 3** — make the router the single decision point; retire the
  `_build_exports_from_audit` offer heuristics; relocate/register the reasoner
  properly.
- **Phase 4** — retire whatever is fully superseded (dead endpoints, the block
  costume) only after the predefined path is proven in production.

## 9. Decisions — RESOLVED (see top of doc)

1. Single front door: everything through the orchestrator; agent kept as an arm.
2. Lift `project_reasoner` into the orchestration layer; drop the block wrapper.
3. DELIVER form: intent default + explicit user override.
4. Steps + materialization via API calls (`/v1/execute` + render endpoints);
   no inlining.

Consequences of "everything through the orchestrator + API calls":
- The orchestrator is the ONLY entry for `/v1/chat/stream`; the current
  `select_agent_for_message` bypass is removed once the pilot proves out.
- Both arms dispatch the SAME way — via API calls to blocks/steps — so a plan
  step and a direct API caller hit identical endpoints. One execution surface.
- The orchestrator is thin: classify -> pick arm -> issue API calls -> collect
  -> DELIVER. It holds no domain logic; blocks/steps do the work.

## 10b. Sandboxes (two distinct ones — both covered)

- **Code-execution sandbox** — `app/core/sandbox.py` (`run_sandboxed`) +
  `app/blocks/sandbox.py` (`SandboxBlock`, 512 MB / 5 s CPU, POSIX). Isolates
  LLM-GENERATED Python; `formula_executor_v2` routes every snippet through it,
  so the `generate_code` plan step inherits isolation. Deterministic schedule
  steps run no generated code, so they need it only if a step generates code.
- **Staging sandbox** — the `ProjectSession` (fresh + ephemeral per request).
  Plan steps write staged state into `session.data`; nothing touches the real
  project or is delivered until DELIVER decides (intent-gated). This is the
  "stage then apply per request" concept — implemented in Phase 1.

## 10c. Reasoning lives in the agent / reasoner — NOT the orchestrator blocks

Verified: `smart_orchestrator` is intent CLASSIFICATION (keyword regex, or a
learned classifier when `routing_mode` != keyword) — no generative/multi-step
reasoning. `orchestrator.py` is chain plumbing (type validation + transforms).
The actual reasoning is in the heavy-reasoning tool-agent and in
`project_reasoner` (whose PLAN phase is an LLM call). So "the orchestrator" must
ABSORB reasoning; today it has none.

## 10d. Direction: DYNAMIC reasoning over predefined route-templates (operator, 2026-07-02)

Operator preference: build DYNAMIC reasoning, not predefined routes/logics.
Reconciliation — Phase 1's machinery is exactly the substrate a dynamic reasoner
needs, so it is NOT wasted:
- KEEP: the typed step vocabulary (`extract_document`/`build_wbs`/`cost_load`/
  `render_artifact`), `PlanExecutor`, the `ProjectSession` staging, intent-gated
  DELIVER + materialization, the SSE streaming.
- CHANGE: the PLAN source. Instead of a fixed `build_schedule_plan(context)`
  template (a "predefined route"), the orchestrator's reasoning does an LLM PLAN
  call that emits an `ExecutionPlan` over the SAME typed steps — exactly
  `project_reasoner`'s PLAN phase. Dynamic (the LLM chooses steps) but BOUNDED
  (only valid step types, executed deterministically) — safer than free-form
  tool-calling (no tool-drift/hallucination of steps), more flexible than
  templates. This becomes Phase 2.
- The fixed template stays only as a deterministic fallback when the LLM planner
  is unavailable.

## 10. Guardrails (must not break)

- Live chat keeps working throughout; the tool-agent stays live until the
  predefined path is proven in production.
- Do not delete `project_reasoner` or the export endpoints until superseded and
  verified.
- Preserve RAG grounding (project-excerpt injection) in every path.
- No regression on the cost-BOQ inline offer that already ships.
- No auto-execution: a deliverable is produced only when intent asks for one.
