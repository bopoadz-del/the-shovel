# Build Spec — The Fork Discipline-Agent Layer (base + hats)

**Audience:** an engineer/coding-agent implementing this without further guidance.
**Reference implementation to mirror:** `bopoadz-del/TEKsystems_GlobalRetailMNC`,
commit `5617712282e4a1edf658fc66d96d436973580ed2` ("retail discipline agents (base + hats)").
Read that commit's `backend/app/retailops/agents/**`, `reasoning/**`, `learning/**`
first — it is the **vendor-neutral contract** we are adopting. This spec adapts it to
The Fork (construction) and grafts it onto our EXISTING wired engine. Do not rebuild
what already works.

---

## 0. Goal (one sentence)

Give The Fork ONE base construction agent that wears swappable **discipline hats**
(Planning, Commercial, Contracts, QAQC, Procurement), driven by declarative JSON
manifests using the same schema as RetailOps, so grounding/honesty/formulas/handoffs
are defined per-discipline and auditable — WITHOUT rewriting the existing engine,
calculators, RAG, or breaking the live pilot.

## 1. What already exists in The Fork — MAP to it, do NOT replace

| Existing piece | File(s) | How it maps into the manifest |
|---|---|---|
| Agent runtime + tools + grounding gates + dynamic intent | `app/agents/runtime.py` (`Agent`, `tool_definitions()`, `_run_tool_call()`, `_INTENT_TOOL_MAP`, `_cost_grounding_gate`, `_standards_advisory`, `_postprocess_answer`, dynamic `understand_intent`) | The base agent's runtime host; hat activation overlays system prompt + allowed tools |
| Deterministic calculators (35) | `app/lib/construction_formulas.py` (`CALCULATORS`, `run_calculation`, `construction_calc` tool) | `playbook.formulas[].implemented_by = "app.lib.construction_formulas.<fn>"` |
| Wired reasoning + learning engine (93/93 tests) | `app/core/construction_knowledge.py` (ActivityGraph, DependencyGraph, WorkflowTemplateLibrary, CrossDomainReasoner, ConstructionLearningEngine, `calculate_evm/payment/tender/risk`) | The reasoner/handoff substrate + `context_sources.kind=learning_profiles`. **NEVER clobber this — it passes `tests/comprehensive_engine_test.py` 93/93.** |
| Construction container actions (~40) | `app/containers/construction/**` (boq, drawing_qto, spec_analyzer, generate_wbs, commissioning_checklist, ...) | `allowed_actions[]` = registered action ids per hat |
| RAG + curated knowledge | `app/core/rag/**`, `docs/knowledge/*.md` (FIDIC, OSHA 1926, WBDG packs), GK project `curated_kb` | `context_sources.kind=project_documents` and `official_guidance` |
| Grounding/honesty doctrine | cost-grounding gate + standards advisory in runtime; `app/agents/configs/project-assistant.md` | `verification` + `failure_policy` blocks in each manifest |

## 2. What to build (mirror RetailOps file-for-file, adapted)

Create under `app/agents/`:

- `manifest.schema.json` — PORT verbatim from RetailOps, changing only the
  `discipline` enum to: `["base","planning","commercial","contracts","qaqc","procurement"]`,
  and `$id`/title to The Fork. Keep every other field (activation, context_sources,
  allowed_actions, handoffs, memory, verification, failure_policy, playbook).
- `models.py` — PORT the Pydantic typed manifest models (`ActivationConfig`,
  `ContextSource`, `HandoffRule`, `MemorySaveRule`, `Verification`, `FailurePolicy`,
  `Playbook`, `AgentManifest`). Reuse our deterministic base model if we have one;
  otherwise copy RetailOps' `_DeterministicModel`.
- `catalog.py` — loads + validates all manifests at import (fail loudly on schema
  violation), exposes `get_agent_catalog()`, `get_base()`, `get_hat(discipline)`.
- `formulas.py` — `allowed_actions_for_hat(hat)` and `executable_formula_ids()`;
  bind each `playbook.formulas[].implemented_by` string to the real callable in
  `app.lib.construction_formulas.CALCULATORS` (and `construction_knowledge` for
  evm/payment/tender/risk). Fail if a manifest references a formula/action that does
  not resolve — this is the guard that keeps manifests honest.
- `manifests/fork.base.json` and one per hat:
  `fork.hat.planning.json`, `fork.hat.commercial.json`, `fork.hat.contracts.json`,
  `fork.hat.qaqc.json`, `fork.hat.procurement.json`.

## 3. The construction hats (content of each manifest)

Base (`fork.base.json`): the shared cortex — mirror `retail.base.json`. Set
`verification.grounding="documents_or_learning_only"`, `cite_required=true`,
`refuse_unaided_figures=true`; `memory.never_persist=["llm_raw_text","secrets",
"cross_tenant_data","unverified_claims"]`; base `allowed_actions` = document export /
generate brief / get-learning-summary equivalents that already exist.

| Hat | discipline | Owns (playbook) | allowed_actions (real ids) | context_sources |
|---|---|---|---|---|
| Planning | planning | WBS build-up, schedule logic, EVM (CPI/SPI/EAC) | `generate_wbs`, schedule/EVM actions, `construction_calc:calculate_evm` | project_documents, learning_profiles (DurationCalibration), official_guidance (WBDG PM) |
| Commercial | commercial | cost build-ups, interim payment/IPC, cash flow | `construction_calc:cost_buildup_*`, `calculate_payment`, boq cost actions | project_documents (priced BOQ), official_guidance (rates in curated_kb) |
| Contracts | contracts | change management, claims, EOT, FIDIC determinations | change-order/variation actions | official_guidance (FIDIC notes), project_documents (contract) |
| QAQC | qaqc | commissioning, handover, ITP, OPR/BOD/Systems Manual | `commissioning_checklist`, validation_pipeline | official_guidance (WBDG Cx/O&M, OSHA), project_documents |
| Procurement | procurement | tender evaluation, estimation, lead times | `evaluate_tender`, procurement list actions | learning_profiles (ProcurementLeadTimeLearner), official_guidance |

Cross-hat `handoffs` (declare in manifests): Commercial -> Contracts `when` a cost
variation implies a change order (carry the figure + evidence); Planning -> Commercial
`when` a schedule change hits cost; QAQC -> Contracts `when` a non-conformance is
contractual.

## 4. Reasoner / activation — THE ONE DELIBERATE DIVERGENCE FROM RETAIL

RetailOps' reasoner scores intent by **keywords only**. Do NOT copy that as the
decision-maker — it is the brittle keyword classifier The Fork already replaced.
Instead:

1. Use `activation.triggers` (keyword/intent/action_id) as a cheap **fast-path**.
2. Use our existing **dynamic LLM `understand_intent`** (see `app/core/dynamic_reasoning.py`
   / the smart orchestrator) as the actual hat-activation decision.
3. Respect `activation.mode` (manual/auto/hybrid), `default_enabled`, and `put_off_by`
   (user toggle disables a hat for the turn — "we always bend the rules, just flag it").
4. Multiple hats may activate; base is always on. On activation, overlay the hat's
   system-prompt guidance + restrict tools to base+active-hat `allowed_actions`.

Build this as a thin adapter that plugs into `runtime.py` — it selects hats, then the
existing tool-forcing (`_INTENT_TOOL_MAP`) and grounding post-processing run unchanged.

## 5. Learning — REUSE, do not rebuild

Map `context_sources.kind=learning_profiles` to the EXISTING wired
`ConstructionLearningEngine` (DurationCalibration, ProcurementLeadTimeLearner) in
`construction_knowledge.py`. Advisory only, verified-outcome only. Do NOT port
RetailOps' `learning/engine.py` — we already have the equivalent, wired and tested.

## 6. Non-negotiable doctrine (enforce, and add tests for each)

- **Real or honest-error.** Missing context -> `failure_policy.on_missing_context`
  (dependency_required / ask_clarifying), never a fabricated answer.
- **Deterministic figures only.** Every number comes from a `construction_calc`
  formula or a cited document. `refuse_unaided_figures=true` must actually refuse a
  cost/quantity answer with no evidence (this is the live cost-grounding gate).
- **No LLM-text learning.** Never persist LLM free text / unverified claims.
- **Standards are advisory (flag, don't block).** Deviations are highlighted, the task
  proceeds (existing `_standards_advisory`).
- **Don't break the pilot.** Ship the whole layer behind an env flag (mirror
  `ORCHESTRATOR_PREDEFINED`), default OFF, so pilot behaviour is unchanged until enabled.
- **Provider routing.** Any agent LLM call routes through the configured provider
  chain (Groq -> Ollama fallback); do NOT add DeepSeek/Anthropic/OpenAI directly.
- **No emojis anywhere** (code, comments, manifests, UI, commits).
- **Do NOT modify `construction_knowledge.py`'s wired classes** — they pass
  `comprehensive_engine_test.py` 93/93. Extend via new files only.

## 7. Tests to write (`tests/`)

- `test_discipline_manifests.py` — every manifest validates against the schema; every
  `discipline` is unique; base has `kind=base`, hats have `extends=fork.base`.
- `test_manifest_bindings.py` — every `playbook.formulas[].implemented_by` resolves to
  a real callable; every `allowed_actions[]` resolves to a registered container action.
  (This is the honesty guard — a manifest that lies fails CI.)
- `test_hat_activation.py` — real construction prompts activate the right hat(s):
  "compute EVM ..." -> planning; "interim payment due ..." -> commercial; "change order
  impact ..." -> contracts; "commissioning checklist ..." -> qaqc; "evaluate these three
  bids" -> procurement; a general question activates base only.
- `test_grounding_enforced.py` — a cost/quantity ask with no retrieved evidence and no
  calc REFUSES (does not fabricate); a calc-backed ask returns the deterministic figure.
- `test_flag_off_is_noop.py` — with the env flag OFF, behaviour is byte-for-byte the
  current pilot path.

## 8. Acceptance criteria (definition of done)

1. All manifests load + validate at import; CI green including the binding tests.
2. Every formula/action referenced by a manifest resolves to real code (no dangling ids).
3. Dynamic activation picks the correct hat on the 6 probe prompts above.
4. Grounding gate refuses unaided figures; deterministic calcs still return exact values.
5. `comprehensive_engine_test.py` still 93/93 (engine untouched).
6. Flag OFF = current pilot behaviour unchanged; flag ON = hats active.
7. One PR, squash-merged to main, deployed, and a live smoke on the 6 probes.

## 9. Sequence (suggested)

1. Port schema + models + catalog + formulas (bindings) — with `test_manifest_bindings`.
2. Author base manifest, then the 5 hats — with `test_discipline_manifests`.
3. Activation adapter into runtime (dynamic understand_intent) — `test_hat_activation`.
4. Wire grounding/failure enforcement to the existing gates — `test_grounding_enforced`.
5. Flag-gate + `test_flag_off_is_noop`; PR; deploy; live smoke.

Keep it small and reviewable; the contract is the owner's RetailOps commit, the wiring is
into The Fork's existing engine. When in doubt, prefer mapping over rewriting.
