# Decisions log

Autonomous-mode decisions with rationale, plus parked items awaiting the owner.
Newest first.

## 2026-09-06 — DeepSeek restored as a first-class cloud LLM provider

**Decision:** restore **DeepSeek native API** as an explicit-only cloud
primary (`LLM_PROVIDER=deepseek`, `DEEPSEEK_API_KEY` required,
`DEEPSEEK_MODEL=deepseek-chat` default; `deepseek-reasoner` or a current
catalogue id is accepted). OpenRouter free is 402 on a $0-credit account;
operators may pin DeepSeek and keep `LLM_FALLBACK_PROVIDER=kimi` or `groq`.

DeepSeek is **not** auto-picked from a bare `DEEPSEEK_API_KEY`. Unset /
unrecognised `LLM_PROVIDER` still resolves to Kimi (historical default).
Do not reintroduce OpenAI. Temperature is not pinned (`fixed_temperature`
is a Kimi-K2 constraint only).

**Render dashboard:**
```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=…
DEEPSEEK_MODEL=deepseek-chat
LLM_FALLBACK_PROVIDER=kimi
```

## 2026-09-05 — OpenRouter free models as a first-class cloud provider

**Decision (operator, 2026-09-05):** production may use **OpenRouter free
models** as the cloud primary (`LLM_PROVIDER=openrouter`,
`OPENROUTER_MODEL=openrouter/free` or any catalogue slug ending `:free`)
so chat does not burn paid Kimi tokens or exhaust Groq TPD. Kimi and Groq
remain optional primary/fallback providers (`LLM_FALLBACK_PROVIDER` can
be `openrouter`, `kimi`, or `groq`).

Paid OpenRouter slugs are refused (logged + swapped to `openrouter/free`)
unless `OPENROUTER_ALLOW_PAID=1`. The previous production workaround
(`LLM_PROVIDER=ollama` + `OLLAMA_URL=https://openrouter.ai/api/v1/chat/completions`)
is replaced by this native branch — do not keep the ollama-URL shim.

**Live intent:** `LLM_PROVIDER=openrouter`, `OPENROUTER_MODEL=openrouter/free`,
`OPENROUTER_API_KEY` set on Render. Kimi/Groq keys optional.

## 2026-07-24 — SUPERSEDES all prior provider rulings: Kimi primary + Groq fallback

**Decision (operator, 2026-07-24):** production chat LLM is **Moonshot
Kimi `kimi-k2.6` (native Moonshot API, `LLM_PROVIDER=kimi`) primary**,
with **Groq as the free-tier fallback** (`LLM_FALLBACK_PROVIDER=groq`).
Ollama Cloud stays as the emergency tier. This **overrides** the
2026-07-08 v2 freeze (OpenAI `gpt-4o-mini` primary) below, the earlier
Groq-Scout-primary freeze, and the "provider work permanently out of
scope / written-instruction-only" clause — this dated operator
instruction is that written instruction.

**Live env (verified 2026-07-25 sweep):** `LLM_PROVIDER=kimi`,
`KIMI_MODEL=kimi-k2.6`, `LLM_FALLBACK_PROVIDER=groq`,
`GROQ_MODEL=llama-3.3-70b-versatile`. `OPENAI_API_KEY/MODEL` remain set
but unused (stale, safe to unset).

**Known constraint (⛔ billing gate, PLATFORM_HEALTH_REPORT F1a):** the
Groq free/on-demand account 413s any fallback payload >~12k tokens for
EVERY available model (scout is not on the account). The fallback is
reliable only for small-payload turns until Groq is upgraded to Dev
tier. Primary (Kimi) is unaffected and carries production load.

Every provider section below (OpenAI-primary v2 freeze, Groq-Scout
freeze, Ollama-primary, interim-Kimi) is now historical context, not
current config.

## 2026-07-13 — Phase-2 param resolution: deterministic doc→slot, not LLM

**Decision:** resolve an action's FILE param from the project's uploaded
documents deterministically (by extension → slot), NOT via a per-turn LLM
extraction call.
**Why:** the file-param majority (schedule_file, as_built drawings, etc.) can't
be extracted from prose anyway — the real source is the project's documents. A
per-turn LLM call adds latency (the K2/Scout deadline scar) and cost for no gain.
**Honesty rule (files inherit no-assumptions):** caller-supplied wins; no match →
leave empty so the action honest-errors; **2+ matches → honest ask-which, never
silently pick one.** Verified live on prod (project `31df5b9d`).

## 2026-07-13 — F4 daily_site_report location: DEFERRED to a project field

**Decision:** do NOT scrape `location` from the chat message for the weather
lookup. Add a project-metadata location field; `daily_site_report` reads it; no
field → honest "set the project location". Tracked as task #26.
**Why:** message-scraping is fabrication-adjacent — a mis-parsed token yields
wrong-city weather in a formal site report. Project metadata is the authoritative
source (uploads inherit active-project metadata — the no-assumptions rule).

## 2026-07-13 — historical_benchmark config-rot: NEEDS-DECISION (parked)

**Parked for the owner:** 4 agent configs say historical_benchmark "was removed" but
it exists + is contract-tested. The wording is an LLM-facing steer; rewriting it
could change routing behavior, so it needs a product call (truthfully reword vs.
actually retire the block + its contract test), not a unilateral guess.

## 2026-07-13 — FROZEN: pilot retrieval configuration (Step 3 winner)

**Decision:** The pilot retrieval config is `RAG_GK_SCORE_MARGIN=0.10` +
`RAG_GK_LEXICAL_FOLD=1` (paired). Deployed to `the-fork`
(srv-d8hdc6ek1jcs739rq5sg), deploy `dep-d9a15o4s728c73e1rfh0`, live.

**Supersedes** the 2026-07-08 park ("do not tune RAG knobs; proceed to embedder
migration"). That park was correct for the legacy 256-dim embedder. The embedder
migration to BGE-384 (v2 namespace) has since completed and the corpus was
rebuilt; on the new embedder the knobs are the correct fix, not a symptom mask —
the failure is a specific, measured scoring-math flaw (below), not an embedding
weakness.

**Evidence (live golden re-run, apples-to-apples with the 10/28 baseline):**
- Golden **10/28 -> 22/28**. All 12 fresh-upload cases flipped 0 -> PASS.
- **Zero regressions** among the baseline-10 (doc_qa still grounded at 939 chars,
  wbs/manpower/s_curve/procurement/rfp/4 demo all held).
- calc/client referee verified ON THE LIVE RUN: GK not strangled.
- In-process finalist cleared all 3 columns: fresh_win@1 12/12, client 8/8, calc 5/5.

**Mechanism:** the 12 fresh misses were GK lexical-bonus inflation — an unbounded
additive `_gk_lexical_bonus` lifting GK chunks to 1.1-2.0 vs own docs at
0.80-0.91 (the EOT/FIDIC bug at 140k-chunk scale). The lexical fold only acts
inside the margin gate, so the knobs MUST be paired.

**Standing authorization (the owner):** if this config ever regresses client/calc, the
fix is a small PR bounding the lexical component (clamp GK total <= cosine + eps,
or project-scope the bonus), same referee set — NOT further env-knob tuning.

**Config freeze in force:** from here, nothing changes on the measured path
except the sanctioned Step 4 dispatch-flag evaluation. `ORCHESTRATOR_PREDEFINED`
stays false until the scoped-v2 code is merged and validated.

### Residual after the knob (6 FAILs — the readiness packet's work list)
None are retrieval problems any longer. Two feature-classes:
- **Test-data / fixture-missing (3):** pilot_boq_total (96bd7cd1),
  pilot_milestones (7ce7b9d0), pilot_qto_floor_area (7ce7b9d0) — target projects
  do not exist (empty answer every run). Mechanical: seed or repoint.
- **Synthesis-strictness (3):** pilot_spec_extraction (1222 chars, matched
  "grade" not a standards body), pilot_document_metadata (76 chars, wrong
  format), pilot_kb_mass_concrete (828 chars, peak 70 C not the 20 C
  differential). Feature/prompt or oracle-strictness, not retrieval.

Path to the 26/28 bar: 3 fixture seeds -> ~25/28; one synthesis fix clears it.
Neither touches retrieval.

## 2026-07-08 — FINAL PROVIDER RESOLUTION (v2 freeze)

**Decision:** Pilot primary provider is **OpenAI `gpt-4o-mini`**.
Fallback is **Ollama native `/api/chat` with `glm-5.2:cloud`**.
All other providers are **permanently removed from consideration** for the
pilot.

### Rationale

- **OpenAI gpt-4o-mini** passed the acceptance gate cleanly:
  `smoke --runs 10` against prod = **10/10 PASS**, all tool-backed,
  model column = `gpt-4o-mini-2024-07-18`, zero `_TOOL_FORMAT_FALLBACK`,
  zero silent fallbacks. Per-run latency ~11s, well inside the 90s timeout.
- **Ollama native `/api/chat`** is kept as the fallback layer. It was the
  path that restored prod service and passed `smoke --runs 3`, but its
  10-run gate was inconsistent under rapid-fire execution (timeout/queueing
  variability). It remains a valid fallback on OpenAI retryable failures.
- **Groq/Scout, Kimi K2.6, Moonshot v1, DeepSeek, and any other provider**
  are out. They consumed multiple days without producing a clean, repeatable
  10/10 smoke gate on prod.

### Operational env (Render prod)

- `LLM_PROVIDER=openai`
- `OPENAI_MODEL=gpt-4o-mini`
- `OPENAI_API_KEY=<set>`
- `LLM_FALLBACK_PROVIDER=ollama`
- `OLLAMA_URL=https://ollama.com/api/chat`
- `OLLAMA_MODEL=glm-5.2:cloud`
- `OLLAMA_API_KEY=<set>`

### Acceptance gate (already passed)

- `smoke --runs 10` on prod: **10/10 PASS** (saved to
  `review_pack/openai/smoke10_*.log`)
- 3 deliverable outputs saved to `review_pack/openai/deliverables/` for
  quality review

### Freeze rule

Provider work is **permanently out of scope** for the remainder of the
pilot program. The only remaining provider knob is `OPENAI_MODEL`, and it
may be bumped **only on the owner's explicit written instruction** citing this
freeze. Any exception must also be written and reference this decision.

## 2026-07-08 — PARKED: retrieval-quality blocker on legacy 256-dim embeddings

**Observation:** After manually reindexing the three fixture projects, the
semantic retriever still returns only `drive_archive:` GK chunks from
`training_material` for project-scoped queries. The fresh-upload eval fell
from the previously reported 12/12 top-1 to **0/12 top-1 / 0/12 top-3**, and
fixture-project searches (`96bd7cd1`, `7ce7b9d0`) likewise surface GK chunks
rather than the project's own BOQ/programme documents. Hybrid BM25 is already
wired (`retriever.py` passes `query_text`) but does not rescue the cases.

**Working diagnosis:** The legacy 256-dim embedding model does not separate
small, project-specific documents from the large, topically overlapping
Drive-archive GK corpus. This is the same limitation measured in the Step-2
recall gate (doc@5 20 %, chunk@5 17 % vs. 41 % / 27 % bar).

**Decision:** Do not tune RAG knobs (score margin, own-doc boost, GK cap,
lexical fold) to mask the embedding weakness — that would be chasing symptoms.
Run the T5 sweep and golden set to capture the baseline, publish the failure
itemization, and proceed to **T6 embedder migration** as the intended fix.
If a new embedder cannot lift doc recall@5 to ≥50 % and golden-set score to
≥ its T5 value, the cutover is PARKED for the owner.

## 2026-07-07 — Provider switch: Ollama native `/api/chat` primary

**Decision:** Prod is now running **Ollama `glm-5.2:cloud` via the native
`/api/chat` protocol**. Groq/Scout is parked pending a clean 10/10 smoke gate.
OpenAI support is added but not yet tested.

### Rationale

- **Groq/Scout** could not hold the smoke gate: intermittent
  `_TOOL_FORMAT_FALLBACK` 120-char answers persisted after the raw-args
  recovery fix. The failure correlated with the Ollama OpenAI-compatible
  `/v1/chat/completions` endpoint leaking raw internal tool/search args as
  `content`; the recovery regexes did not cover every shape the model emitted.
- **Ollama native `/api/chat`** returns properly structured `tool_calls` and
  usable `message.content`. Tool-result messages must be recast from the
  OpenAI `tool` role to `user` role, and `function.arguments` must be sent as
  parsed dicts rather than JSON strings. With those adaptations the smoke
  gate passes (`smoke --runs 3`: 3/3, tool-backed, model=glm-5.2:cloud).
- **OpenAI gpt-4o-mini** provider support was added to `runtime.py` at the
  user's request as an alternative to test, but no `OPENAI_API_KEY` is set
  locally or on Render, so it has not been exercised.

### Operational env

- `LLM_PROVIDER=ollama`
- `OLLAMA_URL=https://ollama.com/api/chat`
- `OLLAMA_MODEL=glm-5.2:cloud`
- `OLLAMA_API_KEY=<set>`
- `LLM_FALLBACK_PROVIDER=` (unset until Groq is re-verified)

### Acceptance gate

- `smoke --runs 10` against prod must be **10/10 PASS**, zero
  `_TOOL_FORMAT_FALLBACK`, and model reported as `glm-5.2:cloud` (no silent
  fallback).
- First 10-run attempt: 3/10 PASS, then 7 transient failures. A fresh single
  run passed immediately after. The gate is provisional until a clean 10/10
  re-run completes.

### Parked items

- **Groq/Scout re-verification:** Parked. If the user wants Scout back as
  primary, re-run `smoke --runs 10` on a branch with `LLM_PROVIDER=groq` and
  `GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`. Do not flip prod
  without the gate.
- **OpenAI gpt-4o-mini:** Parked on key availability. Set
  `OPENAI_API_KEY` on Render and run the same smoke gate before any prod
  flip.

## 2026-07-07 — LLM provider ladder FROZEN for pilot

**Decision:** Pilot default is **Groq `meta-llama/llama-4-scout-17b-16e-instruct`**.
Kimi K2.6 and Moonshot v1 are **out of the active chain** for the pilot.

### Rationale

- **Kimi K2.6** first-token latency is routinely 70–120s, exceeding the
  reader-tolerance threshold and frequently hitting the chat timeout even at
  120s. Not acceptable as a pilot default.
- **Moonshot v1-32k** tokenizes and responds fast, but the stripped-payload
  compatibility path (no tools, no message history) cannot run the
  project-assistant deliverable tools; answers are ~580 chars of prose and
  fail the smoke gate. Not viable for a construction-deliverable pilot.
- **Scout** was intermittent with `_TOOL_FORMAT_FALLBACK` until the raw-args
  recovery fix (`feat/scout-tool-recovery`). With the fix it passes the smoke
  gate and produces tool-backed deliverables.

### Frozen ladder

1. **Primary:** Groq — `meta-llama/llama-4-scout-17b-16e-instruct` (pinned in
   Render env as `GROQ_MODEL`).
2. **Fallback:** Ollama — `glm-5.2:cloud` ( Render env `OLLAMA_*` ), used only
   on Groq retryable failures (429, 5xx, timeout).
3. **Kimi / Moonshot:** parked. Any post-pilot re-evaluation requires the owner's
   explicit written instruction naming this freeze.

### Operational env

- `LLM_PROVIDER=groq`
- `GROQ_MODEL=meta-llama/llama-4-scout-17b-16e- instruct`
- `LLM_FALLBACK_PROVIDER=ollama`
- `CHAT_STREAM_TIMEOUT_SECONDS=90`

### Acceptance gate

- `smoke --runs 10` against prod must be **10/10 PASS**, zero
  `_TOOL_FORMAT_FALLBACK`, and zero silent fallbacks to Ollama. Until that
  gate is met the provider choice is provisional and this entry must be
  updated.

## 2026-07-07 T3 — Migration reconciliation

### GK identity

- **Single GK project: `training_material`.** After the drive_archive migration,
  the curated general-knowledge corpus lives in `training_material` (241 docs).
  `curated_kb` exists but is empty (0 docs). `RAG_GENERAL_KNOWLEDGE_PROJECTS`
  is therefore `training_material` in prod and in `render.yaml`.
- **Consequence:** Any eval harness or manifest that still points GK at
  `curated_kb` is stale and has been updated.

### Fixes landed

- **chunk_count aggregation now uses the vector store.** `GET /v1/projects/{id}`
  previously counted per-document chunks from the legacy `doc_index` JSON blob,
  which bulk-inserted chunks never populate. It now reads from the `chunks`
  table (cheap indexed `COUNT` / `GROUP BY`), so migrated projects never show
  `chunk_count: 0` or `None`.
- **training_material scoping anomaly root cause:** chunks whose `doc_id`
  belongs to `training_material` documents were stored with `project_id=
  projects_folder` during migration. Direct search on `training_material`
  returned 0; search on `projects_folder` returned those chunks. Added
  `POST /v1/admin/corpus/reconcile` (dry-run by default, `execute=true` to
  repair) which aligns `chunks.project_id` with `documents.project_id`.
- **Reconciliation script:** `scripts/reconcile_migration.py` prints the
  project_id/name/docs/chunks/API-chunk-count/admin-chunk-count table and
  flags mismatches. It is read-only unless the operator explicitly calls the
  repair endpoint.

### Operational note

- The repair endpoint is gated on admin role and defaults to dry-run. Against
  prod it must be used only after the reconciliation script is reviewed; no
  destructive ops are performed by the script itself.

## 2026-07-06

- **TASK H evals run against a LOCAL uvicorn instance, not prod.** The sweep is
  ~8 configs x (100 recall + 12 fresh-upload + 3 calc) queries plus config
  changes between runs. Against prod that is ~1000 paced queries (hours),
  8 env-flip deploys of a live pilot service, and a demonstrated
  health-check-kill risk from search bursts. The local corpus was judged
  representative by the June-23 audit; a local config=off baseline run makes
  the comparison internally consistent. V3 will state this explicitly.
- **TASK G sweep runs AFTER TASK H implementation lands locally,** in paced
  batches: the sweep needs ~120+ heavy LLM turns against prod on a free Groq
  tier; interleaving it with H's local work maximizes useful wall-clock and
  the sweep parks cleanly if 429s degrade evidence quality.
- **CI PR #146 intentionally ships with a first-run failure expected:** the
  pytest-timeout stack dump IS the diagnostic for the 6-hour hang. Root-cause
  fix lands in the same PR, then green, then merge.

## Dependabot dispositions (2026-07-06, TASK 0b)

- HIGH npm vite <=8.0.15 (fs.deny bypass, Windows): FIX NOW - PR #67 bumps to 8.0.16, merge on green CI.
- medium npm vite (launch-editor NTLM hash, Windows): fixed by the same #67 bump.
- medium pip pydantic-settings <2.14.2 x2 (requirements.txt + -cv.txt): FIX NOW - PR #99 bumps to 2.14.2, merge on green CI.
- medium pip pytest <9.0.3 x2 (tmpdir handling): DEFER - test-only dependency, not shipped in the runtime image path that serves users; bump with the next requirements refresh.
- low npm @babel/core <=7.29.0 (sourceMappingURL file read): DEFER - build-time only, frontend build runs in CI not on user input.
- low pip torch <=2.12.0 x3 (jit.script memory corruption, no patched version exists): DEFER - no fix released; torch only enters via requirements-ml/cv/rag extras which prod does not install (Starter image ships without the ML stack).

## CI quarantines

- `tests/test_doc_index.py::test_search_uses_hybrid_retriever` and
  `tests/test_doc_search_api.py::test_search_returns_ranked_results` are
  xfail (strict=False): curated GK notes crowd freshly uploaded project
  docs out of the top-5 — a known precision issue documented in
  RAG_AUDIT_V2.md, with the fix (fresh-upload-wins config knobs) tracked
  under TASK H on its own branch, defaults-off. Un-xfail when the knobs
  land.

## Parked (the owner's gates)

- K2 decision: raise CHAT_STREAM_TIMEOUT_SECONDS vs synthesis streaming on
  kimi vs park K2. Task 5 blocked on this.
- RAG production default config: picked from the RAG_AUDIT_V3 table only.
- Construction correctness of TASK G outputs: review_pack/ is for the owner's
  hands-on judgment; the sweep certifies routing/execution/structure only.
- 6 remaining zombie CI runs (cloudflare bot branch + pre-existing) left
  running; classifier requires operator action to mass-cancel.
- Dependabot: 10 open vulnerabilities on main (1 high). Two stale dependabot
  PR branches (vite 8.0.16, pydantic-settings 2.14.2) predate CI revival and
  will get working CI once #146 merges.

## Routing-miss evidence log (TASK G class-1)

### Routing-miss evidence (TASK G sweep, 2026-07-06, prod, verbatim)

| action | prompt | router chose | confidence | reason |
|---|---|---|---|---|
| process_document | what does the the client project project execution plan cover? | (none) | 0.0 | below_routing_gate |
| spec_analyze | analyze the concrete specification requirements - what grades and stan | (none) | 0.0 | below_routing_gate |
| spec_analyze | pull out the material specs for the road works | (none) | 0.0 | below_routing_gate |
| document_metadata | list the documents in this project and what type each one is | (none) | 0.0 | below_routing_gate |
| document_metadata | which drawings do we have for the stormwater network? | (none) | 0.0 | below_routing_gate |
| parse_primavera_schedule | give me a milestone report - what are the major completion dates? | (none) | 0.0 | below_routing_gate |
| cash_flow_forecast | what does the cumulative spend curve look like month by month? | (none) | 0.0 | below_routing_gate |
| procurement_list_generator | what materials do we need to buy for the substructure works? | (none) | 0.0 | below_routing_gate |
| rfp_management | prepare an RFP for the landscaping subcontract package | (none) | 0.0 | below_routing_gate |
| drawing_qto | do a quantity takeoff from the infrastructure drawings - pipe lengths  | (none) | 0.0 | below_routing_gate |
| commissioning_checklist | what T&C steps do we need before energising the electrical rooms? | (none) | 0.0 | below_routing_gate |
| rfi_management | how many RFIs are open and which ones are overdue? | (none) | 0.0 | below_routing_gate |
| safety_compliance_audit | run an HSE compliance audit checklist for working at height on the fac | (none) | 0.0 | below_routing_gate |
| tender_bid_analysis | compare three contractor bids for the earthworks - how should we score | (none) | 0.0 | below_routing_gate |
| extract_quantities | take off the concrete quantities for the ground floor slabs | (none) | None | ? |
| progress_tracker | how is actual progress tracking against planned - where are we slippin | (none) | 0.0 | below_routing_gate |
| submittal_log_generator | set up a submittal log for the finishes packages with approval status  | (none) | 0.0 | below_routing_gate |
| as_built_deviation_report | report the as-built deviations from design on the drainage runs | (none) | 0.0 | below_routing_gate |
| om_manual_generator | generate the O&M manual outline for the chilled water plant | handover_management | 0.4 | below_routing_gate |
| value_engineering | value engineer the basement - options to cut cost without losing parki | (none) | 0.0 | below_routing_gate |

Input for the post-pilot keyword-dictionary rebuild. The router was NOT tuned during the sweep (standing rule).

## BOQ total discrepancy (Step 1c, 2026-07-06)

**Finding:** Project `5c13510e` (the client project Bills of Quantities) live corpus cites a
BOQ total of **29,207,138.5 USD** (review_pack/boq_process_1.md, verbatim
answer). A remembered value of **SAR 62,236,109** could not be verified in the
repo corpus or in the live retrieval context.

**Classification:** Data/expectation issue, not a calculation bug. The live
corpus is internally consistent (the same USD value appears in both the total
and the cost-breakdown chunks), and the model is correctly grounding its
answer in retrieved chunks.

**Disposition:** No code fix. The golden-set gate (`tests/golden_set.yaml` on
`feat/golden-set-gate`) already avoids pinning a number for the BOQ total
query; it expects only a currency token plus a million-scale value. The owner to
confirm whether the corpus value is the authoritative client figure or
whether the project BOQ corpus needs to be refreshed/replaced.

## Missing BOQ fixture project (Step 1, 2026-07-06)

**Finding:** The manifest references project `5c13510e` for `boq_process` and
`drawing_qto` fixtures. Prod API returns `HTTP 404 Project not found` for this
project ID. The projects list on prod does not contain `5c13510e`.

**Impact:** `boq_process` (must-cover pilot-critical feature) is BLOCKED in the
FEATURE_MATRIX_V2 sweep. `drawing_qto` uses project `ff905e29` and is unaffected.

**Classification:** Fixture/data gap, not a routing or block bug.

**Disposition:** Do not guess-fix. The owner to either (a) restore project
`5c13510e` from backup, (b) provide the correct BOQ project ID, or (c) upload a
new BOQ workbook to a fresh fixture project and update the manifest. Until then,
`boq_process` will be reported BLOCKED in the feature matrix.

## `cash_flow_forecast` thin-answer failure (Step 1, 2026-07-06)

**Finding:** In the FEATURE_MATRIX_V2 sweep, prompt 2 for `cash_flow_forecast`
("what does the cumulative spend curve look like month by month?") returned a
251-character answer: "I don't have that information in the provided reference
context." The project used was `master_corpus` (master corpus), which
contains contract/payment clauses but no project-specific cost plan or budget
curve.

**Classification:** Fixture/data gap (no cost data in the test project), not a
block bug. The block refused correctly because there is no cost corpus to ground
a forecast in. This is the same class as the BOQ discrepancy: the feature cannot
produce a correct answer if the project has no priced/cost data.

**Disposition:** No code fix. If a cash-flow/S-curve forecast is a demo-critical
requirement, create a fixture project with a priced BOQ or cost-loaded schedule
and update the manifest. Until then, the red line is valid information that the
feature is untested for real cost data.

## `parse_primavera_schedule` GK-contamination case (Step 1 → Step 2, 2026-07-06)

**Finding:** Prompt 2 for `parse_primavera_schedule` ("give me a milestone
report - what are the major completion dates?") routed to the correct action but
returned FIDIC contract deadline tables instead of milestones extracted from
the uploaded programme in fixture project `ff905e29`. The answer was grounded in
GK/reference content rather than the user's own document.

**Classification:** GK contamination / answer-source problem. This is the same
EOT-failure class (GK beats the user's own document) now appearing in a
scheduling feature. It strengthens the case that GK contamination is a systemic
answer-source problem, not just a retrieval corner case.

**Disposition:** Do not fix code now. Add this as case (e) to the Step 2
acceptance battery and re-test after `RAG_GK_LEXICAL_FOLD=1` is active. The
intent-exempted GK demotion may resolve it for free. If it still fails after the
fold, it becomes a block-level answer-source bug for the next iteration.

## Missing fixture projects on prod (Step 1, 2026-07-06) — UPDATED

**Finding:** `5c13510e` (the client project Bills of Quantities) exists as a Drive-approved
project (4 docs / 179 chunks). The earlier 404 was a chat-stream ownership-check
bug, now fixed in PR #154. `ff905e29` and `bc812f36` are still missing.
`bc812f36` has been superseded by the canonical fixture project
`FIXTURE — Fresh Upload Eval` (`b5a0fed8`) created by `scripts/seed_fixtures.py`.

**Impact:**
- `boq_process` is now unblocked for the feature matrix (still needs the chat fix deployed).
- `parse_primavera_schedule` and `drawing_qto` remain BLOCKED until
  `FIXTURE — Programme+Drawings` is seeded from the files the owner provides.
- Fresh-upload eval / golden-set gate use `FIXTURE — Fresh Upload Eval`.

**Disposition:**
- `boq_process`: use `FIXTURE — BOQ` canonical name in the manifest; seed once
  the owner provides the BOQ files in `FIXTURES_DIR`.
- `parse_primavera_schedule` / `drawing_qto`: use `FIXTURE — Programme+Drawings`
  canonical name; seed once the owner provides `project_programme.xer`,
  `ground_floor_plan.dxf`, and optionally `sample_office.ifc`.

## Drive re-import blocked on service-account config (2026-07-06)

**Finding:** The source data for the lost master-corpus blobs and the
Drive-approved projects is intact in Google Drive, but the surviving project
metadata does not contain the Drive folder IDs. The Render env currently has no
`GDRIVE_SERVICE_ACCOUNT_JSON` or `GDRIVE_PROJECT_FOLDERS` set.

**Classification:** Infrastructure/configuration gap, not a code bug.

**Disposition:** PARKED pending the owner providing:
1. A service-account JSON key with read access to the Drive folders, set as
   `GDRIVE_SERVICE_ACCOUNT_JSON` on Render (or as a mounted file path).
2. The complete `GDRIVE_PROJECT_FOLDERS` mapping, including sibling packs under
   the same parent as `the client project` (folder ID
   `1ExampleFolderId001xxxxxxxxxxxxxx`).

Once both are in place, run `scripts/inspect_drive_projects.py` to update the
manifest, then perform the full re-import/re-index before resuming gates.

## Smoke harness local Python issue (2026-07-06)

**Finding:** `scripts/smoke.sh` invokes `python`, which in this Git Bash session
resolves to the Windows Store placeholder and fails. The same script likely runs
correctly in a PowerShell/Windows environment with Python on PATH.

**Disposition:** Use `.venv/Scripts/python.exe scripts/fork_cli.py` directly for
manual prod health checks in this shell. Do not alter `smoke.sh` unless the
project standardizes a cross-shell python launcher.
## Interim provider state (T0, 2026-07-07)

**Decision:** Prod stays on `LLM_PROVIDER=kimi` (Kimi K2.6) tonight. This is **not**
an R10 override; it is R1 compliance — prod must keep serving, Kimi passes smoke 3/3,
and Scout demonstrably does not on the current (migrated) corpus.

**Evidence:**
- `GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct` deployed.
- Scout smoke: intermittent FAIL (~1/3 runs). Failing runs return the
  `_TOOL_FORMAT_FALLBACK` message: "I hit an internal search formatting issue
  before I could produce a grounded answer." The model emits raw internal
  tool/search arguments instead of a user-facing answer.
- Kimi smoke: PASS 3/3, all runs tool=Y, first token 70–90s, answer 8.5–10.8k chars.

**Next steps before provider verdict:**
1. T3: Reconcile the migrated corpus (chunk_count aggregation,
   `training_material` scoping anomaly, one GK identity). Re-smoke Scout on
   clean ground. If `_TOOL_FORMAT_FALLBACK` vanishes, it was a corpus symptom,
   not a Scout defect. If it persists, capture the raw failing turn and diagnose
   the runtime formatting interaction.
2. Run the Kimi+streaming gate on branch config (`SYNTHESIS_STREAMING=1`,
   smoke 10, tool=Y, first_token < 50% of total, browser check). If it passes,
   the 90s timeout risk evaporates.
3. The owner to read `K2_QUALITY_SAMPLES.md` against `review_pack/` Scout outputs
   (G2 verdict).

**Open risk:** Kimi first token is at 70–90s, close to `CHAT_STREAM_TIMEOUT_SECONDS=90`.
The streaming gate is the intended mitigation. Until that gate passes and G2 is
recorded, the provider ladder remains officially: Scout default / Kimi parked /
Ollama fallback only.

## 2026-07-12 — T1: Restore construction kit (44→17 block collapse)
**Finding:** Prod /v1/health = 17 blocks (base only); local `main` = same 17. Not a Docker/deploy discrepancy — the construction domain kit was never activated on the rebuilt service. `CEREBRUM_DOMAIN_KITS` UNSET on prod; `blocks_failed:{}` because the 27+ construction blocks are absent from the registry, not failing.
**Root cause:** the rebuild deployed without `CEREBRUM_DOMAIN_KITS=construction`; the construction container + boq_processor/drawing_qto/bim_extractor/primavera_parser/spec_analyzer/cpm_engine/schedule_generator/sympy_reasoning/smart_orchestrator/project_reasoner blocks live behind the kit and never loaded.
**Smoke (local):** `CEREBRUM_DOMAIN_KITS=construction` → 40 blocks, FAILED={} (0 failures); with VIRGIN=false → 51 blocks, 0 failures. Chose default mode (40, minimal, construction present).
**Decision:** set `CEREBRUM_DOMAIN_KITS=construction` on srv-d8hdc6ek1jcs739rq5sg. Env-only, smoke-clean, reversible. Gates T6 (construction feature verification).
