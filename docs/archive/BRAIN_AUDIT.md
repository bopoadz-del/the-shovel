# The_Fork — Deep Brain Audit (pilot-readiness)

2026-07-13. Static audit of every item in the "brain" — the project-assistant agent, the
orchestrator/predefined path, and every block/action they route to, plus RAG. Method: 8-agent
static fleet against a fixed rubric (what / how / wiring / real-vs-weak-vs-fake / tested /
verdict), followed by an empirical pass on the critical path against the client project oracles.

Verdicts: **PILOT-READY** · **NEEDS-WORK** · **WEAK** · **BROKEN/FAKE**.

---

## Executive verdict — how far from pilot

**The engine is real. The deliverable layer is not yet wired to it.**

- The **foundations are genuinely pilot-grade**: the CPM math, the RAG injection guards, the agent
  tool-loop + `allowed_blocks` gate, pgvector/HNSW/BM25/RRF storage, the sandbox, and the core BOQ
  arithmetic are real, correct, and well-tested. The codebase has a visibly honest culture — it
  *errors or nulls* rather than fabricating (repeated "no fabricated fallback" comments).
- The **#1 blocker is architectural**: the deterministic path that would produce **grounded**
  deliverables (the 21-action Scoped-Dispatch allowlist) is **dead code** — the live intent
  classifier can only name one workflow (`generate_wbs`). Every other deliverable request falls
  through to a **freeform LLM** that, for financial/contractual documents, **invents figures and
  self-certifies them** ("5-stage validation ✅, Confidence: High").
- A **handful of specific items are broken or fake** and must be fixed or quarantined before pilot:
  `as_built_deviation_report` (hardcoded stub driving a sign-off), `resource_histogram` (silent
  all-zero), `procurement_optimizer` (no supplier data source), `value_engineering` (fixed
  catalogue), plus two trivial UI key-bugs.

**Distance from pilot:** not "rebuild," but a focused **P0 layer**: (1) fold the *gated* deliverables
into the assistant as real tools (kills the dead-allowlist problem AND the fabrication problem at
once — this is the "orchestrator under the assistant" work), (2) add grounding verification so no
figure is self-certified, (3) fix/quarantine the ~5 broken/fake items, (4) close the RAG fusion seam
+ verify the GK knob. Everything else is P1/P2 polish.

---

## The keystone finding (read this first)

Live path (React app → `/v1/chat/stream`):
```
understand_intent  →  vocabulary = { "schedule" → generate_wbs }  ONLY
   ├─ "schedule"  → predefined bespoke WBS plan  (the ONE live predefined route)
   └─ everything else → None → project-assistant agent (freeform tool loop)
```
- `SCOPED_DISPATCH_ALLOWLIST` (21 deliverables), `_run_container_action`, `_synthesize_answer`
  (grounded, anti-hallucination prompt, graceful fallback) — **all built, all unit-tested, never
  executed in prod.** The tests call `run_workflow` directly, bypassing the classifier, so they pass
  while the feature is unreachable ("tests the unit, not the wire").
- Consequence: the container's **honest, gated** deliverable methods (payment cert errors without a
  contract value; cash-flow refuses to fabricate EV/AC; claims requires a real rate) are bypassed;
  the freeform LLM produces the deliverable from thin air instead.
- The flag (`ORCHESTRATOR_PREDEFINED`) is therefore **nearly moot** for these 20 deliverables — they
  are unreachable via predefined either way. (This corrects the earlier "flag on preserves
  deliverables" reasoning.)

**Fix = the orchestrator-under-assistant fold-in**: expose each *real* container deliverable as a
first-class assistant tool (like `generate_wbs`/`commissioning_checklist` already are), running the
gated container method + grounded synthesis. Reachable, honest, testable end-to-end.

---

## Full matrix

### Foundation — PILOT-READY (real, correct, tested)
| Item | Note |
|---|---|
| `pm_computations` (CPM math) | genuine forward/backward pass, real float, hand-verified tests |
| agent tool-loop + `allowed_blocks` gate | airtight hard-gate, best negative-path coverage in the codebase |
| RAG injection (`inject.py`) | real confidence threshold + identifier-miss gate + token budget; best-tested file |
| vector_store (pgvector/HNSW/BM25/RRF) | genuine ANN+BM25+RRF, HNSW guard, embedding-identity fail-loud |
| embeddings | dim probed at runtime, L2-norm, fail-loud; fake mode unreachable in prod |
| `boq_processor` arithmetic | real qty×rate parsing, never fabricates a total (the client project demo BOQ = SAR 62.2M) |
| `drawing_qto` text/title-block | real, tested against real the client project fixture PDFs |
| `formula_executor_v2` | real LLM codegen in a real RestrictedPython sandbox, well-tested |
| `procurement_list_generator` | real BOQ/quantity-driven, honest rate lookup (None not fabricated) |
| `recommendation_template` | real rule engine, honestly labeled |
| container calculators: `payment_certificate`, `variation_order_manager`, `cash_flow_forecast`, `risk_register`, `submittal_log` (action-level) | real math, correctly gated / honestly nulled |

### NEEDS-WORK (real but gaps)
| Item | Gap | Sev |
|---|---|---|
| `retriever.py` fusion | RRF computed then DISCARDED; `retrieve_with_filter` re-ranks on raw score → hybrid attenuated to ~semantic-only on the prod path | HIGH |
| GK contamination / merge | audit (repo config) says GK dominates top-1, knobs off; conflicts with the MCP-set live env — **verify empirically** | HIGH (verify) |
| `validation_pipeline` (as a gate) | auto-validation blind to `formula_executor_v2` output shape (the exact bug class it exists to catch); enforcement advisory-only, no code gate | HIGH |
| `generate_wbs` activities | template per project-type; **critical path is a tiebreak artifact** (`dur+z`); "Hall A/B" labels leak into road/sewer; durations ignore project scale | HIGH (trust) |
| `understand_intent` | `ORCHESTRATOR_INTENT_MODEL=glm-5.2:cloud` may mismatch live `LLM_PROVIDER` → silent-dark classifier; no metric distinguishes "none" from "broken" | HIGH (verify) |
| `search_project_documents` (tool) | no relevance floor (asymmetric with inject.py threshold) | Med |
| chat.py path-2 | no predefined re-dispatch (agents.py has it) — classification wasted | Med |
| `boq_processor` silent-zero | currency-suffixed rate → 0, undercount, no warning; `validate_boq` guard exists but WIRED NOWHERE | Med |
| `drawing_qto` DXF geometry | real entity math but ZERO tests + unfiltered layer sums + area×3m volume | HIGH (for DXF QTO) |
| `spec_analyzer` | real extraction + OCR, but ZERO extraction-fidelity tests | Med |
| `sympy_reasoning` | qty-variance real+wired; z-score/benchmark path orphaned (no data feeds it) | Med |
| `cash_flow_forecast` | real, but no dispatch path supplies `contract_value` → errors unless LLM extracts it | Med |
| `rfi_generator` | real+gated, but signal is thin (extraction-quality flags, not engineering clarifications) | Med |
| UI panels: risk_register, submittal_log | wrong dict key → correct count over EMPTY list; 1-line fix each | Med |

### WEAK / BROKEN / FAKE (fix or quarantine before pilot)
| Item | Verdict | Sev |
|---|---|---|
| Scoped-Dispatch allowlist (21 actions) | **DEAD CODE** — never runs in prod (keystone) | HIGH |
| `as_built_deviation_report` | **FAKE** — `_compare_as_built_to_design` returns hardcoded "Column grid A-1, 18mm" for every file pair, drives REJECTED/APPROVED sign-off | **CRITICAL** |
| Financial freeform fabrication | `payment_certificate` / `claims_builder` on the freeform path invent figures + self-certify "Confidence: High" | **HIGH** |
| `resource_histogram` | **BROKEN** — all-zero non-time-phased output on any real .xer, prints "balanced"; ZERO tests | HIGH |
| `procurement_optimizer` | decorative — nothing in the app ever supplies `suppliers` → empty plan, false "success"; common trigger | HIGH |
| `historical_benchmark` | misleading — no-op `record` returns success; no labour fields yet prompt promises man-hrs/m³ (manpower histogram broken); 3 configs say "removed" but it's live | HIGH |
| `value_engineering` | fixed catalogue of 4 substitutions with fixed % deltas | Med |
| `commissioning_checklist` | static hardcoded checklist; `spec_file` accepted, never read | Med |
| `change_order_impact` | keyword classifier + fixed markups | Med |
| `forensic_delay_analysis` | real XER diffing but hardcoded $5000/day; runtime broken (never got files) | Med |
| `esg_sustainability_report` | Social + Governance pillars are HARDCODED ZEROS (ignore inputs), shown as an A/B/C ESG rating | HIGH (if surfaced) |
| `tender_bid_analysis` | real scoring BUT silently substitutes 3 FAKE demo bids if <2 supplied, recommends award against them, no flag | HIGH |
| `daily_site_report` | voice extraction real; weather 100% hardcoded ("sunny 35°C" always); manpower/equipment/materials/quality = empty stubs | Med |
| `om_manual_generator` | FAKE/template — 7-item TBC demo list + boilerplate | Med |
| `carbon_footprint_calculator` | real calc, but UNREACHABLE via chat (dead wiring) | Med |
| `safety_compliance_audit` | real YOLO+regex pipeline, best-tested, but unreachable/errored at runtime | Med |
| `construction_v2` | already DELETED (PR #204) | — |

### Q&A / lookup path — PILOT-READY (all real)
`chat`, `process_document`, `process_contract`, `process_specification_full`, `spec_analyze`,
`benchmark_lookup`, `recommend`, `health_check`, `status` — all genuine delegation / real extraction,
no fabrication. The **read/answer path is solid**; the **generate/deliverable path is the problem.**

---

## Cross-cutting themes
1. **Deterministic/gated code is honest; the freeform LLM path fabricates.** The fix is to route
   deliverables to the gated methods (the fold-in), not to trust freeform synthesis.
2. **Grounding is prompt-only** everywhere except the RAG identifier-miss gate and the tool-loop
   `_auto_validate`. `_synthesize_answer` (the dead allowlist) and freeform narration have no
   post-hoc numeric-fidelity check — worst exactly where it matters (financial deliverables).
3. **Test coverage is inversely correlated with risk** — the broken/fake items (`resource_histogram`,
   `as_built`, DXF geometry, spec extraction) are the least tested.
4. **Config/comment drift** — dead classifiers, "removed" blocks still live, comments describing
   aspirational designs that don't match the wiring.

---

## Road to pilot (prioritized)

**P0 — blocks pilot**
1. **Orchestrator-under-assistant fold-in** — expose the *real* gated deliverables as assistant tools
   (cash_flow, payment_certificate, variation_order, risk_register, submittal_log, procurement_list,
   rfi, commissioning, forensic_delay) + prompt triggers; revive/replace the dead allowlist.
2. **Quarantine the fakes** — `as_built_deviation_report` (stub → error/flag), `resource_histogram`
   (fix or disable), `procurement_optimizer` (honest "no supplier DB" response), `value_engineering`
   (relabel as indicative). No fake output reaches a QS.
3. **Grounding gate on synthesis** — no figure self-certified; block/flag numbers the tool result
   didn't contain; stop the "Confidence: High" stamp on assumed values.

**P1 — quality**
4. Close the RAG fusion seam (rank on hybrid/RRF end-to-end) + **verify the GK knob live**.
5. Wire `validate_boq` into `boq_processor`; close the `validation_pipeline` formula_executor blind spot.
6. `generate_wbs`: disclose/replace the tiebreak-artifact critical path; fix zone labels for linear works.
7. `historical_benchmark`: reconcile the "removed" narrative; add real labour fields or stop promising them.
8. Two UI panel key-fixes (risk, submittal). Thread `contract_value` into cash_flow dispatch.

**P2 — polish**
9. Tests for the risk-correlated gaps (DXF geometry, spec extraction, resource_histogram).
10. `understand_intent` provider/model alignment + a silent-dark metric.

---

## Empirical pass (live the client project verification, 2026-07-13) — CORRECTS THE STATIC KEYSTONE

Live fork_cli runs against prod-config chat on `master_corpus`. Model live = **gpt-4o-mini**.

| Probe | Result | Correction to static audit |
|---|---|---|
| GK grounding — "which ASTM standards in the the client project spec?" | grounded in a REAL the client project spec doc (Vol 2 Specs Part 2, score 0.756), **not** the FIDIC GK note | GK contamination NOT dominating this query; grounding works |
| payment_certificate (no data) | `mode: predefined` → real gated method → **honest error** "needs contract_value" | **Allowlist is REACHABLE, not dead code.** No fabrication. |
| payment_certificate (WITH "contract_value 25000000, work_done 42%") | **still** honest error — the values were NOT extracted from the message into the action params | **Real gap #1: parameter extraction missing** — gated calculators can't compute even when given data |
| cash_flow_forecast | `mode: predefined` → **honest error** "contract value required" | reachable + honest; no freeform fabrication |
| as_built_deviation_report (no files) | `mode: predefined` → **"0 deviations, 100% conformance, APPROVED"** | **Real gap #2: fabricated success** — a false APPROVED sign-off with nothing compared (distinct from the file-path "Column grid A-1" stub) |
| esg_sustainability_report | `mode: predefined` → scored report, **"overall score 73.3"** | fabricates a composite score from the hardcoded-zero social/governance pillars, live |

### Keystone, CORRECTED
The static fleet concluded the Scoped-Dispatch allowlist was "dead code / deliverables fall through
to freeform fabrication." **Live testing refutes this**: deliverable requests DO route through
`mode: predefined` to the real gated container methods (`action_router.needs_planning` →
`_stream_from_predefined`, not `understand_intent`). The static read of the routing was stale.
The REAL problems are narrower and sharper:
1. **Parameter extraction is missing** — the predefined dispatch reaches the right method but never
   parses the user's stated figures (`contract_value`, `%`) into the action's params, so honest
   methods error even when the data was provided. This is the #1 usability blocker for the gated
   calculators.
2. **A subset of methods fabricate a fake "success"** instead of erroring — `as_built` (false
   APPROVED), `esg` (composite score over zeroed pillars), plus the static-confirmed fakes
   (`resource_histogram`, `tender_bid` demo bids, `om_manual` template, `daily_site` weather,
   `procurement_optimizer`, `value_engineering`, `commissioning`).

This makes the program MORE tractable: reachability is largely solved; the work is (a) wire param
extraction into predefined dispatch, and (b) convert every fabricating method to real-or-honest-error.
The full row-by-row live proof is Phase 3; the ledger (Phase 0b) is the master checklist.
