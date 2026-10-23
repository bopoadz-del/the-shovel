# Routing Matrix — intent × threshold × path (W5)

The single committed source of truth for how a chat message is routed. The logic
lives in `app/core/action_router.py`; this table makes it reviewable and is
locked by `tests/test_routing_matrix.py` (every row is asserted).

## Decision tree

1. `SmartOrchestratorBlock` classifies the message → `matched_actions`
   `[{action, confidence, keywords_matched}]` (descending confidence).
2. `best_action(result)` → top `(action, confidence)`.
3. **Planning path** (heavy-reasoning / predefined) iff
   `needs_planning(action, confidence)` = `action ∈ GENERATIVE_INTENTS AND
   confidence ≥ 0.2` (`GENERATIVE_ROUTING_THRESHOLD`).
4. Else **fast chat path**, with a domain hint when `confidence ≥ 0.4`
   (`HINT_CONFIDENCE_THRESHOLD`) and the action has a registered hint.
5. Else **fast chat path**, no hint.

The two-gate design: the low 0.2 gate lets sparse generative phrasings ("generate
a WBS for a 10-floor tower", one keyword = 0.2) reach planning, while the
`GENERATIVE_INTENTS` whitelist is the safety — a non-generative action stays on
the fast path even at high confidence.

## Flag reconciliation (`ORCHESTRATOR_PREDEFINED`)

- **ON (`=1`, current prod):** planning-path messages are served by the
  predefined orchestrator (`_stream_from_predefined`, deterministic container
  actions + synthesis).
- **OFF:** the same intents route to the heavy-reasoning **agent tool loop**
  instead — no predefined dispatch. `needs_planning` is unchanged either way (the
  flag chooses the *executor*, not *whether* to plan). Locked by the
  noop-when-off test.

## Generative intents → planning path (confidence ≥ 0.2)

| Domain | Action | Typical file / trigger |
|--------|--------|------------------------|
| Schedule | `generate_wbs` | brief / project params (no file) |
| Schedule | `parse_primavera_schedule` | `.xer` / `.xml` (P6) |
| Schedule | `forensic_delay_analysis` | baseline + updated `.xer` |
| Schedule | `resource_histogram` | `.xer` with resources |
| Cost | `estimate_costs`, `cash_flow_forecast` | BOQ xlsx / brief |
| Cost | `procurement_list_generator`, `procurement_optimizer` | BOQ / spec |
| Cost | `payment_certificate` | valuation inputs |
| Specs | `process_specification_full` | spec pdf/docx |
| BIM | `bim_analysis`, `bim_clash_detection`, `bim_extractor` | IFC |
| Drawings | `drawing_qto` | DXF / vector PDF |
| Contracts | `claims_builder`, `rfi_generator`, `value_engineering` | contract docs |
| Contracts | `change_order_impact`, `variation_order_manager` | VO / contract |
| Generic | `intelligent_workflow`, `sympy_reason` | multi-step / variance |

## Fast path + hint (non-generative, confidence ≥ 0.4)

Actions with a registered `ACTION_HINTS` entry but NOT in `GENERATIVE_INTENTS`
stay single-shot with a domain hint — e.g. `boq_process`, `spec_analyze`,
`qa_qc_inspection`, `commissioning_checklist`, `progress_tracker`,
`safety_compliance_audit`, `risk_register_auto_populate`,
`carbon_footprint_calculator`. These answer in one LLM call (sub-2s); they don't
need the tool loop.

## The historical misses (context)

DECISIONS.md (TASK G, 2026-07-06) logged 20 routing misses at confidence 0.0 —
messages the keyword matcher didn't score at all, so they fell to the fast path.
That is a *recall* gap in the orchestrator's keyword vocabulary, separate from
this matrix (which governs what happens *once* an action is matched). The
vocabulary patch (`test_router_vocabulary_patch.py`) addressed a batch; residual
zero-confidence phrasings degrade safely to grounded Q&A, never to a wrong tool.
