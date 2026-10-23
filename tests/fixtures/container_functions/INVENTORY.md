# Construction container actions — inventory

Synthetic-only. No live project names, figures, or document text.

Source of truth: `ConstructionContainer.get_actions()` plus `route()` handlers.
Dropped from the real-action list: `health_check`, `status` (metadata).
Aliased names that call through a real handler are noted, not counted twice.

## Chat reachability (runtime.py)

Agents with `construction` in `allowed_blocks` get:

1. Generic tool `construction` — `{input, params: {action, …}}` for every
   container action.
2. Own typed tools (same agents): `generate_wbs`, `commissioning_checklist`,
   `cash_flow_forecast`, `resource_histogram`, `look_ahead`, `evm_calculate`,
   plus `construction_calc` (owned by another agent).
3. Predispatch (runtime intercept, no model tool pick): `look_ahead`,
   `generate_wbs`, `cash_flow_forecast`, `payment_certificate`,
   `rfi_generator`, `job_requisition`, `safety_briefing`,
   `commissioning_checklist`, `claims_builder`, `om_manual_generator`,
   `as_built_deviation_report`, `rfp_draft`, `wir_form`,
   `resource_histogram`.

Standalone block tools (not the container action name) when the agent
lists that block: `drawing_qto`, `boq_processor`, `primavera_parser`,
`spec_analyzer`, `bim_extractor`.

## Agents with `construction`

construction-pm, quantity-surveyor, contracts-manager, project-assistant,
bim-analyst, smart-orchestrator, document-analyst, safety-officer,
heavy-reasoning, supervision-proposal, validation.

Agents without the container (cannot call generic `construction`):
self-coding, external-mcp, learning, document-ingestion
(document-ingestion still has drawing_qto / boq_processor / primavera_parser /
spec_analyzer as own blocks).

## Aliases that call through (not separate actions)

| Alias | Calls |
|---|---|
| inspection_request | wir_form |
| rfp_management | rfp_draft |
| bim_extractor | bim_extract |
| contract_review | process_contract |
| safety_audit | safety_compliance_audit |
| carbon_report | generate_carbon_report (via carbon_footprint_calculator) |
| procurement | procurement_analysis |
| cost_estimate | generate_cost_estimate |
| analyze_spec | analyze_spec_section |
| schedule_risk | analyze_schedule_risk |
| payment_cert / generate_pay_app | payment_certificate (cm_step_aliases) |
| procurement_plan / link_procurement | procurement_list_generator |
| earned_value / cost_variance | evm_calculate |
| rfi_generation / generate_rfi | rfi_generator |
| primavera_parse | PrimaveraParserBlock (twin of parse_primavera_schedule) |

Not owned here: `construction_calc`, `formula_execute`, `sympy_reason`.

## Action table

Columns: chat reach / agents / det vs LLM / needs uploaded file.

| Action | Chat reach | Agents | Det/LLM | File? |
|---|---|---|---|---|
| look_ahead | own tool + generic + predispatch | construction-capable | det | yes (.xer) |
| procurement_list_generator | generic | construction-capable; QS prompt names it | det | no (BOQ/qty in payload) |
| drawing_qto | own block tool + generic action + file predispatch | any with drawing_qto or construction | det (block) | yes (dxf/dwg/pdf) |
| variation_order_manager | generic | construction-capable | det | optional contract file |
| payment_certificate | generic + IPC predispatch | construction-capable; contracts-manager prompt | det | no (figures) |
| generate_wbs | own tool + generic + predispatch | construction-capable; heavy-reasoning prompt | det | no |
| cash_flow_forecast | own tool + generic + predispatch | construction-capable; PM prompt | det | optional .xer |
| rfi_generator | generic + predispatch | construction-capable | det | no (issues or message) |
| evm_calculate | own tool + generic | construction-capable | det | no (PV/EV/AC) |
| boq_process | generic; chat also `boq_processor` block | construction-capable + QS/PM/ingestion | det (block) | yes (xlsx/csv/pdf) |
| parse_primavera_schedule | generic; chat also `primavera_parser` | construction-capable + PM/orchestrator | det | yes (.xer/.xml) |
| primavera_parse | generic (block delegate) | same | det | yes (.xer) |
| extract_quantities | generic | construction-capable | det | no (measurements list) |
| estimate_costs | generic | construction-capable | det | no (quantities/BOQ) |
| progress_tracker | generic | construction-capable | det | no |
| commissioning_checklist | own tool + generic + predispatch | construction-capable | det (templates) | optional spec |
| resource_histogram | own tool + generic + predispatch | construction-capable | det | yes (.xer) |
| claims_builder | generic + predispatch | construction-capable | det | optional schedule/contract |
| change_order_impact | generic | construction-capable; contracts-manager | det | no |
| tender_bid_analysis | generic | construction-capable | det | no |
| forensic_delay_analysis | generic | construction-capable | det | yes (two .xer) |
| warranty_maintenance_schedule | generic | construction-capable | det | no |
| submittal_log_generator | generic | construction-capable | det | no |
| risk_register_auto_populate | generic | construction-capable; safety-officer | det | no |
| procurement_optimizer | generic | construction-capable | det | no |
| procurement_analysis | generic | construction-capable | det | no |
| evm twin progress_tracker | generic | — | det | no |
| wir_form | generic + predispatch | construction-capable | det | no |
| job_requisition | generic + predispatch | construction-capable | det | no |
| safety_briefing | generic + predispatch | construction-capable | det | no |
| rfp_draft | generic + predispatch | construction-capable | det | no |
| qa_qc_inspection | generic | construction-capable | det + optional photo model | optional photos |
| process_document | generic | construction-capable | det (dispatch by type) | yes |
| process_contract | generic | construction-capable; contracts-manager | det + optional LLM extract | yes or text |
| process_specification_full | generic | construction-capable | det (spec_analyzer) | yes |
| spec_analyze | generic; own `spec_analyzer` block | construction-capable | det (block) | yes or text |
| bim_analysis | generic | construction-capable; bim-analyst | det (bim block) | yes (.ifc) |
| bim_extract | generic; own `bim_extractor` | construction-capable; bim-analyst | det | yes (.ifc) |
| bim_clash_detection | generic | construction-capable; bim-analyst | det | yes (.ifc) |
| as_built_deviation_report | generic + predispatch | construction-capable | det | optional drawings |
| carbon_footprint_calculator | generic | construction-capable | det | no (BOQ/qty) |
| safety_compliance_audit | generic | construction-capable; safety-officer | det | optional text/file |
| esg_sustainability_report | generic | construction-capable; safety-officer | det | no |
| om_manual_generator | generic + predispatch | construction-capable | det | no |
| digital_twin_sync | generic | construction-capable | NEEDS-EXTERNAL | yes / live sync |
| cde_post_rfi | generic | construction-capable | NEEDS-EXTERNAL (CDE) | no |
| cde_poll_events | generic | construction-capable | NEEDS-EXTERNAL (CDE) | no |
| intelligent_workflow | generic | construction-capable | det router | optional file |
| auto_pipeline | generic | construction-capable; PM/doc-analyst | det chain | yes |
| jetson_dispatch | generic | construction-capable | det validate; pending hardware | no |
| daily_site_report | generic | construction-capable | det + optional voice/ocr | optional |
| value_engineering | generic | construction-capable | det | no |
| track_progress | route-only (not in get_actions) | construction-capable via route | det | photos + bim |
| extract_measurements | route-only | via route | det | drawing |
| generate_construction_report | route-only | via route | det | yes |
| chat | generic | construction-capable | LLM | no |
| orchestrate | generic | construction-capable | det keyword router | no |
| learn | generic | construction-capable | det (learning_engine) | no |
| recommend | generic | construction-capable | det (recommendation_template) | no |
| benchmark_lookup | generic | construction-capable | det (historical_benchmark) | no |
| construction_calc | own tool | construction-capable | det | no — other agent |
| formula_execute | generic | construction-capable | LLM codegen — other agent | no |
| sympy_reason | generic | construction-capable | det/LLM — other agent | no |

construction-capable = the 11 agents listed above.

## Phase 1 verdicts (ConstructionContainer direct)

Central fail = `status=success` without a deliverable, or silent invention.
`construction_calc` / `formula_execute` / `sympy_reason` are not owned here.

| Action | Verdict | Note |
|---|---|---|
| look_ahead | WORKS | inclusive 21-day window; missing/.xer/0 days error; skip when primavera_parser unavailable |
| procurement_list_generator | WORKS | empty error; 140_000 / 16-week steel |
| drawing_qto | WORKS | missing/wrong-type error; 10×5 DXF = 50 m²; skip when drawing_qto/ezdxf unavailable |
| variation_order_manager | WORKS | no invented Clause XX; "Draft a variation order" without facts is error |
| payment_certificate | WORKS | IPC arithmetic |
| generate_wbs | WORKS | building brief delivers tree; empty brief left as existing contract |
| cash_flow_forecast | WORKS | S-curve month 1 |
| rfi_generator | WORKS | empty error |
| evm_calculate | WORKS | SPI/CPI hand-derived |
| boq_process | WORKS | CSV line total; skip when boq_processor unavailable |
| parse_primavera_schedule / primavera_parse | WORKS | 3-activity XER / missing file; skip when primavera_parser unavailable |
| extract_quantities | WORKS | empty error |
| estimate_costs / cost_estimate | WORKS | empty error |
| progress_tracker | WORKS | empty error; proxy SPI |
| commissioning_checklist | WORKS | named system; default pack left as existing contract |
| resource_histogram | WORKS | 200+400+100 = 700 h TASKRSRC |
| claims_builder | WORKS | empty error; 35_000 event_sum |
| change_order_impact | WORKS | 50_000 → 67_500 |
| tender_bid_analysis | WORKS | lowest 950_000 |
| forensic_delay_analysis | WORKS | missing schedules error |
| warranty_maintenance_schedule | WORKS | empty error; 24×30 days expiry |
| submittal_log_generator | WORKS | empty error; no invented QA/QC Plan |
| risk_register_auto_populate | WORKS | empty error; scores 56.0 / 20.0 |
| procurement_optimizer | WORKS | empty does not rank ghosts |
| procurement_analysis | WORKS | empty errors at list_generation |
| wir_form / inspection_request | WORKS | alias reaches wir |
| job_requisition | WORKS | empty error; lighting title only if text matches |
| safety_briefing | WORKS | empty error |
| rfp_draft | WORKS | empty error |
| qa_qc_inspection | WORKS | empty errors or asks |
| process_document / process_contract / process_specification_full | WORKS | missing input error |
| spec_analyze / analyze_spec | WORKS | missing input error |
| bim_analysis / bim_extract / bim_clash_detection | WORKS | missing IFC error |
| as_built_deviation_report | WORKS | empty error; 10.05 vs 10.0 major |
| carbon_footprint_calculator | WORKS | empty error; 12×250 + 2000×2.3 = 7600 |
| safety_compliance_audit | WORKS | empty does not invent findings |
| esg_sustainability_report | WORKS | BOQ rows score |
| om_manual_generator | WORKS | empty / systems-only error; equipment_list delivers tag |
| digital_twin_sync | NEEDS-EXTERNAL | prepared_not_pushed (existing honesty) |
| cde_post_rfi / cde_poll_events | NEEDS-EXTERNAL | no config → error |
| jetson_dispatch | NEEDS-EXTERNAL | pending_hardware |
| auto_pipeline | WORKS | missing file error |
| intelligent_workflow | — | not yet a deliverable case |
| daily_site_report | WORKS | date-only does not invent weather/headcount |
| value_engineering | WORKS | empty error; conservative 6_500 |
| track_progress | WORKS | empty error (route-only) |
| extract_measurements | WORKS | empty error (route-only) |
| generate_construction_report | WORKS | empty error (route-only) |
| schedule_risk | WORKS | empty file error (alias → parse) |
| chat / orchestrate / learn / recommend / benchmark_lookup | WORKS | empty is error or empty-honest |

## Phase 2

Live chat probes need `FORK_TOKEN`. If unset, Phase 2 is blocked.
Project name if run: `FIXTURE-containers-<date>` (never master_corpus).
Conversation ids: `ctr-<action>-<epoch>`.
