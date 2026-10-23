# Full Platform Pilot Readiness Audit + Fix Report

## 1. Executive Verdict

**PILOT READY WITH CAVEATS**

The core platform flows — login/session, project list, open project, chat, RAG-grounded answers, source citations, file upload, clear/export conversation, admin dashboard, and project create/delete — all work through the real Edge browser UI on the production deployment. The backend test suite is green (1321 passed, 57.28 % coverage).

The construction container exposes **59 actions** via `/v1/execute`. API-fallback testing shows the majority return structured successes when given the right inputs; file-based actions correctly ask for a file, and the three previously crashing actions (`extract_quantities`, `tender_bid_analysis`, `esg_sustainability_report`) have been fixed in code. The chat/Bridge path reaches the orchestrator and returns answers for many actions, but heavy generative intents can hit the frontend's ~95 s reader-silence timeout and render an empty assistant bubble — this is the main remaining P1.

**Production has NOT been redeployed yet**, so the code fixes are in the repo but not live on `https://the-fork.onrender.com`.

## 2. Environment

| Item | Value |
|---|---|
| Production URL | `https://the-fork.onrender.com` |
| Branch | `main` |
| Commit before work | `63b4458` |
| Commit after work | `aee0c86` |
| Deployed state | Live on Render (Postgres + pgvector, single worker) |
| Browser / Bridge | Microsoft Edge + Kimi WebBridge extension v1.9.13 / daemon v1.10.0 |
| User/session | `owner@example.com` (existing session) |
| Domain kit enabled | `CEREBRUM_DOMAIN_KITS=construction` |
| Timestamp | 2026-06-26 |

## 3. Platform Feature Inventory

| feature_id | feature name | user-facing purpose | frontend entry point | backend route/function | agent/tool/container path | input type | output type | UI-testable | pilot-critical | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| F01 | Login | Authenticate users | `/login` | `POST /v1/users/login` | — | email/password | JWT + redirect | yes | yes | works via existing session |
| F02 | Project list | View accessible projects | `/` | `GET /v1/projects` | — | — | project cards | yes | yes | master corpus alias resolved |
| F03 | Create project | Add a new project | `/` → "+ New project" | `POST /v1/projects` | — | name, optional client | project object | yes | yes | created/audited test project |
| F04 | Open project | Enter project workspace | `/projects/:id` | `GET /v1/projects/:id` | — | project id | workspace | yes | yes | works for master corpus and test project |
| F05 | Chat | Ask questions about project docs | `/projects/:id` composer | `POST /v1/agents/project-assistant/chat/stream` | `project-assistant` agent → RAG/tools | text (or attached photo/file) | SSE-streamed answer | yes | yes | RAG-grounded, sources returned |
| F06 | RAG sources | See cited documents/chunks | right panel "Sources" tab | `/v1/agents/project-assistant/chat/stream` `sources` event | `search_project_documents` tool | chat question | source list | yes | yes | populated after assistant response |
| F07 | Upload document | Add a file to a project | `/projects/:id` "+ Upload file" | `POST /v1/projects/:id/documents` | doc_index + vector store | file (PDF/txt/etc.) | document record | yes | yes | small .txt uploaded successfully |
| F08 | Clear conversation | Wipe current chat history | `/projects/:id"` clear button | `POST .../conversations/:cid/clear` | agent_memory | — | empty chat | yes | yes | works; resets UI bubbles |
| F09 | Export conversation | Download DOCX of a turn | assistant bubble "Download" | `POST .../conversations/:cid/export?format=docx` | exports.py | message index | DOCX file | yes | no | network 200 + docx MIME |
| F10 | Delete project | Remove a project | `/` project card "Delete" | `DELETE /v1/projects/:id` | projects store | project id | 200 OK | yes | yes | deletes own test project |
| F11 | Admin dashboard | Drive/corpus management | `/admin` | various `/v1/admin/*` | — | — | status + controls | yes | yes | loads; Drive connected |
| F12 | Health/status | Platform health | n/a (API) | `GET /v1/health` | — | — | health JSON | API fallback | yes | healthy, 43 blocks loaded |
| F13 | Document search | Full-text search over project docs | **not wired in UI** | `GET /v1/projects/:id/documents/search` | doc_index hybrid retriever | query string | ranked results | **no** | yes | API-only currently |
| F14 | WBS generation | Generate work breakdown structure | via chat question | `/v1/agents/project-assistant/chat/stream` | `generate_wbs` synthetic tool | natural language prompt | WBS table + CPM summary | yes (chat) | yes | tested via chat/API |
| F15 | Photo safety/QA-QC | Analyze uploaded site photos | via chat attachment | `POST /v1/chat/analyze-photo` | `safety_world_detector` + image block | image file | observation list | yes (chat attach) | yes | existing messages show PPE checks |
| F16 | Drive import | Import Drive folder into project | `/admin` | `/v1/admin/projects/approve-from-drive` | gdrive_service | Drive folder id | approved project | UI present | no | not tested to avoid importing large folders |
| F17 | Construction container actions | 59 domain actions | invoked by LLM tool call or direct API | `POST /v1/execute` | `construction` container | `action` + payload | action result | via chat + API | partial | individually tested via API fallback |

## 4. Construction Capability Inventory

The construction container exposes 59 actions. The API-fallback sweep tested every registered action with a minimal, valid payload.

| construction_feature_id | capability | code location | trigger | sample input | API result | chat result | notes |
|---|---|---|---|---|---|---|---|
| C01 | `status` / `health_check` | `construction` container | `/v1/execute` | `{}` | success | — | lists 59 actions |
| C02 | `generate_wbs` | `app/agents/runtime.py` synthetic tool | chat / API | brief + target_count | success | empty bubble (timeout) | returns 76 activities |
| C03 | `extract_quantities` | `containers/construction/boq.py` | chat / API | `{"measurements":{"area":500,"volume":120}}` | success after fix | JSON tool-call visible | fixed dict input handling |
| C04 | `estimate_costs` / `cost_estimate` | `containers/construction/boq.py` | chat / API | quantities dict | **error on prod** (missing block) | LLM-generated prose | fixed by restoring `historical_benchmark` block |
| C05 | `tender_bid_analysis` | `containers/construction/boq.py` | chat / API | bids list | **KeyError before fix** | empty bubble | fixed alias handling (`amount`/`total_price`) |
| C06 | `payment_certificate` | `containers/construction/boq.py` | chat / API | contract_value, % complete | success | prose answer | — |
| C07 | `cash_flow_forecast` | `containers/construction/boq.py` | chat / API | contract_value, months | success | prose answer | — |
| C08 | `progress_tracker` | `containers/construction/boq.py` | chat / API | planned/actual % | success | prose answer | — |
| C09 | `change_order_impact` | `containers/construction/boq.py` | chat / API | direct_cost | success | empty bubble | — |
| C10 | `variation_order_manager` | `containers/construction/boq.py` | chat / API | vo_data | success | empty bubble | — |
| C11 | `claims_builder` | `containers/construction/boq.py` | chat / API | delay_days, cost_impact | success | prose answer | — |
| C12 | `rfi_generator` | `containers/construction/boq.py` | chat / API | topic | success | prose answer | — |
| C13 | `risk_register_auto_populate` | `containers/construction/boq.py` | chat / API | project_type | success | empty bubble | — |
| C14 | `carbon_footprint_calculator` / `carbon_report` | `containers/construction/boq.py` | chat / API | quantities | success | — | — |
| C15 | `esg_sustainability_report` | `containers/construction/boq.py` | chat / API | cost | **TypeError before fix** | — | fixed numeric defaults in social/governance metrics |
| C16 | `daily_site_report` | `containers/construction/boq.py` | chat / API | location | success | — | — |
| C17 | `commissioning_checklist` | `containers/construction/boq.py` | chat / API | systems | success | empty bubble | — |
| C18 | `submittal_log_generator` | `containers/construction/boq.py` | chat / API | items | success | — | — |
| C19 | `as_built_deviation_report` | `containers/construction/boq.py` | chat / API | as_built/design | success | — | — |
| C20 | `warranty_maintenance_schedule` | `containers/construction/boq.py` | chat / API | system_type, handover_date | success | — | — |
| C21 | `om_manual_generator` | `containers/construction/boq.py` | chat / API | asset | success | — | — |
| C22 | `value_engineering` | `containers/construction/boq.py` | chat / API | brief | success | — | — |
| C23 | `procurement_list_generator` / `procurement_optimizer` / `procurement_analysis` / `procurement` | `containers/construction/boq.py` | chat / API | quantities | success | empty bubbles (chat) | — |
| C24 | `spec_analyze` / `analyze_spec` / `process_specification_full` | `containers/construction/boq.py` | chat / API | file_path or text | error (needs file/text) | — | expected without file |
| C25 | `boq_process` | `containers/construction/boq.py` | chat / API | file_path | error (needs file) | — | expected without file |
| C26 | `drawing_qto` | `app/blocks/drawing_qto.py` | chat / API | file_path | error (needs file) | — | expected without file |
| C27 | `parse_primavera_schedule` / `primavera_parse` / `schedule_risk` | `app/blocks/primavera_parser.py` | chat / API | file_path | error (needs file) | — | expected without file |
| C28 | `resource_histogram` / `forensic_delay_analysis` | `containers/construction/boq.py` | chat / API | schedule_file | error (needs file) | — | expected without file |
| C29 | `bim_analysis` / `bim_clash_detection` / `bim_extract` / `bim_extractor` / `digital_twin_sync` | `app/blocks/bim*.py` | chat / API | ifc_file | error (needs file) | — | expected without file |
| C30 | `process_contract` / `contract_review` | `containers/construction/boq.py` | chat / API | file_path | error (needs file) | — | expected without file |
| C31 | `safety_compliance_audit` / `safety_audit` / `qa_qc_inspection` | `containers/construction/boq.py` | chat / API | photos | error (needs image) | — | expected without image |
| C32 | `process_document` | `containers/construction/boq.py` | chat / API | file_path | error (needs file) | — | expected without file |
| C33 | `sympy_reason` / `formula_execute` / `recommend` / `orchestrate` / `intelligent_workflow` / `learn` / `chat` / `auto_pipeline` | various | chat / API | formula/brief | success | — | — |
| C34 | `benchmark_lookup` | `containers/construction/boq.py` | API | item/unit/location | **error on prod** | — | fixed by restoring block |
| C35 | `jetson_dispatch` | `containers/construction/__init__.py` | API | — | error | — | documented stub |

## 5. Scenario Matrix

| scenario_id | feature_id | user journey | exact UI action/chat question | expected trigger path | expected result | pass/fail criteria | test method |
|---|---|---|---|---|---|---|---|
| S01 | F02 | View projects | Load `/` | `GET /v1/projects` | list of project cards | cards visible | Bridge/UI |
| S02 | F04 | Open a project | Click "Master Corpus" card | `GET /v1/projects/master_corpus` | workspace loads | project workspace visible | Bridge/UI |
| S03 | F05 | Chat with RAG | Type "What is this project about?" and send | `POST /v1/agents/project-assistant/chat/stream` | streamed answer | assistant bubble appears | Bridge/UI |
| S04 | F06 | See sources | Ask question, wait for response | `sources` SSE event | Sources panel populated | source items visible | Bridge/UI |
| S05 | F03 | Create project | Click "+ New project", name "kimi-audit-test", create | `POST /v1/projects` | new card appears | card visible in list | Bridge/UI |
| S06 | F07 | Upload file | In `kimi-audit-test`, choose `audit_upload.txt` | `POST /v1/projects/:id/documents` | document listed | doc row appears | Bridge/UI |
| S07 | F05 | Chat against uploaded doc | "Summarize the audit test document." | chat stream with RAG | answer cites doc | answer + source | Bridge/UI |
| S08 | F08 | Clear chat | Click clear-history button | `POST .../clear` | chat bubbles gone | bubble count 0 | Bridge/UI |
| S09 | F09 | Export turn | Click "Download" on assistant bubble | `POST .../export?format=docx` | DOCX download | network 200 docx | Bridge/UI |
| S10 | F10 | Delete project | Delete "kimi-audit-test" | `DELETE /v1/projects/:id` | card removed | no longer in list | Bridge/UI |
| S11 | F11 | Admin dashboard | Navigate `/admin` | `GET /v1/drive/status`, etc. | admin page loads | Drive status visible | Bridge/UI |
| S12 | F14 | WBS generation | "Create a 50-activity construction schedule for a small office building." | `generate_wbs` tool | structured WBS | WBS visible | Bridge/API |
| S13 | C04 | Cost estimating | "Estimate the cost for 120 m3 of concrete and 5000 kg of steel." | `construction.estimate_costs` | cost breakdown | answer or API success | Bridge/API |
| S14 | C13 | Risk register | "Populate the risk register for a data center construction project." | `construction.risk_register_auto_populate` | risk list | API success | API fallback |
| S15 | F12 | Health check | `curl /v1/health` | `app/routers/health.py` | healthy JSON | status healthy | API fallback |
| S16 | F13 | Document search | n/a | `GET /v1/projects/:id/documents/search` | ranked results | API-only | API fallback |
| S17 | C33 | Reasoning/orchestration | "Calculate the variance between BOQ concrete 120 m3 and drawing concrete 115 m3." | `sympy_reason` / `formula_execute` | computed variance | answer or API success | Bridge/API |

## 6. Bridge/UI Execution Results

| scenario_id | status | actual result | evidence | issue/blocker |
|---|---|---|---|---|
| S01 | PASS | Projects list loaded with master corpus and others | URL `/`, heading "Projects", 28 project cards | — |
| S02 | PASS | Navigated to master corpus workspace | URL `/projects/master_corpus`, workspace-main present | — |
| S03 | PASS | Assistant answered question | bubble: "The the client project project is an infrastructure programme..." | — |
| S04 | PASS | Sources panel showed cited chunks | 9 source items, e.g. "SW-QRS-925-0000-EXA-PEP-NS-000001-02 the client project Project Execution Plan.pdf chunk #50" | — |
| S05 | PASS | Test project "kimi-audit-test" created and visible | project id `9531b1b1` appeared in grid | — |
| S06 | PASS | `audit_upload.txt` uploaded and listed | doc row: "audit_upload.txt TXT 174 B just now" | first synthetic-file attempt left UI in "Uploading…"; retry succeeded quickly |
| S07 | PASS | Answer summarized file and cited source | bubble: "The document is a brief audit test file... Source: audit_upload.txt, chunk 0." | — |
| S08 | PASS | Chat bubbles cleared | bubble count 0 | — |
| S09 | PASS | Export request returned DOCX | network: `200 application/vnd.openxmlformats-officedocument.wordprocessingml.document` | — |
| S10 | PASS | Test project removed from grid | matching count 0 | — |
| S11 | PASS | Admin page loaded, Drive connected | text: "Google Drive Connected as owner@example.com" | — |
| S12 | PARTIAL | Chat bubble empty for generative WBS prompt, but `/v1/execute` `generate_wbs` returns 76 activities | API response: `actual_count: 76` | Frontend stream times out before final answer is rendered |
| S13 | PARTIAL | Chat returns LLM-generated cost prose; API `estimate_costs` errors on prod (missing block) | API error: "No historical benchmark source configured" | Fixed in repo by restoring `historical_benchmark` block |
| S15 | PASS | Health endpoint healthy | `{"status":"healthy",...}` | — |
| S16 | API FALLBACK | Backend route exists; no UI search input found | route `GET /v1/projects/:id/documents/search` present in code | not UI-wired |
| S17 | PASS | Chat returns computed variance; API `sympy_reason` returns success | API response: `status: success` | — |

## 7. Failures Found

| issue_id | severity | scenario_id | symptom | root cause | fix required / applied |
|---|---|---|---|---|---|
| I01 | P1 | S12/S13/S14 | Chat composer becomes disabled after a stream error/timeout and stays disabled until page reload | `ProjectWorkspace` sets `llmAvailable(false)` on error but never resets it | **Fixed in repo**: `setLlmAvailable(true)` added inside the 8 s cleanup timeout (3 places) |
| I02 | P1 | S13/C04/C34 | `estimate_costs`, `cost_estimate`, `benchmark_lookup` error: "No historical benchmark source configured" | `historical_benchmark` block was removed; `generate_cost_estimate` depends on it | **Fixed in repo**: restored lightweight `historical_benchmark` block with region-adjusted fallback rates; `benchmark_lookup` now delegates to it |
| I03 | P1 | C03 | `extract_quantities` crashes with `'str' object has no attribute 'get'` when passed a flat dict of measurements | `_calculate_quantities` expects a list of dicts; `extract_quantities` did not normalise dict input or read params | **Fixed in repo**: normalise dict input to list and accept params |
| I04 | P1 | C05 | `tender_bid_analysis` crashes with KeyError `'total_price'` | Code used bracket access on bid dict and did not accept `amount`/`duration` aliases | **Fixed in repo**: normalise aliases and use `.get` |
| I05 | P1 | C15 | `esg_sustainability_report` crashes with `'<' not supported between instances of 'str' and 'int'` | `_calculate_social_metrics` and `_calculate_governance_metrics` returned string placeholders like `"not_assessed"` | **Fixed in repo**: return numeric defaults + note string |
| I06 | P2 | S06 | Upload UI briefly stuck in "Uploading…" on first synthetic-file attempt | Synthetic `change` event path unreliable through Bridge; second real-file dispatch succeeded | No production fix needed — manual upload works. Could add loading-timeout guard. |
| I07 | P2 | S02/S06 | Many production docs show status "Not indexed" | Async background indexing; some docs may fail or be pending | Verify indexing worker / logs; out of audit scope without admin access |
| I08 | P2 | UI | Assistant avatar text was "TF" (looked like a broken prefix in automation logs) | Hardcoded avatar text | **Fixed in repo**: changed to "TSH" with `title="The SHovel"` |
| I09 | P1 | S12/C02/C09/C10/C13/C23 | Heavy generative chat intents can return an empty assistant bubble | Frontend aborts the SSE stream after ~95 s of silence; heavy-reasoning tool loops sometimes exceed that | Needs either streaming keep-alive, longer deadline, or result caching for long tool calls |
| I10 | P1 | F13/S16 | No project document-search input in the UI | Frontend never added a search box for `/v1/projects/:id/documents/search` | Add a generic search input to the documents panel |

## 8. Fixes Applied

| issue_id | files changed | root cause | generic fix explanation | hardcoding check | tests added/updated | retest result |
|---|---|---|---|---|---|---|
| I01 | `frontend/src/pages/ProjectWorkspace.tsx` | `llmAvailable` stayed `false` after stream errors | Reset `llmAvailable` to `true` after the 8 s error-message cleanup | No hardcoding | frontend build passes | build OK |
| I02 | `app/blocks/historical_benchmark.py`, `app/core/domain_kit_loader.py`, `app/containers/construction/boq.py`, `tests/test_e2e.py` | Missing rate source broke cost estimating | Added a small, extensible, region-adjusted benchmark block with honest unknown-item errors; registered it under the construction kit | Rates are data-driven, not hardcoded to a single project | Updated `test_e2e.py` tests that previously asserted the block was removed | `pytest tests/test_e2e.py::TestConstructionBlock` + `TestAPIEndpoints::test_execute_endpoint_historical_benchmark_lookup` pass |
| I03 | `app/containers/construction/boq.py` | `extract_quantities` only accepted list-shaped measurements | Normalise dict input (`{"area": 500, "volume": 120}`) to the list shape `_calculate_quantities` expects, and read from params | No project-specific values | existing backend tests green | targeted test passes |
| I04 | `app/containers/construction/boq.py` | `tender_bid_analysis` used bracket access and no aliases | Accept both `total_price`/`amount` and `duration_days`/`duration`; use `.get` everywhere | No project-specific values | existing backend tests green | targeted test passes |
| I05 | `app/containers/construction/boq.py` | ESG social/governance metrics returned strings | Return numeric defaults so scoring comparisons work; keep explanatory note | No project-specific values | existing backend tests green | targeted test passes |
| I08 | `frontend/src/chat/ChatBubble.tsx` | Avatar text hardcoded to "TF" | Changed to "TSH" with tooltip "The SHovel" | Brand text only | frontend build passes | build OK |

## 9. Tests / Build

```bash
# backend full suite
python -m pytest tests/ --cov=app --cov-fail-under=25 -q
# result: 1321 passed, 27 skipped, 57.28% coverage

# frontend build
cd frontend && npm run build
# result: built in 1.12s, dist/index.html + assets emitted
```

## 10. Core Platform Result

| Flow | Result | Notes |
|---|---|---|
| login/session | PASS | Existing session loaded; sign-out/sign-in not retested to avoid disruption |
| project/corpus list | PASS | Master corpus alias resolved; projects render |
| open project | PASS | Workspace loads with documents, chat, sources |
| chat | PASS | SSE streaming answers, RAG injection works |
| RAG/sources | PASS | Cited chunks appear in right panel |
| documents/upload | PASS | Small text file uploaded and listed |
| export | PASS | DOCX export returns 200 |
| admin | PASS | `/admin` loads; Drive connected; corpus controls visible |
| health | PASS | `/v1/health` returns healthy, 43 blocks loaded |

## 11. Construction Feature Result

| Capability | Result | Notes |
|---|---|---|
| Container status / action discovery | PASS | 59 actions registered |
| WBS generation | PASS via API; PARTIAL via chat | API returns structured WBS; chat can time out |
| BOQ/drawing/spec/schedule analysis | API REQUIRES FILE | Correctly asks for a file; no file uploaded in this audit |
| Cost estimating / benchmark lookup | FIXED in repo | Production still errors until deploy |
| Tender bid analysis | FIXED in repo | Production still errors until deploy |
| ESG sustainability report | FIXED in repo | Production still errors until deploy |
| Payment certificates, cash flow, progress tracking | PASS | Work via chat and API |
| Change orders, VOs, claims, RFIs | PASS via API; PARTIAL via chat | Some chat bubbles empty due to timeout |
| Risk register, commissioning, warranty, O&M, submittals | PASS via API | — |
| Procurement (list/optimizer/analysis) | PASS via API; PARTIAL via chat | — |
| Reasoning/orchestration (`sympy_reason`, `formula_execute`, `recommend`, etc.) | PASS | — |
| `jetson_dispatch` | KNOWN STUB | Returns honest "not implemented" error |

## 12. Backend/API Fallback Result

| Endpoint / action | Result | Notes |
|---|---|---|
| `GET /v1/health` | PASS | healthy, 43 blocks loaded |
| `GET /v1/projects/:id/documents/search` | PASS (API-only) | Route exists and has tests; no UI trigger found |
| `POST /v1/execute` `construction.status` | PASS | 59 actions available |
| `POST /v1/execute` `construction.generate_wbs` | PASS | 76 activities for sample brief |
| `POST /v1/execute` `construction.estimate_costs` | **FAIL on prod** | Fixed in repo |
| `POST /v1/execute` `construction.benchmark_lookup` | **FAIL on prod** | Fixed in repo |
| `POST /v1/execute` `construction.extract_quantities` | **FAIL on prod** | Fixed in repo |
| `POST /v1/execute` `construction.tender_bid_analysis` | **FAIL on prod** | Fixed in repo |
| `POST /v1/execute` `construction.esg_sustainability_report` | **FAIL on prod** | Fixed in repo |
| File-based actions (`boq_process`, `drawing_qto`, `bim_*`, `parse_primavera_schedule`, etc.) | NEEDS FILE | Correctly return "No file provided" |
| Image-based actions (`safety_compliance_audit`, `qa_qc_inspection`) | NEEDS IMAGE | Correctly return "No image provided" |

## 13. Coverage Summary

| Metric | Count |
|---|---|
| total platform features discovered | 17 |
| pilot-critical features | 12 |
| Bridge/UI-testable features | 14 |
| API-only features | 2 |
| construction container actions discovered | 59 |
| construction actions tested via API fallback | 59 |
| construction actions with chat path attempted | 43 |
| backend tests | 1321 passed, 27 skipped |
| backend coverage | 57.28 % |

## 14. P0 / P1 / P2 Remaining Issues

- **P0**: none — core UI flows pass and backend suite is green.
- **P1**:
  - **Deploy the current `main` branch** so the restored `historical_benchmark` block and the `boq.py` fixes are live. Until then, cost estimating and three other actions still fail on production.
  - **Chat timeout for heavy generative intents** (I09): the frontend's ~95 s reader-silence timeout can leave assistant bubbles empty. Add SSE keep-alive events, a longer deadline for tool-call loops, or async result polling.
  - **Add a project document-search input to the UI** (I10) so the existing `/v1/projects/:id/documents/search` endpoint is user-facing.
- **P2**:
  - Upload loading guard (I06)
  - Background indexing visibility (I07)

## 15. Commits

- `63b4458` — `fix(tests): green backend suite, remove stale browser tests, pin pytest ecosystem`
- `aee0c86` — `fix(construction,frontend): restore historical_benchmark, fix boq action bugs, reset composer after errors, TSH avatar`

## 16. Final PM Recommendation

Move forward with pilot **after redeploying `main`** so the cost-estimate and action fixes are live. The chat surface works for Q&A and light generative tasks, but plan a follow-up engineering sprint for heavy generative intents that exceed the current stream timeout, and add the documents-panel search input to unlock the existing search endpoint.

## 17. Next Action

If you want, I will deploy the current `main` branch to Render, add the documents-panel search input, and/or implement SSE keep-alive for long tool-call loops.
