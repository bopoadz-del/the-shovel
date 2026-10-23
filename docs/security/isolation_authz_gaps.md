# Isolation authz gaps (Step 0 — UNPROVEN vs authz matrix sweep)

Date: 2026-09-19. Reviewed against `666e223`, then F-doors against `60c8055`. Mapping only unless a later section says a cell was proven.
Do not call the platform ready or secure from this list.

## Agent F privileged doors (added on `60c8055`, not a live probe)

These are **not** in the four-cell authz MATRIX sweep. They are covered by
`tests/test_no_agent_hands_a_user_the_server.py` and the two-user cells in
`tests/test_no_user_reaches_another_users_data.py`. Live confirm from two
browser accounts waits for `theshovel.ai` `build_sha >= 60c8055`.

| Surface | Plain user A | Plain user B |
| --- | --- | --- |
| `POST /v1/execute` `{block: mcp_consumer\|local_drive\|web\|webhook\|google_drive\|onedrive\|code\|sandbox}` | 403 | 403 |
| `GET /v1/agents/{self-coding\|external-mcp\|document-ingestion}` | 404 | 404 |
| `POST /v1/agents/{name}/chat[/stream]` for those names | 404 | 404 |
| `POST /v1/chat/stream` `"agent"` pin to those names | SSE error `not available` | same |

A new `app/blocks/*.py` that starts a process or reaches the network fails
CI unless the block `name` is in `PRIVILEGED_BLOCKS` or the reviewed
allowlist next to `test_every_block_that_starts_a_process_is_privileged_or_reviewed`.

## Diff

- Router routes enumerated: **156**
- Covered by authz matrix sweep: **4**
- **UNPROVEN (missing from sweep): 152**
- Of UNPROVEN, other tenancy/isolation tests exist: **17**
- Of UNPROVEN, no isolation HTTP test found in the brief's list: **135**

### Covered by the sweep

- `GET /v1/projects/{project_id}`
- `GET /v1/projects/{project_id}/documents`
- `DELETE /v1/projects/{project_id}`
- `POST /v1/projects/{project_id}/documents`

### Other tenancy tests (still UNPROVEN vs the sweep)

These files were read: `test_projects_tenancy.py`, `test_project_ask_tenancy.py`,
`test_chat_project_memory_tenancy.py` (helper, not HTTP), `test_workflows_tenancy.py`,
`test_rag_search_tenancy.py`, `test_upload_shared_project_access.py`,
`test_project_detail_admin_access.py`, `test_chat_open_access_gate.py`,
`test_step0_retrieval_isolation.py` (retriever unit, no HTTP routes),
`test_postgres_isolation.py` (DB layer, no HTTP routes).

- `POST /chat` — test_chat_project_memory_tenancy covers helper, not HTTP cross-user. No conversation_id on this model.
- `POST /v1/chat` — Alias of /chat
- `POST /v1/projects` — Covered as setup in tenancy tests; not in authz MATRIX surfaces
- `GET /v1/projects` — test_projects_tenancy list isolation
- `POST /v1/projects/{project_id}/conversations/{conversation_id}/clear` — test_project_detail_admin_access mutation. Stored row with project_id=None still clears if ws- prefix passes.
- `GET /v1/projects/{project_id}/memory` — test_projects_tenancy cross-tenant 404. Shared-project stranger also 404 (owner-only).
- `POST /v1/projects/{project_id}/drive/import` — test_project_ask_tenancy test_drive_import_cross_tenant_404
- `POST /v1/rag/search` — test_rag_search_tenancy. Not in authz MATRIX.
- `GET /v1/rag/gk-status` — Gate on project_id query; then enumerates all GK project docs. test_rag_search_tenancy.
- `POST /v1/project/ask` — test_project_ask_tenancy
- `POST /v1/workflows` — Can stamp another user's project_id on own workflow row. test_workflows_tenancy covers list/get/delete/run IDOR.
- `GET /v1/workflows` — MATRIX-missing; other tenancy yes
- `GET /v1/workflows/{workflow_id}` — 
- `DELETE /v1/workflows/{workflow_id}` — 
- `POST /v1/workflows/{workflow_id}/run` — 
- `GET /v1/agents/conversations/{conversation_id}/messages` — test_chat_open_access_gate. Non-ws missing row allowed (empty 200) after check returns.
- `POST /v1/agents/{name}/chat/stream` — test_chat_open_access_gate. Body-only ids (no path id).

### No isolation test found (UNPROVEN / no_test)

135 routes. Highest-risk first is in the next section; full list follows by file.

## Top 10 highest-risk UNPROVEN gaps (mapping judgment, not exploits)

Severity here is **potential blast radius if the untested path is wrong**, not a confirmed finding.

### 1. `POST /chain` (score 90)

- File: `chain.py`
- Guard: `require_user`; admin: privileged_blocks_only
- Ownership: none — step inputs may name any project/file
- Flags: id_no_owner
- Why ranked: Orchestrator chain; same privilege gate as execute

### 2. `POST /execute` (score 90)

- File: `execute.py`
- Guard: `require_user`; admin: privileged_blocks_only
- Ownership: none — body.input/params may carry project_id/document_id with no store check
- Flags: id_no_owner
- Why ranked: Any signed-in user; raise_if_privileged_block on PRIVILEGED_BLOCKS (Agent F); construction/rag/etc. still run with caller-supplied ids when not privileged

### 3. `GET /mcp/sse` (score 90)

- File: `mcp.py`
- Guard: `require_api_key`; admin: privileged_blocks_only
- Ownership: none — MCP call_tool executes blocks with caller arguments
- Flags: id_no_owner
- Why ranked: Same execute surface as /v1/execute via tools. /mcp/messages Mount in main.py uses validate_key only (not JWT-normalized).

### 4. `POST /v1/chain` (score 90)

- File: `chain.py`
- Guard: `require_user`; admin: privileged_blocks_only
- Ownership: none
- Flags: id_no_owner
- Why ranked: Alias of /chain

### 5. `POST /v1/execute` (score 90)

- File: `execute.py`
- Guard: `require_user`; admin: privileged_blocks_only
- Ownership: none — delegates to /execute
- Flags: id_no_owner
- Why ranked: EXISTS. Guard=require_user (JWT or API key→system user). Not admin. Not require_api_key-only. PRIVILEGED_BLOCKS 403 to a plain user (Agent F).

### 6. `POST /v1/auth/check` (score 75)

- File: `auth.py`
- Guard: `require_api_key`; admin: no
- Ownership: body.api_key — any caller can check ANY key's permission
- Flags: id_no_owner
- Why ranked: No admin; key-oracle

### 7. `GET /v1/auth/usage` (score 75)

- File: `auth.py`
- Guard: `require_api_key`; admin: no
- Ownership: query key/api_key — any caller can read ANY key usage
- Flags: id_no_owner
- Why ranked: No admin; usage leak across keys

### 8. `POST /v1/auth/validate` (score 75)

- File: `auth.py`
- Guard: `require_api_key`; admin: no
- Ownership: body.api_key — any authenticated caller can validate ANY key
- Flags: id_no_owner
- Why ranked: No admin gate. Cross-key probe: caller A validates caller B's key

### 9. `POST /v1/memory/{action}` (score 75)

- File: `memory.py`
- Guard: `require_api_key`; admin: flush/keys admin-only
- Ownership: body.key — get/set/delete/exists have NO per-user namespace
- Flags: id_no_owner
- Why ranked: Shared in-process cache. Non-admin can get/set/delete any key. flush/keys require admin.

### 10. `POST /v1/projects/{project_id}/conversations/{conversation_id}/export` (score 70)

- File: `exports.py`
- Guard: `require_user`; admin: no
- Ownership: _check_owner only; get_messages(conversation_id) unscoped
- Flags: project_only_no_child
- Why ranked: HIGH: conversation transcript export if caller can open ANY project they can _check_owner

## Flagged rows (all UNPROVEN vs sweep)

### `unguarded_non_public`

- `GET /stats` — Block inventory + version; no auth; not a health probe
- `GET /v1/system/health` — Full monitoring health_report; no auth; richer than /health
- `GET /blocks` — Full block catalog + ui_schema; no auth
- `GET /blocks/{block_name}` — Block config + instance.get_stats(); no auth
- `GET /v1/blocks` — Alias of GET /blocks
- `GET /v1/blocks/{block_name}` — Alias of GET /blocks/{name}

### `id_no_owner`

- `POST /execute` (`require_user`) — PRIVILEGED_BLOCKS 403 to a plain user (Agent F); nested ids granted in #623
- `POST /v1/execute` (`require_user`) — same as /execute
- `POST /chain` (`require_user`) — Orchestrator chain; same privilege gate as execute
- `POST /v1/chain` (`require_user`) — Alias of /chain
- `POST /upload` (`require_api_key`) — File lands in DATA_DIR even when project_id missing/unowned. API key→SYSTEM_USER_ID. No child-id.
- `POST /v1/upload` (`require_api_key`) — Alias
- `POST /v1/auth/validate` (`require_api_key`) — No admin gate. Cross-key probe: caller A validates caller B's key
- `POST /v1/auth/check` (`require_api_key`) — No admin; key-oracle
- `GET /v1/auth/usage` (`require_api_key`) — No admin; usage leak across keys
- `POST /v1/memory/{action}` (`require_api_key`) — Shared in-process cache. Non-admin can get/set/delete any key. flush/keys require admin.
- `GET /v1/projects/{project_id}/drive/index-folder/job/{job_id}` (`require_user`) — Any authenticated user who knows job_id+project_id can poll. Job may include error strings.
- `POST /v1/workflows` (`require_user`) — Can stamp another user's project_id on own workflow row. test_workflows_tenancy covers list/get/delete/run IDOR.
- `GET /mcp/sse` (`require_api_key`) — Same execute surface as /v1/execute via tools. /mcp/messages Mount in main.py uses validate_key only (not JWT-normalized).
- `GET /v1/hydration/latest` (`require_api_key`) — Any authenticated principal can read any project's latest hydration row
- `GET /v1/hydration/history` (`require_api_key`) — Same
- `POST /v1/hydration/run` (`require_api_key`) — Comment says admin-triggered; CODE has no admin check
- `POST /v1/feedback/route` (`require_api_key`) — Writes learning_engine pattern under any project_id. Cross-tenant write to routing corpus.

### `project_only_no_child`

- `DELETE /v1/projects/{project_id}/documents/{document_id}` — Strict == path id; master-corpus alias may 404 own docs. Child check exists but alias-blind.
- `POST /v1/projects/{project_id}/export/cost-boq` — Shared-project stranger can export. document_id from another project is readable if file_path set.
- `POST /v1/projects/{project_id}/price-boq` — Same child-id hole
- `POST /v1/projects/{project_id}/export/schedule-from-brief` — Any accessible project + victim conversation_id stages WBS
- `POST /v1/projects/{project_id}/export/schedule-from-document` — document_ids list unscoped
- `POST /v1/projects/{project_id}/conversations/{conversation_id}/export/schedule` — WBS of any conversation_id
- `POST /v1/projects/{project_id}/conversations/{conversation_id}/export` — HIGH: conversation transcript export if caller can open ANY project they can _check_owner
- `POST /v1/projects/{project_id}/export/schedule-from-boq` — Same document_id hole

### `path_body_id`

- `POST /v1/agents/{name}/chat` — Stream sibling uses include_admin_approved. Body project_id vs conversation_id can diverge; conv check first. Owner-only project_id 404s shared-project users on this path.
- `POST /v1/agents/{name}/chat/stream` — test_chat_open_access_gate. Body-only ids (no path id).

## Full UNPROVEN list (missing from authz matrix)

### `admin.py` (30)

- `GET /v1/admin/debug/doc-extract` guard=`require_api_key`
- `GET /v1/admin/debug/document-download` guard=`require_api_key`
- `POST /v1/admin/debug/doc-reindex` guard=`require_api_key`
- `GET /v1/admin/debug/doc-reindex/job/{job_id}` guard=`require_api_key`
- `POST /v1/admin/projects/{project_id}/approve` guard=`require_api_key`
- `POST /v1/admin/documents/{old_id}/supersede` guard=`require_api_key`
- `GET /v1/admin/projects/archived` guard=`require_api_key`
- `POST /v1/admin/projects/{project_id}/restore` guard=`require_api_key`
- `POST /v1/admin/projects/{project_id}/purge` guard=`require_api_key`
- `GET /v1/admin/training/list` guard=`require_api_key`
- `GET /v1/admin/training/download` guard=`require_api_key`
- `POST /v1/admin/debug/project-reindex` guard=`require_api_key`
- `POST /v1/admin/drive/download-proof` guard=`require_api_key`
- `POST /v1/admin/drive/ingest-proof` guard=`require_api_key`
- `POST /v1/admin/training/generate-scenarios` guard=`require_api_key`
- `GET /v1/admin/training/job/{job_id}` guard=`require_api_key`
- `POST /v1/admin/debug/migrate-sqlite` guard=`require_api_key`
- `GET /v1/admin/debug/pilot-preflight` guard=`require_api_key`
- `POST /v1/admin/debug/sentry-smoke` guard=`require_api_key`
- `GET /v1/admin/corpus/collections` guard=`require_api_key`
- `POST /v1/admin/corpus/bulk-insert` guard=`require_api_key`
- `POST /v1/admin/corpus/repair-document-sizes` guard=`require_api_key`
- `POST /v1/admin/corpus/delete-docs` guard=`require_api_key`
- `POST /v1/admin/corpus/reconcile` guard=`require_api_key`
- `GET /v1/admin/corpus/coverage` guard=`require_api_key`
- `GET /v1/admin/drive/scan` guard=`require_api_key`
- `POST /v1/admin/projects/approve-from-drive` guard=`require_api_key`
- `POST /v1/admin/projects/approve-from-drive/_bg` guard=`require_api_key`
- `GET /v1/admin/dead-letter` guard=`require_api_key`
- `POST /v1/admin/debug/sweep-plaintext` guard=`require_api_key`

### `agents.py` (5)

- `GET /v1/agents/conversations/{conversation_id}/messages` guard=`require_user` [OTHER_TENANCY]
- `GET /v1/agents` guard=`require_user`
- `GET /v1/agents/{name}` guard=`require_user`
- `POST /v1/agents/{name}/chat` guard=`require_user`
- `POST /v1/agents/{name}/chat/stream` guard=`require_user` [OTHER_TENANCY]

### `auth.py` (8)

- `POST /v1/auth/validate` guard=`require_api_key`
- `POST /v1/auth/keys` guard=`require_api_key`
- `DELETE /v1/auth/keys/{api_key}` guard=`require_api_key`
- `GET /v1/auth/keys` guard=`require_api_key`
- `POST /v1/auth/keys/revoke` guard=`require_api_key`
- `POST /v1/auth/keys/rotate` guard=`require_api_key`
- `POST /v1/auth/check` guard=`require_api_key`
- `GET /v1/auth/usage` guard=`require_api_key`

### `blocks.py` (4)

- `GET /blocks` guard=`none`
- `GET /blocks/{block_name}` guard=`none`
- `GET /v1/blocks` guard=`none`
- `GET /v1/blocks/{block_name}` guard=`none`

### `chain.py` (2)

- `POST /chain` guard=`require_user`
- `POST /v1/chain` guard=`require_user`

### `chat.py` (4)

- `POST /chat` guard=`require_user` [OTHER_TENANCY]
- `POST /chat/stream` guard=`require_user`
- `POST /v1/chat` guard=`require_user` [OTHER_TENANCY]
- `POST /v1/chat/stream` guard=`require_user`

### `chat_photos.py` (1)

- `POST /v1/chat/analyze-photo` guard=`require_user`

### `connectors.py` (5)

- `POST /v1/projects/{project_id}/connectors/aconex` guard=`require_user`
- `GET /v1/projects/{project_id}/connectors` guard=`require_user`
- `POST /v1/projects/{project_id}/connectors/aconex/sync` guard=`require_user`
- `POST /v1/projects/{project_id}/connectors/aconex/rfi` guard=`require_user`
- `POST /v1/projects/{project_id}/connectors/aconex/events` guard=`require_user`

### `debug.py` (2)

- `GET /debug/env` guard=`require_api_key`
- `GET /v1/debug/env` guard=`require_api_key`

### `doc_search.py` (1)

- `GET /v1/projects/{project_id}/documents/search` guard=`require_user`

### `doc_types.py` (4)

- `GET /v1/document-types` guard=`require_api_key`
- `POST /v1/document-types` guard=`require_user`
- `DELETE /v1/document-types/{name}` guard=`require_user`
- `POST /v1/document-types/classify` guard=`require_api_key`

### `drive.py` (8)

- `GET /v1/drive/connect` guard=`require_user`
- `GET /v1/drive/callback` guard=`none`
- `GET /v1/drive/status` guard=`require_user`
- `POST /v1/drive/disconnect` guard=`require_user`
- `GET /v1/drive/files` guard=`require_user`
- `POST /v1/projects/{project_id}/drive/index-folder` guard=`require_user`
- `GET /v1/projects/{project_id}/drive/index-folder/job/{job_id}` guard=`require_user`
- `POST /v1/projects/{project_id}/drive/import` guard=`require_user` [OTHER_TENANCY]

### `execute.py` (2)

- `POST /execute` guard=`require_user`
- `POST /v1/execute` guard=`require_user`

### `exports.py` (10)

- `POST /v1/projects/{project_id}/export/schedule` guard=`require_user`
- `POST /v1/projects/{project_id}/export/cost-boq` guard=`require_user`
- `POST /v1/projects/{project_id}/price-boq` guard=`require_user`
- `POST /v1/projects/{project_id}/export/cost-schedule` guard=`require_user`
- `POST /v1/projects/{project_id}/export/schedule-from-brief` guard=`require_user`
- `POST /v1/projects/{project_id}/export/schedule-from-document` guard=`require_user`
- `POST /v1/projects/{project_id}/export/evm` guard=`require_user`
- `POST /v1/projects/{project_id}/conversations/{conversation_id}/export/schedule` guard=`require_user`
- `POST /v1/projects/{project_id}/conversations/{conversation_id}/export` guard=`require_user`
- `POST /v1/projects/{project_id}/export/schedule-from-boq` guard=`require_user`

### `feedback.py` (1)

- `POST /v1/feedback/route` guard=`require_api_key`

### `health.py` (7)

- `GET /livez` guard=`none`
- `GET /v1/upload-limits` guard=`none`
- `GET /health` guard=`none`
- `GET /ready` guard=`none`
- `GET /stats` guard=`none`
- `GET /v1/health` guard=`none`
- `GET /v1/system/health` guard=`none`

### `hydration.py` (3)

- `GET /v1/hydration/latest` guard=`require_api_key`
- `GET /v1/hydration/history` guard=`require_api_key`
- `POST /v1/hydration/run` guard=`require_api_key`

### `mcp.py` (2)

- `GET /mcp/info` guard=`require_api_key`
- `GET /mcp/sse` guard=`require_api_key`

### `memory.py` (2)

- `GET /v1/memory/stats` guard=`require_api_key`
- `POST /v1/memory/{action}` guard=`require_api_key`

### `monitoring.py` (6)

- `GET /metrics` guard=`none`
- `GET /v1/metrics` guard=`require_api_key`
- `GET /v1/leaderboard` guard=`require_api_key`
- `GET /v1/recommend` guard=`require_api_key`
- `GET /v1/predict` guard=`require_api_key`
- `POST /v1/metrics/record` guard=`require_api_key`

### `project.py` (1)

- `POST /v1/project/ask` guard=`require_user` [OTHER_TENANCY]

### `projects.py` (17)

- `POST /v1/projects` guard=`require_user` [OTHER_TENANCY]
- `PATCH /v1/projects/{project_id}` guard=`require_user`
- `POST /v1/projects/from-drive` guard=`require_user`
- `GET /v1/projects` guard=`require_user` [OTHER_TENANCY]
- `GET /v1/projects/{project_id}/documents/{document_id}/preview` guard=`require_user`
- `GET /v1/projects/{project_id}/documents/{document_id}/preview/raw` guard=`require_user`
- `POST /v1/projects/{project_id}/conversations/{conversation_id}/clear` guard=`require_user` [OTHER_TENANCY]
- `GET /v1/projects/{project_id}/conversations` guard=`require_user`
- `POST /v1/projects/{project_id}/progress` guard=`require_user`
- `GET /v1/projects/{project_id}/memory` guard=`require_user` [OTHER_TENANCY]
- `POST /v1/projects/{project_id}/memory` guard=`require_user`
- `DELETE /v1/projects/{project_id}/memory/{key}` guard=`require_user`
- `PATCH /v1/projects/{project_id}/documents/{document_id}` guard=`require_user`
- `DELETE /v1/projects/{project_id}/documents/{document_id}` guard=`require_user`
- `GET /v1/projects/{project_id}/audit` guard=`require_user`
- `GET /v1/governance` guard=`require_user`
- `POST /v1/governance/purge` guard=`require_user`

### `rag.py` (2)

- `POST /v1/rag/search` guard=`require_api_key` [OTHER_TENANCY]
- `GET /v1/rag/gk-status` guard=`require_api_key` [OTHER_TENANCY]

### `redline.py` (1)

- `POST /v1/projects/{project_id}/documents/{document_id}/redlines` guard=`require_user`

### `schedule.py` (5)

- `POST /v1/schedule/generate` guard=`require_user`
- `POST /v1/schedule/cpm` guard=`require_user`
- `POST /v1/schedule/manpower` guard=`require_user`
- `POST /v1/schedule/fasttrack` guard=`require_user`
- `POST /v1/schedule/export` guard=`require_user`

### `static.py` (3)

- `GET /` guard=`none`
- `GET /api` guard=`none`
- `GET /{full_path:path}` guard=`none`

### `upload.py` (4)

- `POST /upload` guard=`require_api_key`
- `POST /v1/upload` guard=`require_api_key`
- `POST /ingest` guard=`require_api_key`
- `POST /ingest-via-block` guard=`require_api_key`

### `usage.py` (2)

- `GET /v1/usage/today` guard=`require_user`
- `GET /v1/usage` guard=`require_user`

### `users.py` (5)

- `POST /v1/users/register` guard=`none`
- `GET /v1/users/verify-email` guard=`none`
- `POST /v1/users/resend-verification` guard=`none`
- `POST /v1/users/login` guard=`none`
- `GET /v1/users/me` guard=`require_user`

### `workflows.py` (5)

- `POST /v1/workflows` guard=`require_user` [OTHER_TENANCY]
- `GET /v1/workflows` guard=`require_user` [OTHER_TENANCY]
- `GET /v1/workflows/{workflow_id}` guard=`require_user` [OTHER_TENANCY]
- `DELETE /v1/workflows/{workflow_id}` guard=`require_user` [OTHER_TENANCY]
- `POST /v1/workflows/{workflow_id}/run` guard=`require_user` [OTHER_TENANCY]

## Phase 1 skeleton (not written this run)

`tests/test_no_user_reaches_another_users_data.py` does not exist yet.
Preferred this run: document gaps. A collect-all skip file would skip incorrectly.
Phase 1 should extend the authz matrix so a new unguarded id-route fails CI,
and add two-user cross-id cases for the Top 10 above first.

## Peer-conflict blockers

- None encountered. Step 0 did not edit `app/core/rag/*`, formula audits,
  `app/containers/construction/`, `master_corpus`, or `FIXTURE-containers-*`.
- `scan_exception_pass.py` line-number allowlist: not touched (docs only).

