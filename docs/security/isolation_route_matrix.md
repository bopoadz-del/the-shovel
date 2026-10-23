# Isolation route matrix (Step 0 — mapping only)

Date: 2026-09-19. Reviewed against `666e223` (`origin/main` after rebase). Synthetic review of `app/routers/*.py` only.
This is an enumeration, not a verdict. Do not treat any row as ready or secure.

## Counts

- **Total HTTP routes in `app/routers/*.py`:** 156
- **`admin.py`:** 30
- **Guarded** (`require_user` or `require_api_key`): 136 (`require_user`=75, `require_api_key`=61)
- **No dependency guard:** 20 (of which flagged `public`=14, `unguarded_non_public`=6)
- **Flags:** `id_no_owner`=17; `project_only_no_child`=8; `path_body_id`=2; `frontend_only`=0
- **Authz MATRIX sweep covers:** 4 routes
- **Missing from authz matrix (UNPROVEN vs that sweep):** 152
- **Of those, some other tenancy test exists:** 17
- **No isolation/tenancy HTTP test found:** 135

`hat_frames.py` and `chat_watchdog.py` define no HTTP routes (stream wrappers).
`/mcp/messages` is a Starlette Mount in `app/main.py` (not a router decorator); gated by `auth_manager.validate_key` only.

## Guard meanings

- `none` — no FastAPI `Depends` auth.
- `require_api_key` — JWT **or** legacy API key (`app/dependencies.py`). API-key principals get `user_id=SYSTEM_USER_ID`. `role` comes from the key record (`CEREBRUM_MASTER_KEY` is minted `admin`).
- `require_user` — JWT user **or** API key as system user. Same SYSTEM_USER collapse.
- Admin check is a **second** gate (`_require_admin` / `role == "admin"`) and is recorded separately.

## Flag legend

| Flag | Meaning |
| --- | --- |
| `public` | No auth; treated as deliberate public (health, SPA, OAuth callback, auth signup). |
| `unguarded_non_public` | No auth and not a documented public probe. |
| `id_no_owner` | Request carries a resource id with no owner/access check. |
| `project_only_no_child` | Project access checked; document/conversation id not proven to belong to that project. |
| `path_body_id` | Path vs body identifiers can diverge; which one is authoritative is noted. |
| `frontend_only` | HTTP layer has no guard (none found). |

## Authz matrix sweep coverage

`tests/test_authz_matrix_sweep.py` exercises four surfaces × owner/admin/stranger × private/shared:

| Surface | Route |
| --- | --- |
| open | `GET /v1/projects/{project_id}` |
| docs | `GET /v1/projects/{project_id}/documents` |
| upload | `POST /v1/projects/{project_id}/documents` |
| delete | `DELETE /v1/projects/{project_id}` |

Everything else is **UNPROVEN** against that sweep. See `isolation_authz_gaps.md`.

## /v1/execute (from code, not a live probe)

`POST /v1/execute` and `POST /execute` exist in `app/routers/execute.py`.
Guard: `auth: dict = Depends(require_user)`.
Additional: `raise_if_privileged_block` for every name in `PRIVILEGED_BLOCKS`
(`code`, `sandbox`, `mcp_consumer`, `local_drive`, `web`, `webhook`,
`google_drive`, `onedrive`) — Agent F `b70763a` / `08055d1` on `60c8055`.
A plain user gets **403** on those six-plus-two blocks. Orchestrator steps scanned.
Nested `project_id` / `document_id` must be accessible (#623). Body
`input`/`params` otherwise pass to `block.execute`.
A plain user (JWT role=user) can still invoke every **non-privileged** loaded block.

## Agent F privileged hats (from code, not a live probe)

Reviewed against `60c8055`. Not a ready/secure claim. Live dual-account
confirm waits for `build_sha >= 60c8055` (live was still `ec6d380` when
this row was written).

- `GET /v1/agents` omits `self-coding`, `external-mcp`, `document-ingestion` for a plain user.
- `GET /v1/agents/{name}` and `POST /v1/agents/{name}/chat[/stream]` for those three names: **404**.
- `POST /v1/chat/stream` with `"agent"` set to one of those three: SSE **error** event (`not available`), no `route`/`final` to that hat.
- Both synthetic users in `tests/test_no_user_reaches_another_users_data.py` must see the same denials.

A new block that starts a process or reaches the network fails CI unless
it is in `PRIVILEGED_BLOCKS` or the reviewed allowlist in
`tests/test_no_agent_hands_a_user_the_server.py`.

## Full matrix

### `admin.py` (30)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/admin/debug/doc-extract` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | query project_id+document_id; admin bypasses tenant; extracts any doc. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/debug/document-download` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | query ids; admin; checks doc.project_id==project_id then decrypts. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/debug/doc-reindex` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin reindex any document. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/debug/doc-reindex/job/{job_id}` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin job poll; job_id unscoped beyond admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/projects/{project_id}/approve` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin approve any project_id. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/documents/{old_id}/supersede` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin; path old_id global. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/projects/archived` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin lists all archived. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/projects/{project_id}/restore` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin restore any. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/projects/{project_id}/purge` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin HARD purge any project. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/training/list` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin training files. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/training/download` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin download training artifact. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/debug/project-reindex` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin reindex any project. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/drive/download-proof` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin Drive proof. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/drive/ingest-proof` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin Drive ingest proof. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/training/generate-scenarios` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/training/job/{job_id}` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin job. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/debug/migrate-sqlite` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin destructive migrate. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/debug/pilot-preflight` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/debug/sentry-smoke` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/corpus/collections` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin corpus listing. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/corpus/bulk-insert` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin write corpus. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/corpus/repair-document-sizes` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/corpus/delete-docs` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin delete corpus docs. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/corpus/reconcile` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/corpus/coverage` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/drive/scan` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin Drive scan. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/projects/approve-from-drive` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin create/approve. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/projects/approve-from-drive/_bg` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin background twin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `GET` | `/v1/admin/dead-letter` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |
| `POST` | `/v1/admin/debug/sweep-plaintext` | `require_api_key` | yes _require_admin | admin global — no tenant ownership (intentional operator bypass) | — | UNPROVEN | admin. Plain user 403 if role!=admin. JWT user with role=user 403. API key without role=admin 403. UNPROVEN vs matrix. |

### `agents.py` (5)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/agents/conversations/{conversation_id}/messages` | `require_user` | no | _enforce_conversation_access (ws-* → project accessible; else stored project_id) | — | OTHER_TENANCY | test_chat_open_access_gate. Non-ws missing row allowed (empty 200) after check returns. |
| `GET` | `/v1/agents` | `require_user` | no | n/a (registry) | — | UNPROVEN |  |
| `GET` | `/v1/agents/{name}` | `require_user` | privileged hats 404 | n/a | — | OTHER_TENANCY | Includes system_prompt. `self-coding` / `external-mcp` / `document-ingestion` 404 for a plain user (Agent F). |
| `POST` | `/v1/agents/{name}/chat` | `require_user` | privileged hats 404 | conversation_id: _enforce_conversation_access; project_id: get_project owner-only (NO include_admin_approved) except master_corpus alias | path_body_id | OTHER_TENANCY | Privileged names 404 for a plain user (Agent F). Stream sibling uses include_admin_approved. Body project_id vs conversation_id can diverge; conv check first. Owner-only project_id 404s shared-project users on this path. |
| `POST` | `/v1/agents/{name}/chat/stream` | `require_user` | no | conversation_id: _enforce; project_id: get_project include_admin_approved except MC alias | path_body_id | OTHER_TENANCY | test_chat_open_access_gate. Body-only ids (no path id). |

### `auth.py` (8)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/auth/validate` | `require_api_key` | no | body.api_key — any authenticated caller can validate ANY key | id_no_owner | UNPROVEN | No admin gate. Cross-key probe: caller A validates caller B's key |
| `POST` | `/v1/auth/keys` | `require_api_key` | yes _require_admin | n/a (creates key) | — | UNPROVEN | Admin mint |
| `DELETE` | `/v1/auth/keys/{api_key}` | `require_api_key` | yes _require_admin | path api_key; admin global | — | UNPROVEN | Admin revoke by URL |
| `GET` | `/v1/auth/keys` | `require_api_key` | yes _require_admin | n/a | — | UNPROVEN | Lists all keys |
| `POST` | `/v1/auth/keys/revoke` | `require_api_key` | yes _require_admin | body key | — | UNPROVEN | Admin revoke |
| `POST` | `/v1/auth/keys/rotate` | `require_api_key` | yes _require_admin | body key | — | UNPROVEN | Admin rotate |
| `POST` | `/v1/auth/check` | `require_api_key` | no | body.api_key — any caller can check ANY key's permission | id_no_owner | UNPROVEN | No admin; key-oracle |
| `GET` | `/v1/auth/usage` | `require_api_key` | no | query key/api_key — any caller can read ANY key usage | id_no_owner | UNPROVEN | No admin; usage leak across keys |

### `blocks.py` (4)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/blocks` | `none` | no | n/a | unguarded_non_public | UNPROVEN | Full block catalog + ui_schema; no auth |
| `GET` | `/blocks/{block_name}` | `none` | no | n/a | unguarded_non_public | UNPROVEN | Block config + instance.get_stats(); no auth |
| `GET` | `/v1/blocks` | `none` | no | n/a | unguarded_non_public | UNPROVEN | Alias of GET /blocks |
| `GET` | `/v1/blocks/{block_name}` | `none` | no | n/a | unguarded_non_public | UNPROVEN | Alias of GET /blocks/{name} |

### `chain.py` (2)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/chain` | `require_user` | privileged_blocks_only | none — step inputs may name any project/file | id_no_owner | UNPROVEN | Orchestrator chain; same privilege gate as execute |
| `POST` | `/v1/chain` | `require_user` | privileged_blocks_only | none | id_no_owner | UNPROVEN | Alias of /chain |

### `chat.py` (4)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/chat` | `require_user` | no | project_id in body: drop if get_project_accessible fails (_with_project_memory/_with_doc_search) | — | OTHER_TENANCY | test_chat_project_memory_tenancy covers helper, not HTTP cross-user. No conversation_id on this model. |
| `POST` | `/chat/stream` | `require_user` | no | NONE — request.project_id unused; no memory/doc inject | — | UNPROVEN | Streaming chat block; project_id accepted but ignored; no tenant leak path via this handler |
| `POST` | `/v1/chat` | `require_user` | no | same as /chat | — | OTHER_TENANCY | Alias of /chat |
| `POST` | `/v1/chat/stream` | `require_user` | privileged pin refused | conversation_id: _enforce_conversation_access; project_id: get_project_accessible drop | — | OTHER_TENANCY | Main UI stream. Pinning `self-coding` / `external-mcp` / `document-ingestion` emits an error event (Agent F). Conversation check imported from agents. Body conversation_id vs project_id can disagree — conv check is authoritative for ws-* |

### `chat_photos.py` (1)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/chat/analyze-photo` | `require_user` | no | none (ephemeral temp file; no project) | — | UNPROVEN | Any signed-in user; no tenant id |

### `connectors.py` (5)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/projects/{project_id}/connectors/aconex` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN |  |
| `GET` | `/v1/projects/{project_id}/connectors` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN |  |
| `POST` | `/v1/projects/{project_id}/connectors/aconex/sync` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN | CDE pull into project |
| `POST` | `/v1/projects/{project_id}/connectors/aconex/rfi` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN |  |
| `POST` | `/v1/projects/{project_id}/connectors/aconex/events` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN |  |

### `debug.py` (2)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/debug/env` | `require_api_key` when mounted | yes role==admin + non-prod | n/a | — | OTHER_TENANCY | Production: SPA reserved prefix → 404 empty, no env fields. Dest: 403 to a plain user. |
| `GET` | `/v1/debug/env` | `require_api_key` when mounted | yes (delegates) | n/a | — | OTHER_TENANCY | Production: 404 empty (`v1/` reserved). Dest: 403 to a plain user. |

### `doc_search.py` (1)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/projects/{project_id}/documents/search` | `require_user` | no | get_project include_admin_approved | — | UNPROVEN | Read-open. Not in matrix. |

### `doc_types.py` (4)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/document-types` | `require_api_key` | no | n/a (global registry) | — | UNPROVEN |  |
| `POST` | `/v1/document-types` | `require_user` | yes role==admin | n/a (process-global) | — | UNPROVEN |  |
| `DELETE` | `/v1/document-types/{name}` | `require_user` | yes role==admin | n/a | — | UNPROVEN |  |
| `POST` | `/v1/document-types/classify` | `require_api_key` | no | n/a | — | UNPROVEN |  |

### `drive.py` (8)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/drive/connect` | `require_user` | no | state signed with auth.user_id | — | UNPROVEN |  |
| `GET` | `/v1/drive/callback` | `none` | no | signed state HMAC carries user_id | public | UNPROVEN | OAuth redirect; signed state is the gate. Deliberate public. |
| `GET` | `/v1/drive/status` | `require_user` | no | load_token(auth.user_id) | — | UNPROVEN |  |
| `POST` | `/v1/drive/disconnect` | `require_user` | no | clear_token(auth.user_id) | — | UNPROVEN |  |
| `GET` | `/v1/drive/files` | `require_user` | no | caller's Drive token | — | UNPROVEN |  |
| `POST` | `/v1/projects/{project_id}/drive/index-folder` | `require_user` | no | get_project(user_id) owner-only | — | UNPROVEN |  |
| `GET` | `/v1/projects/{project_id}/drive/index-folder/job/{job_id}` | `require_user` | no | job dict project_id == path; NO project ownership check | id_no_owner | UNPROVEN | Any authenticated user who knows job_id+project_id can poll. Job may include error strings. |
| `POST` | `/v1/projects/{project_id}/drive/import` | `require_user` | no | get_project(user_id) owner-only | — | OTHER_TENANCY | test_project_ask_tenancy test_drive_import_cross_tenant_404 |

### `execute.py` (2)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/execute` | `require_user` | privileged_blocks_only | nested project_id/document_id: get_project_accessible / get_document grant (#623) | id_no_owner | OTHER_TENANCY | Plain user 403 on PRIVILEGED_BLOCKS (code/sandbox/mcp_consumer/local_drive/web/webhook/google_drive/onedrive). Nested ids 404 if inaccessible. |
| `POST` | `/v1/execute` | `require_user` | privileged_blocks_only | same as /execute | id_no_owner | OTHER_TENANCY | Alias. Guard=require_user (JWT or API key→system user). Not require_api_key-only. |

### `exports.py` (10)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/projects/{project_id}/export/schedule` | `require_user` | no | _check_owner include_admin_approved (open-read) | — | UNPROVEN |  |
| `POST` | `/v1/projects/{project_id}/export/cost-boq` | `require_user` | no | _check_owner; body.document_id via get_document WITHOUT project match | project_only_no_child | UNPROVEN | Shared-project stranger can export. document_id from another project is readable if file_path set. |
| `POST` | `/v1/projects/{project_id}/price-boq` | `require_user` | no | _check_owner; body.document_id get_document no project match | project_only_no_child | UNPROVEN | Same child-id hole |
| `POST` | `/v1/projects/{project_id}/export/cost-schedule` | `require_user` | no | _check_owner | — | UNPROVEN |  |
| `POST` | `/v1/projects/{project_id}/export/schedule-from-brief` | `require_user` | no | _check_owner; body.conversation_id load_conversation_wbs with NO project bind | project_only_no_child | UNPROVEN | Any accessible project + victim conversation_id stages WBS |
| `POST` | `/v1/projects/{project_id}/export/schedule-from-document` | `require_user` | no | _check_owner; body.document_ids passed to extract_schedule_feed with NO belonging check | project_only_no_child | UNPROVEN | document_ids list unscoped |
| `POST` | `/v1/projects/{project_id}/export/evm` | `require_user` | no | _check_owner | — | UNPROVEN | Body periods; no child id |
| `POST` | `/v1/projects/{project_id}/conversations/{conversation_id}/export/schedule` | `require_user` | no | _check_owner only; conversation_id not bound to project | project_only_no_child | UNPROVEN | WBS of any conversation_id |
| `POST` | `/v1/projects/{project_id}/conversations/{conversation_id}/export` | `require_user` | no | _check_owner only; get_messages(conversation_id) unscoped | project_only_no_child | UNPROVEN | HIGH: conversation transcript export if caller can open ANY project they can _check_owner |
| `POST` | `/v1/projects/{project_id}/export/schedule-from-boq` | `require_user` | no | _check_owner; body.document_id get_document no project match | project_only_no_child | UNPROVEN | Same document_id hole |

### `feedback.py` (1)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/feedback/route` | `require_api_key` | no | body.project_id recorded with NO access check | id_no_owner | UNPROVEN | Writes learning_engine pattern under any project_id. Cross-tenant write to routing corpus. |

### `health.py` (7)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/livez` | `none` | no | n/a | public | UNPROVEN | Process liveness; no I/O; deliberate public |
| `GET` | `/v1/upload-limits` | `none` | no | n/a | public | UNPROVEN | Config only; documented unauthenticated |
| `GET` | `/health` | `none` | no | n/a | public | UNPROVEN | Render liveness; build_sha + db/embedder probes |
| `GET` | `/ready` | `none` | no | n/a | public | UNPROVEN | Readiness; 503 if DB down |
| `GET` | `/stats` | `none` | no | n/a | unguarded_non_public | UNPROVEN | Block inventory + version; no auth; not a health probe |
| `GET` | `/v1/health` | `none` | no | n/a | public | UNPROVEN | Health + observability payload |
| `GET` | `/v1/system/health` | `none` | no | n/a | unguarded_non_public | UNPROVEN | Full monitoring health_report; no auth; richer than /health |

### `hydration.py` (3)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/hydration/latest` | `require_api_key` | no | query project_id with NO ownership check | id_no_owner | UNPROVEN | Any authenticated principal can read any project's latest hydration row |
| `GET` | `/v1/hydration/history` | `require_api_key` | no | query project_id unscoped | id_no_owner | UNPROVEN | Same |
| `POST` | `/v1/hydration/run` | `require_api_key` | no | body.project_ids unscoped; triggers learning_engine hydrate | id_no_owner | UNPROVEN | Comment says admin-triggered; CODE has no admin check |

### `mcp.py` (2)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/mcp/info` | `require_api_key` | no | n/a (tool catalog) | — | UNPROVEN |  |
| `GET` | `/mcp/sse` | `require_api_key` | privileged_blocks_only | none — MCP call_tool executes blocks with caller arguments | id_no_owner | UNPROVEN | Same execute surface as /v1/execute via tools. /mcp/messages Mount in main.py uses validate_key only (not JWT-normalized). |

### `memory.py` (2)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/memory/stats` | `require_api_key` | no | none (shared process cache stats) | — | UNPROVEN | Any API-key/JWT; not tenant-scoped |
| `POST` | `/v1/memory/{action}` | `require_api_key` | flush/keys admin-only | body.key — get/set/delete/exists have NO per-user namespace | id_no_owner | UNPROVEN | Shared in-process cache. Non-admin can get/set/delete any key. flush/keys require admin. |

### `monitoring.py` (6)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/metrics` | `none` | no | n/a | public | UNPROVEN | Prometheus scrapes; documented aggregates only |
| `GET` | `/v1/metrics` | `require_api_key` | yes role==admin | n/a | — | UNPROVEN | Per-block timings |
| `GET` | `/v1/leaderboard` | `require_api_key` | no | n/a | — | UNPROVEN | Provider reliability; any authenticated caller |
| `GET` | `/v1/recommend` | `require_api_key` | no | n/a | — | UNPROVEN | Provider recommend; any authenticated |
| `GET` | `/v1/predict` | `require_api_key` | no | n/a | — | UNPROVEN | Predictive failover; any authenticated |
| `POST` | `/v1/metrics/record` | `require_api_key` | no | n/a | — | UNPROVEN | Any caller can inject provider metrics |

### `project.py` (1)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/project/ask` | `require_user` | no | session.user_id == caller; project_id dropped if not accessible | — | OTHER_TENANCY | test_project_ask_tenancy |

### `projects.py` (21)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/projects` | `require_user` | no | creates row owned by auth.user_id | — | OTHER_TENANCY | Covered as setup in tenancy tests; not in authz MATRIX surfaces |
| `PATCH` | `/v1/projects/{project_id}` | `require_user` | no | _owned_or_404 owner-only (default read_only=False) | — | UNPROVEN | Mutating; not in matrix |
| `POST` | `/v1/projects/from-drive` | `require_user` | no | new row owned by caller; Drive token is caller's | — | UNPROVEN | Any user; origin=user_drive_import |
| `GET` | `/v1/projects` | `require_user` | admin sees all | list_projects(user_id) + include_admin_approved; admin unscoped | — | OTHER_TENANCY | test_projects_tenancy list isolation |
| `GET` | `/v1/projects/{project_id}` | `require_user` | admin via role in _owned_or_404 | _owned_or_404 read_only=True + role | — | COVERED | MATRIX surface 'open'. Also test_projects_tenancy + test_project_detail_admin_access + test_chat_open_access_gate |
| `GET` | `/v1/projects/{project_id}/documents` | `require_user` | admin via role | _owned_or_404 read_only=True + role; lists docs of that project | — | COVERED | MATRIX surface 'docs' |
| `GET` | `/v1/projects/{project_id}/documents/{document_id}/preview` | `require_user` | admin via role | project _owned_or_404 read_only + doc must be in citeable owner set (workspace/MC/GK) | — | UNPROVEN | Child belonging is citeable-set, not strict path-project. Private foreign doc 404s. |
| `GET` | `/v1/projects/{project_id}/documents/{document_id}/preview/raw` | `require_user` | admin via role | same as preview | — | UNPROVEN | PDF bytes |
| `DELETE` | `/v1/projects/{project_id}` | `require_user` | owner OR admin | get_project scoped unless admin; 403 if not owner/admin | — | COVERED | MATRIX surface 'delete'. Soft-archive. |
| `POST` | `/v1/projects/{project_id}/conversations/{conversation_id}/clear` | `require_user` | owner OR admin | project owner/admin + ws- prefix bind + stored conv.project_id in {None,'',pid,resolved} | — | OTHER_TENANCY | test_project_detail_admin_access mutation. Stored row with project_id=None still clears if ws- prefix passes. |
| `GET` | `/v1/projects/{project_id}/conversations` | `require_user` | owner OR admin | project owner/admin; lists convs for pid+alias | — | UNPROVEN | Owner-only (stranger on shared = 403). Not in matrix. |
| `POST` | `/v1/projects/{project_id}/documents` | `require_user` | admin via role | _owned_or_404 read_only=True (open-access upload) | — | COVERED | MATRIX surface 'upload'. Also test_upload_shared_project_access |
| `POST` | `/v1/projects/{project_id}/progress` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN | Not in matrix |
| `GET` | `/v1/projects/{project_id}/memory` | `require_user` | no | _owned_or_404 owner-only | — | OTHER_TENANCY | test_projects_tenancy cross-tenant 404. Shared-project stranger also 404 (owner-only). |
| `POST` | `/v1/projects/{project_id}/memory` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN | Write facts; not in matrix |
| `DELETE` | `/v1/projects/{project_id}/memory/{key}` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN | Not in matrix |
| `PATCH` | `/v1/projects/{project_id}/documents/{document_id}` | `require_user` | owner OR admin | project owner/admin + doc.project_id in {path, storage_id} | — | UNPROVEN | Child belonging checked |
| `DELETE` | `/v1/projects/{project_id}/documents/{document_id}` | `require_user` | no | _owned_or_404 owner-only + doc.project_id == path project_id (NO alias) | project_only_no_child | UNPROVEN | Strict == path id; master-corpus alias may 404 own docs. Child check exists but alias-blind. |
| `GET` | `/v1/projects/{project_id}/audit` | `require_user` | no | _owned_or_404 owner-only | — | UNPROVEN | Audit trail |
| `GET` | `/v1/governance` | `require_user` | yes role==admin | n/a (policy metadata) | — | UNPROVEN | Admin-only status |
| `POST` | `/v1/governance/purge` | `require_user` | yes role==admin | global purge by age — all tenants | — | UNPROVEN | Admin global. Phase 2 forbids live purge. Not in matrix. |

### `rag.py` (2)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/rag/search` | `require_api_key` | no | _searchable_project_or_404 → get_project_accessible | — | OTHER_TENANCY | test_rag_search_tenancy. Not in authz MATRIX. |
| `GET` | `/v1/rag/gk-status` | `require_api_key` | no | _searchable_project_or_404; also lists GK docs globally after gate | — | OTHER_TENANCY | Gate on project_id query; then enumerates all GK project docs. test_rag_search_tenancy. |

### `redline.py` (1)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/projects/{project_id}/documents/{document_id}/redlines` | `require_user` | no | get_project(user_id) owner-only + doc.project_id == path | — | UNPROVEN | Owner-only (shared stranger 404). Child belonging checked. No alias. |

### `schedule.py` (5)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/schedule/generate` | `require_user` | no | none (brief in body; no project) | — | UNPROVEN | Compute from body |
| `POST` | `/v1/schedule/cpm` | `require_user` | no | none | — | UNPROVEN |  |
| `POST` | `/v1/schedule/manpower` | `require_user` | no | none | — | UNPROVEN |  |
| `POST` | `/v1/schedule/fasttrack` | `require_user` | no | none | — | UNPROVEN |  |
| `POST` | `/v1/schedule/export` | `require_user` | no | none | — | UNPROVEN | Writes workbook path from body activities |

### `static.py` (3)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/` | `none` | no | n/a | public | UNPROVEN | SPA index; deliberate public |
| `GET` | `/api` | `none` | no | n/a | public | UNPROVEN | API name/version/block count; no tenant data |
| `GET` | `/{full_path:path}` | `none` | no | n/a | public | UNPROVEN | SPA fallback; reserved prefixes 404 |

### `upload.py` (4)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/upload` | `require_api_key` | no | optional Form project_id: get_project(user_id) or skip index; file stored regardless | id_no_owner | UNPROVEN | File lands in DATA_DIR even when project_id missing/unowned. API key→SYSTEM_USER_ID. No child-id. |
| `POST` | `/v1/upload` | `require_api_key` | no | same as /upload | id_no_owner | UNPROVEN | Alias |
| `POST` | `/ingest` | `require_api_key` | no | none (temp files; no project) | — | UNPROVEN | Document engine pipeline; any API-key principal |
| `POST` | `/ingest-via-block` | `require_api_key` | no | none | — | UNPROVEN | Same pipeline via BLOCK_REGISTRY |

### `usage.py` (2)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GET` | `/v1/usage/today` | `require_user` | no | usage_tracker.daily_total(auth.user_id) | — | UNPROVEN | Self-scoped |
| `GET` | `/v1/usage` | `require_user` | no | usage_tracker.history(auth.user_id) | — | UNPROVEN | Self-scoped |

### `users.py` (5)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/users/register` | `none` | no | n/a | public | UNPROVEN | Open in dev; prod gated by ALLOW_OPEN_REGISTRATION |
| `GET` | `/v1/users/verify-email` | `none` | no | purpose JWT user_id | public | UNPROVEN | Token is the gate |
| `POST` | `/v1/users/resend-verification` | `none` | no | email lookup; always 202 | public | UNPROVEN | Anti-oracle |
| `POST` | `/v1/users/login` | `none` | no | n/a | public | UNPROVEN | Constant-time dummy hash |
| `GET` | `/v1/users/me` | `require_user` | no | auth.user_id | — | UNPROVEN | Self |

### `workflows.py` (5)

| Method | Path | Guard | Admin | Ownership | Flags | Matrix | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST` | `/v1/workflows` | `require_user` | no | owner_id=auth.user_id; body.project_id NOT checked | id_no_owner | OTHER_TENANCY | Can stamp another user's project_id on own workflow row. test_workflows_tenancy covers list/get/delete/run IDOR. |
| `GET` | `/v1/workflows` | `require_user` | no | list_workflows(owner_id) | — | OTHER_TENANCY | MATRIX-missing; other tenancy yes |
| `GET` | `/v1/workflows/{workflow_id}` | `require_user` | no | get_workflow(owner_id) | — | OTHER_TENANCY |  |
| `DELETE` | `/v1/workflows/{workflow_id}` | `require_user` | no | delete_workflow(owner_id) | — | OTHER_TENANCY |  |
| `POST` | `/v1/workflows/{workflow_id}/run` | `require_user` | no | get_workflow(owner_id) then chain_execute | — | OTHER_TENANCY |  |

