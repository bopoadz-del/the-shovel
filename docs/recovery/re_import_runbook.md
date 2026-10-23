# Drive re-import runbook

> **RETIRED 2026-10-04.** Hydration no longer discovers, attaches or
> re-indexes documents, so Steps 3-7 below no longer import anything. Re-import
> a project's documents with the admin Drive ingest
> (`.github/workflows/ingest-run-once.yml` → `scripts/p1b_ingest_drive_server.py`).
> See [`docs/INGEST_EXCLUSION_RULE.md`](../INGEST_EXCLUSION_RULE.md).

From "env vars just landed on Render" to "all Drive-linked projects re-imported, re-indexed, and verified".

**Scope:** this runbook only re-imports files from Google Drive. It does **not** flip the GK lexical fold flag, run feature-matrix sweeps, or touch eval oracles. Those resume after this runbook is complete.

**Pre-requisites**
- Render service ID: `srv-d8hdc6ek1jcs739rq5sg`
- Render API key exported as `RENDER_API_KEY`
- Fork admin API key exported as `FORK_API_KEY` (value is `CEREBRUM_MASTER_KEY`)
- A Google Drive service-account JSON key with read access to the project folders
- The parent Drive folder ID that contains the per-project folders

---

## Step 1 â€” Upload the service-account key to Render

You can do this in the Render dashboard (Environment â†’ Add Environment Variable) or via the API. The value must be either an absolute file path inside the container **or** the JSON content on one line.

**Option A: inline JSON (dashboard copy-paste)**

Set `GDRIVE_SERVICE_ACCOUNT_JSON` to the full service-account JSON on a single line.

**Option B: Render API**

```bash
export RENDER_API_KEY="rnd_..."

# Upload the key as a single-line JSON value.
curl -s -X POST \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"key\":\"GDRIVE_SERVICE_ACCOUNT_JSON\",\"value\":$(jq -c . /path/to/service-account.json)}" \
  "https://api.render.com/v1/services/srv-d8hdc6ek1jcs739rq5sg/env-vars"
```

> **STOP:** Do not continue until the deploy that picks up `GDRIVE_SERVICE_ACCOUNT_JSON` is live. Render auto-deploys on env-var changes.

---

## Step 2 â€” Auto-discover the folder mapping

Run the discovery script locally. It uses the service account to list child folders of the parent and proposes a `GDRIVE_PROJECT_FOLDERS` value.

```bash
cd /path/to/The_Fork
export FORK_API_KEY="<FORK_API_KEY -- take it from your .env, never paste it into a doc>"
export GDRIVE_SERVICE_ACCOUNT_JSON='$(cat /path/to/service-account.json)'

.venv/Scripts/python.exe scripts/inspect_drive_projects.py --discover <PARENT_FOLDER_ID>
```

The script prints:
1. A proposed `GDRIVE_PROJECT_FOLDERS` env-var value.
2. A human-readable match table with confidence (`exact` / `fuzzy` / `unmatched`).
3. A list of unmatched Drive folders.
4. An anchor validation warning if `example_infra_pack` is not mapped to `1ExampleFolderId001xxxxxxxxxxxxxx`.

> **STOP â€” operator confirmation required:** Review the proposed mapping. If any project is `unmatched`, locate its folder manually in Drive and add the `project_id:folder_id` pair to the env value before continuing. If the anchor check fails, suspect the wrong parent folder.

---

## Step 3 â€” Set the confirmed `GDRIVE_PROJECT_FOLDERS` on Render

After confirming the mapping, set it on Render.

```bash
export GDRIVE_PROJECT_FOLDERS="projects_folder:<folder-id>,example_estate:<folder-id>,example_estate_2:<folder-id>,example_infra_pack:1ExampleFolderId001xxxxxxxxxxxxxx,5c13510e:<folder-id>"

curl -s -X POST \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"key\":\"GDRIVE_PROJECT_FOLDERS\",\"value\":\"$GDRIVE_PROJECT_FOLDERS\"}" \
  "https://api.render.com/v1/services/srv-d8hdc6ek1jcs739rq5sg/env-vars"
```

> **STOP:** Wait for the Render deploy that picks up `GDRIVE_PROJECT_FOLDERS` to be live. Verify with:
>
> ```bash
> curl -s -H "Authorization: Bearer $RENDER_API_KEY" \
>   "https://api.render.com/v1/services/srv-d8hdc6ek1jcs739rq5sg/env-vars" | \
>   grep -E 'GDRIVE_SERVICE_ACCOUNT_JSON|GDRIVE_PROJECT_FOLDERS'
> ```

---

## Step 4 â€” Decide what to do with existing empty/partial projects

Some projects still have document rows but no file blobs (e.g. `master_corpus` / `projects_folder`, `example_estate`). The Drive hydration path skips files it has already seen, so you must clear the seen-file cache to force a full re-import.

> **STOP â€” operator decision required:** The next step only clears the gdrive seen cache. It is safe. However, re-importing into a project that already has empty document rows will add new documents alongside the old ones, inflating `document_count`. If you want a clean re-import instead, archive the old project first (which hides it but preserves its RAG chunks) and create a new project from Drive. Do **not** hard-delete any project that has live RAG chunks.

Default (safe): clear the seen cache and re-import into the existing projects.

---

## Step 5 â€” Clear the Drive-seen cache

Open a Render web shell for the service and run:

```bash
rm -f /app/data/hydration_gdrive_seen.json
```

This cache is recreated on the next hydration pass.

---

## Step 6 â€” Trigger hydration for all Drive-linked projects

Use the admin hydration endpoint. The request returns immediately; the import runs in the background.

```bash
export FORK_API_KEY="<FORK_API_KEY -- take it from your .env, never paste it into a doc>"

curl -s -X POST \
  -H "Authorization: Bearer $FORK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "project_ids": [
      "projects_folder",
      "example_infra_pack",
      "5c13510e",
      "example_estate",
      "example_estate_2"
    ]
  }' \
  "https://the-fork-jn3t.onrender.com/v1/hydration/run"
```

> **Note:** `master_corpus` is a virtual alias backed by `projects_folder`; hydrating `projects_folder` populates the master corpus.

---

## Step 7 â€” Poll hydration progress

Poll the per-project hydration history. A pass is complete when `latest` shows a recent `finished_at`.

```bash
for pid in projects_folder example_infra_pack 5c13510e example_estate example_estate_2; do
  echo "=== $pid ==="
  curl -s -H "Authorization: Bearer $FORK_API_KEY" \
    "https://the-fork-jn3t.onrender.com/v1/hydration/latest?scope=project&project_id=$pid"
  echo
done
```

Large folders (especially `projects_folder`) may take hours. The learning_engine import is bounded by `HYDRATION_MAX_ATTACH_SIZE` (default 50 MB) and the allowed extension list; files outside those limits are skipped and logged.

---

## Step 8 â€” Verify the re-import against the damage snapshot

Re-run the inspector. Compare `docs`, `files_present`, and `total_chunks` with the pre-re-import snapshot in `docs/recovery/re_import_manifest.md`.

```bash
.venv/Scripts/python.exe scripts/inspect_drive_projects.py
```

Expected outcome for each project:

| project | expected after re-import |
|---|---|
| `projects_folder` | `files_present` == `docs` (> 0); chunks appear after indexing |
| `example_infra_pack` | `files_present` == `docs`; `zero_chunk_docs` == 0 |
| `5c13510e` | `files_present` == `docs`; `zero_chunk_docs` == 0 |
| `example_estate` | `files_present` == `docs`; `zero_chunk_docs` == 0 |
| `example_estate_2` | `files_present` == `docs`; `zero_chunk_docs` == 0 |

---

## Step 9 â€” ZERO_CHUNK tripwire and re-index

If any project shows `zero_chunk_docs > 0`, re-index those documents. First try the per-project re-index endpoint:

```bash
for pid in projects_folder example_infra_pack 5c13510e example_estate example_estate_2; do
  echo "re-indexing $pid ..."
  curl -s -X POST \
    -H "Authorization: Bearer $FORK_API_KEY" \
    "https://the-fork-jn3t.onrender.com/v1/admin/debug/project-reindex?project_id=$pid"
  echo
done
```

> **STOP â€” destructive decision:** `project-reindex` re-extracts and re-chunks every document. It is the right call for projects whose blobs were missing. For projects that already had partial data and working chunks, prefer per-document re-indexing via `/v1/admin/debug/doc-reindex?project_id=<id>&document_id=<id>` to avoid rebuilding healthy indexes.

After re-indexing, re-run the inspector until `zero_chunk_docs` is 0.

---

## Step 10 â€” Final health check

Run a quick chat turn against the master corpus and a Drive-approved project to confirm the normal chat path is healthy.

```bash
.venv/Scripts/python.exe scripts/fork_cli.py chat \
  "what does the the client project project execution plan cover?" \
  --project master_corpus --events

.venv/Scripts/python.exe scripts/fork_cli.py chat \
  "process the bill of quantities and give me the total value" \
  --project 5c13510e --events
```

Both should return a `route` event, a `tool_call`/`tool_result` pair, and an answer with `answer_chars >= 1500`.

---

## What happens next

Once this runbook is green:
1. Run `scripts/seed_fixtures.py` with `FIXTURES_DIR` set to seed `FIXTURE â€” BOQ` and `FIXTURE â€” Programme+Drawings`.
2. Resume the feature-matrix sweep (`scripts/feature_matrix_sweep.py --auto-seed`).
3. Activate the GK lexical fold flag (Step 2) and run the acceptance battery.
4. Continue the golden-set gate and embedder audit as planned.
