# Re-import manifest — Drive-linked projects

Generated: 2026-07-06T17:25 UTC

## Current damage snapshot

| project_id | name | origin | docs | files_present | zero_chunk_docs | total_chunks | folder_id | status |
|---|---|---|---|---|---|---|---|---|
| master_corpus | Master Corpus | admin_drive_approved | 2713 | 0 | N/A | N/A | TBD | **lost blobs** — re-import source |
| projects_folder | Projects Folder | user_create | 2713 | 0 | N/A | N/A | TBD | backing corpus for `master_corpus`; same blob loss |
| example_estate | Ha Long Xanh | admin_drive_approved | 62 | 2 | 61 | 1 | TBD | mostly missing blobs |
| example_estate_2 | Ha Long Xanh | admin_drive_approved | 81 | 81 | 61 | 69 | TBD | blobs present but mostly un-indexed |
| example_infra_pack | the client project | admin_drive_approved | 4 | 4 | 2 | 160 | `1ExampleFolderId001xxxxxxxxxxxxxx` | partial index |
| 5c13510e | the client project Bills of Quantities | admin_drive_approved | 4 | 4 | 1 | 179 | TBD | healthy; used as fixture source |
| training_material | Training Material | user_create | 246 | 5 | 95 | 14 | n/a | user_create, not Drive-linked |

## What the metadata says

- Postgres metadata (project rows, document rows) **survived**.
- `/app/data` persistent disk **survives deploys** (double-deploy survival test passed 2026-07-06).
- File blobs for the earliest imports (`master_corpus` / `projects_folder`, `example_estate`) are gone; later imports (`example_infra_pack`, `5c13510e`, `example_estate_2`) retained blobs.
- Drive folder IDs are **not stored in the database** (no `folder_id` column, no project fact, no surviving audit log on disk). They must come from operator records.

## Required to proceed with re-import

1. `GDRIVE_SERVICE_ACCOUNT_JSON` env var on Render (service-account key with Drive read access).
2. Complete `GDRIVE_PROJECT_FOLDERS` mapping, e.g.:

   ```text
   master_corpus:<folder-id>,projects_folder:<folder-id>,example_estate:<folder-id>,example_estate_2:<folder-id>,example_infra_pack:1ExampleFolderId001xxxxxxxxxxxxxx,5c13510e:<folder-id>
   ```

3. Sibling packs under the same parent as `the client project` (the owner to enumerate).

## Recovery sequence

1. ✅ Persistent disk proven with double-deploy survival test.
2. ⬜ Set `GDRIVE_SERVICE_ACCOUNT_JSON` + `GDRIVE_PROJECT_FOLDERS` on Render.
3. ⬜ Run full re-import / re-index of all Drive-linked projects (use `scripts/inspect_drive_projects.py` to verify counts).
4. ⬜ Re-run `scripts/seed_fixtures.py` for fixtures.
5. ⬜ Resume gate sequence (feature matrix, Step 2 fold activation, golden set, embedder audit).

## Verification targets after re-import

- `master_corpus`: files_present == 2713, total_chunks > 0.
- Each Drive-linked project: zero_chunk_docs == 0.
